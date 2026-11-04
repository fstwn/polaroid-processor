"""Photo-area crop, slice 3: the hand-set window and "no photo crop" (manifest, API, CLI, MCP, review page backend).

The user's window choices live in ``window_fix``, which only a fix writes; ``window`` is the batch's automatic window.
"""
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import tifffile

from polaroid_processor import api, mcp_server, pipeline
from polaroid_processor.cli import main
from polaroid_processor.manifest import Manifest
from polaroid_processor.pipeline import photo_image, reevaluate, settings_from_dict
from polaroid_processor.scan_window import (accept_valid, effective_window, fix_outside, nominal_window, outline_id,
                                            window_current, window_state)

from synth import PX_PER_MM
from test_app import AppClient
from test_photo_crop import ODD, PPI, ROUND, SQUARE, entry, make_named, run, true_window, write_scans
from test_review_server import Page

CORNERS = ("tl", "tr", "br", "bl")


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    return write_scans(tmp_path_factory.mktemp("win_scans"))


@pytest.fixture(scope="module")
def project(src, tmp_path_factory):
    """sq (square window), rd (round window), nw (no window found), exported with photo_crop=tiff."""
    out = tmp_path_factory.mktemp("win_out")
    run(src, out, photo_crop="tiff")
    return out


@pytest.fixture
def out(project, tmp_path):
    dest = tmp_path / "out"
    shutil.copytree(project, dest)
    return dest


def photo_files(out, stem):
    return [p for p in (out / "photo").rglob(f"{stem}.*") if p.is_file()]


def photo_size(path):
    with tifffile.TiffFile(str(path)) as tf:
        h, w = tf.pages[0].shape[:2]
    return w, h


def corners(quad):
    return {n: [float(x), float(y)] for n, (x, y) in zip(CORNERS, quad)}


def hand(out, name):
    """The user's window choice of a scan (None when there is none)."""
    return ((entry(out, name).get("window_fix") or {}).get("window"))


# --------------------------------------------------------------------------- manifest: provenance and revert
def test_a_hand_set_window_is_stored_with_provenance_and_can_be_reverted(out):
    auto = entry(out, "sq.tif")["window"]
    rec = api.fix_scan(str(out), "sq.tif", window_nudges={"br": [-8.0, -6.0]}, author="agent", note="border off")
    e = entry(out, "sq.tif")
    assert e["window"] == auto                                     # the batch's field is not touched by a fix
    w = hand(out, "sq.tif")
    assert w["source"] == "manual" and w["found"] and w["kind"] == "square" and w["author"] == "agent"
    assert w["note"] == "border off" and w["at"]
    assert np.allclose(np.array(w["quad"])[2], np.array(auto["quad"])[2] + [-8.0, -6.0], atol=0.01)
    assert np.allclose(np.array(w["quad"])[[0, 1, 3]], np.array(auto["quad"])[[0, 1, 3]], atol=0.01)
    h = e["history"][-1]
    assert h["action"] == "window" and h["author"] == "agent" and h["window_before"] is None and h["window_after"] == w
    assert effective_window(e) == w and window_current(e) and e["status"] == "ok"
    assert rec["window"]["source"] == "manual" and rec["window"]["state"] == "found" and rec["window"]["author"] == "agent"
    assert not e.get("override")                                   # the print outline is untouched
    # a second edit starts from the hand-set window; revert brings the automatic one back
    api.fix_scan(str(out), "sq.tif", window_nudges={"tl": [4.0, 4.0]}, author="human")
    assert entry(out, "sq.tif")["history"][-1]["window_before"] == w
    api.fix_scan(str(out), "sq.tif", reset_window=True, author="human")
    e = entry(out, "sq.tif")
    assert e["window_fix"] is None and e["window"] == auto and effective_window(e) == auto
    assert e["history"][-1]["action"] == "window-reset" and e["history"][-1]["window_before"]["source"] == "manual"


def test_window_corners_can_be_set_and_the_kind_changed(out):
    truth = true_window("sq", SQUARE)
    api.fix_scan(str(out), "sq.tif", window_sets=corners(truth + [3.0, 2.0]), author="agent")
    w = hand(out, "sq.tif")
    assert np.allclose(np.array(w["quad"]), truth + [3.0, 2.0], atol=0.01) and w["kind"] == "square"
    api.fix_scan(str(out), "sq.tif", window_kind="round", author="agent")
    w2 = hand(out, "sq.tif")
    assert w2["kind"] == "round" and w2["quad"] == w["quad"] and w2["source"] == "manual"


def test_bad_window_requests_are_refused(out, src, tmp_path):
    with pytest.raises(api.UsageError, match="inside the print outline"):
        api.fix_scan(str(out), "sq.tif", window_sets={"tl": [5.0, 5.0]})
    with pytest.raises(api.UsageError, match="convex"):
        api.fix_scan(str(out), "sq.tif", window_sets={"tl": [10 ** 9, 5.0]})
    with pytest.raises(api.UsageError, match="window_kind"):
        api.fix_scan(str(out), "sq.tif", window_kind="oval")
    with pytest.raises(api.UsageError, match="cannot be combined"):
        api.fix_scan(str(out), "sq.tif", no_photo_crop=True, window_nudges={"tl": [1, 1]})
    with pytest.raises(api.UsageError, match="window_set"):
        api.fix_scan(str(out), "sq.tif", window_sets={"top": [1, 1]})
    with pytest.raises(api.UsageError, match="no hand-set window"):
        api.fix_scan(str(out), "sq.tif", reset_window=True)           # nothing to take away
    assert entry(out, "sq.tif").get("window_fix") is None             # nothing was stored
    off = tmp_path / "off"
    run(src, off)                                                    # photo crop off
    with pytest.raises(api.UsageError, match="photo_crop on"):
        api.fix_scan(str(off), "sq.tif", no_photo_crop=True)


def test_counter_clockwise_window_corners_are_refused(out):
    """Corners in the order tl, bl, br, tr make a convex quadrilateral, but the photo would be mirrored."""
    tl, tr, br, bl = true_window("sq", SQUARE)
    with pytest.raises(api.UsageError, match="clockwise"):
        api.fix_scan(str(out), "sq.tif", window_sets=corners([tl, bl, br, tr]))
    assert entry(out, "sq.tif").get("window_fix") is None


# --------------------------------------------------------------------------- hand-set window and outline fixes
def test_a_hand_set_window_survives_an_outline_fix_that_keeps_it_inside_and_the_export_uses_it(src, out):
    api.fix_scan(str(out), "sq.tif", window_sets=corners(true_window("sq", SQUARE) + [[12, 9], [-10, 9], [-10, -9], [12, -9]]),
                 author="agent")
    w = hand(out, "sq.tif")
    time.sleep(1.1)
    api.fix_scan(str(out), "sq.tif", shifts={"right": [3.0]}, author="agent")      # the outline moves a little
    e = entry(out, "sq.tif")
    assert hand(out, "sq.tif") == w and window_current(e) and e["window"]["outline"] != outline_id(Manifest.effective_quad(e))
    assert e["status"] == "ok"
    before = photo_size(out / "photo" / "sq.tif")
    summary = run(src, out)                                          # the run measures the automatic window, not the hand-set one
    e = entry(out, "sq.tif")
    assert hand(out, "sq.tif") == w and e["window"]["source"] == "auto" and summary["exported"] >= 1
    # the photo is cut from the hand-set window: 22 px narrower and 18 px lower than the automatic one
    width, height = photo_size(out / "photo" / "sq.tif")
    assert width - before[0] == pytest.approx(-22, abs=3) and height - before[1] == pytest.approx(-18, abs=3)
    assert run(src, out)["exported"] == 0 and entry(out, "sq.tif")["export_key"]    # and the key belongs to it


def test_a_hand_set_window_the_outline_moved_away_from_is_flagged_not_dropped(src, out):
    api.fix_scan(str(out), "sq.tif", window_nudges={"br": [-4.0, -4.0]}, author="agent")
    w = hand(out, "sq.tif")
    assert entry(out, "sq.tif")["status"] == "ok"
    time.sleep(1.1)
    api.fix_scan(str(out), "sq.tif", shifts={"right": [-200.0]}, author="agent", note="print is narrower")
    e = entry(out, "sq.tif")
    assert hand(out, "sq.tif") == w and fix_outside(e) and effective_window(e) == e["window"]     # kept, not in force
    assert e["status"] == "needs_review" and "photo window set by hand lies outside the print outline" in e["warnings"]
    assert [h["action"] for h in e["history"][-2:]] == ["window", "fix"]                       # nothing was logged as dropped
    assert pipeline.window_record(e)["outside_outline"] and not pipeline.window_record(e).get("accepted")
    run(src, out)                                                  # a run leaves the choice alone and stays flagged
    e = entry(out, "sq.tif")
    assert hand(out, "sq.tif") == w and e["status"] == "needs_review" and fix_outside(e)
    # the outline back to the detected one: the window holds again and the scan is ok
    api.fix_scan(str(out), "sq.tif", reset=True, author="agent")
    e = entry(out, "sq.tif")
    assert not fix_outside(e) and effective_window(e) == w and e["status"] == "ok"
    # or the choice goes: back to the automatic window
    api.fix_scan(str(out), "sq.tif", shifts={"right": [-200.0]}, author="agent")
    assert entry(out, "sq.tif")["status"] == "needs_review"
    api.fix_scan(str(out), "sq.tif", reset_window=True, author="agent")
    assert entry(out, "sq.tif").get("window_fix") is None and not fix_outside(entry(out, "sq.tif"))


def test_an_automatic_window_is_still_found_again_when_the_outline_changes(src, out):
    old = entry(out, "sq.tif")["window"]
    time.sleep(1.1)
    api.fix_scan(str(out), "sq.tif", shifts={"right": [3.0]}, author="agent")
    assert not window_current(entry(out, "sq.tif"))
    run(src, out)
    w = entry(out, "sq.tif")["window"]
    assert w["source"] == "auto" and w["outline"] != old["outline"] and window_current(entry(out, "sq.tif"))


# --------------------------------------------------------------------------- the run does not overwrite a window fix
@pytest.mark.parametrize("where", ["a scan was detected", "the sizes are checked"])
def test_a_window_fix_saved_during_a_detect_run_is_kept_and_exported(src, out, monkeypatch, where):
    """A fix lands while a run re-detects every scan: after the scan's own export (the first), or while the sizes are
    checked (the second). The run must keep the choice, and the files must be cut from it with a key that says so."""
    before = photo_size(out / "photo" / "sq.tif")
    fixes = []

    def fix():
        if not fixes:
            fixes.append(True)      # first: the fix judges the scan again, which calls reference_sizes
            fixes.append(api.fix_scan(str(out), "sq.tif", window_nudges={"br": [-30.0, -24.0]}, author="agent"))

    def progress(key, what):
        if key == "sq.tif" and where == "a scan was detected":
            fix()

    real_sizes = pipeline.reference_sizes

    def reference_sizes(manifest):
        sizes = real_sizes(manifest)
        if where == "the sizes are checked":
            fix()
        return sizes

    monkeypatch.setattr(pipeline, "reference_sizes", reference_sizes)
    api.process([str(src)], str(out), api.make_settings(workers=1, ppi=PPI, redetect=True), progress=progress)
    monkeypatch.undo()
    assert fixes
    e = entry(out, "sq.tif")
    w = hand(out, "sq.tif")
    assert w["source"] == "manual" and w["author"] == "agent" and effective_window(e) == w and e["status"] == "ok"
    assert [h["action"] for h in e["history"]].count("window") == 1
    width, height = photo_size(out / "photo" / "sq.tif")
    # one corner moved: the crop's size is that of the mean edge lengths, half of the corner's move (30 and 24 px)
    assert width - before[0] == pytest.approx(-15, abs=3) and height - before[1] == pytest.approx(-12, abs=3)
    assert run(src, out)["exported"] == 0 and photo_size(out / "photo" / "sq.tif") == (width, height)


def test_a_no_photo_crop_choice_saved_during_a_run_is_kept(src, out, monkeypatch):
    fixes = []

    def progress(key, what):
        if key == "sq.tif" and not fixes:
            fixes.append(api.fix_scan(str(out), "sq.tif", no_photo_crop=True, author="agent"))

    api.process([str(src)], str(out), api.make_settings(workers=1, ppi=PPI, redetect=True), progress=progress)
    e = entry(out, "sq.tif")
    assert window_state(hand(out, "sq.tif")) == "none" and e["status"] == "ok" and not photo_files(out, "sq")
    assert e["window"]["found"]                                       # the automatic window is measured as ever


def test_override_hand_window_and_redetect_with_the_window_outside_the_detected_outline(src, out):
    """The hand-set window is judged against the outline the user fixed, not the one detected again."""
    det = np.array(entry(out, "sq.tif")["detection"]["quad"])
    api.fix_scan(str(out), "sq.tif", shifts={"left": [40.0]}, author="agent")        # outward: the fixed outline is larger
    new = true_window("sq", SQUARE)
    new[0][0] = new[3][0] = min(det[0][0], det[3][0]) - 15                           # outside the detected outline
    api.fix_scan(str(out), "sq.tif", window_sets=corners(new), author="agent")
    w = hand(out, "sq.tif")
    api.process([str(src)], str(out), api.make_settings(workers=1, ppi=PPI, redetect=True))
    e = entry(out, "sq.tif")
    assert e["override"] and hand(out, "sq.tif") == w and effective_window(e) == w and e["status"] == "ok"
    assert [h["action"] for h in e["history"]].count("window") == 1
    assert run(src, out)["exported"] == 0


def test_turning_the_crop_off_and_on_keeps_a_hand_set_window(src, out):
    api.fix_scan(str(out), "sq.tif", window_nudges={"tl": [6.0, 6.0]}, author="agent")
    w = hand(out, "sq.tif")
    run(src, out, photo_crop="off")
    assert hand(out, "sq.tif") == w
    run(src, out, photo_crop="tiff")
    e = entry(out, "sq.tif")
    assert hand(out, "sq.tif") == w and effective_window(e) == w and e["status"] == "ok"
    assert photo_files(out, "sq")


def test_a_changed_scan_drops_the_hand_set_window_and_the_choice_of_no_photo_crop(src, out, tmp_path):
    inp = tmp_path / "in"
    shutil.copytree(src, inp)
    api.fix_scan(str(out), "sq.tif", window_nudges={"tl": [6.0, 6.0]}, author="agent")
    api.fix_scan(str(out), "nw.tif", no_photo_crop=True, author="agent")
    assert hand(out, "sq.tif") and hand(out, "nw.tif")
    tifffile.imwrite(str(inp / "sq.tif"), make_named("rd")[0], photometric="rgb")      # another picture under the same name
    run(inp, out)
    e = entry(out, "sq.tif")
    assert e.get("window_fix") is None and e["history"][-1]["action"] == "window-drop" and e["history"][-1]["author"] == "system"
    assert e["history"][-1]["window_before"]["source"] == "manual"
    assert window_state(hand(out, "nw.tif")) == "none"               # an unchanged scan keeps its choice


# --------------------------------------------------------------------------- no photo crop
def test_no_photo_crop_is_ok_and_exports_no_photo_files(src, out):
    assert entry(out, "nw.tif")["status"] == "needs_review"
    api.fix_scan(str(out), "nw.tif", no_photo_crop=True, author="agent", note="no photo on this print")
    e = entry(out, "nw.tif")
    w = hand(out, "nw.tif")
    assert w["source"] == "manual" and not w["found"] and w["reason"] == "no photo crop by choice" and w["author"] == "agent"
    assert not e["window"]["found"]                                  # the automatic window is as it was
    assert e["status"] == "ok" and not any(x.startswith("photo window") for x in e["warnings"])
    assert e["history"][-1]["action"] == "window-none"
    run(src, out)
    e = entry(out, "nw.tif")
    assert e["status"] == "ok" and hand(out, "nw.tif") == w and not photo_files(out, "nw")
    assert (out / "high" / "nw.jpg").exists() and set(e["outputs"]) == {"tiff", "high", "web"}
    # an outline fix keeps the choice
    time.sleep(1.1)
    api.fix_scan(str(out), "nw.tif", shifts={"left": [2.0]}, author="agent", export=True)
    e = entry(out, "nw.tif")
    assert hand(out, "nw.tif") == w and e["status"] == "ok" and not photo_files(out, "nw")


def test_no_photo_crop_on_a_scan_with_a_window_removes_its_photo_files(src, out):
    assert photo_files(out, "sq")
    api.fix_scan(str(out), "sq.tif", no_photo_crop=True, author="human", export=True)
    e = entry(out, "sq.tif")
    assert e["status"] == "ok" and set(e["outputs"]) == {"tiff", "high", "web"} and not photo_files(out, "sq")
    assert photo_files(out, "rd")                                    # others keep theirs
    api.fix_scan(str(out), "sq.tif", reset_window=True, export=True)
    assert photo_files(out, "sq") and entry(out, "sq.tif")["window"]["found"]


# --------------------------------------------------------------------------- missing window: explicit only
def test_an_outline_only_fix_leaves_a_missing_window_needs_review(src, out):
    assert entry(out, "nw.tif")["status"] == "needs_review"
    time.sleep(1.1)
    api.fix_scan(str(out), "nw.tif", shifts={"right": [3.0]}, author="agent")          # no export: judged at once
    e = entry(out, "nw.tif")
    assert e["status"] == "needs_review" and any(w.startswith("photo window not found") for w in e["warnings"])
    run(src, out)
    e = entry(out, "nw.tif")
    assert e["status"] == "needs_review" and not e["window"]["found"] and not photo_files(out, "nw")
    assert (out / "high" / "nw.jpg").exists()                         # the full print is exported as always
    # a repeated fix does not clear it either
    time.sleep(1.1)
    api.fix_scan(str(out), "nw.tif", shifts={"right": [-1.0]}, author="agent", export=True)
    assert entry(out, "nw.tif")["status"] == "needs_review"


def test_accept_still_exports_the_print_without_a_photo_crop(src, out):
    api.fix_scan(str(out), "nw.tif", accept=True, author="agent", note="no window on this print", export=True)
    e = entry(out, "nw.tif")
    acc = e["window_fix"]["accepted"]
    assert e["status"] == "ok" and acc["author"] == "agent" and acc["outline"] == outline_id(Manifest.effective_quad(e))
    assert any(w.startswith("photo window not found") for w in e["warnings"]) and not photo_files(out, "nw")
    assert (out / "high" / "nw.jpg").exists()
    assert run(src, out)["exported"] == 0 and entry(out, "nw.tif")["status"] == "ok"        # a run keeps it ok
    time.sleep(1.1)
    api.fix_scan(str(out), "nw.tif", shifts={"right": [3.0]}, author="agent")       # a later outline fix: accepted no more
    e = entry(out, "nw.tif")
    assert e["status"] == "needs_review" and not accept_valid(e)


def test_an_accept_survives_the_window_being_measured_again(src, out):
    api.fix_scan(str(out), "nw.tif", accept=True, author="agent")
    assert entry(out, "nw.tif")["status"] == "ok"
    api.process([str(src)], str(out), api.make_settings(workers=1, ppi=PPI, redetect=True))      # every window is new
    e = entry(out, "nw.tif")
    assert e["status"] == "ok" and accept_valid(e) and not e["window"]["found"] and e["window"]["accept_needed"]
    run(src, out, flatten="off")                                       # a change of settings: again
    assert entry(out, "nw.tif")["status"] == "ok"


def test_an_accept_given_with_an_outline_fix_or_after_one_holds_for_the_new_outline(src, out):
    time.sleep(1.1)
    api.fix_scan(str(out), "nw.tif", shifts={"right": [3.0]}, accept=True, author="agent")      # fix and accept in one call
    e = entry(out, "nw.tif")
    assert e["status"] == "ok" and accept_valid(e)
    run(src, out)                                                     # the window is looked for at the new outline: not found
    e = entry(out, "nw.tif")
    assert e["status"] == "ok" and accept_valid(e) and not e["window"]["found"] and window_current(e)
    # Save, then Confirm, in separate calls with the export off in between: the window shown is stale and missing
    time.sleep(1.1)
    api.fix_scan(str(out), "nw.tif", shifts={"left": [2.0]}, author="agent")
    e = entry(out, "nw.tif")
    assert e["status"] == "needs_review" and not accept_valid(e)             # the earlier accept lapsed with the outline
    api.fix_scan(str(out), "nw.tif", accept=True, author="agent")
    assert accept_valid(entry(out, "nw.tif"))
    run(src, out)
    e = entry(out, "nw.tif")
    assert e["status"] == "ok" and accept_valid(e) and not e["window"]["found"]


def test_an_accept_on_a_scan_whose_window_was_not_looked_for_yet_stores_no_mark(src, out):
    """The crop was just turned on, or a run was cancelled before the window pass: nothing was shown to accept."""
    m = Manifest(str(out))
    del m.get("nw.tif")["window"]
    m.save()
    api.fix_scan(str(out), "nw.tif", accept=True, author="agent")
    e = entry(out, "nw.tif")
    assert e.get("window_fix") is None and "window" not in e
    run(src, out)                                                     # the window is not found: shown, not covered
    e = entry(out, "nw.tif")
    assert not e["window"]["found"] and e["status"] == "needs_review" and not accept_valid(e)


def test_resetting_the_outline_revokes_an_accept(out):
    """An accept stores the detected outline as the override, so the outline id is the same after the reset."""
    api.fix_scan(str(out), "nw.tif", accept=True, author="agent")
    assert entry(out, "nw.tif")["status"] == "ok"
    api.fix_scan(str(out), "nw.tif", reset=True, author="agent")
    e = entry(out, "nw.tif")
    assert not e.get("override") and not accept_valid(e) and e["status"] == "needs_review"
    api.fix_scan(str(out), "nw.tif", reset=True, accept=True, author="agent")        # reset and accept in one call still works
    assert entry(out, "nw.tif")["status"] == "ok"


def test_a_changed_scan_loses_its_accept_mark_too(src, out, tmp_path):
    inp = tmp_path / "in"
    shutil.copytree(src, inp)
    api.fix_scan(str(out), "nw.tif", accept=True, author="agent")
    assert entry(out, "nw.tif")["window_fix"]["accepted"]
    tifffile.imwrite(str(inp / "nw.tif"), make_named("sq")[0], photometric="rgb")      # another picture under the same name
    run(inp, out)
    e = entry(out, "nw.tif")
    assert e.get("window_fix") is None and not accept_valid(e)
    assert not any(h["action"] == "window-drop" for h in e["history"])               # no window choice was dropped, none logged


def test_an_accept_given_while_a_window_is_found_does_not_cover_a_later_miss(out):
    api.fix_scan(str(out), "rd.tif", accept=True, author="agent")
    assert entry(out, "rd.tif").get("window_fix") is None                            # nothing to accept: nothing stored
    m = Manifest(str(out))                                                            # the next run finds nothing here
    e = m.get("rd.tif")
    e["window"] = {"found": False, "reason": "no border", "kind": None, "format": None, "notes": [],
                   "outline": outline_id(Manifest.effective_quad(e)), "source": "auto", "accept_needed": True}
    reevaluate(m, ["rd.tif"], settings_from_dict(m.data["settings"]))
    assert e["status"] == "needs_review" and any(w.startswith("photo window not found") for w in e["warnings"])
    # a hand-set window over a missing automatic window: Confirm stores no accept, so taking the window away is shown
    api.fix_scan(str(out), "nw.tif", window_kind="square", window_sets=corners(true_window("nw", ODD)), author="agent")
    api.fix_scan(str(out), "nw.tif", accept=True, author="agent")
    assert entry(out, "nw.tif")["status"] == "ok" and not accept_valid(entry(out, "nw.tif"))
    api.fix_scan(str(out), "nw.tif", reset_window=True, author="agent")
    assert entry(out, "nw.tif")["status"] == "needs_review"


def test_an_accept_still_covers_a_hand_window_outside_the_outline_and_lapses_with_the_outline(src, out):
    api.fix_scan(str(out), "sq.tif", window_nudges={"br": [-4.0, -4.0]}, author="agent")
    time.sleep(1.1)
    api.fix_scan(str(out), "sq.tif", shifts={"right": [-200.0]}, author="agent")
    assert entry(out, "sq.tif")["status"] == "needs_review"
    api.fix_scan(str(out), "sq.tif", accept=True, author="agent")
    e = entry(out, "sq.tif")
    assert fix_outside(e) and accept_valid(e) and e["status"] == "ok"
    time.sleep(1.1)
    api.fix_scan(str(out), "sq.tif", shifts={"right": [2.0]}, author="agent")
    assert entry(out, "sq.tif")["status"] == "needs_review"


def test_a_window_fix_during_the_export_pass_leaves_an_honest_key(src, out):
    """A run exports sq again (force: pass B); a window fix lands while it works. The files are cut from the old window,
    the key says so, and the next run cuts them again."""
    before = photo_size(out / "photo" / "sq.tif")
    fixes = []

    def progress(key, what):
        if key == "sq.tif" and what == "exported" and not fixes:
            fixes.append(api.fix_scan(str(out), "sq.tif", window_nudges={"tl": [20.0, 16.0]}, author="agent"))

    api.process([str(src)], str(out), api.make_settings(workers=1, ppi=PPI, force=True), progress=progress)
    assert fixes and hand(out, "sq.tif")
    assert photo_size(out / "photo" / "sq.tif") == before                           # still cut from the old window
    assert run(src, out)["exported"] >= 1
    width, height = photo_size(out / "photo" / "sq.tif")
    assert width - before[0] == pytest.approx(-10, abs=3) and height - before[1] == pytest.approx(-8, abs=3)
    assert run(src, out)["exported"] == 0


def test_a_slice_2_accept_lapses_when_the_window_is_measured_again(src, out):
    """Pins the old-manifest decision: no handling, one Confirm restores it."""
    m = Manifest(str(out))
    e = m.get("nw.tif")
    del e["window"]["accept_needed"]                                                 # a slice 2 window
    m.set_override("nw.tif", Manifest.effective_quad(e), "human", "", "accept")      # accepted the old way
    reevaluate(m, ["nw.tif"], settings_from_dict(m.data["settings"]))
    m.save()
    assert entry(out, "nw.tif")["status"] == "ok"
    api.process([str(src)], str(out), api.make_settings(workers=1, ppi=PPI, redetect=True))
    e = entry(out, "nw.tif")
    assert e["window"]["accept_needed"] and e["status"] == "needs_review"
    api.fix_scan(str(out), "nw.tif", accept=True, author="human")
    assert entry(out, "nw.tif")["status"] == "ok"


def test_a_window_fix_outside_the_outline_is_shown_by_inspect_and_the_records(out):
    api.fix_scan(str(out), "sq.tif", window_nudges={"br": [-4.0, -4.0]}, author="agent")
    w = hand(out, "sq.tif")
    time.sleep(1.1)
    api.fix_scan(str(out), "sq.tif", shifts={"right": [-200.0]}, author="agent")
    rec, _ = api.inspect_scan("sq.tif", str(out), of="window", write=False)
    assert "outside the print outline" in rec["window_note"] and rec["window_kind"] == "square"
    assert rec["window_corners"]["br"] == pytest.approx(list(np.round(w["quad"][2], 1)), abs=0.06)      # the hand corners
    assert rec["window"]["outside_outline"]["quad"] == w["quad"]
    p = Page(out, show_all=True)
    try:
        info = {s["key"]: s for s in p.json("/api/scans")[1]["scans"]}["sq.tif"]["window"]
        assert info["outside"] and info["cut"] == "auto" and info["quad"] == w["quad"]
    finally:
        p.close()


def test_a_window_placed_by_hand_resolves_a_missing_window_and_is_exported(src, out):
    """Where none was found the fix starts from the nominal window of the format, then the corners are set."""
    nominal = nominal_window(Manifest.effective_quad(entry(out, "nw.tif")), "square")
    assert nominal["format"] == "polaroid-600" and nominal["size_mm"] == [76.9, 78.9]
    truth = true_window("nw", ODD)
    api.fix_scan(str(out), "nw.tif", window_kind="square", window_sets=corners(truth), author="agent", note="found by eye")
    e = entry(out, "nw.tif")
    assert e["status"] == "ok" and hand(out, "nw.tif")["size_mm"] == pytest.approx([40, 40], abs=0.3)
    run(src, out)
    w, h = photo_size(out / "photo" / "nw.tif")
    assert w == pytest.approx((40 - 0.6) * PX_PER_MM, abs=0.012 * 40 * PX_PER_MM + 2) and abs(w - h) <= 3
    assert (out / "photo" / "high" / "nw.jpg").exists()
    # revert: the (not found) automatic window is back and the scan needs review again
    api.fix_scan(str(out), "nw.tif", reset_window=True)
    assert entry(out, "nw.tif")["status"] == "needs_review"


def test_the_nominal_window_sits_where_the_synthetic_window_is(out):
    """The nominal placement from the outline alone lands within a millimetre of the true window."""
    quad = Manifest.effective_quad(entry(out, "sq.tif"))
    nom = np.array(nominal_window(quad, "square")["quad"])
    assert np.abs(nom - true_window("sq", SQUARE)).max() < 1.0 * PX_PER_MM
    rnd = np.array(nominal_window(Manifest.effective_quad(entry(out, "rd.tif")), "round")["quad"])
    assert np.abs(rnd - true_window("rd", ROUND)).max() < 1.0 * PX_PER_MM


def test_the_nominal_window_takes_the_format_the_finder_picked(out):
    quad = Manifest.effective_quad(entry(out, "sq.tif"))
    assert nominal_window(quad, "square", "polaroid-go")["format"] == "polaroid-go"
    assert nominal_window(quad, "square", "no-such-format")["format"] == "polaroid-600"     # unknown: from the size


# --------------------------------------------------------------------------- round windows that are not square
def test_a_round_window_that_is_not_square_is_cut_as_the_ellipse_it_is_drawn_as():
    rgb = np.full((400, 400, 3), 200, np.uint8)
    wide = [[50.0, 150.0], [350.0, 150.0], [350.0, 250.0], [50.0, 250.0]]            # 300 x 100
    _, alpha = photo_image(rgb, {"kind": "round", "quad": wide}, 10.0, 0.0, "cubic")
    h, w = alpha.shape
    assert abs(w - 300) <= 1 and abs(h - 100) <= 1
    assert alpha[h // 2, 10] > 250 and alpha[h // 2, w - 11] > 250                    # on the long axis, inside the ellipse
    assert alpha[2, 20] == 0 and alpha[h - 3, w - 21] == 0                            # outside it
    _, square_alpha = photo_image(rgb, {"kind": "round", "quad": [[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]]},
                                  10.0, 0.0, "cubic")
    h, w = square_alpha.shape
    yy, xx = np.mgrid[:h, :w]
    circle = np.clip((h + w) / 4 - np.hypot(xx + 0.5 - w / 2, yy + 0.5 - h / 2) + 0.5, 0, 1)      # a square crop: the circle, as before
    assert np.abs(square_alpha.astype(float) / 255 - circle).max() < 0.02


# --------------------------------------------------------------------------- manifests from before slice 3
def test_windows_from_before_the_accept_mark_keep_the_old_rule(out):
    """Before slice 3 any override counted as accepting a missing window. Such a window has no ``accept_needed``."""
    m = Manifest(str(out))
    e = m.get("nw.tif")
    del e["window"]["accept_needed"]                                 # how a slice 2 manifest holds it
    settings = settings_from_dict(m.data["settings"])
    reevaluate(m, ["nw.tif"], settings)
    assert e["status"] == "needs_review"                             # no override: still a review item
    m.set_override("nw.tif", Manifest.effective_quad(e), "human", "", "accept")      # accepted the old way
    reevaluate(m, ["nw.tif"], settings)
    assert e["status"] == "ok" and any(w.startswith("photo window not found") for w in e["warnings"])
    # a window made by this version is only accepted by an accept
    e["window"]["accept_needed"] = True
    reevaluate(m, ["nw.tif"], settings)
    assert e["status"] == "needs_review"
    m.set_window_accept("nw.tif", outline_id(Manifest.effective_quad(e)), "human")
    reevaluate(m, ["nw.tif"], settings)
    assert e["status"] == "ok"


# --------------------------------------------------------------------------- inspect and CLI
def test_inspect_of_window_draws_the_window_and_shows_the_nominal_one_where_none_was_found(out):
    rec, sheet = api.inspect_scan("sq.tif", str(out), of="window", write=False)
    assert rec["of"] == "window" and rec["window_kind"] == "square" and set(rec["window_corners"]) == set(CORNERS)
    assert rec["window_corners"]["tl"] == pytest.approx(list(np.round(entry(out, "sq.tif")["window"]["quad"][0], 1)), abs=0.06)
    assert sheet.shape[0] > 0 and (sheet[..., 2] > 200).any()          # cyan pixels
    rec, _ = api.inspect_scan("nw.tif", str(out), of="window", write=False)
    assert "nominal placement" in rec["window_note"] and rec["window"]["state"] == "missing"
    rec, _ = api.inspect_scan("rd.tif", str(out), of="window", write=False)
    assert rec["window_kind"] == "round"
    with pytest.raises(api.UsageError, match="of must be"):
        api.inspect_scan("sq.tif", str(out), of="photo")


def test_inspect_of_window_needs_the_photo_crop(src, tmp_path):
    off = tmp_path / "off"
    run(src, off, export=False)
    with pytest.raises(api.UsageError, match="photo_crop is on"):
        api.inspect_scan("sq.tif", str(off), of="window", write=False)


def test_inspect_of_window_on_a_landscape_outline_is_a_usage_error(out):
    m = Manifest(str(out))
    m.set_override("nw.tif", np.array([[100.0, 100.0], [1100.0, 100.0], [1100.0, 900.0], [100.0, 900.0]]), "human")
    m.save()          # no window found there: the nominal one is asked for, and a landscape print has none
    with pytest.raises(api.UsageError, match="landscape"):
        api.inspect_scan("nw.tif", str(out), of="window", write=False)


def test_cli_fix_and_inspect_with_the_window(out, capsys):
    base = entry(out, "sq.tif")["window"]["quad"]
    assert main(["fix", str(out), "sq", "--window-nudge", "br=-5,-5", "--window-set", "tl=%f,%f" % tuple(np.array(base[0]) + 2),
                 "--author", "agent", "--note", "cli", "--json"]) == 0
    capsys.readouterr()
    w = hand(out, "sq.tif")
    assert w["source"] == "manual" and w["author"] == "agent" and np.allclose(w["quad"][2], np.array(base[2]) - 5, atol=0.01)
    assert main(["fix", str(out), "sq", "--reset-window", "--json"]) == 0
    assert main(["fix", str(out), "nw", "--no-photo-crop", "--json"]) == 0
    assert entry(out, "nw.tif")["status"] == "ok"
    capsys.readouterr()
    assert main(["inspect", "nw", "-m", str(out), "--of", "window", "--json"]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["of"] == "window"
    with pytest.raises(SystemExit) as stop:                          # argparse refuses an unknown kind
        main(["fix", str(out), "nw", "--window-kind", "oval"])
    assert stop.value.code == 2


# --------------------------------------------------------------------------- MCP tool functions
def test_mcp_fix_and_inspect_with_the_window(out):
    assert {"window_set", "window_nudge", "window_kind", "no_photo_crop", "reset_window"} <= set(
        next(t for t in mcp_server.TOOLS if t["name"] == "polaroid_fix")["inputSchema"]["properties"])
    assert "of" in next(t for t in mcp_server.TOOLS if t["name"] == "polaroid_inspect")["inputSchema"]["properties"]
    fix_text = next(t for t in mcp_server.TOOLS if t["name"] == "polaroid_fix")["description"]
    assert "window_set" in fix_text and "no_photo_crop" in fix_text and "of=window" in mcp_server.INSTRUCTIONS
    res = mcp_server.tool_fix({"output": str(out), "scan": "sq.tif", "window_nudge": {"bl": [6, -3]}, "export": False,
                               "note": "mcp"}, None)
    assert not res["isError"] and res["structuredContent"]["window"]["source"] == "manual"
    assert hand(out, "sq.tif")["author"] == "agent"                     # the MCP default
    res = mcp_server.tool_inspect({"scan": "sq.tif", "output": str(out), "of": "window", "save": False}, None)
    assert res["structuredContent"]["of"] == "window" and res["content"][1]["type"] == "image"
    res = mcp_server.tool_fix({"output": str(out), "scan": "nw.tif", "no_photo_crop": True, "export": False}, None)
    assert res["structuredContent"]["status"] == "ok" and res["structuredContent"]["window"]["state"] == "none"
    status = mcp_server.tool_status({"output": str(out), "needs_review": False}, None)["structuredContent"]
    states = {s["file"]: s["window"]["state"] for s in status["scans"]}
    assert states == {"sq.tif": "found", "rd.tif": "found", "nw.tif": "none"}


# --------------------------------------------------------------------------- review page backend
@pytest.fixture
def page(out):
    p = Page(out, show_all=True)
    yield p
    p.close()


def test_the_page_records_carry_the_window_only_with_the_crop_on(page, src, tmp_path):
    _, data = page.json("/api/scans")
    by = {s["key"]: s for s in data["scans"]}
    assert by["sq.tif"]["window"]["state"] == "found" and by["sq.tif"]["window"]["kind"] == "square"
    assert len(by["sq.tif"]["window"]["quad"]) == 4 and "nominal" not in by["sq.tif"]["window"]
    assert by["sq.tif"]["window"]["outside"] is False
    nw = by["nw.tif"]["window"]
    assert nw["state"] == "missing" and nw["reason"] and set(nw["nominal"]) == {"square", "round"} and nw["suggest"] == "square"
    off = tmp_path / "off"
    run(src, off, export=False)
    p2 = Page(off, show_all=True)
    try:
        _, data = p2.json("/api/scans")
        assert all("window" not in s for s in data["scans"])
        assert p2.request("/api/scans/sq.tif/tiles?of=window")[0] == 400          # no photo crop: no window tiles
    finally:
        p2.close()


def test_window_tiles_sit_on_the_window_corners(page, out):
    quad = np.array(entry(out, "sq.tif")["window"]["quad"])
    _, specs = page.json("/api/scans/sq.tif/tiles?of=window")
    t = {x["name"]: x for x in specs["tiles"]}
    for name, p in zip(("tl", "tr", "br", "bl"), quad):
        assert t[name]["x0"] + t[name]["window"][0] // 2 == round(p[0]) and t[name]["y0"] + t[name]["window"][1] // 2 == round(p[1])
        assert "of=window" in t[name]["url"]
    status, headers, raw = page.request(t["tl"]["url"])
    assert status == 200 and raw[:4] == b"\x89PNG"
    # a scan without a window: the tiles sit on the nominal window of the asked kind
    _, specs = page.json("/api/scans/nw.tif/tiles?of=window&kind=round")
    assert "kind=round" in specs["tiles"][0]["url"]
    nom = np.array(nominal_window(Manifest.effective_quad(entry(out, "nw.tif")), "round")["quad"])
    t = {x["name"]: x for x in specs["tiles"]}
    assert t["tl"]["x0"] + 240 == round(nom[0][0]) and t["tl"]["y0"] + 240 == round(nom[0][1])
    assert page.request("/api/scans/nw.tif/tiles?of=window&kind=oval")[0] == 400
    assert page.request("/api/scans/nw.tif/tiles?of=elsewhere")[0] == 400


def test_the_page_saves_a_window_marks_none_and_resets(page, out):
    quad = np.array(entry(out, "sq.tif")["window"]["quad"])
    moved = quad + [[4, 3], [0, 0], [0, 0], [0, 0]]
    status, rec = page.json("/api/scans/sq.tif/window", {"corners": corners(moved), "kind": "square"})
    assert status == 200 and rec["window"]["source"] == "manual" and rec["rev"] == 1
    w = hand(out, "sq.tif")
    assert np.allclose(w["quad"], moved, atol=0.01) and w["author"] == "human" and w["note"].startswith("review page:")
    assert page.json("/api/scans/sq.tif/window", {"reset": True})[1]["window"]["source"] == "auto"
    status, rec = page.json("/api/scans/nw.tif/window", {"none": True})
    assert status == 200 and rec["window"]["state"] == "none" and rec["status"] == "ok"
    assert page.json("/api/scans/nw.tif/window", {"corners": {"tl": [1, 1]}})[0] == 400
    assert page.json("/api/scans/nw.tif/window", {})[0] == 400
    # Confirm accepts a missing window
    status, rec = page.json("/api/scans/rd.tif/confirm", {})
    assert status == 200
    api.fix_scan(str(out), "nw.tif", reset_window=True)
    status, rec = page.json("/api/scans/nw.tif/confirm", {})
    assert status == 200 and rec["status"] == "ok" and rec["window"]["accepted"] and rec["window"]["state"] == "missing"


def test_the_page_saves_outline_and_window_in_one_request_or_not_at_all(page, out):
    quad = np.array(entry(out, "sq.tif")["window"]["quad"])
    det = Manifest.effective_quad(entry(out, "sq.tif"))
    # a window outside the new outline: the outline is not stored either
    bad = {"corners": corners(det + [[0, 0], [0, 0], [0, 0], [0, 0]]), "window": {"corners": corners(quad + 5000), "kind": "square"}}
    status, body = page.json("/api/scans/sq.tif/save", bad)
    assert status == 400 and "inside the print outline" in body["error"]
    e = entry(out, "sq.tif")
    assert not e.get("override") and e.get("window_fix") is None and not e["history"]
    # both together
    outline = det + [[0, 0], [-6, 0], [-6, 0], [0, 0]]
    status, rec = page.json("/api/scans/sq.tif/save", {"corners": corners(outline),
                                                       "window": {"corners": corners(quad + [[3, 3], [0, 0], [0, 0], [0, 0]]), "kind": "square"}})
    e = entry(out, "sq.tif")
    assert status == 200 and e["override"] and hand(out, "sq.tif") and [h["action"] for h in e["history"]] == ["fix", "window"]


def test_the_app_serves_the_window_routes(out, tmp_path):
    c = AppClient(tmp_path)
    try:
        assert c.json("/api/project/open", {"output": str(out)})[0] == 200
        _, data = c.json("/api/fix/scans")
        assert {s["key"]: s["window"]["state"] for s in data["scans"]} == {"sq.tif": "found", "rd.tif": "found", "nw.tif": "missing"}
        status, specs = c.json("/api/fix/scans/nw.tif/tiles?of=window&kind=square")
        assert status == 200 and specs["tiles"][0]["url"].startswith("/api/fix/scans/nw.tif/tile/")
        assert c.request(specs["tiles"][0]["url"])[0] == 200
        status, rec = c.json("/api/fix/scans/nw.tif/window", {"none": True})
        assert status == 200 and rec["window"]["state"] == "none"
        # the page itself carries the window controls and the script is valid for the browser
        page = c.request("/fix")[2].decode()
        assert 'id="winbox"' in page and "strokeWindow" in page
    finally:
        c.close()


def test_grid_cards_carry_the_window_box_and_the_tag_of_a_window_choice(out, tmp_path):
    c = AppClient(tmp_path)
    try:
        assert c.json("/api/project/open", {"output": str(out)})[0] == 200
        by = {s["key"]: s for s in c.json("/api/project/scans")[1]["scans"]}
        box = np.array(by["sq.tif"]["window"]["box"])             # fractions of the crop preview
        assert by["sq.tif"]["window"]["kind"] == "square" and box.shape == (4, 2)
        assert box[0] == pytest.approx([5.5 / 88, 6.3 / 107], abs=0.012) and box[2] == pytest.approx([82.4 / 88, 85.2 / 107], abs=0.012)
        assert by["rd.tif"]["window"]["kind"] == "round" and "window" not in by["nw.tif"]
        assert all("window_fix" not in s for s in by.values())          # no choice, no tag
        api.fix_scan(str(out), "sq.tif", window_nudges={"tl": [30, 30]}, author="agent")
        api.fix_scan(str(out), "nw.tif", no_photo_crop=True, author="agent")
        by = {s["key"]: s for s in c.json("/api/project/scans")[1]["scans"]}
        moved = by["sq.tif"]["window"]
        assert moved["source"] == "manual" and moved["box"][0][0] > box[0][0] + 0.01
        assert by["sq.tif"]["window_fix"] == "hand" and by["nw.tif"]["window_fix"] == "none" and "window_fix" not in by["rd.tif"]
    finally:
        c.close()


def test_the_pages_scripts_are_valid_javascript(tmp_path):
    import re
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    root = Path(api.__file__).parent
    page = (root / "review" / "static" / "index.html").read_text(encoding="utf-8")
    js = {"review.js": re.search(r"<script>\n(.*?)</script>", page, re.S).group(1),
          "app.js": (root / "app" / "static" / "app.js").read_text(encoding="utf-8")}
    for name, code in js.items():
        f = tmp_path / name
        f.write_text(code, encoding="utf-8")
        done = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
