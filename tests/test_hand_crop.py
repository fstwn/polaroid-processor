"""Hand crop and item kind, slice 1 (core): the manifest field, the rules in format, scale, review, window and
export, the re-run behaviour, and ``fix`` / ``inspect`` in the API, the CLI and the MCP server.

The decisions are in ``docs/SCOPE.md``, "Hand crop and item kind: decisions". A scan whose detection fails (here a
faint rectangle on noise) is cropped by hand and gets an item kind: ``600`` / ``go`` (a print of that format) or
``other`` (any flat item, exported exactly as drawn). The choice lives in ``entry["item_fix"]``, which only ``fix``
writes. Scans without a choice keep their results, records and export keys.
"""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from polaroid_processor import api, cli, item as item_mod, mcp_server, pipeline
from polaroid_processor.geometry import rectify
from polaroid_processor.imageio import load_rgb
from polaroid_processor.manifest import Manifest
from polaroid_processor.pipeline import Settings, reference_sizes, resolve_scales, result_record
from polaroid_processor.scale import MM_PER_INCH, NO_SCALE, Scale
from polaroid_processor.scan_window import effective_window, window_current

from synth import PX_PER_MM, make_scan

PPI = PX_PER_MM * MM_PER_INCH
CORNERS = ("tl", "tr", "br", "bl")


def box(x0, y0, x1, y1):
    return {"tl": [x0, y0], "tr": [x1, y0], "br": [x1, y1], "bl": [x0, y1]}


TICKET = box(400, 300, 1100, 1500)          # on ticket.tif / lift.tif, 1600 x 2000 px
SHEET = box(500, 600, 1900, 2200)           # on sheet.tif, 2400 x 2900 px (the size of the prints)


def quad(c):
    return np.array([c[n] for n in CORNERS], float)


def failing(h, w, seed):
    """Noise with a faint patterned rectangle: no print outline can be detected in it."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 4, size=(h, w, 3), dtype=np.uint8)
    x0, y0, x1, y1 = int(w * 0.25), int(h * 0.15), int(w * 0.69), int(h * 0.75)
    img[y0:y1, x0:x1] = (np.linspace(4, 20, x1 - x0)[None, :, None] + rng.integers(0, 3, size=(y1 - y0, x1 - x0, 3))
                         ).astype(np.uint8)
    return img


def write_src(folder, prints=3):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(prints):
        rgb, _ = make_scan("white", 0.3 * i, seed=70 + i)
        tifffile.imwrite(str(folder / f"p{i}.tif"), rgb, photometric="rgb")
    tifffile.imwrite(str(folder / "sheet.tif"), failing(2900, 2400, 1), photometric="rgb")
    tifffile.imwrite(str(folder / "ticket.tif"), failing(2000, 1600, 2), photometric="rgb")
    tifffile.imwrite(str(folder / "lift.tif"), failing(2000, 1600, 3), photometric="rgb", resolution=(400, 400),
                     resolutionunit="INCH")
    return folder


def entry(out, name):
    return Manifest(str(out)).get(name)


def hist(out, name):
    return [h["action"] for h in entry(out, name)["history"]]


def process(src, out, **kw):
    kw.setdefault("workers", 1)
    return api.process([str(src)], str(out), api.make_settings(**kw))


def dpi_of(path):
    """The resolution tag of an export in ppi (None without one)."""
    path = Path(path)
    if path.suffix == ".tif":
        with tifffile.TiffFile(str(path)) as tf:
            tag = tf.pages[0].tags.get("XResolution")
            if tag is None:
                return None
            unit = tf.pages[0].tags.get("ResolutionUnit")
            num, den = tag.value
            v = num / den
            return v * 2.54 if unit is not None and int(unit.value) == 3 else v
    with Image.open(path) as im:
        if im.format == "JPEG" and im.info.get("jfif_unit") != 1:      # JFIF density with no unit: no resolution
            return None
        d = im.info.get("dpi")
    return float(d[0]) if d else None


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    return write_src(tmp_path_factory.mktemp("hc_scans"))


@pytest.fixture(scope="module")
def base(src, tmp_path_factory):
    """Three prints (detected), a sheet the size of the prints, a ticket and a lift (tagged 400): the last three fail."""
    out = tmp_path_factory.mktemp("hc_out")
    process(src, out, export=False)
    return out


@pytest.fixture
def out(base, tmp_path):
    dest = tmp_path / "out"
    shutil.copytree(base, dest)
    return dest


def test_the_fixture_scans_fail_detection(base):
    for name in ("sheet.tif", "ticket.tif", "lift.tif"):
        e = entry(base, name)
        assert e["status"] == "failed" and e["detection"] is None and e["error"].startswith("detection failed")
    assert all(entry(base, f"p{i}.tif")["status"] in ("ok", "needs_review") for i in range(3))


# --------------------------------------------------------------------------- shared checks and the start rectangle
def test_crop_corner_checks():
    good = quad(TICKET)
    assert np.array_equal(item_mod.check_crop_quad(good, 1600, 2000), good)
    rot = good.copy()
    rot[:, 0] = 100 + (good[:, 0] - 400) * 0.9      # a slightly rotated quad is fine
    assert item_mod.check_crop_quad(np.array([[300, 400], [1000, 380], [1010, 1500], [310, 1520]], float), 1600, 2000) is not None
    bad = {
        "inside the scan": np.array([[-1, 300], [1100, 300], [1100, 1500], [400, 1500]], float),
        "inside the scan (far side)": np.array([[400, 300], [1100, 300], [1100, 2001], [400, 1500]], float),
        "clockwise": good[[0, 3, 2, 1]],
        "must not cross": np.array([[400, 300], [1100, 1500], [1100, 300], [400, 1500]], float),
        "side top is 40 px": np.array([[400, 300], [440, 300], [440, 1500], [400, 1500]], float),
        "at most 45": np.array([[800, 300], [1100, 820], [580, 1120], [280, 600]], float),      # 60 deg
        "four finite": np.array([[400, 300], [np.nan, 300], [1100, 1500], [400, 1500]], float),
    }
    for why, q in bad.items():
        with pytest.raises(ValueError, match=why.split(" (")[0]):
            item_mod.check_crop_quad(q, 1600, 2000)
    with pytest.raises(ValueError, match="four finite"):
        item_mod.check_crop_quad(np.zeros((3, 2)), 1600, 2000)
    # the same checks through the API raise UsageError
    with pytest.raises(api.UsageError, match="clockwise"):
        api.check_crop_quad(good[[0, 3, 2, 1]], 1600, 2000)


def test_a_crop_rotated_by_exactly_45_degrees_is_allowed():
    a = np.deg2rad(45)
    c, s = np.cos(a), np.sin(a)
    pts = np.array([[-300, -300], [300, -300], [300, 300], [-300, 300]], float) @ np.array([[c, -s], [s, c]]).T + [800, 1000]
    item_mod.check_crop_quad(pts, 1600, 2000)
    a = np.deg2rad(46)
    c, s = np.cos(a), np.sin(a)
    pts = np.array([[-300, -300], [300, -300], [300, 300], [-300, 300]], float) @ np.array([[c, -s], [s, c]]).T + [800, 1000]
    with pytest.raises(ValueError, match="at most 45"):
        item_mod.check_crop_quad(pts, 1600, 2000)


def test_start_rectangles():
    r = item_mod.start_rectangle(1600, 2000, "other")      # centred, 80 % of the scan
    assert r.tolist() == [[160, 200], [1440, 200], [1440, 1800], [160, 1800]]
    assert item_mod.start_rectangle(1600, 2000, None).tolist() == r.tolist()
    # a format with a scale: the nominal size, centred
    r = item_mod.start_rectangle(2400, 2900, "600", 20.0)
    assert np.allclose(r.mean(axis=0), [1200, 1450]) and np.allclose(r[1] - r[0], [88 * 20, 0]) \
        and np.allclose(r[3] - r[0], [0, 106.85 * 20])
    r = item_mod.start_rectangle(2400, 2900, "go", 20.0)
    assert np.allclose(r[1] - r[0], [53.5 * 20, 0]) and np.allclose(r[3] - r[0], [0, 65.8 * 20])
    # a print fills most of a scan: the nominal size is not cut to 80 %, only to the scan when it would not fit
    r = item_mod.start_rectangle(2400, 2900, "600", 25.6)
    assert (r[1] - r[0])[0] == pytest.approx(88 * 25.6, abs=0.01) and r[0][0] < 0.8 * 2400 * 0.5 + 200
    r = item_mod.start_rectangle(2400, 2900, "600", 200.0)
    w, h = (r[1] - r[0])[0], (r[3] - r[0])[1]
    assert w <= 2400 + 0.01 and h <= 2900 + 0.01 and w / h == pytest.approx(88 / 106.85, abs=1e-3)
    assert max(w / 2400, h / 2900) == pytest.approx(1.0, abs=1e-3)
    item_mod.check_crop_quad(r, 2400, 2900)      # and it is a crop the checks accept
    # no scale: the 80 % limit, with the aspect ratio
    r2 = item_mod.start_rectangle(2400, 2900, "600", None)
    w, h = (r2[1] - r2[0])[0], (r2[3] - r2[0])[1]
    assert w / h == pytest.approx(88 / 106.85, abs=1e-3) and max(w / 2400, h / 2900) == pytest.approx(0.8, abs=1e-3)
    r = item_mod.start_rectangle(3000, 1000, "go", None)      # a landscape scan: the height limits
    assert (r[3] - r[0])[1] == pytest.approx(800, abs=0.01)


def test_the_start_rectangle_of_a_scan_uses_the_projects_scale_or_the_prints_median(out, src):
    got = api.start_rectangle(str(out), "sheet.tif", "600")      # no ppi, no tag: the median of the three prints
    assert got["px_per_mm"] == pytest.approx(PX_PER_MM, rel=0.01)
    c = np.array([got["corners"][n] for n in CORNERS])
    assert (c[1] - c[0])[0] == pytest.approx(88 * got["px_per_mm"], abs=0.02) and c.mean(axis=0).tolist() == [1200.0, 1450.0]
    got = api.start_rectangle(str(out), "ticket.tif", "600")      # no prints of its size: no scale, the 80 % limit
    assert got["px_per_mm"] is None and got["scan_px"] == [1600, 2000]
    c = np.array([got["corners"][n] for n in CORNERS])
    assert max((c[1] - c[0])[0] / 1600, (c[3] - c[0])[1] / 2000) == pytest.approx(0.8, abs=1e-3)
    assert api.start_rectangle(str(out), "ticket.tif", "other")["corners"]["tl"] == [160.0, 200.0]
    m = Manifest(str(out))
    m.data["settings"]["ppi"] = 500.0
    m.save()
    got = api.start_rectangle(str(out), "ticket.tif", "go")      # an explicit scale wins, also without prints
    assert got["px_per_mm"] == pytest.approx(500 / 25.4)
    with pytest.raises(api.UsageError, match="item"):
        api.start_rectangle(str(out), "ticket.tif", "auto")


# --------------------------------------------------------------------------- the manifest field
def test_a_failed_scan_is_cropped_with_all_four_corners_and_turns_ok_at_once(out):
    """The old bug: ``fix`` raised "scan has no outline yet" even with all four corners given."""
    assert entry(out, "ticket.tif")["status"] == "failed"
    rec = api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human", note="a ticket")
    e = entry(out, "ticket.tif")
    assert rec["type"] == "fix" and rec["status"] == "ok" and e["status"] == "ok" and e["error"] is None
    assert e["override"]["quad"] == quad(TICKET).tolist() and e["override"]["author"] == "human"
    assert e["detection"] is None and e["detection_error"].startswith("detection failed")      # stays, not a failure
    it = e["item_fix"]
    assert it["kind"] == "other" and it["author"] == "human" and it["note"] == "a ticket" and it["at"]
    assert e.get("agent_crop") is None and not it.get("scan_ppi")
    assert hist(out, "ticket.tif") == ["fix", "item"]
    h = e["history"][-1]
    assert h["item_before"] is None and h["item_after"] == it and h["author"] == "human"
    assert rec["item"]["kind"] == "other" and rec["item"]["agent_crop"] is False and rec["quad"][0] == [400.0, 300.0]
    assert rec["detection_error"].startswith("detection failed") and rec["quad_source"] == "override"
    assert "manifest updated" in rec["note"]      # nothing exported without export=True
    assert not (out / "ticket.tif").exists()


def test_the_item_kind_can_be_chosen_and_changed(out):
    rec = api.fix_scan(str(out), "ticket.tif", sets=TICKET, item="600", author="human")
    assert rec["item"]["kind"] == "600" and entry(out, "ticket.tif")["format"]["name"] == "polaroid-600"
    api.fix_scan(str(out), "ticket.tif", item="go", author="human", note="no, a Go")
    e = entry(out, "ticket.tif")
    assert e["item_fix"]["kind"] == "go" and e["format"]["name"] == "polaroid-go" and e["status"] == "ok"
    assert hist(out, "ticket.tif") == ["fix", "item", "item"]
    assert e["history"][-1]["item_before"]["kind"] == "600" and e["history"][-1]["item_after"]["kind"] == "go"
    api.fix_scan(str(out), "ticket.tif", item="other", scan_ppi=800, author="human")
    e = entry(out, "ticket.tif")
    assert e["item_fix"]["kind"] == "other" and e["item_fix"]["scan_ppi"] == 800.0 and e["format"] is None
    api.fix_scan(str(out), "ticket.tif", item="other", author="human")      # nothing changes: no new entry
    assert hist(out, "ticket.tif") == ["fix", "item", "item", "item"]
    assert entry(out, "ticket.tif")["item_fix"]["scan_ppi"] == 800.0      # the own ppi stays while the kind stays other
    api.fix_scan(str(out), "ticket.tif", scan_ppi=0, author="human")      # 0 takes it away
    assert not entry(out, "ticket.tif")["item_fix"].get("scan_ppi")
    api.fix_scan(str(out), "ticket.tif", scan_ppi=500, author="human")
    api.fix_scan(str(out), "ticket.tif", item="600", author="human")      # another kind drops the ppi
    assert not entry(out, "ticket.tif")["item_fix"].get("scan_ppi")


def test_an_item_can_be_set_on_a_detected_scan_and_taken_away(out):
    before = entry(out, "p0.tif")
    quad0 = Manifest.effective_quad(before).tolist()
    api.fix_scan(str(out), "p0.tif", item="other", author="human")
    e = entry(out, "p0.tif")
    assert e["status"] == "ok" and not e.get("override") and Manifest.effective_quad(e).tolist() == quad0      # keeps its outline
    assert e["item_fix"]["kind"] == "other" and e["format"] is None
    api.fix_scan(str(out), "p0.tif", item="auto", author="human", note="back")
    e = entry(out, "p0.tif")
    assert e["item_fix"] is None and e["format"]["name"] == "polaroid-600" and hist(out, "p0.tif")[-1] == "item-reset"
    assert e["history"][-1]["item_before"]["kind"] == "other" and e["history"][-1]["item_after"] is None


def test_bad_requests_are_refused_and_store_nothing(out):
    def refused(match, **kw):
        before = json.dumps(Manifest(str(out)).data["scans"], sort_keys=True)
        with pytest.raises(api.UsageError, match=match):
            api.fix_scan(str(out), kw.pop("scan", "ticket.tif"), **kw)
        assert json.dumps(Manifest(str(out)).data["scans"], sort_keys=True) == before

    refused("all four corners", sets={"tl": [400, 300], "tr": [1100, 300], "br": [1100, 1500]})      # three corners
    refused("all four corners", nudges={"tl": [5, 5]})                       # nothing to nudge yet
    refused("all four corners", shifts={"top": [5]})
    refused("all four corners", accept=True)
    refused("all four corners", item="other")                                # an item needs a crop
    refused("all four corners", scan_ppi=500)
    refused("inside the scan", sets=box(400, 300, 1700, 1500))
    refused("clockwise", sets={"tl": [400, 300], "tr": [400, 1500], "br": [1100, 1500], "bl": [1100, 300]})
    refused("cross", sets={"tl": [400, 300], "tr": [1100, 1500], "br": [1100, 300], "bl": [400, 1500]})
    refused("at least 50 px", sets=box(400, 300, 440, 1500))
    refused("at most 45", sets={"tl": [800, 300], "tr": [1100, 820], "br": [580, 1120], "bl": [280, 600]})
    refused("item must be", sets=TICKET, item="polaroid")
    refused("a first crop needs an item kind", sets=TICKET, item="auto")
    refused("scan_ppi must be", sets=TICKET, scan_ppi=-5)
    refused("scan_ppi must be", sets=TICKET, scan_ppi="fast")
    refused("scan_ppi must be", sets=TICKET, scan_ppi=float("nan"))
    refused("scan_ppi must be", sets=TICKET, scan_ppi=1e9)
    refused("item of kind other", sets=TICKET, item="600", scan_ppi=500)
    refused("fix needs", )
    refused("reset cannot be combined", reset=True, item="other")
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human")
    refused("scan_ppi is the resolution of an item of kind other", item="go", scan_ppi=500)
    refused("scan_ppi is the resolution of an item of kind other", item="auto", scan_ppi=500, scan="p0.tif")
    assert entry(out, "ticket.tif")["status"] == "ok" and hist(out, "ticket.tif") == ["fix", "item"]


def test_the_checks_apply_to_a_print_only_when_it_has_an_item(out):
    """A plain fix of a detected print is as before (also a corner outside the scan); with an item the hand-crop rules hold."""
    p = np.array(entry(out, "p0.tif")["detection"]["quad"])
    outside = {"tl": [-5.0, float(p[0][1])]}
    api.fix_scan(str(out), "p0.tif", sets=outside, author="human")      # as before: no check
    assert entry(out, "p0.tif")["override"]["quad"][0][0] == -5.0
    with pytest.raises(api.UsageError, match="inside the scan"):
        api.fix_scan(str(out), "p0.tif", item="600", author="human", nudges={"tl": [-3, 0]})
    assert entry(out, "p0.tif").get("item_fix") is None
    api.fix_scan(str(out), "p0.tif", item="600", sets={"tl": [10.0, float(p[0][1])]}, author="human")
    with pytest.raises(api.UsageError, match="clockwise"):
        api.fix_scan(str(out), "p0.tif", sets={"tl": [100, 100], "tr": [100, 900], "br": [900, 900], "bl": [900, 100]},
                     author="human")      # now the item is set: this runs counter-clockwise


# --------------------------------------------------------------------------- reset, drop, who writes the field
def test_reset_takes_crop_item_and_own_ppi_away_and_the_scan_is_failed_again(out):
    err = entry(out, "ticket.tif")["error"]
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, item="other", scan_ppi=800, author="human")
    rec = api.fix_scan(str(out), "ticket.tif", reset=True, author="human")
    e = entry(out, "ticket.tif")
    assert e["override"] is None and e["item_fix"] is None
    assert e["status"] == "failed" and e["error"] == err and e["warnings"] == [] and "detection_error" not in e
    assert rec["status"] == "failed" and "item" not in rec
    assert hist(out, "ticket.tif") == ["fix", "item", "reset", "item-reset"]
    assert e["history"][-1]["item_before"]["scan_ppi"] == 800.0 and e["history"][-1]["item_after"] is None
    assert Manifest.effective_quad(e) is None


def test_reset_of_a_detected_scan_with_an_item_goes_back_to_the_automatic_outline(out):
    auto = entry(out, "p1.tif")["detection"]["quad"]
    api.fix_scan(str(out), "p1.tif", shifts={"left": [3.0]}, item="go", author="human")
    assert entry(out, "p1.tif")["format"]["name"] == "polaroid-go"
    api.fix_scan(str(out), "p1.tif", reset=True, author="human")
    e = entry(out, "p1.tif")
    assert e["override"] is None and e["item_fix"] is None and Manifest.effective_quad(e).tolist() == auto
    assert e["format"]["name"] == "polaroid-600" and e["status"] in ("ok", "needs_review")


def test_only_fix_writes_the_item_and_a_run_keeps_it(src, out):
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, item="other", scan_ppi=800, author="human")
    api.fix_scan(str(out), "p0.tif", item="go", author="human")
    before = {k: json.dumps(entry(out, k)[f], sort_keys=True) for k, f in (("ticket.tif", "item_fix"), ("p0.tif", "item_fix"),
                                                                           ("ticket.tif", "override"))}
    process(src, out)
    process(src, out, redetect=True)
    process(src, out, redetect=True, export=False)
    after = {k: json.dumps(entry(out, k)[f], sort_keys=True) for k, f in (("ticket.tif", "item_fix"), ("p0.tif", "item_fix"),
                                                                          ("ticket.tif", "override"))}
    assert after == before
    assert entry(out, "ticket.tif")["status"] == "ok" and entry(out, "p0.tif")["item_fix"]["kind"] == "go"


def test_a_changed_scan_file_drops_crop_and_item_and_is_logged(src, out, tmp_path):
    work = tmp_path / "in"
    shutil.copytree(src, work)
    process(work, out, export=False)        # the manifest now points at this copy
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, item="600", scan_ppi=None, author="human")
    api.fix_scan(str(out), "ticket.tif", item="other", scan_ppi=700, author="human")
    assert entry(out, "ticket.tif")["status"] == "ok"
    tifffile.imwrite(str(work / "ticket.tif"), failing(2000, 1600, 99), photometric="rgb")      # a different picture
    process(work, out, export=False)
    e = entry(out, "ticket.tif")
    assert e["override"] is None and e["item_fix"] is None and e["status"] == "failed" and e["error"].startswith("detection failed")
    assert hist(out, "ticket.tif")[-2:] == ["reset", "item-drop"]
    last = e["history"][-1]
    assert last["author"] == "system" and last["item_before"]["kind"] == "other" and last["item_after"] is None
    # a scan that was not changed keeps its crop (checked below: the next run does not detect it again)


def test_a_changed_file_is_seen_for_a_scan_that_is_not_detected_again(src, out, tmp_path):
    """The fingerprint check runs in the detection job: a hand-cropped scan whose size or time changed gets one."""
    work = tmp_path / "in"
    shutil.copytree(src, work)
    process(work, out, export=False)
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human")
    n = []
    api.process([str(work)], str(out), api.make_settings(workers=1, export=False), on_jobs=n.append)
    assert n == [2]                                    # the two uncropped failed scans only: the crop is not detected again
    tifffile.imwrite(str(work / "ticket.tif"), failing(2000, 1600, 2) + 1, photometric="rgb")      # same size, new content, new time
    n.clear()
    api.process([str(work)], str(out), api.make_settings(workers=1, export=False), on_jobs=n.append)
    assert n == [3] and entry(out, "ticket.tif")["item_fix"] is None and entry(out, "ticket.tif")["status"] == "failed"


# --------------------------------------------------------------------------- re-run behaviour (decision 7)
def test_a_cropped_scan_is_not_detected_again_but_a_failed_one_is(src, out):
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human")
    seen = []
    api.process([str(src)], str(out), api.make_settings(workers=1, export=False), on_jobs=seen.append,
                progress=lambda k, what: seen.append(k))
    assert seen[0] == 2 and sorted(seen[1:]) == ["lift.tif", "sheet.tif"]      # only the failed scans nobody cropped
    e = entry(out, "ticket.tif")
    assert e["status"] == "ok" and e["detection"] is None


def test_redetect_that_fails_leaves_the_scan_ok_and_the_crop_in_force(src, out):
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, item="600", author="human")
    ov = entry(out, "ticket.tif")["override"]
    first = []
    summary = api.process([str(src)], str(out), api.make_settings(workers=1, export=False, redetect=True),
                          on_jobs=first.append)
    e = entry(out, "ticket.tif")
    assert first[0] == 6 and e["status"] == "ok" and e["error"] is None and e["override"] == ov
    assert e["item_fix"]["kind"] == "600" and e["detection"] is None
    assert e["detection_error"].startswith("detection failed")      # stored, not shown as a failure
    assert "ticket.tif" not in summary["failed"] and summary["failed"] == ["lift.tif", "sheet.tif"]
    assert e["warnings"] == [] or all(not w.startswith("detection") for w in e["warnings"])


def test_redetect_that_succeeds_is_stored_and_the_crop_stays_in_force(src, out):
    """A failed scan that is cropped by hand, then detected after all: the detection is kept, the crop wins."""
    p = entry(out, "p0.tif")
    det = p["detection"]
    m = Manifest(str(out))
    e = m.get("p0.tif")      # make p0 look like a scan that failed and was cropped by hand with another outline
    crop = (np.array(det["quad"]) + [40.0, 30.0]).round(2).tolist()
    e.update(detection=None, status="ok", error=None, detection_error="detection failed: x", override={
        "quad": crop, "author": "human", "at": "2026-11-24T10:00:00+01:00", "note": ""},
        item_fix={"kind": "600", "scan_ppi": None, "author": "human", "at": "2026-11-24T10:00:00+01:00", "note": "",
                  "agent_crop": None})
    m.save()
    process(src, out, export=False, redetect=True)
    e = entry(out, "p0.tif")
    assert e["detection"]["quad"] and e["override"]["quad"] == crop and e["item_fix"]["kind"] == "600"
    assert Manifest.effective_quad(e).tolist() == crop and e["status"] in ("ok", "needs_review")
    assert "detection_error" not in e and e["error"] is None


def test_a_fix_landing_during_a_redetect_run_is_kept(src, out):
    """Two writers: the crop is saved while the run detects the other scans again."""
    done = []

    def progress(key, what):
        if not done:
            done.append(key)
            api.fix_scan(str(out), "ticket.tif", sets=TICKET, item="other", scan_ppi=800, author="human")

    api.process([str(src)], str(out), api.make_settings(workers=1, redetect=True), progress=progress)
    e = entry(out, "ticket.tif")
    assert done and e["status"] == "ok" and e["error"] is None and e["override"]["quad"] == quad(TICKET).tolist()
    assert e["item_fix"]["kind"] == "other" and e["item_fix"]["scan_ppi"] == 800.0
    assert hist(out, "ticket.tif").count("fix") == 1 and hist(out, "ticket.tif").count("item") == 1
    assert e["outputs"] and e["scale"]["source"] == "scan-ppi"      # exported from the crop, at the scan's own ppi
    assert (out / "ticket.tif").exists()


def test_a_fix_landing_just_after_the_failed_detection_of_that_scan_wins_over_the_failure(src, out):
    def progress(key, what):
        if key == "ticket.tif" and what != "exported":
            api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human")

    api.process([str(src)], str(out), api.make_settings(workers=1, redetect=True), progress=progress)
    e = entry(out, "ticket.tif")
    assert e["status"] == "ok" and e["error"] is None and e["item_fix"]["kind"] == "other" and e["outputs"]
    assert e["detection_error"].startswith("detection failed")


def test_a_cancelled_run_loses_nothing(src, out):
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, item="other", scan_ppi=800, author="agent")
    before = entry(out, "ticket.tif")
    summary = api.process([str(src)], str(out), api.make_settings(workers=1, redetect=True), cancel=lambda: True)
    e = entry(out, "ticket.tif")
    assert summary["cancelled"] and e["item_fix"] == before["item_fix"] and e["override"] == before["override"]
    assert e["status"] == "needs_review" and not (out / "ticket.tif").exists()
    api.process([str(src)], str(out), api.make_settings(workers=1))      # and the next run goes on
    assert (out / "ticket.tif").exists()


# --------------------------------------------------------------------------- agents (decision 8)
def test_a_crop_an_agent_places_needs_review_until_a_human_confirms_or_saves_it(out):
    rec = api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="agent", note="from the sheet")
    e = entry(out, "ticket.tif")
    assert rec["status"] == "needs_review" and e["warnings"][-1] == "cropped by agent"
    assert e["agent_crop"]["author"] == "agent" and e["agent_crop"]["note"] == "from the sheet"
    assert rec["item"]["agent_crop"] is True
    # the agent's own accept and re-save do not clear it
    api.fix_scan(str(out), "ticket.tif", accept=True, author="agent")
    assert entry(out, "ticket.tif")["status"] == "needs_review"
    api.fix_scan(str(out), "ticket.tif", nudges={"tl": [2.0, 2.0]}, author="agent", note="nudged")
    e = entry(out, "ticket.tif")
    assert e["status"] == "needs_review" and e["agent_crop"]["note"] == "nudged"
    api.fix_scan(str(out), "ticket.tif", item="600", author="agent")
    assert entry(out, "ticket.tif")["status"] == "needs_review" and entry(out, "ticket.tif")["warnings"][-1] == "cropped by agent"
    # a human's Confirm clears it
    api.fix_scan(str(out), "ticket.tif", accept=True, author="human", note="looks right")
    e = entry(out, "ticket.tif")
    assert e["status"] == "ok" and e["agent_crop"] is None and "cropped by agent" not in e["warnings"]
    assert [h["action"] for h in e["history"]][-2:] == ["accept", "agent-crop-clear"]
    assert e["history"][-2]["author"] == "human" and e["history"][-1]["author"] == "human"


def test_only_a_humans_confirm_or_corner_save_clears_the_agent_mark(out):
    """Decision after the first review (Q1): an item or ppi change alone keeps the mark; corners saved with an item clear it."""
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="agent")
    api.fix_scan(str(out), "ticket.tif", nudges={"br": [3.0, 0.0]}, author="human")
    assert entry(out, "ticket.tif")["status"] == "ok" and entry(out, "ticket.tif")["agent_crop"] is None
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="agent")
    assert entry(out, "ticket.tif")["status"] == "needs_review"
    for kw in ({"item": "go"}, {"item": "other"}, {"scan_ppi": 500}, {"item": "auto"}):      # a human's item or ppi alone
        api.fix_scan(str(out), "ticket.tif", author="human", **kw)
        e = entry(out, "ticket.tif")
        assert e["status"] == "needs_review" and e["agent_crop"] and e["warnings"][-1] == "cropped by agent", kw
    n = len(entry(out, "ticket.tif")["history"])
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, item="go", author="human")      # corners saved, with an item: cleared
    e = entry(out, "ticket.tif")
    assert e["status"] == "ok" and e["agent_crop"] is None and e["item_fix"]["kind"] == "go"
    assert e["history"][-1]["action"] == "agent-crop-clear" and len(e["history"]) > n
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="agent")
    api.fix_scan(str(out), "ticket.tif", reset=True, author="human")      # reset takes everything away
    assert entry(out, "ticket.tif")["status"] == "failed" and entry(out, "ticket.tif")["item_fix"] is None


def test_agent_fixes_on_a_detected_outline_stay_as_they_were(out):
    api.fix_scan(str(out), "p0.tif", shifts={"left": [2.0]}, author="agent")
    e = entry(out, "p0.tif")
    assert e.get("item_fix") is None and "cropped by agent" not in e["warnings"]
    api.fix_scan(str(out), "p0.tif", item="other", author="agent")      # an item on a detected scan is no agent crop
    e = entry(out, "p0.tif")
    assert e["status"] == "ok" and e.get("agent_crop") is None
    api.fix_scan(str(out), "p0.tif", nudges={"tl": [1.0, 1.0]}, author="agent")
    assert entry(out, "p0.tif")["status"] == "ok"


def test_every_mcp_fix_is_an_agents_whatever_author_it_gives(out):
    res = mcp_server.tool_fix({"output": str(out), "scan": "ticket.tif", "set": TICKET, "author": "human",
                               "export": False}, None)
    assert not res["isError"]
    e = entry(out, "ticket.tif")
    assert e["status"] == "needs_review" and e["agent_crop"]["author"] == "human"      # the name is kept, the mark is set
    mcp_server.tool_fix({"output": str(out), "scan": "ticket.tif", "accept": True, "author": "a human, says the agent",
                         "export": False}, None)
    assert entry(out, "ticket.tif")["status"] == "needs_review"
    # the same call through the API with author "human" is a human's
    api.fix_scan(str(out), "ticket.tif", accept=True, author="human")
    assert entry(out, "ticket.tif")["status"] == "ok"
    # in the API an explicit agent=True does it too
    api.fix_scan(str(out), "ticket.tif", nudges={"tl": [1.0, 1.0]}, author="human", agent=True)
    assert entry(out, "ticket.tif")["status"] == "needs_review"
    # author "agent" is an agent's in the API whatever the flag says (decision 8)
    api.fix_scan(str(out), "ticket.tif", accept=True, author="human")
    api.fix_scan(str(out), "ticket.tif", nudges={"tl": [-1.0, -1.0]}, author="agent", agent=False)
    assert entry(out, "ticket.tif")["status"] == "needs_review"


def test_an_agent_crop_is_exported_like_other_flagged_scans_unless_strict(src, out, tmp_path):
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="agent")
    api.fix_scan(str(out), "lift.tif", sets=TICKET, author="human")
    strict = tmp_path / "strict"
    shutil.copytree(out, strict)
    process(src, out)
    assert (out / "ticket.tif").exists() and (out / "lift.tif").exists()
    assert entry(out, "ticket.tif")["status"] == "needs_review"
    process(src, strict, strict=True)
    assert not (strict / "ticket.tif").exists() and (strict / "lift.tif").exists()
    assert not entry(strict, "ticket.tif").get("outputs")


# --------------------------------------------------------------------------- export of Other (decision 5)
def test_other_is_exported_exactly_as_drawn(src, out):
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human", export=True)
    e = entry(out, "ticket.tif")
    rgb = load_rgb(str(src / "ticket.tif"))
    want = rectify(rgb, quad(TICKET), 0.0, "cubic")      # no margin, no unwarp, no shadow removal
    got = tifffile.imread(str(out / "ticket.tif"))
    assert got.shape == want.shape == (1200, 700, 3) and np.array_equal(got, want)
    assert e["output_size_px"] == [700, 1200] and e["flatten"]["apply"] is False and e["shadows"] == {"sides": []}
    assert sorted(e["outputs"]) == ["high", "tiff", "web"] and (out / "high" / "ticket.jpg").exists()
    assert (out / "web" / "ticket.jpg").exists()


def test_other_has_no_photo_crop_no_window_and_no_window_warnings(src, out):
    m = Manifest(str(out))
    m.data["settings"]["photo_crop"] = "tiff"
    m.save()
    process(src, out, photo_crop="tiff")      # the prints get their windows (or the warning that none was found)
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human", export=True)
    api.fix_scan(str(out), "p0.tif", item="other", author="human", export=True)
    for name in ("ticket.tif", "p0.tif"):
        e = entry(out, name)
        assert e["status"] == "ok" and not any(w.startswith("photo window") for w in e["warnings"])
        assert not any(k.startswith("photo_") for k in e["outputs"])
        assert effective_window(e) is None
        assert "window" not in result_record(Manifest(str(out)), name)
    assert not list((out / "photo").rglob("ticket.*")) and not list((out / "photo").rglob("p0.*"))
    assert api.fix_scan(str(out), "ticket.tif", author="human", accept=True)["status"] == "ok"
    with pytest.raises(api.UsageError, match="photo window"):
        api.inspect_scan("ticket.tif", str(out), of="window", write=False)
    with pytest.raises(api.UsageError, match="window options do not apply"):
        api.fix_scan(str(out), "p0.tif", no_photo_crop=True, author="human")


def test_the_older_photo_files_of_a_scan_set_to_other_are_deleted_by_the_next_export(tmp_path):
    from test_photo_crop import SQUARE, make_named, write_scans
    folder = write_scans(tmp_path / "w_in")
    out = tmp_path / "w_out"
    process(folder, out, photo_crop="tiff", ppi=PPI)
    assert (out / "photo" / "sq.tif").exists() and (out / "photo" / "high" / "sq.jpg").exists()
    api.fix_scan(str(out), "sq.tif", item="other", author="human", export=True)
    e = entry(out, "sq.tif")
    assert e["status"] == "ok" and not any(k.startswith("photo_") for k in e["outputs"])
    assert not (out / "photo" / "sq.tif").exists() and not (out / "photo" / "high" / "sq.jpg").exists()
    assert (out / "photo" / "rd.tif").exists()      # the others keep theirs
    # as drawn: the size of the detected outline, no margin
    q = Manifest.effective_quad(e)
    w, h = api.quad_size(q)
    assert e["output_size_px"] == [round(w), round(h)]
    api.fix_scan(str(out), "sq.tif", item="auto", author="human", export=True)      # back to a print: the photo crop returns
    assert (out / "photo" / "sq.tif").exists() and "photo_tiff" in entry(out, "sq.tif")["outputs"]


def test_item_600_or_go_names_the_format_of_the_photo_window(tmp_path):
    from test_photo_crop import write_scans
    folder = write_scans(tmp_path / "w_in")
    out = tmp_path / "w_out"
    process(folder, out, photo_crop="tiff", ppi=PPI)
    assert entry(out, "sq.tif")["window"]["format"] == "polaroid-600"
    assert not entry(out, "sq.tif")["window"].get("named")
    api.fix_scan(str(out), "sq.tif", item="600", author="human")      # named: measured again with that format only
    assert not window_current(entry(out, "sq.tif"))
    process(folder, out, ppi=PPI)
    e = entry(out, "sq.tif")
    assert e["window"]["named"] and e["window"]["found"] and e["window"]["format"] == "polaroid-600" and window_current(e)
    api.fix_scan(str(out), "sq.tif", item="go", author="human")      # a Go: the window of a 600 does not hold
    process(folder, out, ppi=PPI)
    e = entry(out, "sq.tif")
    assert e["window"]["format"] == "polaroid-go" and not e["window"]["found"]      # only the Go format was tried
    assert e["status"] == "needs_review" and any(w.startswith("photo window not found") for w in e["warnings"])
    assert e["format"]["name"] == "polaroid-go"
    api.fix_scan(str(out), "sq.tif", item="auto", author="human")      # the choice is taken away: the shape decides again
    assert not window_current(entry(out, "sq.tif"))
    process(folder, out, ppi=PPI)
    e = entry(out, "sq.tif")
    assert e["window"]["found"] and e["window"]["format"] == "polaroid-600" and not e["window"].get("named")


def test_the_format_placeholder_follows_the_item_kind(src, out):
    meta = {"title": "{format}|{stem}"}
    process(src, out, export=False, metadata=meta)
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human", export=True)
    api.fix_scan(str(out), "sheet.tif", sets=SHEET, item="go", author="human", export=True)
    api.fix_scan(str(out), "lift.tif", sets=TICKET, item="600", author="human", export=True)
    api.fix_scan(str(out), "p0.tif", item="other", author="human", export=True)
    api.fix_scan(str(out), "p1.tif", item="go", author="human", export=True)
    api.fix_scan(str(out), "p2.tif", author="human", accept=True, export=True)

    def title(name):
        with tifffile.TiffFile(str(out / name)) as tf:
            return tf.pages[0].tags[270].value

    assert title("ticket.tif") == "|ticket" and title("p0.tif") == "|p0"      # Other: empty
    assert title("sheet.tif") == "Go|sheet" and title("p1.tif") == "Go|p1"      # named, whatever the shape says
    assert title("lift.tif") == "600|lift" and title("p2.tif") == "600|p2"
    # a change of kind is a new export (the item is in the key), and it carries the new {format}
    api.fix_scan(str(out), "p1.tif", item="auto", author="human", export=True)
    assert title("p1.tif") == "600|p1"


# --------------------------------------------------------------------------- scale of Other (decision 6)
@pytest.fixture
def cropped(out):
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human")
    api.fix_scan(str(out), "lift.tif", sets=TICKET, author="human")
    api.fix_scan(str(out), "sheet.tif", sets=SHEET, author="human")
    return out


def scales(out, **kw):
    m = Manifest(str(out))
    settings = pipeline.settings_from_dict({**m.data.get("settings", {}), **kw})
    return resolve_scales(m, list(m.scans), settings)


def test_the_scale_of_other_in_the_decided_order(cropped):
    s = scales(cropped)
    assert s["ticket.tif"] == NO_SCALE and s["ticket.tif"].source == "none" and not s["ticket.tif"].known      # no tag, no prints
    assert s["lift.tif"].source == "tiff-tag-unverified" and s["lift.tif"].ppi == pytest.approx(400)      # the tag as it is
    assert s["sheet.tif"].source == "batch-median" and s["sheet.tif"].px_per_mm == pytest.approx(PX_PER_MM, rel=0.01)
    # the project's ppi or profile before the median and the tag
    s = scales(cropped, ppi=500.0)
    assert all(s[k].source == "ppi" and s[k].ppi == 500 for k in ("ticket.tif", "lift.tif", "sheet.tif"))
    # the scan's own ppi before everything
    api.fix_scan(str(cropped), "sheet.tif", scan_ppi=1200, author="human")
    api.fix_scan(str(cropped), "lift.tif", scan_ppi=900, author="human")
    s = scales(cropped, ppi=500.0)
    assert s["sheet.tif"].source == "scan-ppi" and s["sheet.tif"].ppi == pytest.approx(1200)
    assert s["lift.tif"].ppi == pytest.approx(900) and s["ticket.tif"].source == "ppi"


def test_the_median_needs_three_prints_of_the_same_scan_size(cropped):
    m = Manifest(str(cropped))
    m.entry("p2.tif")["detection"] = None      # two prints left in the group
    m.entry("p2.tif")["status"] = "failed"
    s = resolve_scales(m, ["sheet.tif"], pipeline.settings_from_dict(m.data["settings"]))
    assert s["sheet.tif"].source == "none"      # no tag either: no scale at all


def test_scale_unverified_is_a_warning_and_never_review(cropped):
    for name, source in (("ticket.tif", "none"), ("lift.tif", "tiff-tag-unverified")):
        e = entry(cropped, name)
        assert e["status"] == "ok" and "scale_unverified" in e["warnings"] and e["scale"]["source"] == source
    e = entry(cropped, "sheet.tif")
    assert e["status"] == "ok" and "scale_unverified" not in e["warnings"] and "scale_inferred:batch-median" in e["warnings"]
    api.fix_scan(str(cropped), "ticket.tif", scan_ppi=800, author="human")
    assert entry(cropped, "ticket.tif")["warnings"] == []      # an own ppi is verified by the user


def test_the_resolution_tags_follow_the_scale_rules(src, cropped):
    api.fix_scan(str(cropped), "sheet.tif", scan_ppi=1200, author="human")
    process(src, cropped)
    for name, want in (("ticket.tif", None), ("lift.tif", 400.0), ("sheet.tif", 1200.0)):
        e = entry(cropped, name)
        files = {k: cropped / v for k, v in e["outputs"].items()}
        for k, f in files.items():
            got = dpi_of(f)
            if want is None:
                assert got is None, (name, k)      # no usable resolution: no tag at all
            elif k == "tiff":
                assert got == pytest.approx(want, abs=0.05), (name, k)
            else:      # a derivative is smaller: the tag says how large it is
                f_px = Image.open(f).size[0] / (e["output_size_px"][0])
                assert got == pytest.approx(want * f_px, rel=0.02), (name, k)
    assert entry(cropped, "ticket.tif")["export_px_per_mm"] is None and entry(cropped, "ticket.tif")["output_ppm"] == {}


def test_other_never_enters_the_batch_scale_or_the_reference_sizes(cropped):
    base_scales = {k: v for k, v in scales(cropped).items() if k.startswith("p")}
    m = Manifest(str(cropped))
    # five prints and an Other of the same scan size: the reference sizes know the prints only
    for i in range(3, 6):
        e = json.loads(json.dumps(m.get("p0.tif")))
        e["input"] = e["input"].replace("p0", f"p{i}")
        m.scans[f"p{i}.tif"] = e
    refs = reference_sizes(m)
    assert list(refs) == [(2400, 2900, "polaroid-600")] and (2400, 2900, "other") not in refs
    n = len(pipeline._prints_by_group(m)[(2400, 2900)])
    assert n == 6       # three of the project and three added; the sheet, a crop of kind other, is not counted
    # an Other that is huge cannot move the median of the prints
    api.fix_scan(str(cropped), "sheet.tif", sets=box(10, 10, 2390, 2890), author="human")
    assert {k: v for k, v in scales(cropped).items() if k.startswith("p")} == base_scales
    assert (2400, 2900, "other") not in reference_sizes(Manifest(str(cropped)))


def test_a_forced_format_counts_like_a_hand_fixed_print(cropped):
    api.fix_scan(str(cropped), "sheet.tif", sets=SHEET, item="600", author="human")
    m = Manifest(str(cropped))
    found = pipeline._prints_by_group(m)[(2400, 2900)]
    assert len(found) == 4 and [f.name for _, f in found if f is not None] == ["polaroid-600"]
    assert pipeline._ref_group(m.get("sheet.tif")) == (2400, 2900, "polaroid-600")
    s = scales(cropped)["sheet.tif"]
    assert s.source == "batch-median"
    # the format replaces the shape guess in the scale: this 1400 x 1600 crop is a Go for the user
    api.fix_scan(str(cropped), "sheet.tif", item="go", author="human")
    m = Manifest(str(cropped))
    px_go = 1400 / 53.5 / 2 + 1600 / 65.8 / 2
    only_sheet = Manifest(str(cropped))
    for k in ("p0.tif", "p1.tif", "p2.tif"):
        only_sheet.scans.pop(k)
    only_sheet.scans["sheet.tif"]["meta"] = {"width": 1, "height": 1, "dpi_tag": None}
    got = resolve_scales(only_sheet, ["sheet.tif"], pipeline.settings_from_dict(only_sheet.data["settings"]))["sheet.tif"]
    assert got.source == "single-scan" and got.px_per_mm == pytest.approx(px_go, rel=1e-6)
    assert entry(cropped, "sheet.tif")["format"]["name"] == "polaroid-go"


# --------------------------------------------------------------------------- export key, retag and the metadata tie
def q0():
    return np.array([[10.0, 20.0], [1000.5, 21.0], [1001.0, 1300.25], [9.5, 1299.0]])


def test_keys_without_an_item_are_the_keys_of_before():
    s, sc = Settings(), Scale(25.6, "ppi")
    assert s.export_key(q0(), sc) == "2fecc94a02844b31" and s.export_base(q0(), sc) == "5e4aa68ee0da87b8"
    assert s.export_key(q0(), sc, item=None) == "2fecc94a02844b31"
    assert s.export_key(q0(), sc, None, False, None, {}) == "2fecc94a02844b31"      # no choice: nothing in the key
    assert s.export_key(q0(), sc, item={"kind": "bogus"}) == "2fecc94a02844b31"


def test_an_item_enters_the_key_only_when_set():
    s, sc = Settings(), Scale(25.6, "ppi")
    k = {n: s.export_key(q0(), sc, item=i) for n, i in (("none", None), ("600", {"kind": "600"}), ("go", {"kind": "go"}),
                                                       ("other", {"kind": "other"}),
                                                       ("other500", {"kind": "other", "scan_ppi": 500.0}),
                                                       ("other600", {"kind": "other", "scan_ppi": 600.0}))}
    assert len(set(k.values())) == len(k)
    b = {n: s.export_base(q0(), sc, item=i) for n, i in (("none", None), ("600", {"kind": "600"}), ("other", {"kind": "other"}))}
    assert len(set(b.values())) == 3


def test_the_key_of_other_leaves_out_what_it_does_not_use():
    sc = Scale(25.6, "ppi")
    other = {"kind": "other"}
    a = Settings().export_key(q0(), sc, item=other)
    assert a == Settings(margin_mm=0.5, flatten="off", shadows="off").export_key(q0(), sc, {"left": {"coef": [1, 2]}}, False,
                                                                                {"found": True, "quad": [[1, 1]]}, other)
    assert a != Settings(derivatives=()).export_key(q0(), sc, item=other)
    assert a != Settings().export_key(q0() + 1, sc, item=other)
    # the base of Other has no scale in it at all: a changed ppi or profile is a retag, never a new export
    b = Settings().export_base(q0(), Scale(25.6, "ppi"), item=other)
    assert b == Settings(ppi=500.0).export_base(q0(), Scale(10.0, "ppi"), item={"kind": "other", "scan_ppi": 500.0})
    assert b == Settings().export_base(q0(), Scale(10.0, "scan-ppi"), item={"kind": "other", "scan_ppi": 900.0})
    assert b == Settings().export_base(q0(), Scale(10.0, "batch-median"), item=other)
    assert Settings().export_key(q0(), Scale(25.6, "ppi"), item=other) != Settings().export_key(q0(), Scale(10.0, "ppi"), item=other)


def test_a_new_scale_retags_an_export_that_has_tags_and_exports_again_one_that_has_none(src, cropped):
    """The tie of retag and metadata rewrite: tags are patched in place, a missing tag cannot be, so the scan is exported again."""
    one = {"title": "{stem} one"}
    process(src, cropped, metadata=one)      # ticket: no tag; sheet: the median of the prints; lift: its own tag
    assert dpi_of(cropped / "ticket.tif") is None and dpi_of(cropped / "lift.tif") == pytest.approx(400.0, abs=0.05)
    px = lambda n: tifffile.imread(str(cropped / n)).tobytes()

    def title(n):
        with tifffile.TiffFile(str(cropped / n)) as tf:
            return tf.pages[0].tags[270].value

    # the ticket gets a ppi of its own: it had no tag to patch, so it is exported again (with the metadata)
    api.fix_scan(str(cropped), "ticket.tif", scan_ppi=800, author="human")
    s = process(src, cropped, metadata=one)
    assert s["exported"] == 1 and s["retagged"] == 0 and s["metadata_updated"] == 0
    assert dpi_of(cropped / "ticket.tif") == pytest.approx(800.0, abs=0.05) and title("ticket.tif") == "ticket one"
    assert entry(cropped, "ticket.tif")["export_px_per_mm"] == pytest.approx(800 / 25.4)
    # the sheet gets one and the title changes in the same run: tags patched, headers rewritten, pixels untouched
    sheet_px, sheet_bytes = px("sheet.tif"), (cropped / "web" / "sheet.jpg").read_bytes()
    api.fix_scan(str(cropped), "sheet.tif", scan_ppi=1200, author="human")
    s = process(src, cropped, metadata={"title": "{stem} two"})
    assert s["exported"] == 0 and s["retagged"] == 1 and s["metadata_updated"] == 6
    assert dpi_of(cropped / "sheet.tif") == pytest.approx(1200.0, abs=0.05) and px("sheet.tif") == sheet_px
    assert title("sheet.tif") == "sheet two" and title("ticket.tif") == "ticket two"
    assert dpi_of(cropped / "web" / "sheet.jpg") == pytest.approx(1200.0 * Image.open(cropped / "web" / "sheet.jpg").size[0]
                                                                  / entry(cropped, "sheet.tif")["output_size_px"][0], rel=0.01)
    # taking the ppi away leaves no scale at all: that cannot be patched in, the ticket is exported again without a tag
    api.fix_scan(str(cropped), "ticket.tif", scan_ppi=0, author="human")
    s = process(src, cropped, metadata={"title": "{stem} two"})
    assert s["exported"] == 1 and dpi_of(cropped / "ticket.tif") is None and title("ticket.tif") == "ticket two"
    assert entry(cropped, "ticket.tif")["export_px_per_mm"] is None


def test_a_retag_of_other_needs_no_new_pixels_at_any_ratio(src, cropped):
    api.fix_scan(str(cropped), "ticket.tif", scan_ppi=500, author="human")
    process(src, cropped)
    px = tifffile.imread(str(cropped / "ticket.tif")).tobytes()
    api.fix_scan(str(cropped), "ticket.tif", scan_ppi=3200, author="human")      # six times as much
    s = process(src, cropped)
    assert s["exported"] == 0 and s["retagged"] == 1
    assert dpi_of(cropped / "ticket.tif") == pytest.approx(3200, abs=0.1) and tifffile.imread(str(cropped / "ticket.tif")).tobytes() == px


def test_a_changed_item_exports_again(src, cropped):
    process(src, cropped)
    s = process(src, cropped)
    assert s["exported"] == 0 and s["retagged"] == 0      # nothing to do
    api.fix_scan(str(cropped), "ticket.tif", item="600", author="human")
    s = process(src, cropped)
    assert s["exported"] == 1
    got = tifffile.imread(str(cropped / "ticket.tif"))
    assert got.shape[:2] != (1200, 700)      # a print: cut with the margin, 0.1 mm each side
    assert got.shape[0] < 1200 and got.shape[1] < 700


# --------------------------------------------------------------------------- the app and the review page without a 500
def test_the_app_counts_and_export_states_of_a_scan_cropped_by_hand(src, out):
    from polaroid_processor.app.project import Project
    proj = Project(str(src), str(out))
    snap = proj.snapshot()
    assert snap["counts"]["failed"] == 3
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human")
    snap = proj.snapshot()
    rec = {r["key"]: r for r in snap["scans"]}["ticket.tif"]
    assert snap["counts"]["failed"] == 2 and rec["status"] == "ok" and rec["item"] == "other" and rec["quad_source"] == "override"
    assert rec["error"] is None and rec["size_px"] == [700, 1200] and rec["format"] is None
    assert snap["counts"]["to_export"] >= 1
    assert proj._export_inputs() and any(p.endswith("ticket.tif") for p in proj._export_inputs())      # an export run takes it
    proj.start("export")
    proj._thread.join(timeout=300)
    assert proj.run.state == "finished" and proj.run.errors == 0
    snap = proj.snapshot()
    rec = {r["key"]: r for r in snap["scans"]}["ticket.tif"]
    assert rec["export"] == "done" and (out / "ticket.tif").exists()
    assert not any(r["key"] == "ticket.tif" and r["export"] in ("none", "outdated") for r in snap["scans"])
    # the preview is made as drawn: no margin
    from polaroid_processor.preview import preview_key
    proj.make_preview("ticket.tif")
    pv = Manifest(str(out)).get("ticket.tif")["preview"]
    assert pv["key"] == preview_key(quad(TICKET), None, False, 0.0, "off") and pv["size"] == [350, 600]
    assert {r["key"]: r for r in proj.snapshot()["scans"]}["ticket.tif"]["stale"] is False
    # it was exported without a resolution tag: a scale cannot be patched in, so the export is outdated, not "tags"
    api.fix_scan(str(out), "ticket.tif", scan_ppi=800, author="human")
    rec = {r["key"]: r for r in proj.snapshot()["scans"]}["ticket.tif"]
    assert Manifest(str(out)).get("ticket.tif")["export_px_per_mm"] is None and rec["export"] == "outdated"


def test_the_review_page_lists_a_cropped_scan_and_serves_its_tiles(out):
    from test_review_server import Page
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="agent")
    page = Page(out, show_all=True)
    try:
        status, listing = page.json("/api/scans")
        assert status == 200
        rec = {r["key"]: r for r in listing["scans"]}["ticket.tif"]
        assert rec["item"]["kind"] == "other" and rec["item"]["agent_crop"] is True and rec["status"] == "needs_review"
        assert "item" not in {r["key"]: r for r in listing["scans"]}["p0.tif"]      # scans as detected: records as before
        lift = {r["key"]: r for r in listing["scans"]}["lift.tif"]      # slice 2: a scan nothing was found on opens the place step
        assert lift["quad"] is None and lift["status"] == "failed" and lift["detected"] is False and "item" not in lift
        status, tiles = page.json("/api/scans/ticket.tif/tiles")
        assert status == 200 and len(tiles["tiles"]) == 9
        status, _, png = page.request("/api/scans/ticket.tif/tile/tl.png")
        assert status == 200 and png[:4] == b"\x89PNG"
        status, body = page.json("/api/scans/lift.tif/tiles")      # its overview only: the corner tiles need corners
        assert status == 200 and [t["name"] for t in body["tiles"]] == ["overview"]
        assert page.request("/api/scans/lift.tif/tile/tl.png")[0] == 404
        assert page.request("/api/scans/lift.tif/tile/overview.png")[2][:4] == b"\x89PNG"
        # a human's Confirm clears the agent mark
        status, rec = page.json("/api/scans/ticket.tif/confirm", {})
        assert status == 200 and rec["status"] == "ok" and rec["item"]["agent_crop"] is False
        # and a save with four corners by a human is a hand crop
        status, rec = page.json("/api/scans/ticket.tif/save", {"corners": TICKET})
        assert status == 200 and rec["status"] == "ok"
        status, body = page.json("/api/scans/ticket.tif/save", {"corners": box(400, 300, 440, 1500)})
        assert status == 400 and "50 px" in body["error"]
    finally:
        page.close()


def test_the_review_page_with_a_photo_crop_has_no_window_for_other(out):
    from test_review_server import Page
    m = Manifest(str(out))
    m.data["settings"]["photo_crop"] = "tiff"
    m.save()
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human")
    page = Page(out, show_all=True)
    try:
        status, listing = page.json("/api/scans")
        assert status == 200
        assert "window" not in {r["key"]: r for r in listing["scans"]}["ticket.tif"]
        status, body = page.json("/api/scans/ticket.tif/tiles?of=window")
        assert status == 400 and "photo window" in body["error"]
    finally:
        page.close()


# --------------------------------------------------------------------------- inspect of a scan without an outline
def test_inspect_of_a_failed_scan_shows_the_whole_scan_with_a_grid_and_the_start_rectangle(out):
    rec, sheet = api.inspect_scan("ticket.tif", str(out))
    assert rec["failed"] is True and rec["start_rectangle"] is True and rec["scan_px"] == [1600, 2000]
    assert rec["corners"]["tl"] == [160.0, 200.0] and rec["corners"]["br"] == [1440.0, 1800.0]
    assert rec["item"] == "other" and rec["status"] == "failed" and rec["error"].startswith("detection failed")
    assert rec["grid"] == 100 and "fix" in rec["hint"] and Path(rec["image"]).is_file()
    assert sheet.shape[0] > 1000 and sheet.shape[2] == 3
    assert sheet.shape[1] == 1280      # the long edge (2000 px) is 1600 px (+ the header bar on the height)
    rec, _ = api.inspect_scan("sheet.tif", str(out), item="600", grid=250, write=False)
    c = rec["corners"]
    assert rec["item"] == "600" and rec["grid"] == 250 and rec["image"] is None
    assert (c["tr"][0] - c["tl"][0]) / (c["bl"][1] - c["tl"][1]) == pytest.approx(88 / 106.85, abs=1e-3)
    assert rec["px_per_mm"] == pytest.approx(PX_PER_MM, rel=0.01)      # sized by the median of the prints
    c0 = np.array([c[n] for n in CORNERS])
    assert (c0[1] - c0[0])[0] == pytest.approx(88 * rec["px_per_mm"], abs=0.05)
    with pytest.raises(api.UsageError, match="item for inspect"):
        api.inspect_scan("ticket.tif", str(out), item="auto")
    with pytest.raises(api.UsageError, match="item must be"):
        api.inspect_scan("ticket.tif", str(out), item="x")


def test_inspect_of_a_failed_scan_without_a_manifest(src, tmp_path):
    rec, _ = api.inspect_scan(str(src / "ticket.tif"), image=str(tmp_path / "t.png"))
    assert rec["failed"] and rec["corners"]["tl"] == [160.0, 200.0] and (tmp_path / "t.png").is_file()
    rec, _ = api.inspect_scan(str(src / "p0.tif"), write=False)      # a print detects as ever
    assert "failed" not in rec and rec["quad_source"] == "auto"


def test_inspect_of_a_cropped_failed_scan_is_the_normal_sheet_at_the_crop(out):
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human")
    rec, _ = api.inspect_scan("ticket.tif", str(out), write=False)
    assert "failed" not in rec and rec["quad_source"] == "override" and rec["corners"]["tl"] == [400.0, 300.0]
    assert rec["item"]["kind"] == "other"


def test_a_failed_overview_does_not_write_into_the_input_folder(src, out):
    with pytest.raises(api.UsageError, match="input folder"):
        api.inspect_scan(str(src / "ticket.tif"), image=str(src / "x.png"))
    assert not (src / "x.png").exists()


# --------------------------------------------------------------------------- the CLI and the MCP server
def run_cli(capsys, *argv):
    rc = cli.main([str(a) for a in argv])
    out = capsys.readouterr()
    recs = [json.loads(l) for l in out.out.splitlines() if l.startswith("{")]
    return rc, recs, out.err


def test_cli_crops_a_failed_scan_and_inspects_it(out, capsys):
    rc, recs, _ = run_cli(capsys, "inspect", "ticket.tif", "-m", out, "--json", "--item", "go")
    assert rc == 0 and recs[0]["failed"] and recs[0]["item"] == "go"
    rc, _, _ = run_cli(capsys, "inspect", "ticket.tif", "-m", out)
    assert rc == 0
    printed = capsys.readouterr()
    rc, recs, _ = run_cli(capsys, "fix", out, "ticket.tif", "--set", "tl=400,300", "--set", "tr=1100,300", "--set", "br=1100,1500",
                          "--set", "bl=400,1500", "--item", "other", "--scan-ppi", "800", "--author", "agent", "--note", "cli",
                          "--json")
    assert rc == 0 and recs[-1]["type"] == "fix" and recs[-1]["status"] == "needs_review"
    assert recs[-1]["item"]["kind"] == "other" and recs[-1]["item"]["scan_ppi"] == 800.0 and recs[-1]["item"]["agent_crop"] is True
    rc, recs, _ = run_cli(capsys, "fix", out, "ticket.tif", "--accept", "--author", "human", "--json")
    assert rc == 0 and recs[-1]["status"] == "ok"
    rc, recs, _ = run_cli(capsys, "fix", out, "ticket.tif", "--reset", "--json")
    assert rc == 0 and recs[-1]["status"] == "failed"


def test_cli_errors_are_exit_2(out, capsys):
    rc, recs, _ = run_cli(capsys, "fix", out, "ticket.tif", "--set", "tl=400,300", "--json")
    assert rc == 2 and "all four corners" in recs[-1]["error"]
    rc, recs, _ = run_cli(capsys, "fix", out, "ticket.tif", "--set", "tl=400,300", "--set", "tr=1100,300", "--set", "br=1100,1500",
                          "--set", "bl=400,1500", "--scan-ppi", "-3", "--json")
    assert rc == 2 and "scan_ppi" in recs[-1]["error"]
    rc, recs, _ = run_cli(capsys, "fix", out, "ticket.tif", "--set", "tl=-4,300", "--set", "tr=1100,300", "--set", "br=1100,1500",
                          "--set", "bl=400,1500", "--json")
    assert rc == 2 and "inside the scan" in recs[-1]["error"]
    with pytest.raises(SystemExit):
        cli.main(["fix", str(out), "ticket.tif", "--item", "polaroid"])
    capsys.readouterr()
    assert entry(out, "ticket.tif")["status"] == "failed"


def test_cli_human_line_shows_the_item(out, capsys):
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human")
    rc = cli.main(["status", str(out)])
    text = capsys.readouterr().out
    assert rc == 0 and "item other" in text and "ticket.tif" in text


def test_mcp_inspects_and_crops_a_failed_scan(out):
    res = mcp_server.tool_inspect({"scan": "ticket.tif", "output": str(out)}, None)
    assert not res["isError"] and res["structuredContent"]["failed"] and res["content"][1]["type"] == "image"
    res = mcp_server.tool_inspect({"scan": "ticket.tif", "output": str(out), "item": "600", "grid": 200, "save": False}, None)
    assert res["structuredContent"]["item"] == "600" and res["structuredContent"]["grid"] == 200
    res = mcp_server.tool_fix({"output": str(out), "scan": "ticket.tif", "set": TICKET, "item": "other", "scan_ppi": 800,
                               "note": "mcp", "export": True}, None)
    assert not res["isError"]
    rec = res["structuredContent"]
    assert rec["status"] == "needs_review" and rec["item"]["agent_crop"] is True and rec["summary"]["exported"] == 1
    assert (out / "ticket.tif").exists()
    res = mcp_server.tool_status({"output": str(out), "needs_review": True}, None)
    got = {r["file"]: r for r in res["structuredContent"]["scans"]}
    assert got["ticket.tif"]["item"]["kind"] == "other" and "cropped by agent" in got["ticket.tif"]["warnings"]
    with pytest.raises(api.UsageError, match="all four corners"):
        mcp_server.tool_fix({"output": str(out), "scan": "lift.tif", "set": {"tl": [1, 1]}, "export": False}, None)
    tools = {t["name"]: t for t in mcp_server.TOOLS}
    assert {"item", "scan_ppi"} <= set(tools["polaroid_fix"]["inputSchema"]["properties"])
    assert "item" in tools["polaroid_inspect"]["inputSchema"]["properties"]


# --------------------------------------------------------------------------- old manifests, records as before
def test_a_scan_without_a_choice_has_the_records_of_before(base, src):
    m = Manifest(str(base))
    for k in ("p0.tif", "p1.tif", "p2.tif", "ticket.tif"):
        rec = result_record(m, k)
        assert "item" not in rec and "detection_error" not in rec
        assert "item_fix" not in m.get(k)


def test_an_old_manifest_without_the_field_works(out, src):
    """A manifest made before this feature has no ``item_fix`` anywhere: runs, fixes and counts work as before."""
    m = Manifest(str(out))
    assert all("item_fix" not in e and "detection_error" not in e for _, e in m.items())
    seen = []
    api.process([str(src)], str(out), api.make_settings(workers=1, export=False), on_jobs=seen.append)
    assert seen == [3]      # the three failed scans are tried again, as before
    api.fix_scan(str(out), "p2.tif", shifts={"top": [1.0]}, author="human")      # a plain fix, no item
    e = entry(out, "p2.tif")
    assert "item_fix" not in e and e["override"]
    assert api.status(str(out))["counts"]["failed"] == 3


def test_app_runs_keep_a_cropped_scan_and_export_it(src, out):
    from polaroid_processor.app.project import Project
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, scan_ppi=800, author="human")
    api.fix_scan(str(out), "sheet.tif", sets=SHEET, author="agent")
    proj = Project(str(src), str(out))
    proj.start("detect")
    proj._thread.join(timeout=300)
    assert proj.run.state == "finished" and entry(out, "ticket.tif")["status"] == "ok"      # Detect does not fail it again
    assert entry(out, "sheet.tif")["status"] == "needs_review"
    proj.start("all")
    proj._thread.join(timeout=300)
    assert proj.run.state == "finished" and proj.run.errors == 1      # only the lift fails to detect, as ever
    snap = proj.snapshot()
    by = {r["key"]: r for r in snap["scans"]}
    assert by["ticket.tif"]["export"] == "done" and by["sheet.tif"]["export"] == "done" and by["ticket.tif"]["status"] == "ok"
    assert by["sheet.tif"]["status"] == "needs_review" and by["sheet.tif"]["agent_crop"] is True
    assert snap["counts"]["failed"] == 1 and snap["counts"]["needs_review"] >= 1      # only the lift is left failed
    assert dpi_of(out / "ticket.tif") == pytest.approx(800.0, abs=0.05)


def test_a_tiff_without_a_resolution_has_none_after_its_headers_are_written(tmp_path):
    """tifffile writes 1/1 with the unit "none" when it is given no resolution: that is no resolution, and the header
    writer drops those tags. A real resolution (unit centimetre or inch) is left alone."""
    from polaroid_processor import metadata as M
    from polaroid_processor.meta_patch import set_metadata
    a = np.zeros((20, 30, 3), np.uint8)
    none, some = tmp_path / "none.tif", tmp_path / "some.tif"
    tifffile.imwrite(str(none), a, photometric="rgb", metadata=None, software=False)
    tifffile.imwrite(str(some), a, photometric="rgb", metadata=None, software=False, resolution=(100.0, 100.0),
                     resolutionunit="CENTIMETER")
    fm = M.fill({"title": "x"}, stem="x", project="p", format_name="", make="", model="", software="s")
    for p in (none, some, none):      # twice: a rewrite of the file does the same
        set_metadata(str(p), fm)
    with tifffile.TiffFile(str(none)) as tf:
        assert not {282, 283, 296} & {t.code for t in tf.pages[0].tags.values()}
        assert tf.pages[0].tags[270].value == "x"
    with tifffile.TiffFile(str(some)) as tf:
        t = {x.code: x.value for x in tf.pages[0].tags.values()}
        assert t[282] == (100, 1) and t[296] == 3
    assert np.array_equal(tifffile.imread(str(none)), a)


# =========================================================================== review round 1
def edit(out, name, **fields):
    m = Manifest(str(out))
    m.get(name).update(fields)
    m.save()


def as_drawn(src, out, name):
    """True when the export of ``name`` is the plain rectification of its outline (no margin, no unwarp, no shadows)."""
    e = entry(out, name)
    want = rectify(load_rgb(str(src / name)), Manifest.effective_quad(e), 0.0, "cubic")
    got = tifffile.imread(str(out / name))
    return got.shape == want.shape and bool(np.array_equal(got, want))


def print_size(out, name):
    """The size an export of a print of that outline has: the outline less the margin on every side."""
    from polaroid_processor.geometry import output_size
    e = entry(out, name)
    return list(output_size(Manifest.effective_quad(e), 0.1 * e["scale"]["px_per_mm"]))


def test_r1a_an_item_set_during_the_run_after_the_scans_own_export_is_cut_as_set(src, out):
    fired = []      # the scale is explicit (ppi): the stamp must not trust a known scale alone

    def progress(key, what):
        if key == "p0.tif" and what == "detected" and not fired:
            fired.append(api.fix_scan(str(out), "p0.tif", item="other", author="human"))

    api.process([str(src)], str(out), api.make_settings(workers=1, redetect=True, ppi=PPI), progress=progress)
    assert fired and entry(out, "p0.tif")["item_fix"]["kind"] == "other"
    assert as_drawn(src, out, "p0.tif")
    assert process(src, out, ppi=PPI)["exported"] == 0      # and the key says so


def test_r1b_an_item_taken_away_during_the_run_gives_a_print(src, out):
    process(src, out, export=False, ppi=PPI)
    api.fix_scan(str(out), "p0.tif", item="other", author="human", export=True)
    quad0 = Manifest.effective_quad(entry(out, "p0.tif"))
    assert entry(out, "p0.tif")["output_size_px"] == [round(v) for v in api.quad_size(quad0)]
    fired = []

    def progress(key, what):
        if key == "p0.tif" and what == "detected" and not fired:
            fired.append(api.fix_scan(str(out), "p0.tif", item="auto", author="human"))

    api.process([str(src)], str(out), api.make_settings(workers=1, redetect=True, ppi=PPI), progress=progress)
    assert fired and entry(out, "p0.tif").get("item_fix") is None
    assert entry(out, "p0.tif")["output_size_px"] == print_size(out, "p0.tif")
    assert process(src, out, ppi=PPI)["exported"] == 0


def test_r1c_a_replaced_scan_file_is_not_cut_with_the_item_of_the_old_picture(src, out, tmp_path):
    work = tmp_path / "in"
    shutil.copytree(src, work)
    process(work, out, export=False)
    api.fix_scan(str(out), "p0.tif", item="other", author="human", export=True)      # a detected scan, set to Other, no override
    rgb, _ = make_scan("white", 0.7, seed=123)
    tifffile.imwrite(str(work / "p0.tif"), rgb, photometric="rgb")      # a new scan of a print under the same name
    process(work, out)
    e = entry(out, "p0.tif")
    assert e.get("item_fix") is None and e["status"] in ("ok", "needs_review")
    assert e["output_size_px"] == print_size(out, "p0.tif")      # cut as a print: with margin
    assert "curves" in e["detection"]      # and traced, as a print is
    assert "item-drop" in hist(out, "p0.tif")
    assert process(work, out)["exported"] == 0


def test_r1_r12_the_job_ignores_override_and_item_when_the_file_changed(src, out, tmp_path):
    """The detection job cuts what the file holds now: an override and an item described the old picture."""
    from polaroid_processor.scan_window import outline_id
    api.fix_scan(str(out), "p1.tif", shifts={"left": [3.0]}, author="human")
    api.fix_scan(str(out), "p0.tif", item="other", author="human")
    for name, kind in (("p1.tif", None), ("p0.tif", "other")):
        e = entry(out, name)
        job = {"key": name, "path": str(src / name), "detect": True, "detection": None,
               "override": e["override"]["quad"] if e.get("override") else None, "item": e.get("item_fix"), "export": True,
               "scale": None, "window": None, "window_fix": None, "fingerprint": "s32:not-the-fingerprint-of-this-file",
               "export_settings": pipeline._export_settings(api.make_settings(workers=1), tmp_path / name),
               "detect_params": api.make_settings().detect_params, "preview": None}
        res = pipeline._job(job)
        assert "error" not in res, res
        assert res["export_outline"] == outline_id(np.array(res["detection"]["quad"])) and res["export_kind"] is None
        job["fingerprint"] = res["fingerprint"]      # the file is the one the manifest knows: the choices hold
        res = pipeline._job(job)
        assert res["export_kind"] == kind
        if e.get("override"):
            assert res["export_outline"] == outline_id(np.array(e["override"]["quad"]))


def test_r1_a_curled_print_set_to_other_is_cut_straight_without_shadow_removal(tmp_path):
    from test_flatten_shading import bowed_scan
    rgb, _ = bowed_scan()
    folder = tmp_path / "in"
    folder.mkdir()
    tifffile.imwrite(str(folder / "curl.tif"), rgb, photometric="rgb")
    out = tmp_path / "out"
    process(folder, out, ppi=PPI)
    e = entry(out, "curl.tif")
    assert e["flatten"]["apply"] is True and "curves" in e["detection"]
    api.fix_scan(str(out), "curl.tif", item="other", author="human", export=True)
    e = entry(out, "curl.tif")
    assert e["flatten"]["apply"] is False and e["shadows"] == {"sides": []}
    assert as_drawn(folder, out, "curl.tif")


def test_r2_a_plain_fixed_print_whose_detection_failed_does_not_become_other(out):
    api.fix_scan(str(out), "p0.tif", shifts={"left": [2.0]}, author="human")
    edit(out, "p0.tif", detection=None, status="failed", error="detection failed: no plausible print outline")
    api.fix_scan(str(out), "p0.tif", accept=True, author="human")
    e = entry(out, "p0.tif")
    assert e.get("item_fix") is None and e["status"] == "ok" and e["format"]["name"] == "polaroid-600"
    api.fix_scan(str(out), "p0.tif", shifts={"left": [1.0]}, author="human")
    assert entry(out, "p0.tif").get("item_fix") is None
    api.fix_scan(str(out), "p0.tif", accept=True, author="agent")      # an agent's accept: nothing to mark
    assert entry(out, "p0.tif").get("item_fix") is None and entry(out, "p0.tif")["status"] == "ok"
    # decision 8 still holds for it: an agent moves a crop of a scan without a detected outline: a human looks
    api.fix_scan(str(out), "p0.tif", shifts={"left": [1.0]}, author="agent", note="moved")
    e = entry(out, "p0.tif")
    assert e.get("item_fix") is None and e["status"] == "needs_review" and e["warnings"][-1] == "cropped by agent"
    assert e["agent_crop"]["note"] == "moved"
    api.fix_scan(str(out), "p0.tif", item="auto", author="agent")      # an agent's item auto does not clear it (R6)
    assert entry(out, "p0.tif")["status"] == "needs_review"
    api.fix_scan(str(out), "p0.tif", accept=True, author="human")
    e = entry(out, "p0.tif")
    assert e["status"] == "ok" and e["agent_crop"] is None and e.get("item_fix") is None
    assert "agent-crop-clear" in hist(out, "p0.tif")


def test_r2_the_first_crop_still_needs_a_kind(out):
    with pytest.raises(api.UsageError, match="needs an item kind"):
        api.fix_scan(str(out), "ticket.tif", sets=TICKET, item="auto", author="human")
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human")
    api.fix_scan(str(out), "ticket.tif", item="auto", author="human")      # later: a crop without a kind is a plain outline
    e = entry(out, "ticket.tif")
    assert e.get("item_fix") is None and e["override"] and e["status"] == "ok"


def test_r6_the_agent_mark_has_its_own_field_and_only_a_human_clears_it(out):
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="agent")
    e = entry(out, "ticket.tif")
    assert e["agent_crop"]["author"] == "agent" and "agent_crop" not in e["item_fix"]
    api.fix_scan(str(out), "ticket.tif", item="auto", author="agent")
    e = entry(out, "ticket.tif")
    assert e.get("item_fix") is None and e["agent_crop"] and e["status"] == "needs_review"
    api.fix_scan(str(out), "ticket.tif", item="other", author="agent")
    assert entry(out, "ticket.tif")["status"] == "needs_review"
    api.fix_scan(str(out), "ticket.tif", item="auto", author="human")      # an item change alone is not a save
    assert entry(out, "ticket.tif")["agent_crop"] and entry(out, "ticket.tif")["status"] == "needs_review"
    api.fix_scan(str(out), "ticket.tif", accept=True, author="human")
    assert entry(out, "ticket.tif")["agent_crop"] is None and entry(out, "ticket.tif")["status"] == "ok"


def test_r6_the_agent_mark_is_logged_dropped_with_the_crop_and_reset(src, out, tmp_path):
    work = tmp_path / "in"
    shutil.copytree(src, work)
    process(work, out, export=False)
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="agent")
    h = entry(out, "ticket.tif")["history"][-1]
    assert h["action"] == "agent-crop" and h["agent_crop_before"] is None and h["agent_crop_after"]["author"] == "agent"
    tifffile.imwrite(str(work / "ticket.tif"), failing(2000, 1600, 77), photometric="rgb")
    process(work, out, export=False)
    e = entry(out, "ticket.tif")
    assert e.get("agent_crop") is None and e.get("item_fix") is None and e["status"] == "failed"
    assert e["history"][-1]["action"] == "agent-crop-drop" and e["history"][-1]["author"] == "system"
    assert "detection_error" not in e      # R3: the old picture's message goes with its crop
    api.fix_scan(str(out), "lift.tif", sets=TICKET, author="agent")
    api.fix_scan(str(out), "lift.tif", reset=True, author="human")
    e = entry(out, "lift.tif")
    assert e["agent_crop"] is None and "agent-crop-clear" in hist(out, "lift.tif")


def test_r10_every_change_of_the_mark_is_logged_so_the_merge_sees_it(out):
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="agent")
    other_writer = Manifest(str(out))
    n = len(entry(out, "ticket.tif")["history"])
    api.fix_scan(str(out), "ticket.tif", item="other", author="human")      # a human's item alone: no change, no entry
    api.fix_scan(str(out), "ticket.tif", scan_ppi=0, author="human")
    assert len(entry(out, "ticket.tif")["history"]) == n and entry(out, "ticket.tif")["agent_crop"]
    api.fix_scan(str(out), "ticket.tif", accept=True, author="human")      # the Confirm clears it, and the entry says so
    e = entry(out, "ticket.tif")
    assert e["agent_crop"] is None and e["history"][-1]["action"] == "agent-crop-clear"
    assert e["history"][-1]["agent_crop_before"]["author"] == "agent" and e["history"][-1]["agent_crop_after"] is None
    other_writer.refresh()
    assert "ticket.tif" in other_writer.fixed      # the batch judges it again


def test_r3_reset_keeps_the_error_of_a_scan_that_was_never_cropped(out):
    edit(out, "lift.tif", error="worker process crashed on this scan")
    api.fix_scan(str(out), "lift.tif", reset=True, author="human")
    e = entry(out, "lift.tif")
    assert e["status"] == "failed" and e["error"] == "worker process crashed on this scan"
    # a pending scan (never run) is closed to a reset too: see test_r22


def test_r3_a_crop_over_any_earlier_error_restores_it_on_reset(out):
    edit(out, "lift.tif", error="worker process crashed on this scan")
    api.fix_scan(str(out), "lift.tif", sets=TICKET, item="600", author="human")
    e = entry(out, "lift.tif")
    assert e["status"] == "ok" and e["error"] is None and e["detection_error"] == "worker process crashed on this scan"
    assert e["scale"] and e["format"]
    api.fix_scan(str(out), "lift.tif", reset=True, author="human")
    e = entry(out, "lift.tif")
    assert e["status"] == "failed" and e["error"] == "worker process crashed on this scan" and e["warnings"] == []
    assert "detection_error" not in e and "scale" not in e and "format" not in e      # R16: nothing of the crop stays


def test_r4_a_scan_that_was_never_detected_stays_closed(out):
    m = Manifest(str(out))
    m.entry("ghost.tif")
    m.save()
    with pytest.raises(api.UsageError, match="not been detected"):
        api.fix_scan(str(out), "ghost.tif", sets=TICKET, author="human")
    with pytest.raises(api.UsageError, match="not been detected"):
        api.inspect_scan("ghost.tif", str(out), write=False)
    assert "item_fix" not in entry(out, "ghost.tif") and "override" not in entry(out, "ghost.tif")


def test_r5_a_detected_print_at_the_scan_edge_set_to_600_can_be_confirmed(out):
    p = np.array(entry(out, "p0.tif")["detection"]["quad"])
    api.fix_scan(str(out), "p0.tif", sets={"tl": [-5.0, float(p[0][1])]}, author="human")      # as before: no check
    api.fix_scan(str(out), "p0.tif", item="600", author="human")      # an item on an outline that is not a hand crop: no check
    api.fix_scan(str(out), "p0.tif", accept=True, author="human")      # Confirm changes no corner: no check
    assert entry(out, "p0.tif")["status"] in ("ok", "needs_review") and entry(out, "p0.tif")["item_fix"]["kind"] == "600"
    with pytest.raises(api.UsageError, match="inside the scan"):
        api.fix_scan(str(out), "p0.tif", nudges={"tl": [-1.0, 0.0]}, author="human")      # moving a corner is checked


def test_r8_inspect_of_a_print_set_to_other_draws_no_flatten_curve(out, monkeypatch):
    import types
    m = Manifest(str(out))
    m.get("p0.tif")["detection"]["curves"] = {}
    m.save()
    calls = []
    monkeypatch.setattr(api, "flatten_plan", lambda *a, **k: types.SimpleNamespace(apply=True))
    monkeypatch.setattr(api.flat, "curves_from_dict", lambda d: {})
    monkeypatch.setattr(api.flat, "outline_points", lambda *a, **k: calls.append(1) or np.array([[0, 0], [5, 5], [9, 9]], float))
    api.inspect_scan("p0.tif", str(out), write=False)
    assert calls      # a print follows its curve in the sheet
    calls.clear()
    api.fix_scan(str(out), "p0.tif", item="other", author="human")
    api.inspect_scan("p0.tif", str(out), write=False)
    assert not calls      # Other is exported straight


def test_r9_a_scan_cropped_during_the_run_is_exported_in_that_run(src, out, monkeypatch):
    fixes = []
    real = pipeline.reference_sizes

    def reference_sizes(manifest):
        sizes = real(manifest)
        if not fixes:
            fixes.append(True)      # while the sizes are checked: the scales are worked out already
            fixes.append(api.fix_scan(str(out), "ticket.tif", sets=TICKET, scan_ppi=800, author="human"))
        return sizes

    monkeypatch.setattr(pipeline, "reference_sizes", reference_sizes)
    summary = api.process([str(src)], str(out), api.make_settings(workers=1))
    monkeypatch.undo()
    e = entry(out, "ticket.tif")
    assert len(fixes) == 2 and e["status"] == "ok" and e["outputs"] and (out / "ticket.tif").exists()
    assert dpi_of(out / "ticket.tif") == pytest.approx(800.0, abs=0.05) and "ticket.tif" not in summary["failed"]


def test_r9_an_item_set_during_the_run_to_a_detected_scan_is_cut_as_set(src, out, monkeypatch):
    fixes = []
    real = pipeline.reference_sizes

    def reference_sizes(manifest):
        sizes = real(manifest)
        if not fixes:
            fixes.append(True)
            fixes.append(api.fix_scan(str(out), "p0.tif", item="other", scan_ppi=500, author="human"))
        return sizes

    monkeypatch.setattr(pipeline, "reference_sizes", reference_sizes)
    api.process([str(src)], str(out), api.make_settings(workers=1))
    monkeypatch.undo()
    assert len(fixes) == 2 and as_drawn(src, out, "p0.tif")
    assert dpi_of(out / "p0.tif") == pytest.approx(500.0, abs=0.05)      # at the scale of the fix, not the old one
    assert process(src, out)["exported"] == 0


def test_a_changed_file_whose_new_content_detects_gets_a_print(src, out, tmp_path):
    work = tmp_path / "in"
    shutil.copytree(src, work)
    process(work, out, export=False)
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, item="600", author="agent")
    rgb, _ = make_scan("white", 0.2, seed=99)
    tifffile.imwrite(str(work / "ticket.tif"), rgb, photometric="rgb")      # a real print where the ticket was
    process(work, out)
    e = entry(out, "ticket.tif")
    assert e["override"] is None and e.get("item_fix") is None and e.get("agent_crop") is None and e["detection"]["quad"]
    assert e["status"] in ("ok", "needs_review") and e["outputs"] and e["output_size_px"] == print_size(out, "ticket.tif")
    assert {"reset", "item-drop", "agent-crop-drop"} <= set(h["action"] for h in e["history"])
    assert "detection_error" not in e


def test_r11_the_window_check_runs_with_the_final_kind(out):
    m = Manifest(str(out))
    m.data["settings"]["photo_crop"] = "tiff"
    m.save()
    before = json.dumps(Manifest(str(out)).data["scans"], sort_keys=True)
    with pytest.raises(api.UsageError, match="window options do not apply"):      # the first crop is Other
        api.fix_scan(str(out), "ticket.tif", sets=TICKET, no_photo_crop=True, author="human")
    assert json.dumps(Manifest(str(out)).data["scans"], sort_keys=True) == before
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human")
    rec = api.fix_scan(str(out), "ticket.tif", reset=True, reset_window=False, no_photo_crop=False, author="human")
    assert rec["status"] == "failed"
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human")
    api.fix_scan(str(out), "p0.tif", no_photo_crop=True, author="human")
    rec = api.fix_scan(str(out), "p0.tif", reset=True, reset_window=True, author="human")      # not refused on a reset
    assert rec["type"] == "fix"


def test_r13_an_unknown_item_kind_does_not_break_the_review_list(out):
    from test_review_server import Page
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human")
    edit(out, "ticket.tif", item_fix={"kind": "bogus", "author": "x"})
    page = Page(out, show_all=True)
    try:
        status, listing = page.json("/api/scans")
        assert status == 200 and "item" not in {r["key"]: r for r in listing["scans"]}["ticket.tif"]
    finally:
        page.close()
    assert result_record(Manifest(str(out)), "ticket.tif").get("item") is None


def test_r14_the_nominal_window_follows_the_named_format(tmp_path):
    from test_photo_crop import write_scans
    from polaroid_processor.scan_window import nominal_window
    folder = write_scans(tmp_path / "w_in")
    out = tmp_path / "w_out"
    process(folder, out, photo_crop="tiff", ppi=PPI)
    api.fix_scan(str(out), "nw.tif", item="go", author="human")
    quad0 = Manifest.effective_quad(entry(out, "nw.tif"))
    rec, _ = api.inspect_scan("nw.tif", str(out), of="window", write=False)
    want = nominal_window(quad0, rec["window_kind"], "polaroid-go")["quad"]
    assert np.allclose([rec["window_corners"][n] for n in CORNERS], np.round(want, 1), atol=0.06)
    from polaroid_processor.review.server import ReviewApp
    info = ReviewApp(str(out), show_all=True)._window_info(Manifest(str(out)), entry(out, "nw.tif"), quad0, None)
    assert np.allclose(info["nominal"]["square"], nominal_window(quad0, "square", "polaroid-go")["quad"])
    api.fix_scan(str(out), "nw.tif", window_nudges={"br": [-3.0, -3.0]}, author="human")
    assert entry(out, "nw.tif")["window_fix"]["window"]["format"] == "polaroid-go"


def test_r15_a_plain_reset_is_what_it_was_before(out):
    """No re-judge with the photo crop off (so no profile lookup), and no message copied into a scan that never had a crop."""
    m = Manifest(str(out))
    m.data["settings"]["profile"] = "no-such-profile"
    m.save()
    api.fix_scan(str(out), "p0.tif", reset=True, author="human")      # before: no KeyError
    api.fix_scan(str(out), "ticket.tif", reset=True, author="human")
    e = entry(out, "ticket.tif")
    assert e["status"] == "failed" and "detection_error" not in e and e["error"].startswith("detection failed")
    assert [h["action"] for h in e["history"]] == ["reset"]


def test_r16_odd_arguments(out):
    for bad in (True, False):
        with pytest.raises(api.UsageError, match="scan_ppi must be"):
            api.fix_scan(str(out), "ticket.tif", sets=TICKET, scan_ppi=bad)
    with pytest.raises(api.UsageError, match="set alone"):
        api.fix_scan(str(out), "ticket.tif", sets=TICKET, shifts={"top": [4.0]})
    with pytest.raises(api.UsageError, match="set all four corners"):
        api.fix_scan(str(out), "ticket.tif", sets={"tl": [1, 1]})
    assert entry(out, "ticket.tif")["status"] == "failed"


# =========================================================================== decisions after the first review
def test_q2_concave_and_sliver_crops_are_refused_everywhere():
    concave = np.array([[400, 300], [1100, 300], [1100, 1500], [700, 500]], float)      # a dart: bl lies inside the triangle
    sliver = np.array([[400, 300], [1100, 300], [1300, 330], [600, 330]], float)      # sides 200 and 700 px, 30 px high
    with pytest.raises(ValueError, match="convex"):
        item_mod.check_crop_quad(concave, 1600, 2000)
    with pytest.raises(ValueError, match="too thin"):
        item_mod.check_crop_quad(sliver, 1600, 2000)
    thin_ok = np.array([[400, 300], [1100, 300], [1300, 350], [600, 350]], float)      # 50 px high: allowed
    item_mod.check_crop_quad(thin_ok, 1600, 2000)
    with pytest.raises(api.UsageError, match="convex"):
        api.check_crop_quad(concave, 1600, 2000)


def test_q2_the_refusals_reach_fix_cli_and_mcp_and_store_nothing(out, capsys):
    concave = {"tl": [400, 300], "tr": [1100, 300], "br": [1100, 1500], "bl": [700, 500]}
    sliver = {"tl": [400, 300], "tr": [1100, 300], "br": [1300, 330], "bl": [600, 330]}
    before = json.dumps(Manifest(str(out)).data["scans"], sort_keys=True)
    for bad, why in ((concave, "convex"), (sliver, "too thin")):
        with pytest.raises(api.UsageError, match=why):
            api.fix_scan(str(out), "ticket.tif", sets=bad, author="human")
        with pytest.raises(api.UsageError, match=why):
            mcp_server.tool_fix({"output": str(out), "scan": "ticket.tif", "set": bad, "export": False}, None)
        argv = ["fix", str(out), "ticket.tif", "--json"] + sum([["--set", f"{n}={x},{y}"] for n, (x, y) in bad.items()], [])
        assert cli.main(argv) == 2
        assert why in capsys.readouterr().out
    assert json.dumps(Manifest(str(out)).data["scans"], sort_keys=True) == before
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human")      # a moved corner of a crop is checked too
    with pytest.raises(api.UsageError, match="convex"):
        api.fix_scan(str(out), "ticket.tif", nudges={"bl": [300.0, -1000.0]}, author="human")
    from test_review_server import Page
    page = Page(out, show_all=True)
    try:
        status, body = page.json("/api/scans/ticket.tif/save", {"corners": concave})
        assert status == 400 and "convex" in body["error"]
    finally:
        page.close()


def test_q3_a_crop_with_the_agent_mark_counts_for_no_scale_and_no_reference_size(cropped):
    """It counts again once a human has confirmed it."""
    api.fix_scan(str(cropped), "sheet.tif", item="600", author="human")      # a human's item: an agent may not change it (slice 2), it moves corners only
    api.fix_scan(str(cropped), "sheet.tif", sets=SHEET, item="600", author="agent")
    m = Manifest(str(cropped))
    group = (2400, 2900)
    assert len(pipeline._prints_by_group(m)[group]) == 3 and pipeline._ref_group(m.get("sheet.tif")) == group + ("polaroid-600",)
    ref = group + ("polaroid-600",)
    m.scans["p3.tif"] = json.loads(json.dumps(m.get("p0.tif")))
    assert len(pipeline._prints_by_group(m)[group]) == 4 and ref not in reference_sizes(m)      # four prints and the crop: not five
    m.scans["p4.tif"] = json.loads(json.dumps(m.get("p0.tif")))
    assert len(pipeline._prints_by_group(m)[group]) == 5 and ref in reference_sizes(m)
    # a plain print without an item and with the mark counts for nothing either
    api.fix_scan(str(cropped), "sheet.tif", item="auto", author="human")      # a human's item change alone keeps the mark
    m = Manifest(str(cropped))
    assert m.get("sheet.tif")["agent_crop"] and len(pipeline._prints_by_group(m)[group]) == 3
    api.fix_scan(str(cropped), "sheet.tif", accept=True, author="human")
    m = Manifest(str(cropped))
    assert len(pipeline._prints_by_group(m)[group]) == 4      # confirmed: it counts like a hand-fixed print
    # the inferred scale of the crop itself comes from the prints, not from itself
    assert resolve_scales(m, ["sheet.tif"], pipeline.settings_from_dict(m.data["settings"]))["sheet.tif"].source == "batch-median"


def test_q3_the_marked_crop_cannot_move_the_scale_of_the_prints(cropped):
    base = {k: v for k, v in scales(cropped).items() if k.startswith("p")}
    api.fix_scan(str(cropped), "sheet.tif", item="600", author="human")
    api.fix_scan(str(cropped), "sheet.tif", sets=box(10, 10, 2390, 2890), item="600", author="agent")      # a far too big "600"
    assert {k: v for k, v in scales(cropped).items() if k.startswith("p")} == base
    api.fix_scan(str(cropped), "sheet.tif", accept=True, author="human")      # confirmed, it counts (and moves the median)
    assert {k: v for k, v in scales(cropped).items() if k.startswith("p")} != base


def test_q4_after_a_reset_to_failed_the_exported_files_stay_on_disk(src, out):
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human", export=True)
    files = {k: out / v for k, v in entry(out, "ticket.tif")["outputs"].items()}
    assert files and all(f.is_file() for f in files.values())
    api.fix_scan(str(out), "ticket.tif", reset=True, author="human")
    e = entry(out, "ticket.tif")
    assert e["status"] == "failed" and all(f.is_file() for f in files.values())      # kept, as built
    assert e["outputs"] == {k: str(v.relative_to(out)).replace("\\", "/") for k, v in files.items()}      # and still listed
    s = process(src, out)
    assert all(f.is_file() for f in files.values()) and "ticket.tif" in s["failed"]


def test_q5_an_agent_that_changes_a_humans_crop_marks_it_again(out):
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human")
    assert entry(out, "ticket.tif")["status"] == "ok" and entry(out, "ticket.tif").get("agent_crop") is None
    for kw in ({"nudges": {"tl": [3.0, 3.0]}}, {"shifts": {"right": [4.0]}}, {"sets": {"br": [1105.0, 1500.0]}}):
        api.fix_scan(str(out), "ticket.tif", author="human", accept=True)
        assert entry(out, "ticket.tif")["status"] == "ok"
        api.fix_scan(str(out), "ticket.tif", author="agent", note="moved", **kw)
        e = entry(out, "ticket.tif")
        assert e["status"] == "needs_review" and e["agent_crop"]["note"] == "moved" and e["warnings"][-1] == "cropped by agent"
        assert e["history"][-1]["action"] == "agent-crop" and e["history"][-1]["agent_crop_before"] is None


# --- Q7: a hand crop or an item is made only on the file the manifest knows
def stored(out):
    return json.dumps(Manifest(str(out)).data["scans"], sort_keys=True)


def test_q7_a_crop_or_item_is_refused_when_the_scan_file_changed(src, out, tmp_path):
    work = tmp_path / "in"
    shutil.copytree(src, work)
    process(work, out, export=False, redetect=True)      # every entry now describes a file of the copy
    tifffile.imwrite(str(work / "ticket.tif"), failing(2000, 1600, 55), photometric="rgb")      # another picture, same size
    tifffile.imwrite(str(work / "p0.tif"), make_scan("white", 0.1, seed=5)[0], photometric="rgb")      # a changed print
    before = stored(out)
    for what in ({"scan": "ticket.tif", "sets": TICKET}, {"scan": "p0.tif", "item": "go"},
                 {"scan": "p0.tif", "scan_ppi": 500}, {"scan": "p0.tif", "item": "other"}):
        with pytest.raises(api.UsageError, match="changed since the last run; run process first"):
            api.fix_scan(str(out), author="human", **what)
        with pytest.raises(api.UsageError, match="run process first"):
            mcp_server.tool_fix({"output": str(out), "export": False, **{("set" if k == "sets" else k): v
                                                                           for k, v in what.items()}}, None)
    assert stored(out) == before      # nothing was stored
    # a plain fix and a Confirm of a detected print without an item stay as they were
    api.fix_scan(str(out), "p0.tif", shifts={"left": [1.0]}, author="human")
    api.fix_scan(str(out), "p0.tif", accept=True, author="human")
    assert entry(out, "p0.tif")["override"]
    # a reset stays possible, and after a run the new file takes the crop's place
    api.fix_scan(str(out), "ticket.tif", reset=True, author="human")
    process(work, out, export=False)
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human")
    assert entry(out, "ticket.tif")["status"] == "ok"


def test_q7_a_missing_scan_file_is_refused_and_a_reset_still_works(src, out, tmp_path):
    work = tmp_path / "in"
    shutil.copytree(src, work)
    process(work, out, export=False)
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human")
    (work / "ticket.tif").unlink()
    (work / "lift.tif").unlink()
    before = stored(out)
    with pytest.raises(api.UsageError, match="scan file is missing"):
        api.fix_scan(str(out), "lift.tif", sets=TICKET, author="human")
    with pytest.raises(api.UsageError, match="scan file is missing"):
        api.fix_scan(str(out), "ticket.tif", item="go", author="human")
    assert stored(out) == before
    api.fix_scan(str(out), "ticket.tif", reset=True, author="human")
    assert entry(out, "ticket.tif")["status"] == "failed"


def test_q7_a_changed_time_with_the_same_content_is_accepted(src, out, tmp_path):
    import os
    work = tmp_path / "in"
    shutil.copytree(src, work)
    process(work, out, export=False, redetect=True)
    t = entry(out, "ticket.tif")["mtime"]
    os.utime(work / "ticket.tif", (t + 3600, t + 3600))      # touched: the same bytes
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human")
    assert entry(out, "ticket.tif")["status"] == "ok"
    os.utime(work / "p0.tif", (t + 7200, t + 7200))
    api.fix_scan(str(out), "p0.tif", item="other", author="human")
    assert entry(out, "p0.tif")["item_fix"]["kind"] == "other"


def test_q7_a_file_changed_after_the_crop_was_stored_still_drops_it_at_the_next_run(src, out, tmp_path):
    work = tmp_path / "in"
    shutil.copytree(src, work)
    process(work, out, export=False)
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, scan_ppi=800, item="other", author="human")
    tifffile.imwrite(str(work / "ticket.tif"), failing(2000, 1600, 56), photometric="rgb")
    process(work, out, export=False)
    e = entry(out, "ticket.tif")
    assert e["status"] == "failed" and e.get("item_fix") is None and e["override"] is None
    assert "item-drop" in hist(out, "ticket.tif")


# =========================================================================== review round 2
def marked_with_detection(out):
    """A plain hand-fixed print the agent moved after its detection failed, and a redetect then succeeded: the crop carries
    the mark and the scan has a detection again, with the photo crop off and no item."""
    det = entry(out, "p0.tif")["detection"]
    api.fix_scan(str(out), "p0.tif", shifts={"left": [2.0]}, author="human")
    edit(out, "p0.tif", detection=None, status="failed", error="detection failed: x")
    api.fix_scan(str(out), "p0.tif", shifts={"left": [1.0]}, author="agent")
    assert entry(out, "p0.tif")["status"] == "needs_review" and entry(out, "p0.tif")["agent_crop"]
    edit(out, "p0.tif", detection=det)
    assert entry(out, "p0.tif")["status"] == "needs_review"


@pytest.mark.parametrize("how", ["accept", "shift", "reset"])
def test_r17_clearing_the_mark_judges_the_scan_again_also_with_a_detection(out, how):
    marked_with_detection(out)
    kw = {"accept": dict(accept=True), "shift": dict(shifts={"left": [1.0]}), "reset": dict(reset=True)}[how]
    api.fix_scan(str(out), "p0.tif", author="human", **kw)
    e = entry(out, "p0.tif")
    assert e["agent_crop"] is None
    assert e["status"] in ("ok", "needs_review") and "cropped by agent" not in e["warnings"]
    if how != "reset":
        assert e["status"] == "ok"


def test_r17_a_plain_reset_without_item_or_mark_still_does_not_judge(out):
    """R15 stays: no judgement, so no scale lookup (a missing profile raises nothing)."""
    m = Manifest(str(out))
    m.data["settings"]["profile"] = "no-such-profile"
    m.save()
    api.fix_scan(str(out), "p0.tif", shifts={"left": [1.0]}, author="human")
    api.fix_scan(str(out), "p0.tif", reset=True, author="human")


def test_r19_a_changed_file_under_other_with_an_own_ppi_is_cut_at_the_scale_of_a_print(src, out, tmp_path):
    from polaroid_processor.scale import MM_PER_INCH
    api.fix_scan(str(out), "p0.tif", item="other", scan_ppi=2000, author="human")
    e = entry(out, "p0.tif")
    job = {"key": "p0.tif", "path": str(src / "p0.tif"), "detect": True, "detection": None, "override": None,
           "item": e["item_fix"], "export": True, "scale": 2000 / MM_PER_INCH, "scale_base": None, "window": None,
           "window_fix": None, "fingerprint": "s32:not-the-fingerprint-of-this-file",
           "export_settings": pipeline._export_settings(api.make_settings(workers=1), tmp_path / "x"),
           "detect_params": api.make_settings().detect_params, "preview": None}
    res = pipeline._job(job)
    assert "error" not in res, res
    assert res["export_kind"] is None and res["export_px_per_mm"] == pytest.approx(PX_PER_MM, rel=0.01)
    job["scale_base"] = 500 / MM_PER_INCH      # the project's explicit scale is what stands then
    res = pipeline._job(job)
    assert res["export_px_per_mm"] == pytest.approx(500 / MM_PER_INCH)


def test_r23_a_print_does_not_depend_on_an_item_it_once_had(tmp_path):
    from test_flatten_shading import bowed_scan
    rgb, _ = bowed_scan()
    folder = tmp_path / "in"
    folder.mkdir()
    tifffile.imwrite(str(folder / "curl.tif"), rgb, photometric="rgb")
    control, out = tmp_path / "control", tmp_path / "out"
    process(folder, control)
    process(folder, out, export=False)
    api.fix_scan(str(out), "curl.tif", item="other", scan_ppi=2000, author="human")      # a scale far from the print's
    process(folder, out, redetect=True, export=False)      # detected with the item
    api.fix_scan(str(out), "curl.tif", item="auto", author="human", export=True)
    a, b = entry(control, "curl.tif"), entry(out, "curl.tif")
    assert a["detection"]["curves"] == b["detection"]["curves"]
    assert a["export_key"] == b["export_key"] and a["scale"] == b["scale"]
    assert np.array_equal(tifffile.imread(str(control / "curl.tif")), tifffile.imread(str(out / "curl.tif")))


# =========================================================================== review round 2, answers
def test_r18_every_crop_on_a_scan_without_a_detection_is_held(src, out):
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human")
    api.fix_scan(str(out), "ticket.tif", item="auto", author="human")      # a crop without an item
    api.fix_scan(str(out), "sheet.tif", sets=SHEET, author="agent")
    api.fix_scan(str(out), "sheet.tif", item="auto", author="agent")      # a marked crop without an item
    for name in ("ticket.tif", "sheet.tif"):
        assert entry(out, name).get("item_fix") is None and entry(out, name)["override"] and not entry(out, name)["detection"]
    jobs, seen = [], []
    summary = api.process([str(src)], str(out), api.make_settings(workers=1), on_jobs=jobs.append,
                          progress=lambda k, what: seen.append((k, what)))
    assert jobs[0] == 1 and seen[0][0] == "lift.tif"      # only the failed scan nobody cropped is detected again
    assert entry(out, "ticket.tif")["status"] == "ok" and entry(out, "sheet.tif")["status"] == "needs_review"
    assert "ticket.tif" not in summary["failed"] and (out / "ticket.tif").exists() and (out / "sheet.tif").exists()
    jobs.clear()
    again = api.process([str(src)], str(out), api.make_settings(workers=1), on_jobs=jobs.append)
    assert jobs[0] == 1 and again["exported"] == 0 and again["retagged"] == 0      # no needless export either


def test_r18_a_plain_fixed_print_whose_detection_failed_is_held_and_never_flashes_failed(src, out):
    api.fix_scan(str(out), "p0.tif", shifts={"left": [2.0]}, author="human")
    edit(out, "p0.tif", detection=None, status="failed", error="detection failed: x")      # what a failed redetect left
    jobs, seen = [], []
    api.process([str(src)], str(out), api.make_settings(workers=1, export=False), on_jobs=jobs.append,
                progress=lambda k, what: seen.append(k))
    assert "p0.tif" not in seen
    e = entry(out, "p0.tif")
    assert e["status"] == "ok" and e["error"] is None and e["detection_error"] == "detection failed: x"
    assert e.get("item_fix") is None and e["override"]


def test_r18_a_failed_redetect_leaves_a_crop_without_an_item_ok(src, out):
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human")
    api.fix_scan(str(out), "ticket.tif", item="auto", author="human")
    summary = api.process([str(src)], str(out), api.make_settings(workers=1, redetect=True, export=False))
    e = entry(out, "ticket.tif")
    assert e["status"] == "ok" and e["error"] is None and e["detection_error"].startswith("detection failed")
    assert "ticket.tif" not in summary["failed"]


def test_r20_detection_error_goes_only_to_entries_with_an_override_and_no_detection(src, out):
    api.process([str(src)], str(out), api.make_settings(workers=1, export=False))
    for name in ("p0.tif", "p1.tif", "p2.tif", "lift.tif", "sheet.tif", "ticket.tif"):
        assert "detection_error" not in entry(out, name)      # no crop, no field (lift, sheet, ticket are only failed)
    api.fix_scan(str(out), "p0.tif", shifts={"left": [1.0]}, author="human")      # detected print with a plain fix
    api.fix_scan(str(out), "p1.tif", item="other", author="human")
    api.process([str(src)], str(out), api.make_settings(workers=1, export=False, redetect=True))
    assert "detection_error" not in entry(out, "p0.tif") and "detection_error" not in entry(out, "p1.tif")


def test_r21_an_agent_resets_only_a_crop_that_carries_its_mark(out):
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human")      # a human's crop
    before = stored(out)
    with pytest.raises(api.UsageError, match="agent can reset only a crop"):
        api.fix_scan(str(out), "ticket.tif", reset=True, author="agent")
    with pytest.raises(api.UsageError, match="agent can reset only a crop"):
        mcp_server.tool_fix({"output": str(out), "scan": "ticket.tif", "reset": True, "author": "human", "export": False},
                            None)      # every MCP fix is an agent's
    assert stored(out) == before
    api.fix_scan(str(out), "ticket.tif", reset=True, author="human")
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="agent")      # an agent's crop from the start: it may take it back
    api.fix_scan(str(out), "ticket.tif", reset=True, author="agent")
    e = entry(out, "ticket.tif")
    assert e["status"] == "failed" and e["agent_crop"] is None and e["override"] is None
    # a crop a human confirmed is the human's
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="agent")
    api.fix_scan(str(out), "ticket.tif", accept=True, author="human")
    with pytest.raises(api.UsageError, match="agent can reset only a crop"):
        api.fix_scan(str(out), "ticket.tif", reset=True, author="agent")
    assert entry(out, "ticket.tif")["status"] == "ok"
    api.fix_scan(str(out), "ticket.tif", reset=True, author="human")      # a human's reset is always allowed
    assert entry(out, "ticket.tif")["status"] == "failed"


def test_r21_the_marked_crop_without_an_item_and_the_never_cropped_failed_scan(out):
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="agent")
    api.fix_scan(str(out), "ticket.tif", item="auto", author="agent")
    api.fix_scan(str(out), "ticket.tif", reset=True, author="agent")      # still marked: allowed
    assert entry(out, "ticket.tif")["status"] == "failed"
    api.fix_scan(str(out), "lift.tif", reset=True, author="agent")      # nothing was cropped: as before
    assert entry(out, "lift.tif")["status"] == "failed"


def test_r22_a_reset_on_a_pending_scan_is_refused_like_set_and_item(out):
    m = Manifest(str(out))
    m.entry("ghost.tif")      # never detected: no override, not failed
    m.save()
    before = stored(out)
    for author in ("human", "agent"):
        with pytest.raises(api.UsageError, match="not been detected"):
            api.fix_scan(str(out), "ghost.tif", reset=True, author=author)
    assert stored(out) == before and entry(out, "ghost.tif")["history"] == []      # no history entry
    edit(out, "lift.tif", error="worker process crashed on this scan")
    api.fix_scan(str(out), "lift.tif", reset=True, author="human")      # a failed scan that was never cropped: as before
    assert entry(out, "lift.tif")["status"] == "failed"


# =========================================================================== decisions after the third review
def test_t1_an_agent_may_reset_only_a_crop_that_was_an_agents_from_the_start(out):
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human")      # a human's crop
    api.fix_scan(str(out), "ticket.tif", nudges={"tl": [3.0, 3.0]}, author="agent")      # marked again (first review, 5)
    assert entry(out, "ticket.tif")["agent_crop"]
    before = stored(out)
    with pytest.raises(api.UsageError, match="agent can reset only a crop"):
        api.fix_scan(str(out), "ticket.tif", reset=True, author="agent")
    with pytest.raises(api.UsageError, match="agent can reset only a crop"):
        mcp_server.tool_fix({"output": str(out), "scan": "ticket.tif", "reset": True, "export": False}, None)
    assert stored(out) == before
    # an agent's crop from the start, moved again by the agent: still the agent's
    api.fix_scan(str(out), "ticket.tif", reset=True, author="human")
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="agent")
    api.fix_scan(str(out), "ticket.tif", nudges={"br": [2.0, 2.0]}, author="agent")
    api.fix_scan(str(out), "ticket.tif", reset=True, author="agent")
    assert entry(out, "ticket.tif")["status"] == "failed" and entry(out, "ticket.tif")["agent_crop"] is None
    # placed by an agent, confirmed by a human, moved again by an agent: a human's crop for this rule
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="agent")
    api.fix_scan(str(out), "ticket.tif", accept=True, author="human")
    api.fix_scan(str(out), "ticket.tif", nudges={"tl": [1.0, 1.0]}, author="agent")
    assert entry(out, "ticket.tif")["agent_crop"]
    with pytest.raises(api.UsageError, match="agent can reset only a crop"):
        api.fix_scan(str(out), "ticket.tif", reset=True, author="agent")
    # same after a human's save of corners
    api.fix_scan(str(out), "ticket.tif", reset=True, author="human")
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="agent")
    api.fix_scan(str(out), "ticket.tif", nudges={"tl": [1.0, 1.0]}, author="human")
    api.fix_scan(str(out), "ticket.tif", nudges={"tl": [1.0, 1.0]}, author="agent")
    with pytest.raises(api.UsageError, match="agent can reset only a crop"):
        api.fix_scan(str(out), "ticket.tif", reset=True, author="agent")


def test_t1_the_origin_is_kept_in_the_mark_and_an_old_mark_without_it_is_read_as_a_humans(out):
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="agent")
    assert entry(out, "ticket.tif")["agent_crop"]["from_start"] is True
    api.fix_scan(str(out), "p0.tif", shifts={"left": [1.0]}, author="human")
    edit(out, "p0.tif", detection=None, status="failed", error="detection failed: x")
    api.fix_scan(str(out), "p0.tif", shifts={"left": [1.0]}, author="agent")      # a human's crop on a scan without a detection
    assert entry(out, "p0.tif")["agent_crop"]["from_start"] is False
    # a mark made before the field existed: the safe reading
    m = Manifest(str(out))
    m.get("ticket.tif")["agent_crop"].pop("from_start")
    m.save()
    with pytest.raises(api.UsageError, match="agent can reset only a crop"):
        api.fix_scan(str(out), "ticket.tif", reset=True, author="agent")
    api.fix_scan(str(out), "ticket.tif", reset=True, author="human")      # a human's reset clears it
    assert entry(out, "ticket.tif")["agent_crop"] is None and entry(out, "ticket.tif")["status"] == "failed"


def test_t1_the_origin_goes_with_a_changed_file(src, out, tmp_path):
    work = tmp_path / "in"
    shutil.copytree(src, work)
    process(work, out, export=False)
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="agent")
    tifffile.imwrite(str(work / "ticket.tif"), failing(2000, 1600, 78), photometric="rgb")
    process(work, out, export=False)
    e = entry(out, "ticket.tif")
    assert e["agent_crop"] is None and e["status"] == "failed"
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="human")
    api.fix_scan(str(out), "ticket.tif", nudges={"tl": [2.0, 2.0]}, author="agent")      # a human's crop on the new picture
    assert entry(out, "ticket.tif")["agent_crop"]["from_start"] is False


def test_t2_an_agents_reset_on_a_detected_scan_keeps_a_humans_item_or_ppi(out):
    api.fix_scan(str(out), "p0.tif", shifts={"left": [2.0]}, author="human")
    api.fix_scan(str(out), "p0.tif", reset=True, author="agent")      # a plain corner fix, no item or ppi: as before
    assert entry(out, "p0.tif")["override"] is None
    api.fix_scan(str(out), "p1.tif", item="other", scan_ppi=500, author="human")
    before = stored(out)
    with pytest.raises(api.UsageError, match="item choice or an own ppi that a human set"):
        api.fix_scan(str(out), "p1.tif", reset=True, author="agent")
    with pytest.raises(api.UsageError, match="item choice or an own ppi that a human set"):
        mcp_server.tool_fix({"output": str(out), "scan": "p1.tif", "reset": True, "export": False}, None)
    assert stored(out) == before
    # where an agent set the item, its reset is allowed
    api.fix_scan(str(out), "p2.tif", item="go", author="agent")
    api.fix_scan(str(out), "p2.tif", reset=True, author="agent")
    assert entry(out, "p2.tif").get("item_fix") is None
    # a human set the kind: an agent cannot change its ppi any more (slice 2, fourth review), nor reset it
    api.fix_scan(str(out), "p2.tif", item="other", author="human")
    with pytest.raises(api.UsageError, match="a human set"):
        api.fix_scan(str(out), "p2.tif", scan_ppi=700, author="agent")
    with pytest.raises(api.UsageError, match="a human set"):
        api.fix_scan(str(out), "p2.tif", reset=True, author="agent")
    api.fix_scan(str(out), "p2.tif", reset=True, author="human")      # a human's reset is always allowed
    assert entry(out, "p2.tif").get("item_fix") is None
    api.fix_scan(str(out), "p2.tif", item="other", author="agent")
    api.fix_scan(str(out), "p2.tif", scan_ppi=700, author="human")
    with pytest.raises(api.UsageError, match="a human set"):
        api.fix_scan(str(out), "p2.tif", reset=True, author="agent")
    # an item in a manifest that has no record of who set it is read as a human's
    m = Manifest(str(out))
    m.get("p2.tif")["item_fix"] = {"kind": "go", "author": "agent"}
    m.save()
    with pytest.raises(api.UsageError, match="a human set"):
        api.fix_scan(str(out), "p2.tif", reset=True, author="agent")


def _trace_px(src, tmp_path, item, **job_kw):
    """The scale the flatten trace of a detection job used."""
    seen = []
    real = pipeline.flat.trace

    def spy(rgb, quad, px, *a, **k):
        seen.append(px)
        return real(rgb, quad, px, *a, **k)

    pipeline.flat.trace = spy
    try:
        job = {"key": "p0.tif", "path": str(src / "p0.tif"), "detect": True, "detection": None, "override": None,
               "item": item, "export": False, "scale": None, "scale_base": None, "window": None, "window_fix": None,
               "fingerprint": None, "export_settings": pipeline._export_settings(api.make_settings(workers=1), tmp_path / "x"),
               "detect_params": api.make_settings().detect_params, "preview": None, **job_kw}
        res = pipeline._job(job)
    finally:
        pipeline.flat.trace = real
    assert "error" not in res, res
    return seen[0], res


def test_t3_a_forced_format_is_traced_at_its_own_scale_and_only_other_without_the_item(src, tmp_path):
    from polaroid_processor.detection import quad_size
    from polaroid_processor.formats import POLAROID_GO
    from polaroid_processor.scale import scale_from_size
    none_px, res = _trace_px(src, tmp_path, None)
    size = quad_size(np.array(res["detection"]["quad"]))
    assert none_px == pytest.approx(scale_from_size(size).px_per_mm)      # the shape guess: a 600
    go_px, _ = _trace_px(src, tmp_path, {"kind": "go"})      # a redetect of a forced scan whose shape says 600
    assert go_px == pytest.approx(scale_from_size(size, POLAROID_GO).px_per_mm) and go_px > 1.3 * none_px
    other_px, _ = _trace_px(src, tmp_path, {"kind": "other", "scan_ppi": 2000})
    assert other_px == pytest.approx(none_px)      # Other is traced as if it had no item
    base_px, _ = _trace_px(src, tmp_path, {"kind": "other"}, scale_base=500 / 25.4)
    assert base_px == pytest.approx(500 / 25.4)


def test_s2_an_agent_cannot_change_or_remove_a_humans_item_or_ppi(out):
    """Slice 2, decision after the fourth review: --item, --item auto, --scan-ppi and reset, with and without a
    detected outline, also on an agent's own crop whose item or ppi a human changed. Nothing is stored on refusal."""
    gone = "an agent cannot (change or )?take away an item choice or an own ppi that a human set"
    # a human's crop with a human's item and ppi on a scan without an outline
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, item="other", scan_ppi=500, author="human")
    before = stored(out)
    for kw in ({"item": "go"}, {"item": "600"}, {"item": "auto"}, {"scan_ppi": 700}, {"scan_ppi": 0}, {"reset": True},
               {"item": "go", "sets": TICKET}):
        with pytest.raises(api.UsageError, match="agent"):
            api.fix_scan(str(out), "ticket.tif", author="agent", **kw)
        with pytest.raises(api.UsageError, match="agent"):
            mcp_server.tool_fix({"output": str(out), "scan": "ticket.tif", "export": False,
                                 **{("set" if k == "sets" else k): v for k, v in kw.items()}}, None)
        assert stored(out) == before
    with pytest.raises(api.UsageError, match="human set"):
        api.fix_scan(str(out), "ticket.tif", item="auto", author="agent")
    # allowed: the same item again, a corner nudge with the unchanged item (it marks the crop again, first review 5)
    api.fix_scan(str(out), "ticket.tif", item="other", scan_ppi=500, author="agent")
    api.fix_scan(str(out), "ticket.tif", nudges={"tl": [3.0, 3.0]}, item="other", author="agent")
    assert entry(out, "ticket.tif")["item_fix"]["kind"] == "other" and entry(out, "ticket.tif")["item_fix"]["scan_ppi"] == 500.0
    with pytest.raises(api.UsageError, match="agent"):
        api.fix_scan(str(out), "ticket.tif", reset=True, author="agent")
    api.fix_scan(str(out), "ticket.tif", item="go", author="human")      # a human's actions are always allowed
    api.fix_scan(str(out), "ticket.tif", reset=True, author="human")
    assert entry(out, "ticket.tif")["status"] == "failed" and entry(out, "ticket.tif").get("item_fix") is None

    # an agent's own crop (from the start) whose ppi a human changed: no longer the agent's to change, remove or reset
    api.fix_scan(str(out), "lift.tif", sets=TICKET, scan_ppi=None, author="agent")
    assert entry(out, "lift.tif")["agent_crop"]["from_start"] is True and entry(out, "lift.tif")["item_fix"]["by_human"] is False
    api.fix_scan(str(out), "lift.tif", item="go", author="agent")      # while no human touched it, the agent may
    api.fix_scan(str(out), "lift.tif", item="other", author="agent")
    api.fix_scan(str(out), "lift.tif", scan_ppi=600, author="human")      # a human changes the ppi: the crop stays unconfirmed
    assert entry(out, "lift.tif")["agent_crop"] and entry(out, "lift.tif")["item_fix"]["by_human"] is True
    before = stored(out)
    for kw in ({"item": "auto"}, {"item": "go"}, {"scan_ppi": 0}, {"scan_ppi": 900}, {"reset": True}):
        with pytest.raises(api.UsageError, match="agent"):
            api.fix_scan(str(out), "lift.tif", author="agent", **kw)
        assert stored(out) == before
    api.fix_scan(str(out), "lift.tif", reset=True, author="human")
    assert entry(out, "lift.tif")["status"] == "failed"

    # a detected scan: the human's item holds against every agent change; corners with the unchanged item are fine
    api.fix_scan(str(out), "p1.tif", item="go", author="human")
    before = stored(out)
    for kw in ({"item": "other"}, {"item": "600"}, {"item": "auto"}, {"reset": True}):
        with pytest.raises(api.UsageError, match="agent"):
            api.fix_scan(str(out), "p1.tif", author="agent", **kw)
        assert stored(out) == before
    api.fix_scan(str(out), "p1.tif", item="go", shifts={"left": [1.0]}, author="agent")
    assert entry(out, "p1.tif")["item_fix"]["kind"] == "go" and entry(out, "p1.tif")["override"]
    # an item an agent set stays the agent's to change, until a human touches it
    api.fix_scan(str(out), "p2.tif", item="go", author="agent")
    api.fix_scan(str(out), "p2.tif", item="other", scan_ppi=400, author="agent")
    api.fix_scan(str(out), "p2.tif", item="auto", author="agent")
    assert entry(out, "p2.tif").get("item_fix") is None
    # the CLI is an agent with --author agent
    api.fix_scan(str(out), "p2.tif", item="other", author="human")
    assert cli.main(["fix", str(out), "p2.tif", "--item", "auto", "--author", "agent"]) == 2
    assert entry(out, "p2.tif")["item_fix"]["kind"] == "other"
