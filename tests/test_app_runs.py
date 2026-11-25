"""The v2 app, slice 2: run lock, merge before save, export, the drag-to-fix view, live reload."""
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from polaroid_processor import api
from polaroid_processor.cli import main
from polaroid_processor.lock import LOCK_NAME, RunLock, RunLocked, file_mutex, holder
from polaroid_processor.manifest import Manifest

from test_app import PPI, app, finished, scans, settings, src  # noqa: F401  (fixtures and helpers)

CORNERS = ("tl", "tr", "br", "bl")


def dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


HOLD = ("import sys, time; from polaroid_processor.lock import RunLock; "
        "RunLock(sys.argv[1], who='polaroid-mcp.exe').acquire(); print('ready', flush=True); time.sleep(60)")


def lock_holder(out):
    """Another process holding the run lock of ``out``."""
    p = subprocess.Popen([sys.executable, "-c", HOLD, str(out)], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "ready"
    return p


def write_lock(out, pid, who="test"):
    out.mkdir(parents=True, exist_ok=True)
    (out / LOCK_NAME).write_text(json.dumps({"pid": pid, "who": who, "started": "2026-11-01T10:00:00"}),
                                 encoding="utf-8")


# --------------------------------------------------------------------------- run lock
def test_run_lock_is_exclusive_and_released(tmp_path):
    out = tmp_path / "out"
    with RunLock(str(out)):
        assert holder(str(out))["pid"] == os.getpid()
        with pytest.raises(RunLocked, match="another run is working"):
            RunLock(str(out)).acquire()
    assert not (out / LOCK_NAME).exists() and holder(str(out)) is None
    RunLock(str(out)).acquire()          # free again


# --------------------------------------------------------------------------- merge before save
def test_manifest_merges_two_writers(src, tmp_path):
    out = tmp_path / "out"
    api.process([str(src)], str(out), settings(export=False))
    a, b = Manifest(str(out)), Manifest(str(out))
    b.set_override("a_white.tif", np.array(b.get("a_white.tif")["detection"]["quad"]) + 5, "agent", "moved")
    b.entry("b_new.tif")["note"] = "from b"
    b.save()
    a.entry("a_white.tif")["status"] = "needs_review"          # another field of the same scan
    a.set_override("a_white.tif", np.array(a.get("a_gold.tif")["detection"]["quad"]), "human", "also moved")
    a.entry("a_gold.tif")["warnings"] = ["from a"]
    a.entry("c_new.tif")["note"] = "from a"
    a.save()
    m = Manifest(str(out))
    w = m.get("a_white.tif")
    assert w["status"] == "needs_review" and w["override"]["author"] == "human"
    assert [h["author"] for h in w["history"]] == ["agent", "human"]          # both fixes are kept
    assert m.get("b_new.tif")["note"] == "from b" and m.get("c_new.tif")["note"] == "from a"
    assert m.get("a_gold.tif")["warnings"] == ["from a"]
    # a writer that changed nothing takes the other's changes over
    c = Manifest(str(out))
    m.entry("a_gold.tif")["status"] = "failed"
    m.save()
    c.refresh()
    assert c.get("a_gold.tif")["status"] == "failed"


def test_a_fix_made_during_a_run_survives_it(src, tmp_path):
    out = tmp_path / "out"
    api.process([str(src)], str(out), settings(export=False))
    before = np.array(Manifest(str(out)).get("a_white.tif")["detection"]["quad"])
    for i in range(4):
        shutil.copy(src / "a_white.tif", src / f"b_{i}.tif")
    fixed = []

    def fix_once(key, what):
        if not fixed:
            fixed.append(api.fix_scan(str(out), "a_white.tif", shifts={"left": [8.0]}, author="agent"))

    summary = api.process([str(src)], str(out), settings(export=False), progress=fix_once)
    assert fixed and summary["scans"] == 7
    m = Manifest(str(out))
    e = m.get("a_white.tif")
    assert e["override"]["author"] == "agent" and [h["action"] for h in e["history"]] == ["fix"]
    assert not np.allclose(Manifest.effective_quad(e), before, atol=1)
    assert all(m.get(f"b_{i}.tif")["detection"] for i in range(4))             # the run's own work is there too


# --------------------------------------------------------------------------- export
def test_export_stops_on_cancel_and_resumes(src, tmp_path):
    for i in range(6):
        shutil.copy(src / "a_white.tif", src / f"b_{i}.tif")
    out = tmp_path / "out"
    api.process([str(src)], str(out), settings(export=False))
    flag, jobs = threading.Event(), []
    summary = api.process([str(src)], str(out), settings(no_derivatives=True), cancel=flag.is_set,
                          progress=lambda k, what: flag.set() if what == "exported" else None,
                          on_jobs=jobs.append)
    assert summary["cancelled"] and jobs == [1, 8]            # the blank scan is tried again; 8 to export
    first = {k: e["exported_at"] for k, e in Manifest(str(out)).items() if e.get("outputs")}
    assert 1 <= len(first) < 8
    time.sleep(1.1)
    summary = api.process([str(src)], str(out), settings(no_derivatives=True))
    m = Manifest(str(out))
    assert not summary["cancelled"] and sum(1 for _, e in m.items() if e.get("outputs")) == 8
    assert all(m.get(k)["exported_at"] == t for k, t in first.items())       # done ones are not exported again


def test_export_through_the_app_and_only_what_changed(app, src, tmp_path):
    app.json("/api/project/open", {"input": str(src)})
    assert app.json("/api/project/export", {})[0] == 400                      # nothing detected yet
    app.json("/api/project/detect", {})
    prog = app.wait(finished)
    assert prog["counts"]["to_export"] == 2 and prog["counts"]["exported"] == 0
    assert app.json("/api/project/export", {})[0] == 200
    prog = app.wait(finished)
    assert prog["run"]["mode"] == "export" and prog["run"]["total"] == 2 and prog["run"]["done"] == 2
    assert prog["counts"]["exported"] == 2 and prog["counts"]["to_export"] == 0
    out = app.state.project.output
    first = {k: e["exported_at"] for k, e in Manifest(out).items() if e.get("outputs")}
    assert sorted(first) == ["a_gold.tif", "a_white.tif"]
    scans_ = {s["key"]: s for s in app.json("/api/project/scans")[1]["scans"]}
    assert scans_["a_white.tif"]["export"] == "done" and scans_["a_blank.tif"]["export"] is None
    # a fix made from outside (CLI, agent) makes that scan's export outdated, live: no reload needed
    time.sleep(1.1)
    rev = app.json("/api/project/progress")[1]["rev"]
    api.fix_scan(out, "a_white.tif", shifts={"right": [6.0]}, author="agent")
    prog = app.json("/api/project/progress")[1]
    assert prog["rev"] != rev and prog["counts"]["to_export"] == 1
    scans_ = {s["key"]: s for s in app.json("/api/project/scans")[1]["scans"]}
    assert scans_["a_white.tif"]["export"] == "outdated" and scans_["a_white.tif"]["quad_source"] == "override"
    app.json("/api/project/export", {})
    prog = app.wait(lambda p: p["run"]["state"] == "finished" and p["run"]["mode"] == "export")
    assert prog["run"]["total"] == 1 and prog["counts"]["to_export"] == 0
    later = {k: e["exported_at"] for k, e in Manifest(out).items() if e.get("outputs")}
    assert later["a_gold.tif"] == first["a_gold.tif"] and later["a_white.tif"] != first["a_white.tif"]


def test_detect_and_export_in_one_go(app, src):
    app.json("/api/project/open", {"input": str(src)})
    assert app.json("/api/project/all", {})[0] == 200
    prog = app.wait(finished)
    assert prog["run"]["mode"] == "all" and prog["counts"]["exported"] == 2 and prog["counts"]["to_export"] == 0
    assert prog["counts"]["failed"] == 1


# --------------------------------------------------------------------------- the drag view
def test_the_drag_view_is_served_by_the_app_and_a_save_renews_card_and_preview(app, src):
    app.json("/api/project/open", {"input": str(src)})
    app.json("/api/project/detect", {})
    app.wait(finished)
    status, headers, raw = app.request("/fix")
    assert status == 200 and b'window.POLAROID_API = "/api/fix"' in raw and b"Polaroid review" in raw
    status, data = app.json("/api/fix/scans")
    assert status == 200 and [s["key"] for s in data["scans"]] == ["a_blank.tif", "a_gold.tif", "a_white.tif"]
    blank = data["scans"][0]      # hand crop, slice 2: a scan nothing was found on is listed too, without an outline (the page's place step)
    assert blank["status"] == "failed" and blank["quad"] is None and blank["detected"] is False
    status, specs = app.json("/api/fix/scans/a_white.tif/tiles")
    assert status == 200 and len(specs["tiles"]) == 9
    assert specs["tiles"][0]["url"].startswith("/api/fix/scans/") and app.request(specs["tiles"][0]["url"])[0] == 200
    before = {s["key"]: s for s in app.json("/api/project/scans")[1]["scans"]}["a_white.tif"]
    # the stored status is stale on purpose: after the save the card shows the outline as reviewed
    m = Manifest(app.state.project.output)
    m.get("a_white.tif")["status"] = "needs_review"
    m.save()
    quad = np.array([s for s in data["scans"] if s["key"] == "a_white.tif"][0]["quad"])
    corners = {n: [float(x), float(y)] for n, (x, y) in zip(CORNERS, quad)}
    corners["tl"][0] -= 12
    status, rec = app.json("/api/fix/scans/a_white.tif/save", {"corners": corners})
    assert status == 200 and rec["quad_source"] == "override" and rec["status"] == "ok"
    after = {s["key"]: s for s in app.json("/api/project/scans")[1]["scans"]}["a_white.tif"]
    assert after["status"] == "ok" and not after["stale"] and after["preview"]["url"] != before["preview"]["url"]
    assert Manifest(app.state.project.output).get("a_white.tif")["history"][-1]["author"] == "human"
    assert app.json("/api/fix/scans/a_gold.tif/confirm", {})[0] == 200


def test_a_fix_in_the_app_during_a_run_is_kept(app, src):
    app.json("/api/project/open", {"input": str(src)})
    app.json("/api/project/detect", {})
    app.wait(finished)
    for i in range(6):
        shutil.copy(src / "a_gold.tif", src / f"b_{i}.tif")
    assert app.json("/api/project/detect", {})[0] == 200
    quad = np.array([s for s in app.json("/api/fix/scans")[1]["scans"] if s["key"] == "a_white.tif"][0]["quad"])
    corners = {n: [float(x), float(y)] for n, (x, y) in zip(CORNERS, quad + [3, 0])}
    status, _ = app.json("/api/fix/scans/a_white.tif/save", {"corners": corners})       # while the run goes
    assert status == 200
    prog = app.wait(finished)
    assert prog["run"]["total"] == 7
    m = Manifest(app.state.project.output)
    e = m.get("a_white.tif")
    assert np.allclose(e["override"]["quad"], quad + [3, 0], atol=0.01) and e["history"][-1]["author"] == "human"
    assert all(m.get(f"b_{i}.tif")["detection"] for i in range(6))


def test_live_reload_stamps_follow_the_manifest(app, src):
    app.json("/api/project/open", {"input": str(src)})
    app.json("/api/project/detect", {})
    app.wait(finished)
    out = app.state.project.output
    stamp = app.json("/api/fix/stamp")[1]["stamp"]
    rev = {s["key"]: s["rev"] for s in app.json("/api/fix/scans")[1]["scans"]}["a_white.tif"]
    api.fix_scan(out, "a_white.tif", shifts={"top": [4.0]}, author="agent")
    assert app.json("/api/fix/stamp")[1]["stamp"] != stamp
    rec = {s["key"]: s for s in app.json("/api/fix/scans")[1]["scans"]}["a_white.tif"]
    assert rec["rev"] == rev + 1 and rec["review"]["author"] == "agent"


def test_a_fix_by_the_command_line_shows_as_reviewed_in_the_grid(app, src):
    app.json("/api/project/open", {"input": str(src)})
    app.json("/api/project/detect", {})
    app.wait(finished)
    out = app.state.project.output
    m = Manifest(out)
    m.get("a_gold.tif")["status"] = "needs_review"
    m.save()
    assert {s["key"]: s["status"] for s in app.json("/api/project/scans")[1]["scans"]}["a_gold.tif"] == "needs_review"
    api.fix_scan(out, "a_gold.tif", accept=True, author="agent")
    assert {s["key"]: s["status"] for s in app.json("/api/project/scans")[1]["scans"]}["a_gold.tif"] == "ok"


# --------------------------------------------------------------------------- lock: the system holds it
def test_a_lock_file_left_by_a_dead_process_does_not_block(tmp_path):
    out = tmp_path / "out"
    write_lock(out, dead_pid())             # text of a crashed run: nobody holds the system lock on it
    assert holder(str(out)) is None
    with RunLock(str(out)):
        assert holder(str(out))["pid"] == os.getpid()
    out2 = tmp_path / "out2"
    out2.mkdir()
    (out2 / LOCK_NAME).write_text("", encoding="utf-8")
    with RunLock(str(out2)):
        pass


def test_a_killed_run_releases_the_lock(tmp_path):
    out = tmp_path / "out"
    other = lock_holder(out)
    try:
        assert holder(str(out))["pid"] == other.pid and holder(str(out))["who"] == "polaroid-mcp.exe"
        with pytest.raises(RunLocked, match=f"pid {other.pid}.*delete"):
            RunLock(str(out)).acquire()
    finally:
        other.kill()
        other.wait()
    assert holder(str(out)) is None
    RunLock(str(out)).acquire()


TAKE = ("import sys, time; from polaroid_processor.lock import RunLock, RunLocked\n"
        "while time.time() < float(sys.argv[2]): time.sleep(0.001)\n"
        "try:\n    RunLock(sys.argv[1]).acquire(); print('ok', flush=True); time.sleep(3)\n"
        "except RunLocked:\n    print('locked', flush=True)\n")


def test_two_processes_cannot_both_take_the_lock(tmp_path):
    out = tmp_path / "out"
    write_lock(out, dead_pid())             # a stale lock that every one of them finds
    at = time.time() + 4
    procs = [subprocess.Popen([sys.executable, "-c", TAKE, str(out), str(at)], stdout=subprocess.PIPE, text=True)
             for _ in range(4)]
    answers = [p.communicate(timeout=60)[0].strip() for p in procs]
    assert sorted(answers) == ["locked", "locked", "locked", "ok"]


def test_file_mutex_excludes_other_holders(tmp_path):
    path = tmp_path / "m.lock"
    with file_mutex(path):
        with pytest.raises(TimeoutError):
            with file_mutex(path, timeout=0.2):
                pass
    with file_mutex(path, timeout=1):
        pass


def test_a_second_run_is_refused_and_a_stale_lock_does_not_block(src, tmp_path, capsys):
    out = tmp_path / "out"
    other = lock_holder(out)
    try:
        with pytest.raises(RunLocked, match=f"pid {other.pid}"):
            api.process([str(src)], str(out), settings(export=False))
        assert main(["process", str(src), "-o", str(out), "--ppi", str(PPI), "-j", "1"]) == 2
        assert "another run is working" in capsys.readouterr().err
        assert not (out / "polaroid-manifest.json").exists()
    finally:
        other.kill()
        other.wait()
    write_lock(out, dead_pid())
    assert api.process([str(src)], str(out), settings(export=False))["scans"] == 3


def test_a_foreign_run_is_shown_and_blocks_starting(app, src, tmp_path):
    app.json("/api/project/open", {"input": str(src)})
    out = tmp_path / "scans_processed"
    other = lock_holder(out)
    try:
        lock = app.json("/api/project/progress")[1]["lock"]
        assert lock["pid"] == other.pid and lock["who"] == "polaroid-mcp.exe"
        status, data = app.json("/api/project/detect", {})
        assert status == 400 and "another run" in data["error"]
    finally:
        other.kill()
        other.wait()
    assert app.json("/api/project/progress")[1]["lock"] is None
    assert app.json("/api/project/detect", {})[0] == 200
    app.wait(finished)


# --------------------------------------------------------------------------- review follow-ups
def test_an_accept_during_the_size_check_wins_over_the_run_status(src, tmp_path):
    from polaroid_processor.pipeline import ReviewRules
    out = tmp_path / "out"
    strict = settings(export=False)
    strict.rules = ReviewRules(size_tolerance=-1.0)           # every scan is flagged: its size never matches
    api.process([str(src)], str(out), strict)
    assert Manifest(str(out)).get("a_white.tif")["status"] == "needs_review"
    shutil.copy(src / "a_white.tif", src / "b_0.tif")
    fixed = []

    def accept(phase):
        if phase == "checking sizes" and not fixed:
            fixed.append(api.fix_scan(str(out), "a_white.tif", accept=True, author="agent"))

    api.process([str(src)], str(out), strict, on_phase=accept)
    e = Manifest(str(out)).get("a_white.tif")
    assert fixed and e["override"]["author"] == "agent"
    assert e["status"] == "ok"                              # judged on the accepted outline, not the run's
    assert Manifest(str(out)).get("a_gold.tif")["status"] == "needs_review"


def test_fix_with_export_during_a_run_keeps_the_fix_and_says_the_export_was_skipped(src, tmp_path):
    out = tmp_path / "out"
    api.process([str(src)], str(out), settings(no_derivatives=True))
    other = lock_holder(out)
    try:
        rec = api.fix_scan(str(out), "a_white.tif", shifts={"left": [3.0]}, author="agent", export=True)
        assert "fix saved; export skipped" in rec["note"] and "another run" in rec["note"] and not rec["exported"]
        assert [h["action"] for h in rec["history"]] == ["fix"]
        assert main(["fix", str(out), "a_white.tif", "--accept", "--export", "-q"]) == 0
    finally:
        other.kill()
        other.wait()
    assert len(Manifest(str(out)).get("a_white.tif")["history"]) == 2         # no duplicated entries from retries


def test_export_count_follows_settings_and_scale_not_time_stamps(app, src):
    app.json("/api/project/open", {"input": str(src)})
    app.json("/api/project/all", {})
    prog = app.wait(finished)
    assert prog["counts"]["to_export"] == 0
    out = app.state.project.output
    m = Manifest(out)
    m.data["settings"]["margin_mm"] = 0.4                     # a new setting: every export is out of date
    m.save()
    prog = app.json("/api/project/progress")[1]
    assert prog["counts"]["to_export"] == 2
    assert {s["key"]: s["export"] for s in app.json("/api/project/scans")[1]["scans"]}["a_white.tif"] == "outdated"
    m = Manifest(out)
    m.data["settings"]["margin_mm"] = 0.1
    m.data["settings"]["ppi"] = 700.0                         # a deliberate scale, even a close one: new exports
    m.save()
    counts = app.json("/api/project/progress")[1]["counts"]
    assert counts["to_export"] == 2 and counts["to_retag"] == 0
    m = Manifest(out)
    m.data["settings"]["ppi"] = None
    m.save()
    counts = app.json("/api/project/progress")[1]["counts"]
    assert counts["to_export"] == 0 and counts["to_retag"] == 0
    api.fix_scan(out, "a_gold.tif", shifts={"top": [5.0]})    # a fix after the export
    assert app.json("/api/project/progress")[1]["counts"]["to_export"] == 1


def test_stamp_reads_the_file_stamp_only(src, tmp_path):
    from polaroid_processor.review import ReviewApp
    out = tmp_path / "out"
    api.process([str(src)], str(out), settings(export=False))
    review = ReviewApp(str(out), show_all=True)
    first = review.stamp()
    assert first == review.stamp() and first != "0"
    (out / "polaroid-manifest.json").write_text("{ broken", encoding="utf-8")    # content is not parsed
    assert review.stamp() != first


def test_pages_cannot_be_framed_by_other_sites(app):
    status, headers, _ = app.request("/")
    assert headers["X-Frame-Options"] == "SAMEORIGIN" and "frame-ancestors 'self'" in headers["Content-Security-Policy"]


# --------------------------------------------------------------------------- scale-only changes patch tags
def test_a_fix_moves_the_batch_median_and_only_the_fixed_scan_is_encoded_again(tmp_path):
    import tifffile
    from synth import make_scan
    src = tmp_path / "in"
    src.mkdir()
    for i in range(3):                                           # no resolution tag: the scale is inferred
        rgb, _ = make_scan("white", 0.3 * i, seed=70 + i)
        tifffile.imwrite(str(src / f"s_{i}.tif"), rgb, photometric="rgb")
    out = tmp_path / "out"
    inferred = api.make_settings(workers=1, no_derivatives=True)
    api.process([str(src)], str(out), api.make_settings(workers=1, export=False))
    api.fix_scan(str(out), "s_0.tif", shifts={"right": [6.0]}, author="agent")    # three different print sizes
    api.fix_scan(str(out), "s_1.tif", shifts={"right": [-6.0]}, author="agent")
    api.process([str(src)], str(out), inferred)
    m = Manifest(str(out))
    assert {e["scale"]["source"] for _, e in m.items()} == {"batch-median"} and all(e["outputs"] for _, e in m.items())
    before = {k: dict(e) for k, e in m.items()}
    time.sleep(1.1)
    api.fix_scan(str(out), "s_2.tif", shifts={"right": [12.0]}, author="agent")    # the median print grows: the median moves
    from polaroid_processor.app.project import export_plan, export_state
    from polaroid_processor.pipeline import settings_from_dict
    m = Manifest(str(out))
    plan = export_plan(m, settings_from_dict(m.data["settings"]))        # what the app shows before the run
    assert [export_state(m, m.get(k), plan[k], None) for k in ("s_0.tif", "s_1.tif", "s_2.tif")] == [
        "tags", "tags", "outdated"]
    jobs = []
    summary = api.process([str(src)], str(out), inferred, on_jobs=jobs.append)
    m = Manifest(str(out))
    others = ["s_0.tif", "s_1.tif"]
    assert all(m.get(k)["export_key"] != before[k]["export_key"] for k in others), "the median did not move"
    assert jobs == [0, 1] and summary["retagged"] == 2              # one scan encoded again, two patched
    assert m.get("s_2.tif")["exported_at"] != before["s_2.tif"]["exported_at"]
    for k in others:
        e = m.get(k)
        assert e["exported_at"] == before[k]["exported_at"]       # not encoded again
        assert e["export_base"] == before[k]["export_base"] and e["scale"]["px_per_mm"] != before[k]["scale"]["px_per_mm"]
        with tifffile.TiffFile(str(m.abs(e["outputs"]["tiff"]))) as tf:
            num, den = tf.pages[0].tags["XResolution"].value
        assert abs(num / den - e["scale"]["px_per_mm"] * 10) < 0.05 * 10     # the tags carry the new scale
        assert abs(e["export_px_per_mm"] - e["scale"]["px_per_mm"]) < 1e-3
    again = api.process([str(src)], str(out), inferred)           # nothing is left to do
    assert again["exported"] == 0 and again["retagged"] == 0


# --------------------------------------------------------------------------- second review of slice 2
def test_a_deliberate_scale_change_exports_everything_again(tmp_path):
    import tifffile
    from synth import make_scan
    src = tmp_path / "in"
    src.mkdir()
    for i in range(3):
        rgb, _ = make_scan("white", 0.3 * i, seed=80 + i)
        tifffile.imwrite(str(src / f"p_{i}.tif"), rgb, photometric="rgb")
    out = tmp_path / "out"
    api.process([str(src)], str(out), api.make_settings(workers=1, no_derivatives=True))
    jobs = []
    summary = api.process([str(src)], str(out), api.make_settings(workers=1, no_derivatives=True, ppi=PPI),
                          on_jobs=jobs.append)
    assert jobs == [0, 3] and summary["retagged"] == 0 and summary["exported"] == 3      # 3 new exports
    assert {e["scale"]["source"] for _, e in Manifest(str(out)).items()} == {"ppi"}
    again = api.process([str(src)], str(out), api.make_settings(workers=1, no_derivatives=True, ppi=PPI))
    assert again["retagged"] == 0
    jobs.clear()
    api.process([str(src)], str(out), api.make_settings(workers=1, no_derivatives=True, ppi=PPI * 1.02),
                on_jobs=jobs.append)
    assert jobs == [0, 3]                                       # a changed ppi: new exports, not tag patches


def test_a_fix_found_by_a_later_refresh_is_judged_before_the_export_pass(src, tmp_path):
    from polaroid_processor.pipeline import ReviewRules
    out = tmp_path / "out"
    strict = settings(export=False)
    strict.rules = ReviewRules(size_tolerance=-1.0)              # every scan is flagged
    api.process([str(src)], str(out), strict)
    shutil.copy(src / "a_white.tif", src / "b_0.tif")
    done = []

    def accept_gold(phase):
        if phase == "checking a_white.tif" and not done:         # a_gold was judged a moment ago
            done.append(api.fix_scan(str(out), "a_gold.tif", accept=True, author="agent"))
            time.sleep(1.2)                                      # the next flagged scan (b_0) refreshes again

    api.process([str(src)], str(out), strict, on_phase=accept_gold)
    m = Manifest(str(out))
    assert done and m.get("a_gold.tif")["override"]["author"] == "agent"
    assert m.get("a_gold.tif")["status"] == "ok"                 # not the run's older needs_review
    assert m.get("a_white.tif")["status"] == "needs_review" and m.get("b_0.tif")["status"] == "needs_review"


def test_the_summary_shows_a_fix_made_during_the_export(src, tmp_path):
    from polaroid_processor.pipeline import ReviewRules
    out = tmp_path / "out"
    strict = settings(export=False)
    strict.rules = ReviewRules(size_tolerance=-1.0)
    shutil.copy(src / "a_white.tif", src / "b_0.tif")
    api.process([str(src)], str(out), strict)
    exporting = settings(no_derivatives=True)
    exporting.rules = ReviewRules(size_tolerance=-1.0)
    seen = []

    def accept_on_last(key, what):
        if what != "exported":                                   # (the blank scan fails its detection again)
            return
        seen.append(key)
        if len(seen) == 3:                                       # the last export of three
            api.fix_scan(str(out), "a_gold.tif", accept=True, author="agent")

    summary = api.process([str(src)], str(out), exporting, progress=accept_on_last)
    assert len(seen) == 3
    assert "a_gold.tif" not in summary["needs_review"] and summary["counts"]["needs_review"] == 2
    assert Manifest(str(out)).get("a_gold.tif")["status"] == "ok"


def test_a_failed_write_does_not_keep_the_lock(tmp_path, monkeypatch):
    out = tmp_path / "out"

    def boom(*a, **k):
        raise OSError("disk full")

    lock = RunLock(str(out))
    monkeypatch.setattr("polaroid_processor.lock.os.write", boom)
    with pytest.raises(OSError, match="disk full"):
        lock.acquire()
    monkeypatch.undo()
    assert holder(str(out)) is None
    RunLock(str(out)).acquire()


def test_a_probe_holding_the_lock_for_a_moment_does_not_fail_a_starting_run(tmp_path):
    from polaroid_processor.lock import _try_lock, _unlock
    out = tmp_path / "out"
    write_lock(out, dead_pid())
    fd = os.open(out / LOCK_NAME, os.O_RDWR)
    assert _try_lock(fd)                                         # what holder() does for an instant

    def release():
        _unlock(fd)
        os.close(fd)

    threading.Timer(0.06, release).start()
    with RunLock(str(out)):
        assert holder(str(out))["pid"] == os.getpid()


# --------------------------------------------------------------------------- cancel stops the work in flight
import multiprocessing

from pool_jobs import slow_job
from polaroid_processor.pipeline import _clean_partial, _run_pool


def test_cancel_kills_the_jobs_in_flight_and_returns_at_once():
    flag = threading.Event()
    threading.Timer(0.6, flag.set).start()
    t0 = time.time()
    got = list(_run_pool(slow_job, [{"i": i, "t": 30} for i in range(6)], 8, flag.is_set))   # more workers than jobs
    assert got == [] and time.time() - t0 < 10
    time.sleep(0.5)
    assert multiprocessing.active_children() == []              # no worker is left behind


def test_cancel_keeps_the_results_that_are_in():
    flag = threading.Event()
    got = []
    for res in _run_pool(slow_job, [{"i": 0, "t": 0.0}] + [{"i": i, "t": 30} for i in range(1, 5)], 8, flag.is_set):
        got.append(res["key"])
        flag.set()
    assert got == ["0"]


def test_only_about_the_worker_count_is_submitted_at_a_time():
    flag = threading.Event()
    seen = []
    for res in _run_pool(slow_job, [{"i": i, "t": 0.2} for i in range(12)], 2, flag.is_set):
        seen.append(res["key"])
    assert sorted(seen, key=int) == [str(i) for i in range(12)]


def test_cancelling_detection_with_a_pool_bigger_than_the_batch(src, tmp_path):
    for i in range(5):
        shutil.copy(src / "a_white.tif", src / f"b_{i}.tif")
    out = tmp_path / "out"
    flag, jobs = threading.Event(), []
    threading.Timer(0.3, flag.set).start()
    t0 = time.time()
    summary = api.process([str(src)], str(out), api.make_settings(ppi=PPI, workers=8, export=False),
                          cancel=flag.is_set, on_jobs=jobs.append)
    took = time.time() - t0
    assert summary["cancelled"] and jobs == [8] and took < 20
    m = Manifest(str(out))
    done = [k for k, e in m.items() if e.get("detection")]
    assert len(done) < 7                                          # whatever finished is kept, the rest is not done
    time.sleep(0.5)
    assert multiprocessing.active_children() == []
    # the next run does the rest and keeps what was done
    stamp = {k: m.get(k)["detected_at"] for k in done}
    again = api.process([str(src)], str(out), api.make_settings(ppi=PPI, workers=8, export=False), on_jobs=jobs.append)
    m = Manifest(str(out))
    assert not again["cancelled"] and jobs[-1] == 8 - len(done)
    assert all(m.get(k)["detected_at"] == t for k, t in stamp.items())
    assert sum(1 for _, e in m.items() if e.get("detection")) == 7


def test_cancelling_an_export_leaves_no_partial_file_and_the_next_run_finishes(src, tmp_path):
    import tifffile
    for i in range(5):
        shutil.copy(src / "a_white.tif", src / f"b_{i}.tif")
    out = tmp_path / "out"
    api.process([str(src)], str(out), api.make_settings(ppi=PPI, workers=2, export=False))
    flag, jobs = threading.Event(), []

    def on_jobs(n):
        jobs.append(n)
        if len(jobs) == 2:                                        # the export pass starts: cut it off a moment later
            threading.Timer(0.5, flag.set).start()

    exporting = api.make_settings(ppi=PPI, workers=8, derivatives=["web=800:jpg:85"])
    t0 = time.time()
    summary = api.process([str(src)], str(out), exporting, cancel=flag.is_set, on_jobs=on_jobs)
    assert summary["cancelled"] and jobs == [1, 7] and time.time() - t0 < 25
    m = Manifest(str(out))
    for k, e in m.items():
        outs = e.get("outputs") or {}
        if outs:      # an entry that claims an export: its files are whole
            assert e.get("export_key")
            for rel in outs.values():
                assert m.abs(rel).exists()
            tifffile.imread(str(m.abs(outs["tiff"])))
    cut = [k for k, e in m.items() if e.get("detection") and not e.get("export_key")]
    assert cut                                                    # something was cut off, and is not marked exported
    for f in out.rglob("*.tif"):                                  # every file with a final name reads back
        assert tifffile.imread(str(f)).size
    again = api.process([str(src)], str(out), exporting)
    assert not again["cancelled"] and not list(out.rglob("*.part"))
    m = Manifest(str(out))
    assert sum(1 for _, e in m.items() if e.get("export_key")) == 7
    for _, e in m.items():
        if e.get("outputs"):
            tifffile.imread(str(m.abs(e["outputs"]["tiff"])))


def test_temporary_files_of_a_killed_run_are_cleaned_up_but_not_other_part_files(tmp_path):
    out = tmp_path / "out"
    (out / "web").mkdir(parents=True)
    (out / "_overlays").mkdir()
    (out / "_previews").mkdir()
    gone = ["x.tif.part", "x2.tiff.part", "web/y.jpg.part", "web/y2.jpeg.part", "p.png.part", "_overlays/o.jpg.part"]
    stay = ["_previews/z.jpg.part", "polaroid-manifest.json.7.8.part", "keep.tif", "movie.mp4.part",
            "archive.zip.part", "web/download.part"]
    for rel in gone + stay:
        (out / rel).write_bytes(b"x")
    _clean_partial(out)
    left = sorted(str(p.relative_to(out)).replace("\\", "/") for p in out.rglob("*") if p.is_file())
    assert left == sorted(stay)


def test_the_app_cancel_stops_a_detect_run_within_seconds(app, src):
    for i in range(6):
        shutil.copy(src / "a_white.tif", src / f"b_{i}.tif")
    app.json("/api/project/open", {"input": str(src)})
    assert app.json("/api/project/detect", {})[0] == 200
    time.sleep(0.5)
    t0 = time.time()
    assert app.json("/api/project/cancel", {})[0] == 200
    prog = app.wait(lambda p: p["run"]["state"] not in ("running", "cancelling"), timeout=30)
    assert prog["run"]["state"] == "cancelled" and time.time() - t0 < 15
    assert prog["run"]["done"] < prog["run"]["total"]
    app.json("/api/project/detect", {})                           # resumes with the rest
    prog = app.wait(finished)
    assert prog["counts"]["pending"] == 0 and prog["counts"]["detected"] == 0
