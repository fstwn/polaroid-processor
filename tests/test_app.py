"""The v2 app: detect-only runs, crop previews, project list and counts, progress and cancel."""
import io
import json
import shutil
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest
import tifffile
from PIL import Image

from polaroid_processor import api
from polaroid_processor.app import AppState, make_server
from polaroid_processor.app.project import project_summary
from polaroid_processor.cli import main
from polaroid_processor.manifest import Manifest
from polaroid_processor.preview import preview_key

from synth import PX_PER_MM, make_scan

PPI = PX_PER_MM * 25.4


@pytest.fixture(scope="module")
def scans(tmp_path_factory):
    """Two synthetic scans plus a blank one that cannot be detected."""
    d = tmp_path_factory.mktemp("app_scans")
    for i, frame in enumerate(["white", "gold"]):
        rgb, _ = make_scan(frame, 0.3 * i, seed=60 + i)
        tifffile.imwrite(str(d / f"a_{frame}.tif"), rgb, photometric="rgb")
    tifffile.imwrite(str(d / "a_blank.tif"), np.zeros((500, 400, 3), np.uint8), photometric="rgb")
    return d


@pytest.fixture
def src(scans, tmp_path):
    """A fresh copy of the input folder, so a test can add scans."""
    d = tmp_path / "scans"
    shutil.copytree(scans, d)
    return d


def settings(**kw):
    return api.make_settings(ppi=PPI, workers=1, **kw)


def test_detect_only_stores_detections_and_previews(src, tmp_path):
    out = tmp_path / "out"
    summary = api.process([str(src)], str(out), settings(export=False, previews=True))
    assert summary["exported"] == 0 and summary["counts"].get("failed") == 1 and not summary["cancelled"]
    m = Manifest(str(out))
    assert sorted(m.scans) == ["a_blank.tif", "a_gold.tif", "a_white.tif"]
    for key in ("a_white.tif", "a_gold.tif"):
        e = m.get(key)
        assert e["detection"]["quad"] and e["status"] in ("ok", "needs_review") and not e.get("outputs")
        pv = e["preview"]
        with Image.open(m.abs(pv["file"])) as im:
            w, h = im.size
        assert pv["size"] == [w, h] and abs(max(w, h) - 600) <= 1
        assert abs(w / h - 88 / 107) < 0.02          # straightened: the print's aspect, upright
        assert pv["key"] == preview_key(np.array(e["detection"]["quad"]), e["detection"].get("curves"), False, 0.1, "auto")
    assert m.get("a_blank.tif")["status"] == "failed" and "preview" not in m.get("a_blank.tif")
    assert sorted(p.name for p in out.iterdir() if p.suffix == ".tif") == []   # nothing exported
    assert sorted(p.name for p in (out / "_previews").iterdir()) == ["a_gold.jpg", "a_white.jpg"]
    detected_at = m.get("a_white.tif")["detected_at"]

    # a later export reuses the stored detections
    summary = api.process([str(src)], str(out), settings(no_derivatives=True))
    assert summary["exported"] == 2
    m = Manifest(str(out))
    assert m.get("a_white.tif")["detected_at"] == detected_at
    assert all(m.abs(p).exists() for p in m.get("a_white.tif")["outputs"].values())


def test_manifest_save_survives_a_reader(src, tmp_path):
    """The app reads the manifest while a run saves it; Windows refuses the swap under an open reader."""
    for i in range(3):
        shutil.copy(src / "a_white.tif", src / f"b_{i}.tif")
    out = tmp_path / "out"
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                if (out / "polaroid-manifest.json").exists():
                    with open(out / "polaroid-manifest.json", "rb") as f:
                        f.read()
                        time.sleep(0.03)          # holds the file open
                time.sleep(0.01)
            except OSError:
                pass                              # the reader may lose against the swap; the writer must not

    t = threading.Thread(target=reader)
    t.start()
    try:
        api.process([str(src)], str(out), settings(export=False))
    finally:
        stop.set()
        t.join()
    assert len(Manifest(str(out)).scans) == 6


def test_default_run_writes_no_previews(src, tmp_path):
    out = tmp_path / "out"
    api.process([str(src)], str(out), settings(no_derivatives=True))
    assert not (out / "_previews").exists() and "preview" not in Manifest(str(out)).get("a_white.tif")


def test_cancel_stops_new_scans_and_a_rerun_resumes(src, tmp_path):
    for i in range(7):
        shutil.copy(src / "a_white.tif", src / f"b_{i}.tif")
    out = tmp_path / "out"
    flag, seen = threading.Event(), []
    summary = api.process([str(src)], str(out), settings(export=False, previews=True),
                          progress=lambda k, what: (seen.append(k), flag.set()), cancel=flag.is_set,
                          on_jobs=lambda n: seen.append(n))
    assert seen[0] == 10 and summary["cancelled"]
    done = [k for k, e in Manifest(str(out)).items() if e.get("detection")]
    assert 1 <= len(done) < 9
    stamp = {k: Manifest(str(out)).get(k)["detected_at"] for k in done}
    summary = api.process([str(src)], str(out), settings(export=False, previews=True), on_jobs=lambda n: seen.append(n))
    assert seen[-1] == 10 - len(done)               # the scans without a detection (the blank one is tried again)
    assert not summary["cancelled"]
    m = Manifest(str(out))
    assert sum(1 for _, e in m.items() if e.get("detection")) == 9
    assert all(m.get(k)["detected_at"] == t for k, t in stamp.items())      # not detected again


# --------------------------------------------------------------------------- the HTTP side
class AppClient:
    def __init__(self, tmp_path, picked=None):
        self.picked = picked
        self.state = AppState(tmp_path / "config" / "app.json", picker=lambda title, initial: self.picked)
        self.srv = make_server(self.state, 0)
        self.port = self.srv.server_address[1]
        self.t = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.t.start()

    def request(self, path, body=None, headers=None, method=None):
        h = {"Host": f"127.0.0.1:{self.port}", **(headers or {})}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            h.setdefault("Content-Type", "application/json")
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data, headers=h, method=method)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()

    def json(self, path, body=None, **kw):
        status, _, raw = self.request(path, body, **kw)
        return status, json.loads(raw)

    def wait(self, pred, timeout=180):
        end = time.time() + timeout
        while time.time() < end:
            _, prog = self.json("/api/project/progress")
            if pred(prog):
                return prog
            time.sleep(0.2)
        raise AssertionError(f"timeout; last progress {prog}")

    def close(self):
        self.state.shutdown()
        self.srv.shutdown()
        self.srv.server_close()
        self.t.join(timeout=5)


@pytest.fixture
def app(tmp_path):
    c = AppClient(tmp_path)
    yield c
    c.close()


def finished(prog):
    return prog["run"] and prog["run"]["state"] == "finished" and prog["previews_pending"] == 0


def test_page_and_static(app):
    status, headers, raw = app.request("/")
    assert status == 200 and headers["Content-Type"].startswith("text/html") and b"Polaroid" in raw
    for name, ctype in (("app.js", "javascript"), ("app.css", "css")):
        status, headers, _ = app.request(f"/static/{name}")
        assert status == 200 and ctype in headers["Content-Type"]
    assert app.request("/static/..%5Capp.py")[0] == 404 and app.request("/static/nope.js")[0] == 404


def test_rejects_foreign_host_and_origin(app):
    assert app.request("/api/home", headers={"Host": "evil.example"})[0] == 403
    assert app.request("/api/home", headers={"Origin": "http://evil.example"})[0] == 403
    assert app.request("/api/project/close", {}, headers={"Content-Type": "text/plain"})[0] == 400


def test_home_is_empty_and_browse_uses_the_dialog(app, src):
    status, home = app.json("/api/home")
    assert status == 200 and home["projects"] == [] and home["recent_inputs"] == [] and home["current"] is None
    app.picked = str(src)
    assert app.json("/api/browse", {})[1] == {"path": str(src)}
    app.picked = None                  # dialog cancelled
    assert app.json("/api/browse", {})[1] == {"path": None}


def test_open_validates(app, src, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    for body in ({}, {"input": str(tmp_path / "missing")}, {"input": str(empty)},
                 {"input": str(src), "output": str(src)}, {"output": str(empty)}):
        status, data = app.json("/api/project/open", body)
        assert status == 400 and "error" in data, body
    assert app.json("/api/project")[1] == {"project": None}
    assert app.request("/api/project/progress")[0] == 400
    assert not (tmp_path / "scans_processed").exists()      # nothing is created before the run


def test_project_detect_run_fills_the_grid(app, src, tmp_path):
    status, proj = app.json("/api/project/open", {"input": str(src)})
    out = tmp_path / "scans_processed"
    assert status == 200 and proj["output"] == str(out) and proj["input"] == str(src)
    _, prog = app.json("/api/project/progress")
    assert prog["run"] is None and prog["can_run"] and prog["previews_pending"] == 0
    assert prog["counts"]["pending"] == 3 and prog["counts"]["total"] == 3
    assert app.json("/api/project/cancel", {})[0] == 400        # nothing runs

    assert app.json("/api/project/detect", {})[0] == 200
    assert app.json("/api/project/detect", {})[0] == 400        # one run at a time
    prog = app.wait(finished)
    assert prog["run"]["total"] == 3 and prog["run"]["done"] == 3 and prog["run"]["errors"] == 1
    c = prog["counts"]
    assert c["failed"] == 1 and c["ok"] + c["needs_review"] == 2 and c["pending"] == 0 and c["exported"] == 0

    _, data = app.json("/api/project/scans")
    assert [s["key"] for s in data["scans"]] == ["a_blank.tif", "a_gold.tif", "a_white.tif"]
    blank, gold, white = data["scans"]
    assert blank["status"] == "failed" and blank["preview"] is None and blank["error"]
    for s in (gold, white):
        assert s["status"] in ("ok", "needs_review") and not s["stale"] and not s["exported"]
        status, headers, raw = app.request(s["preview"]["url"])
        assert status == 200 and headers["Content-Type"] == "image/jpeg"
        with Image.open(io.BytesIO(raw)) as im:
            assert list(im.size) == s["preview"]["size"]
    assert app.request("/api/project/preview/a_blank.tif.jpg")[0] == 404
    assert app.request("/api/project/preview/..%5Cpolaroid-manifest.json.jpg")[0] == 404

    # a second run only tries the failed scan again; the home page shows the counts
    assert app.json("/api/project/detect", {})[0] == 200
    assert app.wait(lambda p: p["run"]["state"] == "finished")["run"]["total"] == 1
    _, home = app.json("/api/home")
    (p,) = home["projects"]
    assert p["output"] == str(out) and p["open"] and p["counts"]["failed"] == 1 and p["counts"]["total"] == 3
    assert home["recent_inputs"] == [str(src)]
    assert sorted(f.name for f in src.iterdir()) == ["a_blank.tif", "a_gold.tif", "a_white.tif"]   # input untouched
    assert app.json("/api/project/close", {})[0] == 200 and app.json("/api/project")[1] == {"project": None}


def test_cancel_through_the_api(app, src, tmp_path):
    for i in range(7):
        shutil.copy(src / "a_white.tif", src / f"b_{i}.tif")
    app.json("/api/project/open", {"input": str(src)})
    assert app.json("/api/project/detect", {})[0] == 200
    app.wait(lambda p: p["run"]["done"] >= 1)
    status, run = app.json("/api/project/cancel", {})
    assert status == 200 and run["state"] in ("cancelling", "cancelled", "finished")
    prog = app.wait(lambda p: p["run"]["state"] != "cancelling" and p["run"]["state"] != "running")
    if prog["run"]["state"] == "cancelled":
        assert prog["run"]["done"] < prog["run"]["total"]
    n_done = prog["counts"]["ok"] + prog["counts"]["needs_review"] + prog["counts"]["detected"]
    # a new run goes on with the rest
    assert app.json("/api/project/detect", {})[0] == 200
    prog = app.wait(finished)
    assert prog["run"]["total"] == 10 - n_done
    assert prog["counts"]["pending"] == 0 and prog["counts"]["detected"] == 0 and prog["counts"]["failed"] == 1


def test_stale_preview_is_made_again_after_a_fix(app, src, tmp_path):
    app.json("/api/project/open", {"input": str(src)})
    app.json("/api/project/detect", {})
    app.wait(finished)
    before = {s["key"]: s for s in app.json("/api/project/scans")[1]["scans"]}["a_white.tif"]
    api.fix_scan(app.state.project.output, "a_white.tif", shifts={"left": [8.0]})
    _, mid = app.json("/api/project/scans")
    assert {s["key"]: s for s in mid["scans"]}["a_white.tif"]["stale"]
    app.wait(lambda p: p["previews_pending"] == 0 and p["run"]["state"] == "finished")
    after = {s["key"]: s for s in app.json("/api/project/scans")[1]["scans"]}["a_white.tif"]
    assert not after["stale"] and after["preview"]["url"] != before["preview"]["url"]
    assert after["preview"]["size"][0] > before["preview"]["size"][0]       # 8 px wider on the left


def test_open_an_output_folder_made_by_the_cli(app, src, tmp_path):
    out = tmp_path / "cli_out"
    assert main(["process", str(src), "-o", str(out), "--ppi", str(PPI), "--no-derivatives", "-j", "1", "-q"]) == 1
    status, proj = app.json("/api/project/open", {"output": str(out)})
    assert status == 200 and proj["input"] == str(src) and proj["output"] == str(out)
    _, prog = app.json("/api/project/progress")
    assert prog["counts"]["exported"] == 2 and prog["counts"]["failed"] == 1 and prog["counts"]["pending"] == 0
    assert prog["previews_pending"] == 2                   # no previews yet: made in the background
    prog = app.wait(lambda p: p["previews_pending"] == 0)
    assert {s["key"]: s["preview"] is not None for s in app.json("/api/project/scans")[1]["scans"]} == {
        "a_blank.tif": False, "a_gold.tif": True, "a_white.tif": True}
    # a project folder that belongs to another input folder is refused
    other = tmp_path / "other"
    other.mkdir()
    shutil.copy(src / "a_white.tif", other / "a_white.tif")
    status, data = app.json("/api/project/open", {"input": str(other), "output": str(out)})
    assert status == 400 and "already holds" in data["error"]
    # forgetting drops it from the list but not from disk
    assert app.json("/api/project/forget", {"output": str(out)})[0] == 200
    assert app.json("/api/home")[1]["projects"] == [] and (out / "polaroid-manifest.json").exists()


def test_project_summary_of_a_missing_project(tmp_path):
    info = project_summary(str(tmp_path / "gone_in"), str(tmp_path / "gone_out"))
    assert not info["has_manifest"] and not info["input_exists"] and not info["output_exists"]
    assert info["counts"]["total"] == 0


# --------------------------------------------------------------------------- review follow-ups
def fake_running(project):
    """Make a project look as if a run is going (no thread behind it)."""
    from polaroid_processor.app.project import Run
    project.run = Run()
    return project.run


def test_close_and_reopen_during_a_run(app, src, tmp_path):
    app.json("/api/project/open", {"input": str(src)})
    run = fake_running(app.state.project)
    status, data = app.json("/api/project/close", {})
    assert status == 200 and data["stopping"] and run.cancel.is_set()
    out = str(tmp_path / "scans_processed")
    status, data = app.json("/api/project/open", {"input": str(src)})
    assert status == 400 and "still stopping" in data["error"]
    assert app.json("/api/project/open", {"output": out})[0] == 400
    other = tmp_path / "elsewhere"
    assert app.json("/api/project/open", {"input": str(src), "output": str(other)})[0] == 200   # other folder is fine
    app.json("/api/project/close", {})
    run.state = "cancelled"                      # the run thread ends
    status, data = app.json("/api/project/open", {"input": str(src)})
    assert status == 200 and app.json("/api/project/close", {})[1]["stopping"] is False


def test_preview_keeper_waits_for_the_run(src, tmp_path):
    from polaroid_processor.app.project import Project
    out = tmp_path / "out"
    api.process([str(src)], str(out), settings(no_derivatives=True))     # exported, no previews
    p = Project(str(src), str(out))
    run = fake_running(p)
    p.kick_previews()
    p.progress()
    assert p._keeper is None and not (out / "_previews").exists()
    run.state = "finished"
    p.kick_previews()
    p._keeper.join(timeout=120)
    assert sorted(f.name for f in (out / "_previews").iterdir()) == ["a_gold.jpg", "a_white.jpg"]
    p.close()
    p.kick_previews()
    assert not p._keeper.is_alive()


def test_closed_project_does_not_start_again(src, tmp_path):
    from polaroid_processor.app.project import Project
    p = Project(str(src), str(tmp_path / "out"))
    assert p.close()
    with pytest.raises(api.UsageError):
        p.start_detect()


def test_failing_preview_does_not_fail_the_scan(src, tmp_path, monkeypatch):
    from polaroid_processor import pipeline
    out = tmp_path / "out"
    out.mkdir()

    def boom(*a, **k):
        raise RuntimeError("no preview today")

    monkeypatch.setattr(pipeline, "write_preview", boom)
    s = settings(export=False, previews=True)
    res = pipeline._job({"key": "a_white.tif", "path": str(src / "a_white.tif"), "detect": True, "detection": None,
                         "override": None, "export": False, "scale": PX_PER_MM, "detect_params": s.detect_params,
                         "export_settings": pipeline._export_settings(s, out),
                         "preview": {"dir": str(out / "_previews"), "margin_mm": 0.1, "flatten": "auto"}})
    assert "error" not in res and res["detection"]["quad"] and "preview" not in res


def run_with_fake_process(project, monkeypatch, cancelled, done=2):
    def fake(inputs, output, settings, progress=None, cancel=None, on_jobs=None, on_phase=None, **kw):
        on_jobs(2)
        for k in range(done):
            progress(f"s{k}.tif", "detected")
        on_phase("checking sizes")
        time.sleep(0.2)
        return {"cancelled": cancelled}

    monkeypatch.setattr(api, "process", fake)
    project.start_detect()


def test_cancel_at_the_end_of_a_run_does_not_stick(src, tmp_path, monkeypatch):
    from polaroid_processor.app.project import Project
    p = Project(str(src), str(tmp_path / "out"))
    run_with_fake_process(p, monkeypatch, cancelled=True, done=2)       # everything was done: not "cancelled"
    while True:
        try:
            p.cancel()
        except api.UsageError:
            break
        time.sleep(0.01)
    p._thread.join(timeout=10)
    assert p.run.state == "finished" and p.run.record()["phase"] is None and p.run.ended
    run_with_fake_process(p, monkeypatch, cancelled=True, done=1)        # dropped scans: "cancelled"
    p.cancel()
    p._thread.join(timeout=10)
    assert p.run.state == "cancelled"
    assert not p.running


def test_open_refuses_an_output_inside_the_input(app, src):
    status, data = app.json("/api/project/open", {"input": str(src), "output": str(src / "out")})
    assert status == 400 and "inside the input folder" in data["error"]


def test_phases_are_reported(src, tmp_path):
    phases = []
    api.process([str(src)], str(tmp_path / "out"), settings(export=False), on_phase=phases.append)
    assert phases[0] == "checking sizes"
