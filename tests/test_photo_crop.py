"""Photo-area crop, slice 2: settings and export key, the window step at Detect, the review rule, the photo/ export."""
import json
import shutil
import time

import numpy as np
import pytest
import tifffile
from PIL import Image

from polaroid_processor import api
from polaroid_processor.formats import POLAROID_600, POLAROID_GO
from polaroid_processor.manifest import Manifest
from polaroid_processor.pipeline import (PHOTO_BLEED_MM, Settings, settings_from_dict, settings_to_dict)
from polaroid_processor.scale import MM_PER_INCH, Scale
from polaroid_processor import cli, pipeline
from polaroid_processor.scan_window import outline_id, window_current
from polaroid_processor.window import find_window_for_formats, inset_quad

from synth import PRINT_MM, PX_PER_MM, make_scan
from test_window import make_print, window_quad

PPI = PX_PER_MM * MM_PER_INCH
SQUARE = (5.5, 6.3, 76.9, 78.9)       # nominal 600 windows, mm from the print's top left corner
ROUND = (5.5, 7.7, 76.7, 76.7)
ODD = (20.0, 30.0, 40.0, 40.0)        # a window of no format: "not found"
FRAME = np.array([232, 234, 228])


SPECS = {"sq": dict(window_mm=SQUARE), "rd": dict(window_mm=ROUND, round_window=True), "nw": dict(window_mm=ODD)}


def make_named(name, which=("sq", "rd", "nw")):
    """The synthetic scan called ``name`` and the true print outline."""
    i = which.index(name)
    return make_scan("white", 0.3 * (i - 1), seed=40 + i, **SPECS[name])


def write_scans(folder, which=("sq", "rd", "nw")):
    folder.mkdir(parents=True, exist_ok=True)
    for name in which:
        rgb, _ = make_named(name, which)
        tifffile.imwrite(str(folder / f"{name}.tif"), rgb, photometric="rgb")
    return folder


def true_window(name, box):
    """Corners (tl, tr, br, bl) in scan px of the window ``box`` = (left, top, width, height) mm from the print's top left."""
    _, quad = make_named(name)
    left, top, w, h = box
    u, v = (quad[1] - quad[0]) / PRINT_MM[0], (quad[3] - quad[0]) / PRINT_MM[1]      # px per mm along the print's axes
    return np.array([quad[0] + a * u + b * v for a, b in ((left, top), (left + w, top), (left + w, top + h), (left, top + h))])


def run(src, out, **kw):
    kw.setdefault("workers", 1)
    kw.setdefault("ppi", PPI)
    return api.process([str(src)], str(out), api.make_settings(**kw))


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    return write_scans(tmp_path_factory.mktemp("photo_scans"))


@pytest.fixture(scope="module")
def project(src, tmp_path_factory):
    """Three scans exported with photo_crop=tiff: a square window, a round one and one without a window."""
    out = tmp_path_factory.mktemp("photo_out")
    run(src, out, photo_crop="tiff")
    return out


def entry(out, name):
    return Manifest(str(out)).get(name)


# --------------------------------------------------------------------------- settings and key
def test_off_keeps_the_export_keys_of_exports_made_before():
    q = np.array([[10.0, 20.0], [1000.5, 21.0], [1001.0, 1300.25], [9.5, 1299.0]])
    s = Settings()
    # values of the commit before the photo crop; a change here would export every old project again
    assert s.export_key(q, Scale(25.6, "ppi")) == "2fecc94a02844b31"
    assert s.export_base(q, Scale(25.6, "ppi")) == "5e4aa68ee0da87b8"
    win = {"found": True, "kind": "square", "quad": [[1, 1], [9, 1], [9, 9], [1, 9]]}
    assert s.export_key(q, Scale(25.6, "ppi"), window=win) == "2fecc94a02844b31"      # off: the window is not in it


def test_photo_settings_change_the_key():
    q = np.array([[10.0, 20.0], [1000.5, 21.0], [1001.0, 1300.25], [9.5, 1299.0]])
    sc = Scale(25.6, "ppi")
    win = {"found": True, "kind": "square", "quad": [[1, 1], [9, 1], [9, 9], [1, 9]]}
    moved = {**win, "quad": [[1, 1], [9, 1], [9, 9.5], [1, 9]]}
    keys = {}
    for name, kw, w in (("off", {}, None), ("der", dict(photo_crop="derivatives"), win),
                        ("tiff", dict(photo_crop="tiff"), win), ("bleed", dict(photo_crop="tiff", photo_bleed_mm=0.5), win),
                        ("moved", dict(photo_crop="tiff"), moved), ("nowin", dict(photo_crop="tiff"), None)):
        s = Settings(**kw)
        keys[name] = (s.export_key(q, sc, window=w), s.export_base(q, sc, window=w))
    assert len({k[0] for k in keys.values()}) == len(keys) and len({k[1] for k in keys.values()}) == len(keys)


def test_settings_round_trip_and_defaults_stay_out_of_the_manifest():
    d = settings_to_dict(Settings())
    assert "photo_crop" not in d and "photo_bleed_mm" not in d                # a manifest made without it is unchanged
    assert settings_from_dict(d).photo_crop == "off" and settings_from_dict(d).photo_bleed_mm == PHOTO_BLEED_MM
    s = Settings(photo_crop="tiff", photo_bleed_mm=0.5)
    back = settings_from_dict(json.loads(json.dumps(settings_to_dict(s))))
    assert (back.photo_crop, back.photo_bleed_mm) == ("tiff", 0.5)
    assert settings_to_dict(Settings(photo_bleed_mm=0.5))["photo_bleed_mm"] == 0.5     # kept while the crop is off


def test_make_settings_checks_the_photo_options():
    assert api.make_settings(photo_crop="tiff", photo_bleed_mm=0.4).photo_bleed_mm == 0.4
    with pytest.raises(api.UsageError, match="photo_crop"):
        api.make_settings(photo_crop="maybe")
    with pytest.raises(api.UsageError, match="bleed"):
        api.make_settings(photo_bleed_mm=-1)


def test_options_left_out_keep_the_projects_values(src, tmp_path):
    assert api.make_settings().photo_crop is None and api.make_settings().photo_bleed_mm is None
    a = cli.build_parser().parse_args(["process", str(src), "-o", str(tmp_path / "x")])
    assert a.photo_crop is None and a.photo_bleed_mm is None
    out = tmp_path / "out"
    run(src, out, photo_crop="tiff", photo_bleed_mm=0.5)
    files = sorted(p.relative_to(out).as_posix() for p in (out / "photo").rglob("*") if p.is_file())
    stamps = {p: p.stat().st_mtime_ns for p in (out / "photo").rglob("*") if p.is_file()}
    keys = {k: e["export_key"] for k, e in Manifest(str(out)).items()}
    summary = run(src, out)                                       # no photo option: the project keeps tiff / 0.5
    m = Manifest(str(out))
    assert m.data["settings"]["photo_crop"] == "tiff" and m.data["settings"]["photo_bleed_mm"] == 0.5
    assert summary["exported"] == 0
    assert {k: e["export_key"] for k, e in m.items()} == keys
    assert sorted(p.relative_to(out).as_posix() for p in (out / "photo").rglob("*") if p.is_file()) == files
    assert {p: p.stat().st_mtime_ns for p in (out / "photo").rglob("*") if p.is_file()} == stamps
    run(src, out, photo_crop="off")                               # saying it does switch the crop off
    assert "photo_crop" not in Manifest(str(out)).data["settings"] and not any(p.is_file() for p in (out / "photo").rglob("*"))
    run(src, tmp_path / "fresh", export=False)                    # a new project starts with off / 0.3
    assert "photo_crop" not in Manifest(str(tmp_path / "fresh")).data["settings"]


# --------------------------------------------------------------------------- helpers in window.py
def test_find_window_for_formats_finds_a_600_window_that_classify_calls_go():
    img = make_print(POLAROID_600, 12.0)
    res, fmt = find_window_for_formats(img, POLAROID_GO)         # classify() said Go
    assert res.found and fmt.name == "polaroid-600"
    assert any("classify() said polaroid-go" in n for n in res.notes)
    assert np.abs(res.quad - window_quad()).max() / 12.0 < 0.1
    res, fmt = find_window_for_formats(img, POLAROID_600)
    assert res.found and fmt.name == "polaroid-600" and not any("classify" in n for n in res.notes)


def test_find_window_for_formats_not_found_keeps_the_first_formats_reason():
    img = make_print(POLAROID_600, 12.0, no_window=True)
    res, fmt = find_window_for_formats(img, POLAROID_600)
    assert not res.found and fmt.name == "polaroid-600" and res.reason
    assert any(n.startswith("polaroid-go:") for n in res.notes)


def test_inset_quad_moves_every_side_inward():
    q = np.array([[10.0, 10.0], [110.0, 12.0], [108.0, 212.0], [8.0, 210.0]])
    inner = inset_quad(q, 2.0)
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        t = (b - a) / np.linalg.norm(b - a)
        n = np.array([-t[1], t[0]])
        for p in (inner[i], inner[(i + 1) % 4]):
            assert abs(np.dot(p - a, n)) == pytest.approx(2.0, abs=1e-6)
    assert np.allclose(inset_quad(q, 0.0), q)


# --------------------------------------------------------------------------- the window step at Detect
def test_detect_with_the_crop_off_stores_no_window(src, tmp_path):
    out = tmp_path / "out"
    run(src, out, export=False)
    m = Manifest(str(out))
    assert all("window" not in e for _, e in m.items())
    assert "photo_crop" not in m.data["settings"]


def test_window_is_stored_in_scan_coordinates_with_its_outline(project):
    for name, kind in (("sq.tif", "square"), ("rd.tif", "round")):
        e = entry(project, name)
        w = e["window"]
        assert w["found"] and w["kind"] == kind and w["format"] == "polaroid-600" and w["source"] == "auto"
        assert window_current(e) and w["outline"] == outline_id(Manifest.effective_quad(e))
        quad = np.array(w["quad"])
        assert quad.shape == (4, 2)
        truth = true_window(name[:2], SQUARE if kind == "square" else ROUND)      # the synthetic window, through the true outline
        assert np.abs(quad - truth).max() / PX_PER_MM < 0.1
        assert w["size_mm"][0] == pytest.approx(76.9 if kind == "square" else 76.7, abs=0.5)


def test_window_not_found_is_a_review_item_and_the_full_print_is_still_exported(project):
    e = entry(project, "nw.tif")
    assert not e["window"]["found"] and e["status"] == "needs_review"
    assert any(w.startswith("photo window not found: ") for w in e["warnings"])
    assert e["outputs"].keys() == {"tiff", "high", "web"} and e["export_key"]
    m = Manifest(str(project))
    assert all(m.abs(p).exists() for p in e["outputs"].values())
    assert not (project / "photo" / "nw.tif").exists() and not (project / "photo" / "high" / "nw.jpg").exists()
    for name in ("sq.tif", "rd.tif"):
        assert entry(project, name)["status"] == "ok"


def test_window_step_alone_runs_when_the_crop_is_turned_on_and_when_the_outline_changes(src, tmp_path):
    out = tmp_path / "out"
    run(src, out, export=False)                                   # detected, no crop
    before = Manifest(str(out))
    detected = {k: e["detected_at"] for k, e in before.items()}
    jobs = []
    api.process([str(src)], str(out), api.make_settings(workers=1, ppi=PPI, export=False, photo_crop="derivatives"),
                on_jobs=jobs.append)
    m = Manifest(str(out))
    assert jobs == [0]                                             # nothing detected again
    assert {k: e["detected_at"] for k, e in m.items()} == detected
    detected_windows = {k: e["window"] for k, e in m.items()}
    assert all(window_current(e) for _, e in m.items()) and m.get("nw.tif")["status"] == "needs_review"
    assert all(not e.get("outputs") for _, e in m.items())        # Detect stage: nothing exported
    jobs.clear()
    api.process([str(src)], str(out), api.make_settings(workers=1, ppi=PPI, export=False, photo_crop="derivatives"),
                on_jobs=jobs.append)
    assert jobs == [0]
    assert m.get("sq.tif")["window"] == detected_windows["sq.tif"]   # windows are current: not looked for again
    old = m.get("sq.tif")["window"]
    time.sleep(1.1)
    api.fix_scan(str(out), "sq.tif", shifts={"right": [3.0]}, author="agent")      # a hand fix moves the outline
    m = Manifest(str(out))
    assert not window_current(m.get("sq.tif")) and window_current(m.get("rd.tif"))
    jobs.clear()
    api.process([str(src)], str(out), api.make_settings(workers=1, ppi=PPI, export=False, photo_crop="derivatives"),
                on_jobs=jobs.append)
    m = Manifest(str(out))
    assert jobs == [0]
    assert window_current(m.get("sq.tif")) and m.get("sq.tif")["window"]["outline"] != old["outline"]
    assert m.get("sq.tif")["detected_at"] == detected["sq.tif"]
    assert np.abs(np.array(m.get("sq.tif")["window"]["quad"]) - np.array(old["quad"])).max() < 1.5 * PX_PER_MM   # same window


def test_export_measures_a_window_of_another_outline_again(src, project, tmp_path, monkeypatch):
    out = tmp_path / "out"
    shutil.copytree(project, out)
    time.sleep(1.1)
    api.fix_scan(str(out), "sq.tif", shifts={"right": [3.0]}, author="agent")            # the outline moves
    e = entry(out, "sq.tif")
    assert not window_current(e)
    real_pool = pipeline._run_pool

    def lose_window_jobs(fn, jobs, workers, cancel=None):         # the window pass finds nothing (scan lost, or a fix came later)
        return iter(()) if fn is pipeline._window_job else real_pool(fn, jobs, workers, cancel)

    monkeypatch.setattr(pipeline, "_run_pool", lose_window_jobs)
    summary = run(src, out)
    monkeypatch.undo()
    e = entry(out, "sq.tif")
    assert summary["exported"] == 1 and window_current(e) and e["window"]["found"]      # measured in the export job
    assert (out / "photo" / "sq.tif").exists()
    assert run(src, out)["exported"] == 0             # the key belongs to the new window: nothing to do again


def test_a_window_lost_in_the_export_job_is_a_review_item_in_the_same_run(src, project, tmp_path, monkeypatch):
    out = tmp_path / "out"
    shutil.copytree(project, out)
    time.sleep(1.1)
    api.fix_scan(str(out), "sq.tif", shifts={"right": [80.0]}, author="agent")           # an outline the window no longer fits
    real_pool = pipeline._run_pool
    monkeypatch.setattr(pipeline, "_run_pool", lambda fn, jobs, workers, cancel=None:
                        iter(()) if fn is pipeline._window_job else real_pool(fn, jobs, workers, cancel))
    run(src, out)
    monkeypatch.undo()
    e = entry(out, "sq.tif")
    assert window_current(e) and not e["window"]["found"]
    assert any(w.startswith("photo window not found: ") for w in e["warnings"])        # the fix is an override: status stays ok
    assert not (out / "photo" / "sq.tif").exists() and not (out / "photo" / "high" / "sq.jpg").exists()


def test_a_window_lost_in_the_export_job_makes_a_refit_scan_needs_review(tmp_path, monkeypatch):
    one = write_scans(tmp_path / "one", which=("sq",))
    out = tmp_path / "out"
    real_evaluate, real_pool = pipeline.evaluate, pipeline._run_pool

    def evaluate(entry, *a, **kw):                       # needs_review until the (patched) refit has been applied
        status, warnings, fmt = real_evaluate(entry, *a, **kw)
        return ("needs_review" if not (entry.get("detection") or {}).get("refit") else status), warnings, fmt

    def size_refit(entry, *a, **kw):                     # the left side moves 80 px outward: the window no longer fits
        quad = np.array(entry["detection"]["quad"], float)
        quad[[0, 3], 0] -= 80.0
        return quad, {"side": "left", "shift_px": 80.0}

    monkeypatch.setattr(pipeline, "evaluate", evaluate)
    monkeypatch.setattr(pipeline, "size_refit", size_refit)
    monkeypatch.setattr(pipeline, "_run_pool", lambda fn, jobs, workers, cancel=None:
                        iter(()) if fn is pipeline._window_job else real_pool(fn, jobs, workers, cancel))
    run(one, out, photo_crop="tiff", flatten="off")
    monkeypatch.undo()
    e = entry(out, "sq.tif")
    assert e["detection"]["refit"] and not e.get("override") and window_current(e) and not e["window"]["found"]
    assert e["status"] == "needs_review" and any(w.startswith("photo window not found: ") for w in e["warnings"])
    assert not (out / "photo" / "sq.tif").exists() and not (out / "photo" / "high" / "sq.jpg").exists()


# --------------------------------------------------------------------------- accept
def test_accepting_a_scan_without_a_window_exports_the_print_without_a_crop(project, tmp_path):
    out = tmp_path / "out"
    shutil.copytree(project, out)
    rec = api.fix_scan(str(out), "nw.tif", accept=True, note="no window on this print", export=True)
    e = entry(out, "nw.tif")
    assert e["status"] == "ok" and any(w.startswith("photo window not found") for w in e["warnings"])
    assert rec["window"]["found"] is False
    assert not (out / "photo" / "nw.tif").exists() and not (out / "photo" / "high" / "nw.jpg").exists()
    assert (out / "high" / "nw.jpg").exists() and (out / "nw.tif").exists()
    assert entry(out, "sq.tif")["window"]["found"] and (out / "photo" / "sq.tif").exists()


# --------------------------------------------------------------------------- export
def photo_size_px(path):
    with tifffile.TiffFile(str(path)) as tf:
        h, w = tf.pages[0].shape[:2]
    return w, h


def test_square_window_export(project):
    e = entry(project, "sq.tif")
    assert e["outputs"].keys() == {"tiff", "high", "web", "photo_tiff", "photo_high", "photo_web"}
    m = Manifest(str(project))
    assert m.abs(e["outputs"]["photo_tiff"]) == project / "photo" / "sq.tif"
    assert (project / "photo" / "high" / "sq.jpg").exists() and (project / "photo" / "web" / "sq.jpg").exists()
    w, h = photo_size_px(project / "photo" / "sq.tif")
    # window 76.9 x 78.9 mm, cut 0.3 mm inward on every side
    assert w == pytest.approx((76.9 - 0.6) * PX_PER_MM, abs=0.012 * 76.9 * PX_PER_MM)
    assert h == pytest.approx((78.9 - 0.6) * PX_PER_MM, abs=0.012 * 78.9 * PX_PER_MM)
    with tifffile.TiffFile(str(project / "photo" / "sq.tif")) as tf:
        assert tf.pages[0].shape[2] == 3                           # a square window has no alpha
        num, den = tf.pages[0].tags["XResolution"].value
        assert num / den == pytest.approx(PX_PER_MM * 10, abs=0.5)
    # the whole crop is photo: no frame pixels at the border (the frame is bright, the noise photo is not)
    img = tifffile.imread(str(project / "photo" / "sq.tif"))
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]).astype(float)
    assert np.abs(border - FRAME).sum(axis=1).min() > 8
    with Image.open(project / "photo" / "web" / "sq.jpg") as im:
        assert im.size == (w, h)                                   # smaller than the 2048 px derivative: not enlarged
    assert e["output_ppm"]["photo_tiff"] == pytest.approx(e["output_ppm"]["tiff"])


def test_round_window_export_has_alpha_in_the_tiff_and_frame_corners_in_the_jpg(project):
    img = tifffile.imread(str(project / "photo" / "rd.tif"))
    h, w, c = img.shape
    assert c == 4 and abs(w - h) <= 2 and w == pytest.approx((76.7 - 0.6) * PX_PER_MM, abs=0.012 * 76.7 * PX_PER_MM)
    alpha = img[..., 3]
    assert alpha[0, 0] == 0 and alpha[0, -1] == 0 and alpha[-1, 0] == 0 and alpha[h // 2, w // 2] == 255
    assert alpha[h // 2, 6] == 255 and alpha[6, w // 2] == 255      # inside the circle right up to its rim
    with tifffile.TiffFile(str(project / "photo" / "rd.tif")) as tf:
        assert tf.pages[0].extrasamples[0].name == "UNASSALPHA"
    jpg = np.asarray(Image.open(project / "photo" / "high" / "rd.jpg"))
    assert jpg.shape[2] == 3
    for corner in (jpg[2, 2], jpg[2, -3], jpg[-3, 2], jpg[-3, -3]):
        assert np.abs(corner.astype(float) - FRAME).max() < 25      # the frame, not black or transparent
    assert np.abs(jpg[jpg.shape[0] // 2, jpg.shape[1] // 2].astype(float) - FRAME).sum() > 40   # the photo in the middle


def test_derivatives_mode_writes_no_tiff_master_and_bleed_sets_the_size(src, tmp_path):
    out = tmp_path / "out"
    one = src / "sq.tif"
    run(one, out, photo_crop="derivatives", photo_bleed_mm=0.0)
    w0, h0 = Image.open(out / "photo" / "high" / "sq.jpg").size
    assert not (out / "photo" / "sq.tif").exists() and (out / "photo" / "web" / "sq.jpg").exists()
    assert set(entry(out, "sq.tif")["outputs"]) == {"tiff", "high", "web", "photo_high", "photo_web"}
    run(one, out, photo_crop="derivatives", photo_bleed_mm=0.5)       # a new export key: exported again
    w1, h1 = Image.open(out / "photo" / "high" / "sq.jpg").size
    assert w0 - w1 == pytest.approx(1.0 * PX_PER_MM, abs=1.5) and h0 - h1 == pytest.approx(1.0 * PX_PER_MM, abs=1.5)
    run(one, out, photo_crop="tiff", photo_bleed_mm=0.5)              # + master
    assert (out / "photo" / "sq.tif").exists()
    run(one, out, photo_crop="derivatives", photo_bleed_mm=0.5)       # master dropped again
    assert not (out / "photo" / "sq.tif").exists() and (out / "photo" / "high" / "sq.jpg").exists()


def test_turning_the_crop_off_removes_the_photo_files_of_the_scans_exported_again(src, tmp_path):
    out = tmp_path / "out"
    one = src / "sq.tif"
    summary = run(one, out, photo_crop="tiff")
    assert (out / "photo" / "sq.tif").exists() and summary["exported"] == 1
    old_key = entry(out, "sq.tif")["export_key"]
    summary = run(one, out, photo_crop="off")                           # off: the old-style key, exported again
    e = entry(out, "sq.tif")
    assert summary["exported"] == 1 and e["export_key"] != old_key and set(e["outputs"]) == {"tiff", "high", "web"}
    assert not (out / "photo" / "sq.tif").exists() and not (out / "photo" / "high" / "sq.jpg").exists()
    assert not list((out / "photo").rglob("*.part"))


def test_a_window_lost_after_a_fix_leaves_no_stale_photo_files(src, tmp_path):
    out = tmp_path / "out"
    one = src / "sq.tif"
    run(one, out, photo_crop="derivatives")
    assert (out / "photo" / "web" / "sq.jpg").exists()
    # a hand fix far off the print edge: the window is looked for again, not found, and the old files go
    api.fix_scan(str(out), "sq.tif", shifts={"left": [-250.0]}, author="agent", export=True)
    e = entry(out, "sq.tif")
    assert not e["window"]["found"] and set(e["outputs"]) == {"tiff", "high", "web"}
    assert not (out / "photo" / "web" / "sq.jpg").exists() and not (out / "photo" / "high" / "sq.jpg").exists()


def test_scale_only_change_patches_the_photo_outputs_too(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    for i in range(3):                                                  # no resolution tag: the scale is inferred
        rgb, _ = make_scan("white", 0.3 * i, seed=70 + i, window_mm=SQUARE)
        tifffile.imwrite(str(src / f"s_{i}.tif"), rgb, photometric="rgb")
    out = tmp_path / "out"
    kw = dict(workers=1, photo_crop="tiff")
    api.process([str(src)], str(out), api.make_settings(export=False, **kw))
    api.fix_scan(str(out), "s_0.tif", shifts={"right": [6.0]}, author="agent")
    api.fix_scan(str(out), "s_1.tif", shifts={"right": [-6.0]}, author="agent")
    api.process([str(src)], str(out), api.make_settings(**kw))
    m = Manifest(str(out))
    assert all("photo_tiff" in e["outputs"] for _, e in m.items())
    before = {k: dict(e) for k, e in m.items()}
    time.sleep(1.1)
    api.fix_scan(str(out), "s_2.tif", shifts={"right": [12.0]}, author="agent")    # the median print grows
    summary = api.process([str(src)], str(out), api.make_settings(**kw))
    m = Manifest(str(out))
    assert summary["retagged"] == 2
    for k in ("s_0.tif", "s_1.tif"):
        e = m.get(k)
        assert e["exported_at"] == before[k]["exported_at"]            # not encoded again
        assert e["output_ppm"]["photo_tiff"] != before[k]["output_ppm"]["photo_tiff"]
        for name in ("tiff", "photo_tiff"):
            with tifffile.TiffFile(str(m.abs(e["outputs"][name]))) as tf:
                num, den = tf.pages[0].tags["XResolution"].value
            assert num / den == pytest.approx(e["output_ppm"][name] * 10, abs=0.5)
        with Image.open(m.abs(e["outputs"]["photo_web"])) as im:
            assert im.info["dpi"][0] == pytest.approx(e["output_ppm"]["photo_web"] * 25.4, abs=1.0)
    assert api.process([str(src)], str(out), api.make_settings(**kw))["exported"] == 0


def test_dry_run_and_status_show_the_window(src, tmp_path):
    recs = []
    api.process([str(src)], None, api.make_settings(workers=1, ppi=PPI, dry_run=True, photo_crop="derivatives"),
                emit=recs.append)
    by = {r["file"]: r for r in recs if r["type"] == "result"}
    assert by["sq.tif"]["window"]["found"] and by["sq.tif"]["window"]["kind"] == "square"
    assert by["rd.tif"]["window"]["kind"] == "round"
    assert not by["nw.tif"]["window"]["found"] and by["nw.tif"]["status"] == "needs_review"
    plain = []
    api.process([str(src)], None, api.make_settings(workers=1, ppi=PPI, dry_run=True), emit=plain.append)
    assert all("window" not in r for r in plain if r["type"] == "result")
