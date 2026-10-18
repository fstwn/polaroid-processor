"""End-to-end CLI tests on small synthetic scans (no real data needed)."""
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from polaroid_processor.cli import apply_fix, main
from polaroid_processor.formats import POLAROID_600, POLAROID_GO, match_format
from polaroid_processor.manifest import MANIFEST_NAME, Manifest
from polaroid_processor.pipeline import Derivative
from polaroid_processor.scale import MM_PER_INCH, scale_from_tag

from synth import PX_PER_MM, make_scan

PPI = PX_PER_MM * MM_PER_INCH


@pytest.fixture(scope="module")
def scans(tmp_path_factory):
    d = tmp_path_factory.mktemp("scans")
    truths = {}
    for i, frame in enumerate(["white", "gold", "glitter"]):
        rgb, quad = make_scan(frame, 0.3 * (i - 1), seed=10 + i)
        p = d / f"scan_{frame}.tif"
        tifffile.imwrite(str(p), rgb, photometric="rgb", resolution=(600 / 2.54, 600 / 2.54), resolutionunit="CENTIMETER")
        truths[p.name] = quad
    return d, truths


def run(capsys, *args):
    rc = main([str(a) for a in args])
    out = capsys.readouterr().out
    recs = [json.loads(line) for line in out.splitlines() if line.strip()]
    return rc, recs


def test_process_json_outputs(scans, tmp_path, capsys):
    src, truths = scans
    out = tmp_path / "out"
    rc, recs = run(capsys, "process", src, "-o", out, "--json", "--ppi", PPI, "--overlay",
                   "--derivative", "web=800:jpg:85", "--derivative", "png=x0.25:png", "-j", "1")
    assert rc == 0
    results = [r for r in recs if r["type"] == "result"]
    summary = recs[-1]
    assert summary["type"] == "summary" and summary["scans"] == 3 and not summary["failed"]
    for r in results:
        assert r["status"] == "ok", r
        assert r["format"]["name"] == "polaroid-600"
        assert r["scale"]["source"] == "ppi"
        for key in ("tiff", "web", "png", "overlay"):
            assert Path(r["outputs"][key]).exists(), key
        err = np.linalg.norm(np.array(r["quad"]) - truths[r["file"]], axis=1).max()
        assert err <= 0.2 * PX_PER_MM
    # TIFF master carries the true resolution
    with tifffile.TiffFile(results[0]["outputs"]["tiff"]) as tf:
        num, den = tf.pages[0].tags["XResolution"].value
        assert abs(num / den - PX_PER_MM * 10) < 0.5
    with Image.open(results[0]["outputs"]["web"]) as im:
        assert max(im.size) == 800
    assert (out / MANIFEST_NAME).exists()


def test_rerun_skips_up_to_date(scans, tmp_path, capsys):
    src, _ = scans
    out = tmp_path / "out"
    run(capsys, "process", src, "-o", out, "--json", "--ppi", PPI, "--no-derivatives", "-j", "1")
    rc, recs = run(capsys, "process", src, "-o", out, "--json", "--ppi", PPI, "--no-derivatives", "-j", "1")
    assert rc == 0
    assert recs[-1]["exported"] == 0
    assert all(not r["exported"] for r in recs if r["type"] == "result")


def test_fix_records_provenance_and_reexports(scans, tmp_path, capsys):
    src, _ = scans
    out = tmp_path / "out"
    run(capsys, "process", src, "-o", out, "--json", "--ppi", PPI, "--no-derivatives", "-j", "1")
    before = Manifest(str(out)).effective_quad(Manifest(str(out)).get("scan_white.tif"))
    rc, recs = run(capsys, "fix", out, "scan_white", "--nudge", "tl=3,-2", "--shift", "right=-5",
                   "--author", "agent", "--note", "test", "--export", "--json")
    assert rc == 0
    m = Manifest(str(out))
    e = m.get("scan_white.tif")
    after = m.effective_quad(e)
    assert np.allclose(after[0], before[0] + [3, -2], atol=0.01)
    assert e["override"]["author"] == "agent"
    h = e["history"][-1]
    assert h["action"] == "fix" and h["quad_before"] is not None
    res = [r for r in recs if r["type"] == "result" and r["file"] == "scan_white.tif"]
    assert res and res[0]["exported"] and res[0]["quad_source"] == "override"
    # reset goes back to automatic detection
    rc, _ = run(capsys, "fix", out, "scan_white", "--reset", "--json")
    assert Manifest(str(out)).get("scan_white.tif")["override"] is None


def test_dry_run_writes_nothing(scans, tmp_path, capsys):
    src, _ = scans
    out = tmp_path / "dry"
    rc, recs = run(capsys, "process", src, "-o", out, "--dry-run", "--json", "-j", "1")
    assert rc == 0
    assert not out.exists()
    # TIFF tag says 600 dpi, implausible -> inferred from the batch, with a warning
    r = [r for r in recs if r["type"] == "result"][0]
    assert r["scale"]["source"] == "batch-median"
    assert any(w.startswith("scale_inferred") for w in r["warnings"])


def test_inspect_writes_png(scans, tmp_path, capsys):
    src, _ = scans
    img = tmp_path / "insp.png"
    rc, recs = run(capsys, "inspect", src / "scan_gold.tif", "--image", img, "--json")
    assert rc == 0 and img.exists()
    assert set(recs[0]["corners"]) == {"tl", "tr", "br", "bl"}


def test_inspect_refuses_input_folder(scans, capsys):
    src, _ = scans
    assert main(["inspect", str(src / "scan_gold.tif"), "--image", str(src / "x.png")]) == 2
    assert not (src / "x.png").exists()


def test_failed_scan_sets_exit_code(tmp_path, capsys):
    d = tmp_path / "in"
    d.mkdir()
    rng = np.random.default_rng(0)
    tifffile.imwrite(str(d / "empty.tif"), rng.integers(0, 4, size=(1500, 1200, 3), dtype=np.uint8), photometric="rgb")
    rc, recs = run(capsys, "process", d, "-o", tmp_path / "out", "--json", "-j", "1")
    assert rc == 1
    assert recs[-1]["failed"] == ["empty.tif"]


def test_duplicate_names_rejected(tmp_path, capsys):
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        tifffile.imwrite(str(tmp_path / sub / "same.tif"), np.zeros((10, 10, 3), np.uint8), photometric="rgb")
    rc, recs = run(capsys, "process", tmp_path / "a", tmp_path / "b", "-o", tmp_path / "out", "--json")
    assert rc == 2 and recs[0]["type"] == "error"


# ---- small units -------------------------------------------------------------
def test_derivative_parse():
    assert Derivative.parse("web=2048") == Derivative("web", 2048, None, "jpg", 92)
    assert Derivative.parse("half=x0.5:png") == Derivative("half", None, 0.5, "png", 92)
    with pytest.raises(ValueError):
        Derivative.parse("bad=100:gif")


def test_match_format_orientation_and_go():
    m = match_format((88 * 10, 107 * 10), 10)
    assert m.format is POLAROID_600 and m.error < 1e-6
    m = match_format((107 * 10, 88 * 10), 10)  # landscape scan
    assert m.format is POLAROID_600
    m = match_format((53.9 * 10, 66.6 * 10), 10)
    assert m.format is POLAROID_GO


def test_scale_from_tag_plausibility():
    size = (88 * 128.0, 107 * 128.0)
    assert scale_from_tag((600.0, 600.0), size) is None
    s = scale_from_tag((128.0 * 25.4, 128.0 * 25.4), size)
    assert s is not None and s.source == "tiff-tag"


def test_apply_fix_shift_moves_outward():
    q = np.array([[0, 0], [100, 0], [100, 200], [0, 200]], float)
    r = apply_fix(q, shifts={"right": [5]})
    assert np.allclose(r[1], [105, 0]) and np.allclose(r[2], [105, 200])
    r = apply_fix(q, shifts={"top": [-4]})
    assert np.allclose(r[0], [0, 4])


def test_inferred_scale_is_patched_into_outputs(scans, tmp_path, capsys):
    """Without --ppi each scan exports with its own provisional scale; after the batch the
    resolution metadata of every output must equal the batch-median scale."""
    src, _ = scans
    out = tmp_path / "inferred"
    rc, recs = run(capsys, "process", src, "-o", out, "--json", "--derivative", "web=600:jpg", "-j", "1")
    assert rc == 0
    results = [r for r in recs if r["type"] == "result"]
    finals = {r["scale"]["px_per_mm"] for r in results}
    assert len(finals) == 1 and all(r["scale"]["source"] == "batch-median" for r in results)
    final = finals.pop()
    for r in results:
        with tifffile.TiffFile(r["outputs"]["tiff"]) as tf:
            num, den = tf.pages[0].tags["XResolution"].value
            assert abs(num / den / 10 - final) < 0.01
        with Image.open(r["outputs"]["web"]) as im:
            w_full = tifffile.TiffFile(r["outputs"]["tiff"]).pages[0].shape[1]
            expect_dpi = final * 25.4 * im.size[0] / w_full
            assert abs(im.info["dpi"][0] - expect_dpi) <= 1.0
    # a second run changes nothing
    rc, recs = run(capsys, "process", src, "-o", out, "--json", "--derivative", "web=600:jpg", "-j", "1")
    assert recs[-1]["exported"] == 0


def test_review_rules_parallel_size_outlier_and_hint():
    from polaroid_processor.pipeline import ReviewRules, evaluate
    from polaroid_processor.scale import Scale
    ppm = 128.0
    w, h = 88 * ppm, 107 * ppm
    ref = (w, h)
    scale = Scale(ppm, "ppi")
    good = [[0, 0], [w, 0], [w, h], [0, h]]
    status, warnings, _ = evaluate({"detection": {"quad": good, "edges": {}}}, scale, ReviewRules(), ref)
    assert status == "ok" and not warnings
    # bottom side tilted by ~0.5 degrees
    tilted = [[0, 0], [w, 0], [w, h + 0.009 * w], [0, h]]
    status, warnings, _ = evaluate({"detection": {"quad": tilted, "edges": {}}}, scale, ReviewRules(), ref)
    assert status == "needs_review" and any(x.startswith("not_parallel:top-bottom") for x in warnings)
    # 0.5 mm too tall, with a parallel inner line 64 px inside the bottom edge
    tall = [[0, 0], [w, 0], [w, h + 64], [0, h + 64]]
    edges = {"bottom": {"n_points": 16, "n_inliers": 16, "mad_px": 1.0, "bow_px": 0.0, "max_px": 2.0,
                        "inner_px": 62.0, "inner_share": 0.5}}
    status, warnings, _ = evaluate({"detection": {"quad": tall, "edges": edges}}, scale, ReviewRules(), ref)
    assert status == "needs_review"
    assert "size_outlier:height:+0.50mm" in warnings
    assert any(x.startswith("hint:fix --shift bottom=-62") for x in warnings)


def test_review_rule_edge_inside_margin():
    from polaroid_processor.pipeline import ReviewRules, evaluate
    from polaroid_processor.scale import Scale
    ppm = 128.0
    w, h = 88 * ppm, 107 * ppm
    quad = [[0, 0], [w, 0], [w, h], [0, h]]
    base = {"n_points": 20, "n_inliers": 20, "mad_px": 1.0, "bow_px": 0.0, "max_px": 2.0}
    for inside, flagged in ((20.0, False), (40.0, True)):  # 0.16 mm and 0.31 mm against a 0.2 mm margin
        edges = {"left": {**base, "inside_px": inside}}
        status, warnings, _ = evaluate({"detection": {"quad": quad, "edges": edges}}, Scale(ppm, "ppi"),
                                       ReviewRules(), (w, h), margin_mm=0.2)
        assert (status == "needs_review") is flagged
        assert any(x.startswith("edge_inside:left:") for x in warnings) is flagged
