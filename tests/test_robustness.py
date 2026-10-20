"""Crash safety: bright objects on the holder, worker processes that die."""
import os

import cv2
import numpy as np

from polaroid_processor import detect
from polaroid_processor.pipeline import _run_pool

from synth import PX_PER_MM, make_scan


def test_white_circle_on_the_holder_keeps_the_threshold_in_range():
    # Go holders have a white circle beside the print; it pushed the holder noise percentile
    # to ~251 and the second-pass speck threshold past 255 (numpy 2.0 then crashed)
    rgb, truth = make_scan("white", 0.6, seed=4, scan_px=(3000, 3600))
    cv2.circle(rgb, (2850, 3450), 260, (250, 250, 250), -1, cv2.LINE_AA)
    det = detect(rgb)
    assert det.speck_threshold <= 70
    assert np.abs(det.quad - truth).max() / PX_PER_MM < 0.1


def _job(job):
    if job["key"] == "bad":
        os._exit(3)  # a native crash takes the worker process down
    return {"key": job["key"], "path": job["path"], "value": job["key"] * 2}


def test_a_crashing_scan_fails_alone():
    jobs = [{"key": k, "path": f"{k}.tif"} for k in ("a", "bad", "b", "c")]
    res = {r["key"]: r for r in _run_pool(_job, jobs, workers=2)}
    assert set(res) == {"a", "bad", "b", "c"}
    assert "crashed" in res["bad"]["error"]
    assert res["c"]["value"] == "cc" and "error" not in res["a"]


# --------------------------------------------------------------------------- rim refit
def _entry(quad, inner_bottom=None):
    edges = {s: {"n_points": 10, "n_inliers": 10, "mad_px": 0.5, "bow_px": 0.0, "max_px": 1.0} for s in
             ("top", "right", "bottom", "left")}
    edges["bottom"]["inner_px"] = inner_bottom
    return {"detection": {"quad": quad, "edges": edges, "background": "dark"}}


def test_rim_refit_moves_the_side_whose_inner_line_explains_the_oversize():
    from polaroid_processor.pipeline import ReviewRules, Settings, apply_refit, size_refit
    from polaroid_processor.scale import Scale
    px = 100.0
    ref = (88.0 * px, 107.0 * px)
    tall = [[0, 0], [8800, 0], [8800, 10754], [0, 10754]]  # 0.54 mm too tall
    rules, scale = ReviewRules(), Scale(px, "ppi")

    def no_scan():
        raise AssertionError("the inner line suffices; the scan must not be loaded")

    e = _entry(tall, inner_bottom=54.0)
    quad, info = size_refit(e, scale, ref, rules, no_scan)
    assert info["side"] == "bottom" and np.allclose(quad[2:, 1], 10700)
    apply_refit(e, quad, info, None, px, Settings(flatten="off"))
    assert e["detection"]["quad_detected"] == tall and size_refit(e, scale, ref, rules, no_scan) is None
    # too small is never "fixed" by moving inward
    small = [[0, 0], [8800, 0], [8800, 10646], [0, 10646]]
    assert size_refit(_entry(small, inner_bottom=54.0), scale, ref, rules, no_scan) is None


def test_rim_refit_without_evidence_leaves_the_outline():
    from polaroid_processor.pipeline import ReviewRules, size_refit
    from polaroid_processor.scale import Scale
    px = 10.0
    tall = [[20, 20], [900, 20], [900, 1095.4], [20, 1095.4]]
    blank = np.zeros((1200, 1000, 3), np.uint8)
    blank[20:1095, 20:900] = 230  # a plain print on the holder: no inner line, no seam
    assert size_refit(_entry(tall, inner_bottom=30.0), Scale(px, "ppi"), (880.0, 1070.0), ReviewRules(),
                      lambda: blank) is None
