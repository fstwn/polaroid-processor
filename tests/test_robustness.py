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
