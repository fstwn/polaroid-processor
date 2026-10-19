"""Regression against hand-reviewed corners of real scans.

- tests/data/labels.json: scans in input/; the detector was tuned on these. Acceptance target
  from docs/SCOPE.md is every corner within 0.2 mm; the detector meets 0.1 mm since the edge
  bias fix (2026-10-18).
- tests/data/labels_holdout.json: scans from the user's library ($POLAROID_LIBRARY) that were
  never used for tuning; checked against the 0.2 mm target. Never tune parameters on them.

Each test is skipped when its scans are not present.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from polaroid_processor import detect, load_rgb
from polaroid_processor.formats import POLAROID_600

ROOT = Path(__file__).resolve().parents[1]
LABELS = Path(__file__).parent / "data" / "labels.json"
HOLDOUT = Path(__file__).parent / "data" / "labels_holdout.json"
LIBRARY = Path(os.environ.get("POLAROID_LIBRARY", "D:/11_pictures/Pictures"))
TOL_MM = 0.1
HOLDOUT_TOL_MM = 0.2
# input/ folders whose scans also live in the library with identical pixels (checked 2026-10-18)
LIBRARY_COPIES = {"SICK_D3": "05_roller_roids/250729_SICKalps_L2_D3"}


def _scan_path(rel: str) -> Path:
    path = ROOT / "input" / rel
    folder, _, name = rel.partition("/")
    if not path.exists() and folder in LIBRARY_COPIES:
        path = LIBRARY / LIBRARY_COPIES[folder] / name
    return path


def _labels(path: Path = LABELS):
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return sorted(data.get("scans", {}).items())


def _corner_errors_mm(path: Path, label: dict) -> np.ndarray:
    truth = np.array(label["quad"], dtype=float)
    px_per_mm = label.get("px_per_mm") or np.linalg.norm(truth[1] - truth[0]) / POLAROID_600.width_mm
    det = detect(load_rgb(str(path)))
    return np.linalg.norm(det.quad - truth, axis=1) / px_per_mm


@pytest.mark.scans
@pytest.mark.parametrize("rel,label", _labels() or [pytest.param(None, None, marks=pytest.mark.skip("no labels"))])
def test_corners_match_labels(rel, label, request):
    path = _scan_path(rel)
    if not path.exists():
        pytest.skip(f"{rel} not present")
    if label.get("known_failure"):
        request.applymarker(pytest.mark.xfail(reason=f"known detector failure: {label['known_failure']}"))
    err_mm = _corner_errors_mm(path, label)
    assert err_mm.max() <= TOL_MM, f"{rel}: corner errors (mm) {np.round(err_mm, 3).tolist()}"


NO_LABELS = [pytest.param(None, None, marks=pytest.mark.skip("no labels"))]


@pytest.mark.scans
@pytest.mark.parametrize("rel,label", _labels(HOLDOUT) or NO_LABELS)
def test_corners_match_holdout_labels(rel, label, request):
    path = LIBRARY / rel
    if not path.exists():
        pytest.skip(f"{rel} not present (set POLAROID_LIBRARY)")
    if label.get("known_failure"):
        request.applymarker(pytest.mark.xfail(reason=f"known detector failure: {label['known_failure']}"))
    err_mm = _corner_errors_mm(path, label)
    assert err_mm.max() <= HOLDOUT_TOL_MM, f"{rel}: corner errors (mm) {np.round(err_mm, 3).tolist()}"
