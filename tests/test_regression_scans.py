"""Regression against hand-reviewed corners of real scans (tests/data/labels.json).

Skipped when the scans in input/ are not present. Acceptance target from docs/SCOPE.md is
every corner within 0.2 mm; the detector meets 0.1 mm since the edge bias fix (2026-10-18).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from polaroid_processor import detect, load_rgb
from polaroid_processor.formats import POLAROID_600

ROOT = Path(__file__).resolve().parents[1]
LABELS = Path(__file__).parent / "data" / "labels.json"
TOL_MM = 0.1


def _labels():
    if not LABELS.exists():
        return []
    data = json.loads(LABELS.read_text(encoding="utf-8"))
    return sorted(data.get("scans", {}).items())


@pytest.mark.scans
@pytest.mark.parametrize("rel,label", _labels() or [pytest.param(None, None, marks=pytest.mark.skip("no labels"))])
def test_corners_match_labels(rel, label, request):
    path = ROOT / "input" / rel
    if not path.exists():
        pytest.skip(f"{rel} not present")
    if label.get("known_failure"):
        request.applymarker(pytest.mark.xfail(reason=f"known detector failure: {label['known_failure']}"))
    truth = np.array(label["quad"], dtype=float)
    px_per_mm = np.linalg.norm(truth[1] - truth[0]) / POLAROID_600.width_mm
    det = detect(load_rgb(str(path)))
    err_mm = np.linalg.norm(det.quad - truth, axis=1) / px_per_mm
    assert err_mm.max() <= TOL_MM, f"{rel}: corner errors (mm) {np.round(err_mm, 3).tolist()}"
