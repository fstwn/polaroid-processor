import numpy as np
import pytest

from polaroid_processor import detect
from polaroid_processor.detection import DetectionError, DetectParams
from polaroid_processor.geometry import rectify

from synth import PX_PER_MM, make_scan

TOL_PX = 0.2 * PX_PER_MM  # acceptance target: 0.2 mm per corner


@pytest.mark.parametrize("frame", ["white", "gold", "glitter"])
@pytest.mark.parametrize("angle", [0.0, 0.6, -1.5])
def test_corners_within_tolerance(frame, angle):
    rgb, truth = make_scan(frame, angle, seed=hash((frame, angle)) % 1000)
    det = detect(rgb)
    err = np.linalg.norm(det.quad - truth, axis=1)
    assert err.max() <= TOL_PX, f"{frame} {angle}: corner errors {np.round(err, 1)}"
    assert abs(det.skew_deg - angle) < 0.15


def test_rim_line_is_not_taken_as_edge():
    rgb, truth = make_scan("glitter", 0.3, seed=7, rim=True)
    det = detect(rgb)
    w_true = np.linalg.norm(truth[1] - truth[0])
    assert abs(det.size_px[0] - w_true) < TOL_PX * 2


def test_empty_holder_fails_cleanly():
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 4, size=(2900, 2400, 3), dtype=np.uint8)
    with pytest.raises(DetectionError):
        detect(rgb)


def test_rectify_size_and_orientation():
    rgb, truth = make_scan("white", 1.0, seed=3)
    det = detect(rgb)
    out = rectify(rgb, det.quad, margin_px=0)
    w_true = np.linalg.norm(truth[1] - truth[0])
    h_true = np.linalg.norm(truth[3] - truth[0])
    assert abs(out.shape[1] - w_true) <= TOL_PX * 2
    assert abs(out.shape[0] - h_true) <= TOL_PX * 2
    # frame pixels everywhere along the border, no holder
    border = np.concatenate([out[2, :], out[-3, :], out[:, 2], out[:, -3]])
    assert border.mean() > 200


def test_margin_crops_inside():
    rgb, _ = make_scan("white", 0.0, seed=4)
    det = detect(rgb)
    a = rectify(rgb, det.quad, 0)
    b = rectify(rgb, det.quad, 10)
    assert a.shape[0] - b.shape[0] == 20 and a.shape[1] - b.shape[1] == 20


@pytest.mark.parametrize("frame", ["white", "gold"])
def test_rim_band_touching_the_print_is_repaired(frame):
    """The print lies on a holder rim band that emerges at a slight angle under the bottom edge.
    The outer candidates follow the band; the bottom side is refit on the inner candidates."""
    rgb, truth = make_scan(frame, 0.2, seed=21, rim=False, rim_band_deg=0.4)
    det = detect(rgb)
    err = np.linalg.norm(det.quad - truth, axis=1).max() / PX_PER_MM
    assert det.repaired == ("bottom",)
    assert err <= 0.2, err
    # without the repair the bottom follows the band
    plain = detect(rgb, DetectParams(parallel_tol_deg=90))
    assert np.linalg.norm(plain.quad - truth, axis=1).max() / PX_PER_MM > 0.3
