"""Light background (camera shot of a print on paper): synthetic shots with known corners."""
import cv2
import numpy as np
import pytest

from polaroid_processor import detect
from polaroid_processor.light import is_light_background

from synth import make_scan

PX = 12.0  # px per mm
PRINT = (88.0, 107.0)
WINDOW = (79.0, 79.0, 4.5, 6.0)  # width, height, left and top frame


def paper_shot(angle_deg: float = 1.2, seed: int = 0, shadow: bool = True):
    """Return (rgb, true quad): grainy paper, smooth white frame, busy photo window, a soft
    shadow line along two print edges."""
    rng = np.random.default_rng(seed)
    w, h = int(110 * PX), int(130 * PX)
    ss = 4  # supersampled print mask for anti-aliased edges
    grain = cv2.GaussianBlur(rng.normal(0, 4.0, (h, w)).astype(np.float32), (0, 0), 0.8)
    img = np.full((h, w), 232.0, np.float32) + grain
    a = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    centre = np.array([w / 2 + 3.3, h / 2 - 2.1])

    def place(x0, y0, x1, y1):
        local = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) * PX - np.array(PRINT) * PX / 2
        return local @ rot.T + centre

    quad = place(0, 0, *PRINT)
    win = place(WINDOW[2], WINDOW[3], WINDOW[2] + WINDOW[0], WINDOW[3] + WINDOW[1])

    def mask(q):
        m = np.zeros((h * ss, w * ss), np.uint8)
        cv2.fillPoly(m, [np.round(q * ss - 0.5).astype(np.int32).reshape(-1, 1, 2)], 255)
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255

    if shadow:  # shadow line on the paper below and right of the print, darker than the frame
        sh = cv2.GaussianBlur(mask(quad + [4.0, 4.0]), (0, 0), 1.5)
        img -= 22 * sh * (1 - mask(quad))
    pm = mask(quad)
    frame = 223.0 + rng.normal(0, 0.4, (h, w)).astype(np.float32)
    img = img * (1 - pm) + frame * pm
    photo = cv2.GaussianBlur(rng.uniform(30, 200, (h // 8, w // 8)).astype(np.float32), (0, 0), 1.0)
    photo = cv2.resize(photo, (w, h), interpolation=cv2.INTER_NEAREST)
    wm = mask(win)
    img = img * (1 - wm) + photo * wm
    rgb = np.clip(np.repeat(img[..., None], 3, axis=2), 0, 255).astype(np.uint8)
    return rgb, quad


@pytest.mark.parametrize("angle,seed,shadow", [(1.2, 0, True), (-0.7, 1, False), (3.0, 2, True)])
def test_print_on_paper_is_found(angle, seed, shadow):
    rgb, truth = paper_shot(angle, seed, shadow)
    assert is_light_background(rgb)
    det = detect(rgb)
    assert det.background == "light"
    err_mm = np.linalg.norm(det.quad - truth, axis=1).max() / PX
    assert err_mm < 0.15, (det.quad, truth)


def test_holder_scans_stay_on_the_dark_path():
    rgb, _ = make_scan("white", 0.4, seed=1)
    assert not is_light_background(rgb)
    assert detect(rgb).background == "dark"
