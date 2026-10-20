"""Rectification: one perspective warp from the detected quad to an upright rectangle."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .detection import quad_size

INTERPOLATION = {
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order four points as tl, tr, br, bl."""
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


SIDE_CORNERS = {"top": (0, 1), "right": (1, 2), "bottom": (2, 3), "left": (3, 0)}  # tl, tr, br, bl


def shift_side(quad: np.ndarray, side: str, d: float) -> np.ndarray:
    """Move one side of a quad (tl, tr, br, bl) along its normal by ``d`` px, outward positive."""
    q = np.asarray(quad, dtype=np.float64).copy()
    a, b = SIDE_CORNERS[side]
    t = q[b] - q[a]
    n = np.array([-t[1], t[0]]) / np.linalg.norm(t)
    if np.dot((q[a] + q[b]) / 2 - q.mean(axis=0), n) < 0:
        n = -n
    q[a] += d * n
    q[b] += d * n
    return q


def output_size(quad: np.ndarray, margin_px: float = 0.0) -> Tuple[int, int]:
    w, h = quad_size(quad)
    return max(1, int(round(w - 2 * margin_px))), max(1, int(round(h - 2 * margin_px)))


def homography(quad: np.ndarray, margin_px: float = 0.0) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Homography mapping the scan to the upright print, cropped by ``margin_px`` per side.

    Quad corners are continuous coordinates (pixel i spans [i, i+1)); OpenCV samples at
    pixel centres, hence the half-pixel shifts.
    """
    quad = order_quad(quad)
    w, h = quad_size(quad)
    out_w, out_h = output_size(quad, margin_px)
    m = margin_px
    dst = np.array([[-m, -m], [w - m, -m], [w - m, h - m], [-m, h - m]], dtype=np.float64)
    src = quad - 0.5
    dst = dst - 0.5
    H = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    return H, (out_w, out_h)


def rectify(rgb: np.ndarray, quad: np.ndarray, margin_px: float = 0.0, interpolation: str = "cubic") -> np.ndarray:
    H, size = homography(quad, margin_px)
    return cv2.warpPerspective(rgb, H, size, flags=INTERPOLATION[interpolation], borderMode=cv2.BORDER_REPLICATE)


def map_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)
