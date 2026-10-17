"""Synthetic holder scans with known print corners, at 25.6 px/mm (1/5 of the real scans)."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

PX_PER_MM = 25.6
PRINT_MM = (88.0, 107.0)


def make_scan(frame: str = "white", angle_deg: float = 0.4, seed: int = 0,
              scan_px: Tuple[int, int] = (2400, 2900), rim: bool = True,
              rim_band_deg: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rgb uint8 scan, true quad tl/tr/br/bl in continuous pixel coordinates).

    ``rim_band_deg`` adds a grey holder rim band at the bottom edge, tilted by that angle against
    the print. The print lies on top of it: the band is hidden at the left and emerges towards the
    right, separated from the print only by the thin shadow line of the print edge.
    """
    rng = np.random.default_rng(seed)
    w, h = scan_px
    img = rng.integers(0, 4, size=(h, w, 3), dtype=np.uint8)  # holder noise
    pw, ph = PRINT_MM[0] * PX_PER_MM, PRINT_MM[1] * PX_PER_MM
    cx, cy = w / 2 + 7.3, h / 2 - 4.1
    a = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    local = np.array([[-pw / 2, -ph / 2], [pw / 2, -ph / 2], [pw / 2, ph / 2], [-pw / 2, ph / 2]])
    quad = local @ rot.T + [cx, cy]

    if rim:  # holder cut-out lip: thin bright rounded line ~3 mm outside the print
        off = 3.0 * PX_PER_MM
        rl = np.array([[-pw / 2 - off, -ph / 2 - off], [pw / 2 + off, -ph / 2 - off],
                       [pw / 2 + off, ph / 2 + off], [-pw / 2 - off, ph / 2 + off]]) @ rot.T + [cx, cy]
        cv2.polylines(img, [np.round(rl * 16).astype(np.int32).reshape(-1, 1, 2)], True, (70, 70, 70), 5,
                      cv2.LINE_AA, shift=4)
        dust = rng.random((h, w)) < 0.004
        img[dust] = rng.integers(15, 60, size=(int(dust.sum()), 1), dtype=np.uint8)

    if rim_band_deg is not None:
        b = np.deg2rad(rim_band_deg)
        brot = np.array([[np.cos(b), -np.sin(b)], [np.sin(b), np.cos(b)]])
        width = 0.4 * PX_PER_MM
        band = np.array([[-pw / 2, ph / 2], [pw / 2, ph / 2], [pw / 2, ph / 2 + width], [-pw / 2, ph / 2 + width]])
        pivot = np.array([pw / 2, ph / 2 + width])
        band = ((band - pivot) @ brot.T + pivot) @ rot.T + [cx, cy]
        cv2.fillPoly(img, [np.round(band * 16).astype(np.int32).reshape(-1, 1, 2)], (68, 66, 64),
                     cv2.LINE_AA, shift=4)

    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.round((quad - 0.5) * 16).astype(np.int32).reshape(-1, 1, 2)], 255, cv2.LINE_AA, shift=4)
    if frame == "white":
        layer = np.full((h, w, 3), (232, 234, 228), np.uint8)
    elif frame == "gold":
        layer = np.full((h, w, 3), (120, 104, 52), np.uint8)
    elif frame == "glitter":
        layer = rng.integers(0, 6, size=(h, w, 3), dtype=np.uint8)
        sp = rng.random((h, w)) < 0.06
        layer[sp] = rng.integers(20, 120, size=(int(sp.sum()), 3), dtype=np.uint8)
    else:
        raise ValueError(frame)
    # photo window
    win = np.zeros((h, w), np.uint8)
    wl = np.array([[-39, -45], [39, -45], [39, 34], [-39, 34]]) * PX_PER_MM @ rot.T + [cx, cy]
    cv2.fillPoly(win, [np.round(wl).astype(np.int32).reshape(-1, 1, 2)], 255)
    photo = rng.integers(0, 255, size=(h // 32 + 1, w // 32 + 1, 3), dtype=np.uint8)
    photo = cv2.resize(photo, (w, h), interpolation=cv2.INTER_CUBIC)
    layer = np.where(win[..., None] > 0, photo, layer)
    alpha = (mask.astype(np.float32) / 255)[..., None]
    img = (img * (1 - alpha) + layer * alpha).astype(np.uint8)
    if rim_band_deg is not None:  # shadow line of the print edge: about one pixel, partly lit by dust
        nrm = np.array([-np.sin(a), np.cos(a)])
        line = (quad[[3, 2]] + 1.0 * nrm - 0.5) * 16
        shadow = np.zeros((h, w), np.uint8)
        cv2.line(shadow, tuple(int(v) for v in np.round(line[0])), tuple(int(v) for v in np.round(line[1])),
                 255, 1, cv2.LINE_8, shift=4)
        dark = (shadow > 0) & (rng.random((h, w)) < 0.65)
        img[dark] = 4
    return img, quad
