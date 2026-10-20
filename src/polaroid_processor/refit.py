"""Size-proven refits of a holder-scan outline that sits on the holder rim.

When the holder rim runs along one side of the print, the detector can take the rim's outer
edge for the print edge. The outline is then too large along one axis, which only the batch
shows (the median print of the same scan size). Two sources can name the real edge:

- an inner line the detector already found behind that side (``inner_px``): the side moves
  parallel onto it;
- a dark seam behind the rim band (the print's edge shadow between rim and print): looked up
  at full resolution only for such scans, and the side is refit on the rise into the print.

A refit is applied only when exactly one side explains the oversize and the corrected size
matches the batch within ``tol_px``; everything else stays flagged for review.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .detection import quad_size
from .flatten import _side_frames, _side_strip, max_rgb_u8
from .geometry import shift_side

AXIS_SIDES = {"width": ("left", "right"), "height": ("top", "bottom")}
_SIDE_INDEX = {"top": (0, 1), "right": (1, 2), "bottom": (3, 2), "left": (0, 3)}


def oversize(quad: np.ndarray, ref_size: Tuple[float, float], min_px: float) -> Optional[Tuple[str, float]]:
    """(axis, excess px) when the outline is larger than the reference along exactly one axis
    and not smaller along the other."""
    size = quad_size(quad)
    if not all(abs(v / r - 1) < 0.03 for v, r in zip(size, ref_size)):
        return None
    devs = {axis: v - r for axis, v, r in zip(("width", "height"), size, ref_size)}
    big = [a for a, d in devs.items() if d > min_px]
    if len(big) != 1 or any(d < -min_px for d in devs.values()):
        return None
    return big[0], devs[big[0]]


def from_inner_line(edges: Dict[str, dict], quad: np.ndarray, axis: str, excess: float,
                    tol_px: float) -> Optional[Tuple[str, np.ndarray, dict]]:
    found = []
    for side in AXIS_SIDES[axis]:
        depth = (edges.get(side) or {}).get("inner_px")
        if depth and abs(depth - excess) <= tol_px:
            found.append((side, float(depth)))
    if len(found) != 1:
        return None
    side, depth = found[0]
    return side, shift_side(quad, side, -depth), {"side": side, "shift_px": round(-depth, 1), "source": "inner line"}


def _seam_points(mx: np.ndarray, quad: np.ndarray, side: str, depth_px: int, px_per_mm: float,
                 seg_mm: float = 5.0) -> List[np.ndarray]:
    """Points where the print rises out of a dark seam, searched up to ``depth_px`` inside the side."""
    p0, t, n, length = _side_frames(quad)[side]
    strip = _side_strip(mx, p0, t, n, length, 2, depth_px).astype(np.float32)  # rows: outward 2 .. inward depth
    d = np.arange(2, -depth_px, -1, dtype=np.float64)
    seg = int(seg_mm * px_per_mm)
    plateau = int(0.4 * px_per_mm)
    pts = []
    for s0 in range(int(0.1 * length), int(0.9 * length) - seg, seg // 2):
        prof = np.median(strip[:, s0:s0 + seg], axis=1)
        top = float(np.median(prof[-max(3, plateau // 8):]))  # print level at the inner end
        rise = np.flatnonzero(prof >= 0.75 * top)
        if not rise.size or rise[0] < 2:
            continue
        r = int(rise[0])  # where the print starts, coming from the side line
        k0 = max(0, r - plateau)
        k = k0 + int(np.argmin(prof[k0:r]))  # the seam just in front of it
        seam = float(prof[k])
        band = float(prof[:k].max()) if k > 0 else 0.0
        inner = float(np.median(prof[k + plateau // 2: k + plateau]))
        if band < 2 * seam + 10 or inner < 3 * seam + 40:
            continue  # no rim band in front of a dark seam, or no print behind it
        half = (seam + inner) / 2
        after = np.flatnonzero(prof[k:] >= half)
        if not after.size:
            continue
        j = k + int(after[0])
        frac = (half - prof[j - 1]) / max(1e-6, prof[j] - prof[j - 1])
        dd = d[j - 1] - frac  # rows go inward: d decreases
        pts.append(p0 + (s0 + seg / 2) * t + dd * n)
    return pts


def from_seam(rgb: np.ndarray, quad: np.ndarray, axis: str, excess: float, px_per_mm: float,
              tol_px: float) -> Optional[Tuple[str, np.ndarray, dict]]:
    mx = max_rgb_u8(rgb)
    found = []
    for side in AXIS_SIDES[axis]:
        pts = _seam_points(mx, quad, side, int(excess + 0.5 * px_per_mm), px_per_mm)
        p0, t, n, length = _side_frames(quad)[side]
        if len(pts) < 0.6 * (0.8 * length / (2.5 * px_per_mm)):
            continue
        pts = np.array(pts)
        c = pts.mean(0)
        dvec = np.linalg.svd(pts - c)[2][0]
        res = (pts - c) @ np.array([-dvec[1], dvec[0]])
        if np.median(np.abs(res)) > 0.05 * px_per_mm:
            continue
        found.append((side, c, dvec))
    if len(found) != 1:
        return None
    side, c, dvec = found[0]
    # the new side line, intersected with the two neighbouring sides
    q = np.asarray(quad, dtype=np.float64).copy()
    a, b = _SIDE_INDEX[side]
    others = {"top": ((0, 3), (1, 2)), "bottom": ((0, 3), (1, 2)), "left": ((0, 1), (3, 2)), "right": ((0, 1), (3, 2))}[side]
    for idx, (u, v) in zip((a, b), others):
        e = q[v] - q[u]
        s = np.linalg.solve(np.array([dvec, -e]).T, q[u] - c)
        q[idx] = c + s[0] * dvec
    shift = float(np.median((np.array([q[a], q[b]]) - np.asarray(quad)[[a, b]]) @ _side_frames(quad)[side][2]))
    if abs(-shift - excess) > tol_px:
        return None
    return side, q, {"side": side, "shift_px": round(shift, 1), "source": "seam"}

