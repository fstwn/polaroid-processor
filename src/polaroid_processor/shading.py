"""Edge shadows: measure the darkening towards each print edge and divide it out.

Near its edges a print darkens because the edge lifts off the scanner glass and casts a soft
shadow; on white frames this is 3-18 % at 0.25 mm and reaches about 1.5 mm in. On plain
frames the border is otherwise uniform, so the shading can be measured from the border
itself: brightness at depth d from the edge divided by brightness at a reference depth,
per window along the side.

Only windows with the shape of a shadow are measured: plain frame (the same in every quarter
of the window), no steep brightness drop (a graphic edge), flat from 1.5 mm inward. A side is
corrected only when at least half of it could be measured; gaps between measured windows
take their neighbours' values.
Printed frames, dark frames and frames with sheen or ink gradients are left untouched.
The image must have straight, axis-aligned edges (the output of the rectify or unwarp step).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

SIDES = ("top", "right", "bottom", "left")


@dataclass(frozen=True)
class ShadingParams:
    depth_mm: float = 2.5  # corrected band from the edge inward
    ref_mm: Tuple[float, float] = (2.5, 3.5)  # reference depth (unshaded frame)
    step_mm: float = 1.0  # window spacing along a side
    window_mm: float = 4.0  # window length along a side
    uniform_tol: float = 0.04  # max spread of quarter medians (plain frame test)
    jump: float = 0.03  # max drop within 0.15 mm (graphic edge test)
    flat_tol: float = 0.02  # profile from 1.5 mm inward must stay within this of 1
    min_gain: float = 0.7  # never brighten by more than 1 / min_gain
    min_level: float = 30.0  # reference brightness (8 bit) below which nothing is measured
    min_effect: float = 0.01  # sides whose strongest darkening is below this are left alone
    min_share: float = 0.5  # share of a side that must be measurable; otherwise the side is left alone


def _band(img: np.ndarray, side: str, depth: int) -> np.ndarray:
    """View of the band along a side as (depth, length, ...), row 0 at the edge."""
    if side == "top":
        return img[:depth]
    if side == "bottom":
        return img[::-1][:depth]
    if side == "left":
        return np.swapaxes(img[:, :depth], 0, 1)
    return np.swapaxes(img[:, ::-1][:, :depth], 0, 1)


def side_gains(img: np.ndarray, px_per_mm: float, p: ShadingParams = ShadingParams()):
    """Per side: (gain[depth, length] or None, share of the side measured)."""
    depth = int(p.depth_mm * px_per_mm)
    r0, r1 = int(p.ref_mm[0] * px_per_mm), int(p.ref_mm[1] * px_per_mm)
    step, win = max(1, int(p.step_mm * px_per_mm)), max(8, int(p.window_mm * px_per_mm))
    scale = 257.0 if img.dtype == np.uint16 else 1.0
    span = max(2, int(0.15 * px_per_mm))
    start = int(0.3 * px_per_mm)
    flat0 = int(1.5 * px_per_mm)
    out = {}
    for side in SIDES:
        band = _band(img, side, r1).astype(np.float32)
        lum = (band.mean(axis=2) if band.ndim == 3 else band) / scale
        length = lum.shape[1]
        centres = np.arange(win // 2, length - win // 2, step)
        if not centres.size:
            out[side] = (None, 0.0)
            continue
        cols = np.full((depth, len(centres)), np.nan, np.float32)
        for k, c in enumerate(centres):
            seg = lum[:, c - win // 2: c + win // 2]
            ref_band = seg[r0:r1]
            ref = float(np.median(ref_band))
            if ref < p.min_level:
                continue
            q = [np.median(x) for x in np.array_split(ref_band, 4, axis=1)]
            if (max(q) - min(q)) / ref > p.uniform_tol:
                continue
            q = [np.median(x) for x in np.array_split(seg[depth // 3: depth], 4, axis=1)]
            if (max(q) - min(q)) / ref > p.uniform_tol:
                continue
            raw = np.median(seg, axis=1) / ref
            sm = np.convolve(np.pad(raw, 4, mode="edge"), np.ones(9) / 9, mode="valid")
            inner = sm[start:]
            if inner.size > span and (inner[:-span] - inner[span:]).max() > p.jump:
                continue
            if np.abs(sm[flat0:] - 1.0).max() > p.flat_tol:
                continue
            # gain at depth d = darkest level at or inside d: monotonic, and a highlight right at
            # the edge cannot hold the gain at 1
            prof = np.minimum.accumulate(sm[::-1])[::-1][:depth]
            cols[:, k] = np.clip(prof, p.min_gain, 1.0)
        measured = ~np.isnan(cols[0])
        share = float(measured.mean())
        if measured.sum() < 3 or share < p.min_share or (1.0 - np.nanmin(cols)) < p.min_effect:
            out[side] = (None, share)
            continue
        idx = np.flatnonzero(measured)
        # sections without a measurement take their neighbours' values, then a light smoothing
        filled = np.array([np.interp(np.arange(len(centres)), idx, row[idx]) for row in cols], np.float32)
        k = np.ones(5, np.float32) / 5
        filled = np.array([np.convolve(np.pad(r, 2, mode="edge"), k, mode="valid") for r in filled], np.float32)
        x = np.arange(length)
        gain = np.array([np.interp(x, centres, r) for r in filled], np.float32)  # (depth, length)
        out[side] = (gain, share)
    return out


def correct(img: np.ndarray, px_per_mm: float, p: ShadingParams = ShadingParams()) -> Tuple[np.ndarray, Dict]:
    """Return (corrected image, info). Corners where two bands overlap get both corrections."""
    gains = side_gains(img, px_per_mm, p)
    out = img.copy()
    hi = 65535 if img.dtype == np.uint16 else 255
    info = {"sides": [], "max_darkening": {}, "measured": {}}
    for side, (gain, share) in gains.items():
        info["measured"][side] = round(share, 2)
        if gain is None:
            continue
        band = _band(out, side, gain.shape[0])  # a view into `out`
        g = gain[..., None] if band.ndim == 3 else gain
        band[...] = np.clip(band.astype(np.float32) / g + 0.5, 0, hi).astype(img.dtype)
        info["sides"].append(side)
        info["max_darkening"][side] = round(float(1.0 - gain.min()), 3)
    return out, info
