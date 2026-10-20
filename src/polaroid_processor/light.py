"""Print outline on a light background (a camera shot of a print on white paper).

The dark-holder detector relies on a black surround; here the surround is paper as bright as
the frame. Two things stay reliable:

- the photo window: its border is the strongest line in a texture map (|luminance - blur|),
  far stronger than the print edge, so it is found first, per side as the strongest ridge;
- the frame: walking outward from the window line, the frame is a flat luminance plateau; the
  print edge is where the profile leaves it (paper grain, the shadow line or the light rim
  around the print). Measured outside the corners, where pencil marks or photo corners sit.

The plateau exit lands on the print edge: on 42 camera shots of 600 prints the outlines
measure 0.820-0.824 (short / long side), like the holder scans (0.819-0.824). Plain white
frames only; printed frames do not give a plateau.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

SIDES = ("top", "right", "bottom", "left")
_SIDE_CORNERS = {"top": (0, 1), "right": (1, 2), "bottom": (3, 2), "left": (0, 3)}


@dataclass(frozen=True)
class LightParams:
    border_share: float = 0.02  # outer band of the image that decides "light background"
    border_level: float = 128.0  # median max(R, G, B) of that band above this: light background
    sigma: float = 1.5  # texture: |luminance - Gaussian blur(sigma)|
    factor: int = 4  # analysis scale for the photo window
    band: int = 40  # window search: analysis pixels per profile along a side
    skip: float = 0.15  # window search: share of each side ignored at the ends
    ridge: float = 4.0  # the window line must be this many times the median texture
    window_mm: float = 79.0  # photo window of a 600 print; only sets the mm scale of the search
    seg_mm: float = 4.0  # edge search: segment length along a side
    start_mm: float = 0.7  # edge search starts this far outside the window line
    plateau_mm: float = 1.5  # frame level measured over this stretch
    gap_mm: float = 0.5  # ... ending this far behind the position tested
    dark_line_mm: float = 0.6  # a dark line at the edge is searched this far out
    dark_line_depth: float = 0.07  # ... and belongs to the print when this much (relative) below the frame
    max_mm: float = 30.0  # edge search ends this far outside the window line
    end_skip: float = 0.1  # edge search: share of each side ignored at the ends
    min_points: int = 6


class LightDetectionError(ValueError):
    pass


def is_light_background(rgb: np.ndarray, p: LightParams = LightParams()) -> bool:
    """True when the outer band of the image is bright (paper), not a dark holder."""
    h, w = rgb.shape[:2]
    b = max(2, int(p.border_share * min(h, w)))
    step = max(1, min(h, w) // 600)
    band = np.concatenate([rgb[:b:step, ::step].reshape(-1, rgb.shape[2]), rgb[-b::step, ::step].reshape(-1, rgb.shape[2]),
                           rgb[::step, :b:step].reshape(-1, rgb.shape[2]), rgb[::step, -b::step].reshape(-1, rgb.shape[2])])
    mx = band.max(axis=1).astype(np.float32)
    if rgb.dtype == np.uint16:
        mx = mx / 257.0
    return float(np.median(mx)) > p.border_level


def _luminance(rgb: np.ndarray) -> np.ndarray:
    lum = rgb.astype(np.float32).mean(axis=2)
    return lum / 257.0 if rgb.dtype == np.uint16 else lum


def _fit_line(pts: np.ndarray, iters: int = 4):
    """Robust total-least-squares line: (centre, direction, inlier mask, residuals)."""
    keep = np.ones(len(pts), bool)
    res = np.zeros(len(pts))
    c = d = None
    for _ in range(iters):
        if keep.sum() < 3:
            return None
        c = pts[keep].mean(0)
        d = np.linalg.svd(pts[keep] - c)[2][0]
        res = (pts - c) @ np.array([-d[1], d[0]])
        mad = np.median(np.abs(res[keep])) * 1.4826
        keep = np.abs(res) <= max(1.5, 3 * mad)
    return c, d, keep, res


def _intersect(l1, l2) -> np.ndarray:
    (c1, d1), (c2, d2) = l1, l2
    s = np.linalg.solve(np.array([d1, -d2]).T, c2 - c1)
    return c1 + s[0] * d1


def _quad(lines: Dict[str, tuple]) -> np.ndarray:
    return np.array([_intersect(lines["top"], lines["left"]), _intersect(lines["top"], lines["right"]),
                     _intersect(lines["bottom"], lines["right"]), _intersect(lines["bottom"], lines["left"])])


def photo_window(tex: np.ndarray, p: LightParams = LightParams()) -> np.ndarray:
    """Quad (tl, tr, br, bl) of the photo window: per side the strongest texture ridge."""
    h, w = tex.shape
    f = p.factor
    small = cv2.GaussianBlur(cv2.resize(tex, (w // f, h // f), interpolation=cv2.INTER_AREA), (0, 0), 1.0)
    sh, sw = small.shape
    lines = {}
    for side in SIDES:
        along = sw if side in ("top", "bottom") else sh
        depth = int(0.45 * (sh if side in ("top", "bottom") else sw))
        pts = []
        for a0 in range(int(p.skip * along), int((1 - p.skip) * along) - p.band, p.band // 2):
            if side == "top":
                prof = np.median(small[:depth, a0:a0 + p.band], axis=1)
            elif side == "bottom":
                prof = np.median(small[::-1][:depth, a0:a0 + p.band], axis=1)
            elif side == "left":
                prof = np.median(small[a0:a0 + p.band, :depth], axis=0)
            else:
                prof = np.median(small[a0:a0 + p.band, ::-1][:, :depth], axis=0)
            k = int(np.argmax(prof))
            if prof[k] < p.ridge * float(np.median(prof)):
                continue
            c = a0 + p.band / 2
            pts.append({"top": (c, k), "bottom": (c, sh - 1 - k), "left": (k, c), "right": (sw - 1 - k, c)}[side])
        fit = _fit_line(np.array(pts, float)) if len(pts) >= 4 else None
        if fit is None:
            raise LightDetectionError(f"photo window not found ({side})")
        lines[side] = (fit[0] * f + (f - 1) / 2 + 0.5, fit[1])  # analysis pixel centre -> continuous full res
    return _quad(lines)


def edge_points(lum: np.ndarray, window: np.ndarray, px_mm: float, p: LightParams = LightParams()) -> Dict[str, np.ndarray]:
    """Per side, print edge points: from the window line outward, where the luminance (median
    over a segment along the side) leaves the frame plateau.

    Not handled: a soft shadow whose brightness right at the edge matches the frame; the edge
    then lands outward by up to the shadow's width. The texture of the paper would show it,
    but a high-pass response starts 2-3 px inside any edge and moved real outlines inward."""
    out = {}
    centre = window.mean(0)
    h, w = lum.shape
    for side, (a, b) in _SIDE_CORNERS.items():
        p0, p1 = window[a], window[b]
        L = float(np.linalg.norm(p1 - p0))
        t = (p1 - p0) / L
        n = np.array([-t[1], t[0]])
        if np.dot((p0 + p1) / 2 - centre, n) < 0:
            n = -n
        seg = p.seg_mm * px_mm
        d = np.arange(p.start_mm * px_mm, p.max_mm * px_mm, 1.0)
        pin = int(p.plateau_mm * px_mm)
        run = max(2, int(0.15 * px_mm))
        gap = max(2, int(p.gap_mm * px_mm))
        pts = []
        for s0 in np.arange(p.end_skip * L, (1 - p.end_skip) * L - seg, seg / 2):
            S, D = np.meshgrid(np.arange(s0, s0 + seg), d)
            X = (p0[0] + S * t[0] + D * n[0] - 0.5).astype(np.float32)  # continuous -> pixel index
            Y = (p0[1] + S * t[1] + D * n[1] - 0.5).astype(np.float32)
            rows = ((X >= 0) & (X < w - 1) & (Y >= 0) & (Y < h - 1)).all(axis=1)
            if rows.sum() < 2 * pin:
                continue
            prof = np.median(cv2.remap(lum, X, Y, cv2.INTER_LINEAR)[rows], axis=1)
            dd = d[rows]
            level = float(np.median(prof[:pin]))
            noise = float(np.median(np.abs(prof[:pin] - level))) * 1.4826
            thr = max(3.0, 4.0 * noise)
            # reference = the frame just behind each position (a stretch of plateau_mm ending
            # gap_mm before it): the wide bottom frame drifts in brightness across its 20 mm,
            # and a fixed reference from next to the window ends the plateau too early
            first = pin + gap
            ref = np.full(len(prof), level)
            if len(prof) > first:
                win = np.lib.stride_tricks.sliding_window_view(prof, pin)[: len(prof) - first]
                ref[first:] = np.median(win, axis=1)
            dev = np.abs(prof - ref)
            cs = np.convolve((dev > thr).astype(int), np.ones(run, int), mode="valid")
            idx = np.flatnonzero(cs[first:] == run)
            if not idx.size:
                continue
            j = first + int(idx[0])
            frac = float(np.clip((thr - dev[j - 1]) / max(1e-6, dev[j] - dev[j - 1]), 0, 1))
            pos = dd[j - 1] + frac
            if prof[j] < ref[j]:
                # The frame ends darker. A deep dark line (the print's own edge turned away
                # from the light, about 10 % below the frame) belongs to the print: the edge
                # is its darkest point. A shallow shadow cast on the paper (about 5 %) does
                # not: the edge stays where the frame ends. Calibrated on one camera setup
                # (42 shots, reviewer-placed edges); both stay within the 0.2 mm crop margin.
                k_end = min(len(prof), j + int(p.dark_line_mm * px_mm))
                k = j + int(np.argmin(prof[j:k_end]))
                if 0 < k < len(prof) - 1 and k < k_end - 1 and ref[j] - prof[k] > p.dark_line_depth * ref[j]:
                    a, b, c = prof[k - 1], prof[k], prof[k + 1]
                    den = a - 2 * b + c
                    pos = dd[k] + (0.5 * (a - c) / den if den > 1e-6 else 0.0)
            pts.append(p0 + (s0 + seg / 2) * t + pos * n)
        out[side] = np.array(pts, float).reshape(-1, 2)
    return out


def detect_light(rgb: np.ndarray, p: LightParams = LightParams()):
    """Returns (quad tl/tr/br/bl in continuous coordinates, per-side stats)."""
    lum = cv2.GaussianBlur(_luminance(rgb), (0, 0), 1.0)
    raw = _luminance(rgb)
    tex = np.abs(raw - cv2.GaussianBlur(raw, (0, 0), p.sigma))
    window = photo_window(tex, p)
    px_mm = (np.linalg.norm(window[1] - window[0]) + np.linalg.norm(window[3] - window[0])) / 2 / p.window_mm
    pts = edge_points(lum, window, px_mm, p)
    lines, stats = {}, {}
    for side, sp in pts.items():
        fit = _fit_line(sp) if len(sp) >= p.min_points else None
        if fit is None:
            raise LightDetectionError(f"too few edge points ({side}: {len(sp)})")
        c, dvec, keep, res = fit
        lines[side] = (c, dvec)
        # outward positive residuals; bow = middle minus ends, like the holder detector
        centre = window.mean(0)
        nrm = np.array([-dvec[1], dvec[0]])
        if np.dot(c - centre, nrm) < 0:
            res = -res
        along = (sp - c) @ dvec
        r_in, a_in = res[keep], along[keep]
        third = (a_in.max() - a_in.min()) / 3 if keep.sum() > 2 else 0
        mid = np.abs(a_in - (a_in.max() + a_in.min()) / 2) < third / 2
        ends = ~mid
        bow = float(np.median(r_in[mid]) - np.median(r_in[ends])) if mid.any() and ends.any() else 0.0
        stats[side] = {"n_points": int(len(sp)), "n_inliers": int(keep.sum()),
                       "mad_px": float(np.median(np.abs(r_in))), "bow_px": bow, "max_px": float(np.abs(r_in).max())}
    return _quad(lines), stats, pts
