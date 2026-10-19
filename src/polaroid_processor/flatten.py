"""Curved print edges: trace them, decide whether to flatten, and unwarp onto a rectangle.

Curled prints do not lie flat on the scanner glass, so their sides bow by up to about 1 mm.
A straight-sided crop then trims print in some places and shows holder in others. This
module traces each side densely at full resolution, fits a smooth curve, and maps the
curved outline onto a rectangle: the four sides become straight.

Only the border is distorted. The photo window of a curled print stays straight (square
windows) and round (round windows) in a plain rectification, so the correction of each side
fades out within ``fade_mm`` of the edge; further in, the map is the bilinear map between
the corners. With an unlimited fade this is transfinite (Coons) interpolation, which bends
the photo window.

Each side is checked on its own; a side whose trace is not trustworthy (printed graphics or
black glitter at the edge) stays straight. Coordinates are continuous scan pixels: pixel
``i`` spans ``[i, i + 1)``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

SIDES = ("top", "right", "bottom", "left")
# side -> (start corner, end corner) as indices into the quad (tl, tr, br, bl)
SIDE_CORNERS = {"top": (0, 1), "right": (1, 2), "bottom": (3, 2), "left": (0, 3)}


@dataclass(frozen=True)
class FlattenParams:
    step_mm: float = 0.5  # distance between edge samples along a side
    band_mm: float = 1.0  # each sample averages this much of the side
    search_mm: float = 2.0  # searched on each side of the straight outline
    plateau_mm: float = 0.5  # the print must stay bright for this long behind the edge
    end_skip_mm: float = 2.0  # rounded corners: not sampled
    contrast: float = 25.0  # min print - holder level (8 bit) for the brightness profile
    degree: int = 4  # polynomial degree of a side curve
    floor_mm: float = 0.15  # inlier tolerance never below this
    min_share: float = 0.8  # share of samples that must agree with the curve
    max_gap: float = 0.12  # largest stretch of a side (share) without agreeing samples
    max_mad_mm: float = 0.05  # median scatter of samples around the curve
    max_dev_mm: float = 1.6  # a curve that bends further than this is not trusted
    min_bend_mm: float = 0.1  # flatten only when a reliable side bends at least this much
    fade_mm: float = 5.0  # a side's correction fades to zero this far inward (photo window untouched)


@dataclass
class SideCurve:
    """One side: origin p0 (start corner of the straight side), unit tangent t, unit outward
    normal n, length L. A point is ``p0 + s t + offset(s) n``."""

    p0: np.ndarray
    t: np.ndarray
    n: np.ndarray
    length: float
    coef: np.ndarray  # polynomial offset(s / length) in px, numpy.polyval order; zeros = straight
    reliable: bool = False
    reason: str = ""
    stats: dict = field(default_factory=dict)

    def offset(self, s) -> np.ndarray:
        return np.polyval(self.coef, np.asarray(s, float) / self.length)

    def point(self, s) -> np.ndarray:
        s = np.asarray(s, float)
        return self.p0 + s[..., None] * self.t + self.offset(s)[..., None] * self.n

    def bend_px(self) -> float:
        """Peak-to-peak deviation of the curve from the chord between its ends (tilt excluded)."""
        s = np.linspace(0, self.length, 201)
        o = self.offset(s)
        o = o - (o[0] + (o[-1] - o[0]) * s / self.length)
        return float(o.max() - o.min())

    def to_dict(self) -> dict:
        return {"p0": [round(float(v), 3) for v in self.p0], "t": [float(v) for v in self.t],
                "n": [float(v) for v in self.n], "length": round(self.length, 3),
                "coef": [float(v) for v in self.coef], "reliable": self.reliable, "reason": self.reason,
                "stats": self.stats}

    @classmethod
    def from_dict(cls, d: dict) -> "SideCurve":
        return cls(np.array(d["p0"], float), np.array(d["t"], float), np.array(d["n"], float), float(d["length"]),
                   np.array(d["coef"], float), bool(d.get("reliable")), d.get("reason", ""), d.get("stats", {}))


def curves_from_dict(d: Optional[dict]) -> Optional[Dict[str, SideCurve]]:
    if not d or any(s not in d for s in SIDES):
        return None
    return {s: SideCurve.from_dict(d[s]) for s in SIDES}


# --------------------------------------------------------------------------- tracing
def _side_frames(quad: np.ndarray):
    centre = quad.mean(axis=0)
    out = {}
    for side, (a, b) in SIDE_CORNERS.items():
        p0, p1 = quad[a], quad[b]
        t = p1 - p0
        length = float(np.linalg.norm(t))
        t = t / length
        n = np.array([-t[1], t[0]])
        if np.dot((p0 + p1) / 2 - centre, n) < 0:
            n = -n
        out[side] = (p0, t, n, length)
    return out


def _side_strip(img: np.ndarray, p0, t, n, length: float, d_out: int, d_in: int) -> np.ndarray:
    """The band across one side: rows = outward offsets d_out .. -d_in + 1, columns = 1 px along it."""
    s = np.arange(int(length)) + 0.5
    d = np.arange(d_out, -d_in, -1, dtype=np.float64)
    S, D = np.meshgrid(s, d)
    X = (p0[0] + S * t[0] + D * n[0] - 0.5).astype(np.float32)
    Y = (p0[1] + S * t[1] + D * n[1] - 0.5).astype(np.float32)
    return cv2.remap(img, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _window_means(strip: np.ndarray, centres: np.ndarray, band: int) -> np.ndarray:
    """Mean over columns [c - band/2, c + band/2) per centre; shape (rows, len(centres))."""
    cs = np.concatenate([np.zeros((strip.shape[0], 1)), np.cumsum(strip, axis=1, dtype=np.float64)], axis=1)
    a = np.clip((centres - band // 2).astype(int), 0, strip.shape[1])
    b = np.clip(a + band, 0, strip.shape[1])
    return (cs[:, b] - cs[:, a]) / np.maximum(1, b - a)


def _find_edge(prof: np.ndarray, plateau: int, min_contrast: float) -> Optional[float]:
    """Fractional index of the edge in an outside-first profile: the first 50 % crossing
    between holder and print level that is followed by ``plateau`` samples at or above it."""
    n = len(prof)
    lo = float(np.percentile(prof[: max(8, n // 8)], 10))
    hi = float(np.median(prof[n - n // 4:]))
    if hi - lo < min_contrast:
        return None
    half = lo + 0.5 * (hi - lo)
    run = np.convolve((prof >= half).astype(np.int32), np.ones(plateau, np.int32), mode="valid")
    idx = np.flatnonzero(run == plateau)
    if not idx.size or idx[0] == 0:
        return None
    j = int(idx[0])
    return j - 1 + (half - prof[j - 1]) / max(1e-6, float(prof[j] - prof[j - 1])) + 0.5


def _robust_poly(x: np.ndarray, y: np.ndarray, deg: int, floor: float, iters: int = 5):
    ok = np.isfinite(y)
    w = ok.copy()
    coef = None
    for _ in range(iters):
        if w.sum() < 3 * (deg + 1):
            return None, w
        coef = np.polyfit(x[w], y[w], deg)
        r = y - np.polyval(coef, x)
        mad = float(np.median(np.abs(r[w]))) * 1.4826
        w = ok & (np.abs(r) <= max(floor, 3.0 * mad))
    return coef, w


def max_rgb_u8(rgb: np.ndarray) -> np.ndarray:
    ch = cv2.split(rgb)
    mx = ch[0]
    for c in ch[1:]:
        mx = cv2.max(mx, c)
    return mx if mx.dtype == np.uint8 else (mx >> 8).astype(np.uint8)


def trace(rgb: np.ndarray, quad: np.ndarray, px_per_mm: float, thr: int = 30,
          params: FlattenParams = FlattenParams(), mx: Optional[np.ndarray] = None) -> Dict[str, SideCurve]:
    """Trace the four sides of a print around its straight outline ``quad`` (tl, tr, br, bl).

    Each side is sampled every ``step_mm``; a sample's edge is the 50 % crossing of the
    max(R, G, B) profile, or of the share of pixels above ``thr`` for sparse frames.
    """
    p = params
    if mx is None:
        mx = max_rgb_u8(rgb)
    step, band = max(1, int(p.step_mm * px_per_mm)), max(1, int(p.band_mm * px_per_mm))
    d_out = d_in = int(p.search_mm * px_per_mm)
    plateau = max(2, int(p.plateau_mm * px_per_mm))
    skip = p.end_skip_mm * px_per_mm
    curves = {}
    for side, (p0, t, n, length) in _side_frames(quad).items():
        ss = np.arange(skip, length - skip, step)
        strip = _side_strip(mx, p0, t, n, length, d_out, d_in)
        prof_mx = _window_means(strip.astype(np.float32), ss, band)
        prof_sp = _window_means((strip > thr).astype(np.float32), ss, band)
        ds = np.full(len(ss), np.nan)
        for i in range(len(ss)):
            pos = _find_edge(prof_mx[:, i], plateau, p.contrast)
            if pos is None:
                pos = _find_edge(prof_sp[:, i], plateau, 0.05)
            if pos is not None:
                ds[i] = d_out - pos
        coef, inl = _robust_poly(ss / length, ds, p.degree, p.floor_mm * px_per_mm)
        curve = SideCurve(p0, t, n, length, np.zeros(p.degree + 1))
        share = float(inl.sum()) / max(1, len(ss))
        cover = ss[inl]
        gaps = np.diff(np.concatenate([[0.0], cover, [length]])) / length if cover.size else np.array([1.0])
        if coef is not None:
            mad = float(np.median(np.abs((ds - np.polyval(coef, ss / length))[inl]))) / px_per_mm
            dev = float(np.abs(np.polyval(coef, np.linspace(0, 1, 101))).max()) / px_per_mm
        else:
            mad = dev = float("inf")
        curve.stats = {"share": round(share, 2), "max_gap": round(float(gaps.max()), 2),
                       "mad_mm": round(mad, 3) if np.isfinite(mad) else None,
                       "max_dev_mm": round(dev, 2) if np.isfinite(dev) else None}
        if coef is None:
            curve.reason = "too few edge points"
        elif share < p.min_share:
            curve.reason = f"only {share:.0%} of the edge points agree"
        elif gaps.max() > p.max_gap:
            curve.reason = f"{gaps.max():.0%} of the side without edge points"
        elif mad > p.max_mad_mm:
            curve.reason = f"edge points scatter {mad:.3f} mm"
        elif dev > p.max_dev_mm:
            curve.reason = f"curve deviates {dev:.2f} mm"
        else:
            curve.coef, curve.reliable = coef, True
        curves[side] = curve
    return curves


# --------------------------------------------------------------------------- decision
@dataclass
class FlattenPlan:
    apply: bool
    sides: List[str]  # sides that follow a curve
    straight: Dict[str, str]  # side -> why it stays straight (only when applied)
    bend_mm: Dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def plan(curves: Optional[Dict[str, SideCurve]], px_per_mm: float,
         params: FlattenParams = FlattenParams()) -> FlattenPlan:
    """Flatten when at least one reliable side bends by ``min_bend_mm`` or more."""
    if not curves:
        return FlattenPlan(False, [], {}, {})
    bend = {s: round(c.bend_px() / px_per_mm, 3) for s, c in curves.items() if c.reliable}
    apply = any(b >= params.min_bend_mm for b in bend.values())
    straight = {s: c.reason for s, c in curves.items() if not c.reliable} if apply else {}
    return FlattenPlan(apply, sorted(bend, key=SIDES.index) if apply else [], straight, bend)


# --------------------------------------------------------------------------- unwarp
def _corner(c1: SideCurve, c2: SideCurve, guess: np.ndarray, iters: int = 30) -> np.ndarray:
    """Intersection of two side curves near ``guess`` (alternating projection)."""
    s1 = float(np.dot(guess - c1.p0, c1.t))
    s2 = 0.0
    for _ in range(iters):
        s2 = float(np.dot(c1.point(s1) - c2.p0, c2.t))
        s1 = float(np.dot(c2.point(s2) - c1.p0, c1.t))
    return (c1.point(s1) + c2.point(s2)) / 2


def corners(curves: Dict[str, SideCurve], quad: np.ndarray) -> np.ndarray:
    """Corners (tl, tr, br, bl) where the side curves meet."""
    return np.array([
        _corner(curves["top"], curves["left"], quad[0]),
        _corner(curves["top"], curves["right"], quad[1]),
        _corner(curves["bottom"], curves["right"], quad[2]),
        _corner(curves["bottom"], curves["left"], quad[3]),
    ])


def _boundary(c: SideCurve, a: np.ndarray, b: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Points on curve ``c`` from corner a (u = 0) to corner b (u = 1), uniform along the side."""
    sa, sb = float(np.dot(a - c.p0, c.t)), float(np.dot(b - c.p0, c.t))
    return c.point(sa + np.asarray(u, float) * (sb - sa))


def _arclen(c: SideCurve, a, b, n: int = 400) -> float:
    pts = _boundary(c, a, b, np.linspace(0, 1, n))
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def output_size(curves: Dict[str, SideCurve], quad: np.ndarray) -> Tuple[float, float]:
    """Print size (W, H) in px: mean arc length of opposite sides."""
    tl, tr, br, bl = corners(curves, quad)
    W = (_arclen(curves["top"], tl, tr) + _arclen(curves["bottom"], bl, br)) / 2
    H = (_arclen(curves["left"], tl, bl) + _arclen(curves["right"], tr, br)) / 2
    return W, H


def _fade(d: np.ndarray, fade_px: Optional[float], extent: float) -> np.ndarray:
    """Weight of a side's correction at distance ``d`` (px) from it: 1 at the edge, 0 from
    ``fade_px`` inward (raised cosine). None: linear across the print (Coons)."""
    if fade_px is None:
        return 1.0 - d / extent
    x = np.clip(d / fade_px, 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * x))


def coons_map(curves: Dict[str, SideCurve], quad: np.ndarray, margin_px: float = 0.0, grid: int = 16,
              fade_px: Optional[float] = None):
    """Source coordinates for every output pixel (float32 maps for cv2.remap) and the size (W, H).

    Source point = bilinear map of the four corners + each side's deviation from its chord,
    weighted by :func:`_fade` of the distance to that side. The deviation is zero at the
    corners, so the sides map exactly onto the curves for any fade.

    The map is smooth, so it is evaluated on a coarse grid and resized; the error is about
    0.002 px. Coarse node k sits at output pixel ``k * grid + (grid - 1) / 2`` (k = -1 .. K),
    which is where cv2.resize by ``grid`` places it; the extra outer nodes keep the border
    interpolated rather than clamped.
    """
    tl, tr, br, bl = corners(curves, quad)
    W = (_arclen(curves["top"], tl, tr) + _arclen(curves["bottom"], bl, br)) / 2
    H = (_arclen(curves["left"], tl, bl) + _arclen(curves["right"], tr, br)) / 2
    m = margin_px
    ow, oh = max(1, int(round(W - 2 * m))), max(1, int(round(H - 2 * m)))
    kx, ky = np.arange(-1, ow // grid + 2), np.arange(-1, oh // grid + 2)
    u = (kx * grid + (grid - 1) / 2 + 0.5 + m) / W
    v = (ky * grid + (grid - 1) / 2 + 0.5 + m) / H
    T, B = _boundary(curves["top"], tl, tr, u), _boundary(curves["bottom"], bl, br, u)
    L, R = _boundary(curves["left"], tl, bl, v), _boundary(curves["right"], tr, br, v)
    uu, vv = u[:, None], v[:, None]
    dT, dB = T - ((1 - uu) * tl + uu * tr), B - ((1 - uu) * bl + uu * br)  # deviation from the chord
    dL, dR = L - ((1 - vv) * tl + vv * bl), R - ((1 - vv) * tr + vv * br)
    U, V = u[None, :, None], v[:, None, None]
    P = ((1 - U) * (1 - V) * tl + U * (1 - V) * tr + (1 - U) * V * bl + U * V * br
         + _fade(V * H, fade_px, H) * dT[None] + _fade((1 - V) * H, fade_px, H) * dB[None]
         + _fade(U * W, fade_px, W) * dL[:, None] + _fade((1 - U) * W, fade_px, W) * dR[:, None])
    size = (len(kx) * grid, len(ky) * grid)
    mapx = cv2.resize((P[..., 0] - 0.5).astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    mapy = cv2.resize((P[..., 1] - 0.5).astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    return mapx[grid: grid + oh, grid: grid + ow], mapy[grid: grid + oh, grid: grid + ow], (W, H)


INTERPOLATION = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}


def unwarp(rgb: np.ndarray, curves: Dict[str, SideCurve], quad: np.ndarray, margin_px: float = 0.0,
           interpolation: str = "cubic", strip_rows: int = 1024, fade_px: Optional[float] = None) -> np.ndarray:
    """Map the curved outline onto a rectangle, cropped ``margin_px`` inside the edge; the side
    corrections fade out within ``fade_px`` (None: Coons, linear across the print)."""
    mapx, mapy, _ = coons_map(curves, quad, margin_px, fade_px=fade_px)
    out = np.empty(mapx.shape + rgb.shape[2:], rgb.dtype)
    for r0 in range(0, mapx.shape[0], strip_rows):
        sl = slice(r0, r0 + strip_rows)
        m1, m2 = cv2.convertMaps(mapx[sl], mapy[sl], cv2.CV_16SC2)
        out[sl] = cv2.remap(rgb, m1, m2, INTERPOLATION[interpolation], borderMode=cv2.BORDER_REPLICATE)
    return out


def outline_points(curves: Dict[str, SideCurve], quad: np.ndarray, n: int = 200) -> np.ndarray:
    """Closed outline (tl -> tr -> br -> bl) along the curves, for drawing."""
    tl, tr, br, bl = corners(curves, quad)
    u = np.linspace(0, 1, n)
    return np.concatenate([
        _boundary(curves["top"], tl, tr, u),
        _boundary(curves["right"], tr, br, u),
        _boundary(curves["bottom"], bl, br, u)[::-1],
        _boundary(curves["left"], tl, bl, u)[::-1],
    ])
