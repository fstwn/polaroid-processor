"""Curved print edges: trace them, decide whether to flatten, and unwarp onto a rectangle.

Curled prints do not lie flat on the scanner glass, so their sides bow by up to about 1 mm.
A straight-sided crop then trims print in some places and shows holder in others. This
module traces each side densely at full resolution, fits a smooth curve, and maps the
curved outline onto a rectangle: the four sides become straight.

Only the border is distorted. The photo window of a curled print stays straight (square
windows) and round (round windows) in a plain rectification, so the inside maps exactly as
in the straight export (the homography from the fitted lines), and each side's traced offset
is added only near that side, fading out within ``fade_mm``. Near the corners, where the
trace has no samples, the offset is held rather than extrapolated: corners extrapolated from
the curves land up to 0.85 mm off and would skew the photo window.

Each side is checked on its own; a side whose trace is not trustworthy (printed graphics or
black glitter at the edge) stays straight. Coordinates are continuous scan pixels: pixel
``i`` spans ``[i, i + 1)``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .detection import quad_size
from .geometry import output_size

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

    def warp_px(self, px_per_mm: float) -> dict:
        """``fade_px`` and ``hold_px`` for :func:`unwarp` and :func:`outline_points`."""
        return {"fade_px": self.fade_mm * px_per_mm, "hold_px": self.end_skip_mm * px_per_mm}


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


def _fade(d: np.ndarray, fade_px: Optional[float], extent: float) -> np.ndarray:
    """Weight of a side's correction at distance ``d`` (px) from it: 1 at the edge, 0 from
    ``fade_px`` inward (raised cosine). None: linear across the print."""
    if fade_px is None:
        return np.clip(1.0 - d / extent, 0.0, 1.0)
    x = np.clip(d / fade_px, 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * x))


def to_source(curves: Dict[str, SideCurve], quad: np.ndarray, X: np.ndarray, Y: np.ndarray,
              fade_px: Optional[float] = None, hold_px: float = 0.0) -> np.ndarray:
    """Scan coordinates (continuous) of print coordinates (X, Y): (0, 0) is the top-left
    corner of the straight outline, (w, h) its bottom-right, w and h its size.

    Base: the straight export's homography from the quad, so the inside of the print maps
    exactly as without flattening. Each side then adds its traced offset along its outward
    normal, weighted by :func:`_fade` of the distance to that side. Within ``hold_px`` of a
    corner, where the trace has no samples, the offset is held instead of extrapolated.
    """
    w, h = quad_size(quad)
    rect = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    Hm = cv2.getPerspectiveTransform(rect, np.asarray(quad, np.float32))
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    pts = np.stack([X, Y], axis=-1).reshape(-1, 1, 2)
    P = cv2.perspectiveTransform(pts, Hm).reshape(X.shape + (2,))
    along = {"top": (X / w, Y, h), "bottom": (X / w, h - Y, h), "left": (Y / h, X, w), "right": (Y / h, w - X, w)}
    for side, (frac, d, extent) in along.items():
        c = curves[side]
        if not np.any(c.coef):
            continue
        s = np.clip(frac * c.length, hold_px, c.length - hold_px)
        P += (_fade(d, fade_px, extent) * c.offset(s))[..., None] * c.n
    return P


def warp_map(curves: Dict[str, SideCurve], quad: np.ndarray, margin_px: float = 0.0, grid: int = 16,
             fade_px: Optional[float] = None, hold_px: float = 0.0):
    """Source coordinates for every output pixel (float32 maps for cv2.remap); the output has
    the straight export's size (:func:`geometry.output_size`).

    The map is smooth, so it is evaluated on a coarse grid and resized; the error is about
    0.002 px. Coarse node k sits at output pixel ``k * grid + (grid - 1) / 2`` (k = -1 .. K),
    which is where cv2.resize by ``grid`` places it; the extra outer nodes keep the border
    interpolated rather than clamped.
    """
    m = margin_px
    ow, oh = output_size(quad, m)
    kx, ky = np.arange(-1, ow // grid + 2), np.arange(-1, oh // grid + 2)
    X, Y = np.meshgrid(kx * grid + (grid - 1) / 2 + 0.5 + m, ky * grid + (grid - 1) / 2 + 0.5 + m)
    P = to_source(curves, quad, X, Y, fade_px, hold_px) - 0.5
    size = (len(kx) * grid, len(ky) * grid)
    mapx = cv2.resize(P[..., 0].astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    mapy = cv2.resize(P[..., 1].astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    return mapx[grid: grid + oh, grid: grid + ow], mapy[grid: grid + oh, grid: grid + ow]


INTERPOLATION = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}


def unwarp(rgb: np.ndarray, curves: Dict[str, SideCurve], quad: np.ndarray, margin_px: float = 0.0,
           interpolation: str = "cubic", strip_rows: int = 1024, fade_px: Optional[float] = None,
           hold_px: float = 0.0) -> np.ndarray:
    """Map the curved outline onto the straight export's rectangle, cropped ``margin_px``
    inside the edge (see :func:`to_source`)."""
    mapx, mapy = warp_map(curves, quad, margin_px, fade_px=fade_px, hold_px=hold_px)
    out = np.empty(mapx.shape + rgb.shape[2:], rgb.dtype)
    for r0 in range(0, mapx.shape[0], strip_rows):
        sl = slice(r0, r0 + strip_rows)
        m1, m2 = cv2.convertMaps(mapx[sl], mapy[sl], cv2.CV_16SC2)
        out[sl] = cv2.remap(rgb, m1, m2, INTERPOLATION[interpolation], borderMode=cv2.BORDER_REPLICATE)
    return out


def outline_points(curves: Dict[str, SideCurve], quad: np.ndarray, n: int = 200, fade_px: Optional[float] = None,
                   hold_px: float = 0.0) -> np.ndarray:
    """Closed outline (tl -> tr -> br -> bl) of the flattened crop in scan coordinates, for drawing."""
    w, h = quad_size(quad)
    u = np.linspace(0, 1, n)
    X = np.concatenate([u * w, np.full(n, w), (1 - u) * w, np.zeros(n)])
    Y = np.concatenate([np.zeros(n), u * h, np.full(n, h), (1 - u) * h])
    return to_source(curves, quad, X, Y, fade_px, hold_px)
