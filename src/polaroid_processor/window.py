"""Photo window of a print: where the picture sits inside the frame, measured per image.

Input is a straightened print (what ``rectify()`` produces, possibly at reduced scale): the
print edges are the image edges, so the window border lies near its nominal place and is
axis-aligned up to a small rotation and shear against the frame (up to 0.7 deg measured on
the library, 0.3-0.5 deg on typical batches). The finder measures the window itself; the
nominal size per format (``NOMINAL``) is only a plausibility check, never a fallback.

Square windows: per side a Radon-style search over a narrow strip around the nominal border
(sum of the colour-gradient magnitude along tilted lines) gives a coarse line that ignores
artwork and dark photo edges, then per segment of about 1 mm the border is located where the
colour first leaves the frame colour (half way to the strongest step, seen from outside; with a
thin dark line at the border this is the outer flank), and a robust line is fitted through
those points. The four lines give the window quad, so rotation and shear stay in it.

Round windows: the same idea on a circle (Hough search of the gradient magnitude, then rays
from outside inward and a robust ellipse fit).

Coordinates in the result are continuous image pixels (pixel i spans [i, i + 1)), like the
print quads elsewhere. A clean "not found" is better than a wrong window: every check that
fails says why in ``reason``.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .formats import PrintFormat

SIDES = ("top", "right", "bottom", "left")


@dataclass(frozen=True)
class WindowSpec:
    """Nominal window of a format and kind, in mm from the print's outer edge (portrait)."""

    kind: str  # "square" | "round"
    left: float  # distance from the left print edge to the window's left border (round: bounding square)
    top: float
    width: float  # square: width; round: diameter
    height: float  # square: height; round: diameter
    tol: float = 1.0  # accepted deviation of the measured size per side, mm
    zone: float = 2.5  # search zone around each nominal border (round: around the nominal centre), mm


# Measured with this finder on the labelled and sample scans at 1/4 scale (2026-11-02), outer
# border flank of the window, with px/mm = straightened print width / format width:
# 600 holder scans 76.6 x 78.7 mm (sd 0.1; extremes 76.47-77.29 x 78.44-79.29 over 168 prints),
# camera shots 77.1 x 79.1, Go 45.8-46.2 x 46.7-47.0 (two batches), round windows 76.2-77.0 mm
# across. The window's position varies by +-0.7 mm (600) and +-0.2 mm (Go) between prints.
NOMINAL: Dict[Tuple[str, str], WindowSpec] = {
    ("polaroid-600", "square"): WindowSpec("square", 5.5, 6.3, 76.9, 78.9, tol=1.0, zone=2.5),
    ("polaroid-600", "round"): WindowSpec("round", 5.5, 7.7, 76.7, 76.7, tol=1.0, zone=3.0),
    ("polaroid-go", "square"): WindowSpec("square", 3.8, 4.85, 46.0, 46.9, tol=0.5, zone=1.5),
}


@dataclass(frozen=True)
class WindowParams:
    sigma_mm: float = 0.12  # blur before the gradient
    min_sigma_px: float = 1.0
    max_tilt_deg: float = 1.2  # search range of a border's tilt against the image axes
    tilt_step_deg: float = 0.08  # tilt resolution of the coarse search
    coarse_px_per_mm: float = 10.0  # resolution of the coarse line search
    along_skip: float = 0.08  # share of the window length ignored at each end (corners)
    seg_mm: float = 1.0  # segment length along a border
    ref_mm: Tuple[float, float] = (0.6, 1.6)  # frame colour: this far outside the coarse line
    near_mm: Tuple[float, float] = (0.6, 0.9)  # border searched this far outside / inside the coarse line
    dmax_mm: Tuple[float, float] = (0.4, 0.6)  # strongest step taken in this range around the coarse line
    min_contrast: float = 10.0  # colour distance (0-255 scale) the step must reach
    noise_factor: float = 3.0  # ... and this many times the texture of the frame beside it
    min_step_share: float = 0.3  # share of segments (rays) that must show a step at all
    min_inlier: float = 0.30  # share of segments on the fitted line
    min_span: float = 0.6  # inliers must cover this share of the border length
    max_rms_mm: float = 0.12  # residual of the inliers (square)
    max_rms_round_mm: float = 0.24  # residual of the inliers (round)
    max_parallel_deg: float = 0.8  # opposite sides may differ by this much
    max_axis_diff_mm: float = 1.0  # round: the two ellipse diameters may differ by this much (real: 0.6)
    round_px_per_mm: float = 10.0  # resolution of the coarse circle search
    round_rays: int = 360
    seed: int = 1


@dataclass
class WindowResult:
    found: bool
    reason: str = ""
    kind: Optional[str] = None  # "square" | "round"
    quad: Optional[np.ndarray] = None  # tl, tr, br, bl in continuous image px (round: bounding square)
    size_mm: Optional[Tuple[float, float]] = None  # square: width, height; round: long, short ellipse diameter
    circle: Optional[Tuple[float, float, float]] = None  # round: cx, cy, r in px
    ellipse: Optional[Tuple[float, float, float, float, float]] = None  # round: cx, cy, long and short semi axis px, angle deg
    rms_mm: Dict[str, float] = field(default_factory=dict)  # per side (square) / "circle" (round)
    inlier_share: Dict[str, float] = field(default_factory=dict)
    side_deg: Dict[str, float] = field(default_factory=dict)  # square: border angles against the image axes
    rotation_deg: Optional[float] = None  # square: mean tilt of top and bottom
    shear_deg: Optional[float] = None  # square: left and right tilt beyond the rotation
    px_per_mm: float = 0.0
    nominal_mm: Optional[Tuple[float, float]] = None
    notes: List[str] = field(default_factory=list)  # one line per kind that was tried and failed

    def scaled(self, factor: float) -> "WindowResult":
        """A copy for an image ``factor`` times larger (px values scaled, mm values unchanged)."""
        r = copy.deepcopy(self)
        if r.quad is not None:
            r.quad = r.quad * factor
        if r.circle is not None:
            r.circle = tuple(v * factor for v in r.circle)
        if r.ellipse is not None:
            cx, cy, a, b, ang = r.ellipse
            r.ellipse = (cx * factor, cy * factor, a * factor, b * factor, ang)
        r.px_per_mm = self.px_per_mm * factor
        return r

    def to_dict(self) -> dict:
        d = {"found": self.found, "reason": self.reason, "kind": self.kind, "px_per_mm": round(self.px_per_mm, 4),
             "notes": list(self.notes)}
        if self.quad is not None:
            d["quad"] = np.round(self.quad, 3).tolist()
        for k in ("size_mm", "nominal_mm"):
            v = getattr(self, k)
            if v is not None:
                d[k] = [round(float(x), 4) for x in v]
        for k in ("circle", "ellipse"):
            v = getattr(self, k)
            if v is not None:
                d[k] = [round(float(x), 4) for x in v]
        for k in ("rms_mm", "inlier_share", "side_deg"):
            v = getattr(self, k)
            if v:
                d[k] = {a: round(float(b), 5) for a, b in v.items()}
        if self.rotation_deg is not None:
            d["rotation_deg"] = round(float(self.rotation_deg), 5)
            d["shear_deg"] = round(float(self.shear_deg), 5)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "WindowResult":
        """Inverse of ``to_dict`` (values keep the rounding of ``to_dict``)."""
        r = cls(bool(d.get("found")), d.get("reason", ""), d.get("kind"))
        if d.get("quad") is not None:
            r.quad = np.array(d["quad"], float)
        for k in ("size_mm", "nominal_mm", "circle", "ellipse"):
            if d.get(k) is not None:
                setattr(r, k, tuple(d[k]))
        for k in ("rms_mm", "inlier_share", "side_deg"):
            setattr(r, k, dict(d.get(k) or {}))
        r.rotation_deg, r.shear_deg = d.get("rotation_deg"), d.get("shear_deg")
        r.px_per_mm = float(d.get("px_per_mm", 0.0))
        r.notes = list(d.get("notes") or [])
        return r


# ---------------------------------------------------------------- helpers


def _planes(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    if img.dtype == np.uint16:
        f /= 257.0
    return f


def _oriented(a: np.ndarray, side: str) -> np.ndarray:
    """View of ``a`` in which rows run from the print edge inward for ``side`` and columns along it."""
    if side == "top":
        return a
    if side == "bottom":
        return a[::-1]
    t = a.transpose(1, 0, 2) if a.ndim == 3 else a.T
    return t if side == "left" else t[::-1]


def _to_image(side: str, depth: np.ndarray, along: np.ndarray, h: int, w: int) -> np.ndarray:
    """Oriented (depth, along) index coordinates -> image (x, y) index coordinates."""
    depth, along = np.asarray(depth, float), np.asarray(along, float)
    if side == "top":
        return np.stack([along, depth], -1)
    if side == "bottom":
        return np.stack([along, h - 1 - depth], -1)
    if side == "left":
        return np.stack([depth, along], -1)
    return np.stack([w - 1 - depth, along], -1)


def _robust_line(a: np.ndarray, d: np.ndarray, floor: float, seed: int) -> Optional[Tuple[float, float, np.ndarray, np.ndarray]]:
    """d = c0 + c1 * a through outliers: Theil-Sen start, then trimmed least squares.
    Returns (c0, c1, inlier mask, residuals)."""
    n = len(a)
    if n < 6:
        return None
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, 400)
    j = rng.integers(0, n, 400)
    ok = np.abs(a[i] - a[j]) > 1e-9
    if ok.sum() < 5:
        return None
    c1 = float(np.median((d[i] - d[j])[ok] / (a[i] - a[j])[ok]))
    c0 = float(np.median(d - c1 * a))
    keep = np.ones(n, bool)
    res = d - (c0 + c1 * a)
    for _ in range(6):
        mad = float(np.median(np.abs(res[keep] if keep.any() else res))) * 1.4826
        keep = np.abs(res) <= max(floor, 2.5 * mad)
        if keep.sum() < 5:
            return None
        c1, c0 = np.polyfit(a[keep], d[keep], 1)
        res = d - (c0 + c1 * a)
    return float(c0), float(c1), keep, res


def _intersect(p1, d1, p2, d2) -> np.ndarray:
    s = np.linalg.solve(np.array([d1, -d2]).T, p2 - p1)
    return p1 + s[0] * d1


def _step_position(dist: np.ndarray, first: int, last: int, strong: Tuple[int, int], min_step: float, noise: float,
                   p: WindowParams) -> Optional[float]:
    """Where a colour-distance profile (frame colour = 0, seen from outside) first reaches half of its
    strongest step: fractional index into ``dist``, or None when there is no clear, new step.

    ``first``/``last``: indices searched; ``strong``: indices where the strongest step is taken."""
    top = float(dist[strong[0]:strong[1] + 1].max())
    if top < max(min_step, p.noise_factor * noise):
        return None
    half = 0.5 * top
    inside = np.flatnonzero(dist[first:last + 1] >= half)
    if not inside.size or inside[0] == 0:  # no step, or already beyond half at the start (artwork)
        return None
    i = first + int(inside[0])
    below, above = float(dist[i - 1]), float(dist[i])
    frac = (half - below) / (above - below) if above > below else 0.0
    return i - 1 + frac


# ---------------------------------------------------------------- square window


@dataclass
class _Border:
    """One fitted window border (square windows)."""

    ok: bool
    why: str = ""
    point: Optional[np.ndarray] = None  # a point of the line, image index coordinates
    direction: Optional[np.ndarray] = None  # unit direction
    rms_mm: float = 0.0
    inlier_share: float = 0.0


def _coarse_line(strip: np.ndarray, factor: int, px: float, zone_rows: Tuple[int, int],
                 p: WindowParams) -> Tuple[float, float]:
    """Radon-style search: (tilt, row) of the strongest line in ``strip``, where ``row`` is its
    depth at the strip's centre column, in rows of the full-resolution strip.

    The strip is shrunk by ``factor``; the score of a line is the mean colour-gradient magnitude
    along it (gradient across the border, capped so that artwork cannot outvote the border)."""
    small = strip
    if factor > 1:
        small = cv2.resize(strip, (strip.shape[1] // factor, strip.shape[0] // factor), interpolation=cv2.INTER_AREA)
    sigma = max(p.min_sigma_px, p.sigma_mm * px)
    small = cv2.GaussianBlur(small, (0, 0), max(1.0, sigma / factor))
    across = np.zeros_like(small)
    across[1:-1] = (small[2:] - small[:-2]) * 0.5
    grad = np.sqrt((across ** 2).sum(axis=2)).astype(np.float32)
    cap = 2.0 * float(np.median(grad.max(axis=0)))
    if cap > 0:
        grad = np.minimum(grad, cap)
    n_rows, n_cols = grad.shape
    centre = (n_cols - 1) / 2.0
    max_tilt = math.tan(math.radians(p.max_tilt_deg))
    n_tilts = max(7, int(round(2 * p.max_tilt_deg / p.tilt_step_deg)) | 1)
    tilts = np.linspace(-max_tilt, max_tilt, n_tilts)
    score = np.zeros((n_tilts, n_rows), np.float32)
    for i, tilt in enumerate(tilts):
        shear = np.array([[1, 0, 0], [tilt, 1, -tilt * centre]], np.float32)
        along = cv2.warpAffine(grad, shear, (n_cols, n_rows), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
        score[i] = along.mean(axis=1)
    score = cv2.blur(score, (1, max(1, int(round(0.2 * px / factor)))))
    rows = np.arange(n_rows)
    in_zone = (rows >= zone_rows[0] / factor) & (rows <= zone_rows[1] / factor)
    best_tilt, best_row = np.unravel_index(int(np.argmax(np.where(in_zone[None, :], score, -1.0))), score.shape)
    return float(tilts[best_tilt]), float(round(best_row * factor + (factor - 1) / 2.0))


def _flatten(o: np.ndarray, line_depth: int, tilt: float, col0: int, n_cols: int, px: float,
             p: WindowParams) -> Tuple[np.ndarray, int, int]:
    """Cut the rows around the coarse line and shear them so that the line is horizontal.
    Returns (flat colour strip, its first depth row, row of the line inside it)."""
    mm = lambda v: int(round(v * px))  # noqa: E731
    room = int(math.ceil(abs(tilt) * n_cols / 2)) + 3  # rows the shear pulls in from above and below
    first = max(0, line_depth - mm(max(p.ref_mm)) - room)
    last = min(o.shape[0], line_depth + mm(p.near_mm[1]) + room + 1)
    crop = np.ascontiguousarray(o[first:last, col0:col0 + n_cols])
    crop = cv2.GaussianBlur(crop, (0, 0), max(0.7, 0.5 * max(p.min_sigma_px, p.sigma_mm * px)))
    centre = (n_cols - 1) / 2.0
    shear = np.array([[1, 0, 0], [tilt, 1, -tilt * centre]], np.float32)
    flat = cv2.warpAffine(crop, shear, (n_cols, crop.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)
    return flat, first, line_depth - first


def _border_points(flat: np.ndarray, line_row: int, px: float, p: WindowParams) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """Per segment along the flattened strip, the row where the colour leaves the frame colour.
    Returns (segment centre columns, rows, segment count) or None when the strip is unusable."""
    mm = lambda v: int(round(v * px))  # noqa: E731
    seg = max(3, mm(p.seg_mm))
    n_seg = flat.shape[1] // seg
    ref0, ref1 = line_row - mm(p.ref_mm[1]), line_row - mm(p.ref_mm[0])  # frame colour rows
    near0, near1 = line_row - mm(p.near_mm[0]), line_row + mm(p.near_mm[1])  # rows searched for the border
    step0, step1 = line_row - mm(p.dmax_mm[0]), line_row + mm(p.dmax_mm[1])  # rows of the strongest step
    if n_seg < 8 or ref0 < 0 or near1 >= flat.shape[0]:
        return None
    # colour profile of each segment: median over its columns, rows ref0..near1
    profiles = np.median(flat[ref0:near1 + 1, :n_seg * seg].reshape(near1 + 1 - ref0, n_seg, seg, 3), axis=2)
    cols, rows = [], []
    for j in range(n_seg):
        prof = profiles[:, j]
        frame = np.median(prof[:ref1 - ref0], axis=0)
        dist = np.sqrt(((prof - frame) ** 2).sum(axis=1))
        noise = float(np.median(dist[:ref1 - ref0]))
        pos = _step_position(dist, near0 - ref0, near1 - ref0, (step0 - ref0, step1 - ref0), p.min_contrast, noise, p)
        if pos is None:
            continue
        rows.append(ref0 + pos)
        cols.append(j * seg + (seg - 1) / 2.0)
    return np.array(cols), np.array(rows), n_seg


def _find_border(f: np.ndarray, side: str, px: float, zone_mm: Tuple[float, float], along_mm: Tuple[float, float],
                 p: WindowParams) -> _Border:
    """One border. ``zone_mm``: where it may lie, as distance from the print edge; ``along_mm``: range along it."""
    mm = lambda v: int(round(v * px))  # noqa: E731
    o = _oriented(f, side)
    depth_n, along_n = o.shape[:2]
    h, w = f.shape[:2]
    factor = max(1, int(px // p.coarse_px_per_mm))
    col0 = max(0, mm(along_mm[0]))
    n_cols = (min(along_n, mm(along_mm[1])) - col0) // factor * factor
    zone = (mm(zone_mm[0]), mm(zone_mm[1]))
    room = int(math.ceil(math.tan(math.radians(p.max_tilt_deg)) * n_cols / 2)) + 2  # rows a tilted line moves
    row0 = max(0, zone[0] - room - mm(max(p.ref_mm)))
    n_rows = (min(depth_n, zone[1] + room + mm(p.near_mm[1]) + 1) - row0) // factor * factor
    if n_cols < 20 * factor:
        return _Border(False, f"{side}: window outside the image")
    if n_rows < 8 * factor or zone[1] <= zone[0]:
        return _Border(False, f"{side}: search zone outside the image")
    tilt, line_row = _coarse_line(np.ascontiguousarray(o[row0:row0 + n_rows, col0:col0 + n_cols]), factor, px,
                                  (zone[0] - row0, zone[1] - row0), p)
    line_depth = int(row0 + line_row)
    flat, flat_row0, flat_line = _flatten(o, line_depth, tilt, col0, n_cols, px, p)
    found = _border_points(flat, flat_line, px, p)
    if found is None:
        return _Border(False, f"{side}: search zone outside the image or border too short")
    cols, rows, n_seg = found
    if len(cols) < max(8, int(p.min_step_share * n_seg)):
        return _Border(False, f"{side}: no border ({len(cols)} of {n_seg} segments have a step)")
    # the strip was sheared about its centre column: undo it, then fit depth = c0 + c1 * along
    centre = (n_cols - 1) / 2.0
    along = col0 + cols
    depth = flat_row0 + rows + tilt * (cols - centre)
    fit = _robust_line(along, depth, max(0.04 * px, 1.0), p.seed)
    if fit is None:
        return _Border(False, f"{side}: line fit failed")
    c0, c1, inliers, residual = fit
    rms = float(np.sqrt(np.mean(residual[inliers] ** 2))) / px
    seg = max(3, mm(p.seg_mm))
    span = float((along[inliers].max() - along[inliers].min()) / max(1.0, (n_seg - 1) * seg))
    share = float(inliers.sum()) / n_seg
    ends = _to_image(side, c0 + c1 * np.array([0.0, along_n - 1.0]), np.array([0.0, along_n - 1.0]), h, w)
    border = _Border(False, "", ends[0], (ends[1] - ends[0]) / np.linalg.norm(ends[1] - ends[0]), rms, share)
    if share < p.min_inlier:
        border.why = f"{side}: border not straight enough ({share:.0%} of segments on one line)"
    elif span < p.min_span:
        border.why = f"{side}: border seen over only {span:.0%} of its length"
    elif rms > p.max_rms_mm:
        border.why = f"{side}: border residual {rms:.3f} mm"
    else:
        border.ok = True
    return border


def _find_square(f: np.ndarray, px: float, spec: WindowSpec, margin_mm: float, p: WindowParams) -> WindowResult:
    h, w = f.shape[:2]
    res = WindowResult(False, kind="square", px_per_mm=px, nominal_mm=(spec.width, spec.height))
    # nominal borders as distances from each image edge (the image may be cropped by margin_mm)
    print_w, print_h = w / px + 2 * margin_mm, h / px + 2 * margin_mm
    nominal = {"top": spec.top - margin_mm, "left": spec.left - margin_mm,
               "bottom": print_h - spec.top - spec.height - margin_mm, "right": print_w - spec.left - spec.width - margin_mm}
    skip = p.along_skip
    across = (nominal["left"] + skip * spec.width, nominal["left"] + (1 - skip) * spec.width)  # range along top/bottom
    down = (nominal["top"] + skip * spec.height, nominal["top"] + (1 - skip) * spec.height)  # range along left/right
    lines = {}
    for side in SIDES:
        along = across if side in ("top", "bottom") else down
        border = _find_border(f, side, px, (nominal[side] - spec.zone, nominal[side] + spec.zone), along, p)
        res.rms_mm[side], res.inlier_share[side] = border.rms_mm, border.inlier_share
        if not border.ok:
            res.reason = border.why  # one failed border is enough; the caller may try a round window next
            return res
        lines[side] = (border.point, border.direction)
    corners = np.array([_intersect(*lines["top"], *lines["left"]), _intersect(*lines["top"], *lines["right"]),
                        _intersect(*lines["bottom"], *lines["right"]), _intersect(*lines["bottom"], *lines["left"])]) + 0.5
    angle = {}  # tilt of each border against the image axes, deg: top/bottom dy/dx, left/right dx/dy
    for side in ("top", "bottom"):
        d = lines[side][1] if lines[side][1][0] > 0 else -lines[side][1]
        angle[side] = math.degrees(math.atan2(d[1], d[0]))
    for side in ("left", "right"):
        d = lines[side][1] if lines[side][1][1] > 0 else -lines[side][1]
        angle[side] = math.degrees(math.atan2(d[0], d[1]))
    rotation = (angle["top"] + angle["bottom"]) / 2
    shear = (angle["left"] + angle["right"]) / 2 + rotation
    width = (np.linalg.norm(corners[1] - corners[0]) + np.linalg.norm(corners[2] - corners[3])) / 2 / px
    height = (np.linalg.norm(corners[3] - corners[0]) + np.linalg.norm(corners[2] - corners[1])) / 2 / px
    res.quad, res.size_mm, res.side_deg = corners, (float(width), float(height)), angle
    res.rotation_deg, res.shear_deg = rotation, shear
    tb, lr = angle["top"] - angle["bottom"], angle["left"] - angle["right"]
    if abs(tb) > p.max_parallel_deg or abs(lr) > p.max_parallel_deg:
        res.reason = f"borders not parallel (top-bottom {tb:+.2f}, left-right {lr:+.2f} deg)"
    elif abs(width - spec.width) > spec.tol or abs(height - spec.height) > spec.tol:
        res.reason = f"size {width:.1f} x {height:.1f} mm, nominal {spec.width:g} x {spec.height:g} +-{spec.tol:g}"
    else:
        res.found = True
    return res


# ---------------------------------------------------------------- round window


def _circle_search(crop: np.ndarray, origin: np.ndarray, centre: np.ndarray, px: float, spec: WindowSpec,
                   p: WindowParams) -> Tuple[np.ndarray, float]:
    """Coarse Hough search: (centre in image px, radius in mm) of the circle with the strongest
    gradient around it. ``origin``: image px of the crop's corner; ``centre``: nominal centre."""
    scale = min(1.0, p.round_px_per_mm / px)
    work = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else crop
    work = cv2.GaussianBlur(work, (0, 0), max(1.0, p.sigma_mm * px * scale))
    gx = cv2.Sobel(work, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(work, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt((gx ** 2 + gy ** 2).sum(axis=2))
    grad = np.minimum(grad, 3.0 * float(np.percentile(grad, 99)))
    px_work = px * scale
    offsets = np.arange(-spec.zone, spec.zone + 1e-6, 0.4)
    ox, oy = np.meshgrid(offsets, offsets)
    cx = (centre[0] - origin[0]) * scale + ox.ravel() * px_work  # candidate centres, work px
    cy = (centre[1] - origin[1]) * scale + oy.ravel() * px_work
    theta = np.linspace(0, 2 * np.pi, 240, endpoint=False).astype(np.float32)
    best_score, best_centre, best_radius = -1.0, centre, spec.width / 2
    for radius_mm in np.arange(spec.width / 2 - spec.tol - 1.5, spec.width / 2 + spec.tol + 1.5 + 1e-6, 0.3):
        mx = (cx[:, None] + radius_mm * px_work * np.cos(theta)[None]).astype(np.float32)
        my = (cy[:, None] + radius_mm * px_work * np.sin(theta)[None]).astype(np.float32)
        ring = cv2.remap(grad, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0).mean(axis=1)
        k = int(np.argmax(ring))
        if ring[k] > best_score:
            best_score, best_centre, best_radius = float(ring[k]), np.array([cx[k] / scale + origin[0], cy[k] / scale + origin[1]]), float(radius_mm)
    return best_centre, best_radius


def _ray_points(crop: np.ndarray, origin: np.ndarray, centre: np.ndarray, radius_mm: float, px: float,
                p: WindowParams) -> Tuple[np.ndarray, np.ndarray]:
    """Border points on rays from outside inward: (ray angles, radii in px)."""
    crop = cv2.GaussianBlur(crop, (0, 0), max(0.7, 0.5 * max(p.min_sigma_px, p.sigma_mm * px)))
    theta = np.linspace(0, 2 * np.pi, p.round_rays, endpoint=False).astype(np.float32)
    t = np.arange(radius_mm + max(p.ref_mm) + 1.0, radius_mm - max(p.near_mm[1], p.dmax_mm[1]) - 0.6, -1.0 / px)  # mm, outside first
    mx = (centre[0] - origin[0] + (t * px)[None].astype(np.float32) * np.cos(theta)[:, None]).astype(np.float32)
    my = (centre[1] - origin[1] + (t * px)[None].astype(np.float32) * np.sin(theta)[:, None]).astype(np.float32)
    profiles = cv2.remap(crop, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)  # rays, steps, 3
    ref = np.flatnonzero((t >= radius_mm + p.ref_mm[0]) & (t <= radius_mm + p.ref_mm[1]))
    near = np.flatnonzero((t <= radius_mm + p.near_mm[0]) & (t >= radius_mm - p.near_mm[1]))
    strong = np.flatnonzero((t <= radius_mm + p.dmax_mm[0]) & (t >= radius_mm - p.dmax_mm[1]))
    angles, radii = [], []
    for k in range(len(theta)):
        frame = np.median(profiles[k, ref], axis=0)
        dist = np.sqrt(((profiles[k] - frame) ** 2).sum(axis=1))
        noise = float(np.median(dist[ref]))
        pos = _step_position(dist, int(near[0]), int(near[-1]), (int(strong[0]), int(strong[-1])), p.min_contrast, noise, p)
        if pos is None:
            continue
        i = int(math.floor(pos))
        angles.append(theta[k])
        radii.append((t[i] - (pos - i) * (t[i] - t[i + 1])) * px)
    return np.array(angles), np.array(radii)


def _fit_ellipse(pts: np.ndarray, px: float):
    """Robust ellipse through ``pts``: fit, drop points far from it, repeat.
    Returns (ellipse (cx, cy, semi, semi, angle) in the points' frame, residual per point in px, inlier mask) or None."""
    inliers = np.ones(len(pts), bool)
    ellipse, residual = None, None
    for _ in range(5):
        if inliers.sum() < 20:
            break
        (ex, ey), (d1, d2), angle = cv2.fitEllipse(pts[inliers])
        ellipse = (ex, ey, d1 / 2, d2 / 2, angle)
        ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        dx, dy = pts[:, 0] - ex, pts[:, 1] - ey
        rho = np.sqrt(((dx * ca + dy * sa) / (d1 / 2)) ** 2 + ((-dx * sa + dy * ca) / (d2 / 2)) ** 2)
        residual = (rho - 1.0) * math.sqrt((d1 / 2) * (d2 / 2))
        mad = float(np.median(np.abs(residual[inliers]))) * 1.4826
        inliers = np.abs(residual) <= max(0.05 * px, 2.5 * mad)
    return None if ellipse is None else (ellipse, residual, inliers)


def _find_round(f: np.ndarray, px: float, spec: WindowSpec, margin_mm: float, p: WindowParams) -> WindowResult:
    h, w = f.shape[:2]
    res = WindowResult(False, kind="round", px_per_mm=px, nominal_mm=(spec.width, spec.height))
    nominal = np.array([(spec.left + spec.width / 2 - margin_mm) * px, (spec.top + spec.height / 2 - margin_mm) * px])
    reach = (spec.width / 2 + spec.tol + 1.5 + spec.zone + 1.0) * px
    x0, x1 = int(max(0, nominal[0] - reach)), int(min(w, nominal[0] + reach))
    y0, y1 = int(max(0, nominal[1] - reach)), int(min(h, nominal[1] + reach))
    if x1 - x0 < 20 or y1 - y0 < 20:
        res.reason = "window outside the image"
        return res
    crop, origin = f[y0:y1, x0:x1], np.array([x0, y0], float)
    centre, radius_mm = _circle_search(crop, origin, nominal, px, spec, p)
    for _ in range(2):  # the second pass puts the rays around the fitted ellipse instead of the coarse circle
        angles, radii = _ray_points(crop, origin, centre, radius_mm, px, p)
        if len(radii) < p.min_step_share * p.round_rays:
            res.reason = f"no round border ({len(radii)} of {p.round_rays} rays have a step)"
            return res
        pts = np.stack([centre[0] - x0 + radii * np.cos(angles), centre[1] - y0 + radii * np.sin(angles)], -1).astype(np.float32)
        fit = _fit_ellipse(pts, px)
        if fit is None or fit[2].sum() < 0.5 * p.round_rays:
            n_in = 0 if fit is None else int(fit[2].sum())
            res.reason = f"round border not smooth ({n_in} of {p.round_rays} rays on one ellipse)"
            return res
        ellipse, residual, inliers = fit
        centre = np.array([ellipse[0] + x0, ellipse[1] + y0])
        radius_mm = (ellipse[2] + ellipse[3]) / 2 / px
    ex, ey, a, b, angle = ellipse
    if a < b:  # long axis first, like size_mm
        a, b, angle = b, a, angle + 90.0
    rms = float(np.sqrt(np.mean(residual[inliers] ** 2))) / px
    octants = len(np.unique(np.floor(angles[inliers] / (np.pi / 4)))) / 8.0
    cx, cy, radius = ex + x0 + 0.5, ey + y0 + 0.5, (a + b) / 2
    res.rms_mm["circle"], res.inlier_share["circle"] = rms, float(inliers.sum()) / p.round_rays
    res.ellipse, res.circle = (cx, cy, a, b, angle % 180.0), (cx, cy, radius)
    res.quad = np.array([[cx - radius, cy - radius], [cx + radius, cy - radius], [cx + radius, cy + radius], [cx - radius, cy + radius]])
    res.size_mm = (2 * a / px, 2 * b / px)
    diameter = 2 * radius / px
    if octants < 1.0:
        res.reason = "round border seen over only part of its circumference"
    elif rms > p.max_rms_round_mm:
        res.reason = f"round border residual {rms:.3f} mm"
    elif res.size_mm[0] - res.size_mm[1] > p.max_axis_diff_mm:
        res.reason = f"round border is an ellipse ({res.size_mm[0]:.1f} x {res.size_mm[1]:.1f} mm)"
    elif abs(diameter - spec.width) > spec.tol:
        res.reason = f"diameter {diameter:.1f} mm, nominal {spec.width:g} +-{spec.tol:g}"
    else:
        res.found = True
    return res


# ---------------------------------------------------------------- entry


def find_window(img: np.ndarray, px_per_mm: float, fmt: PrintFormat, kind: Optional[str] = None,
                margin_mm: float = 0.0, params: WindowParams = WindowParams(),
                spec: Optional[WindowSpec] = None) -> WindowResult:
    """Find the photo window in a straightened print.

    ``img``: uint8 or uint16 RGB, as ``rectify()`` returns it (cropped by ``margin_mm`` on every
    side), at any scale. ``kind`` limits the search to ``"square"`` or ``"round"``; by default
    square is tried first, then round.

    ``px_per_mm`` must be the print's own scale: straightened image width in px divided by
    ``fmt.width_mm - 2 * margin_mm``. The nominal window sizes were measured on that basis, so
    do not pass a batch scale, whose per-scan spread (0.3 %) would eat the size tolerance. The
    format must be the right one: a Go window passes for a 600 print only if the sizes agree
    within the tolerance, so a wrong format normally ends as "not found".
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img must be an RGB image")
    f = _planes(img)
    kinds = [kind] if kind else ["square", "round"]
    first: Optional[WindowResult] = None
    notes: List[str] = []
    for k in kinds:
        sp = spec if (spec is not None and spec.kind == k) else NOMINAL.get((fmt.name, k))
        if sp is None:
            notes.append(f"{k}: no nominal window for {fmt.name}")
            continue
        r = _find_square(f, px_per_mm, sp, margin_mm, params) if k == "square" else _find_round(f, px_per_mm, sp, margin_mm, params)
        if r.found:
            r.notes = notes
            return r
        notes.append(f"{k}: {r.reason}")
        if first is None:
            first = r
    if first is None:
        return WindowResult(False, "no nominal window for this format", px_per_mm=px_per_mm, notes=notes)
    first.notes = notes
    if len(notes) > 1:
        first.reason = " | ".join(notes)
    return first
