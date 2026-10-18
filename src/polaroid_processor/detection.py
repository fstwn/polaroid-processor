"""Frame-colour-agnostic detection of a print lying in a black scanning holder.

Idea: the holder is near-black and flat. Every print frame (white, gold, printed,
black glitter) contains pixels brighter than the holder noise. So the detector
does not threshold brightness; it asks "is there anything at all here":

1. ``max(R, G, B)`` per pixel, max-pooled into analysis cells (factor ``f``).
2. Feature map: speck density above a threshold, or grey > 22, or saturation > 60.
3. Coarse blob: morphological open (drops the holder rim line, dust, hairs),
   largest component, close, fill.
4. Per side, scan from outside inward and accept the first position where the
   feature stays set for ``run`` cells, then refine on raw speck cells.
5. Refine each point at full resolution: 50 % crossing between holder and print level.
6. Robust (trimmed) line fit per side, moved to the middle third of the side (concave prints);
   corners are the line intersections.
7. Second pass with the threshold raised above the measured holder noise.

Coordinates are full-resolution pixels, continuous, pixel ``i`` spanning ``[i, i+1)``.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

SIDES = ("top", "right", "bottom", "left")


@dataclass(frozen=True)
class DetectParams:
    analysis_long_edge: int = 3600  # analysis image size; sets the cell factor
    thresholds: Tuple[int, ...] = (10, 20, 35, 50, 70)  # speck thresholds tried in order
    noise_margin: int = 6  # second pass threshold = holder p99.5 + margin
    noise_gap: float = 0.015  # holder noise is sampled this far (share of scan size) beyond the outline
    density: float = 0.02  # blurred pixel share of specks for "frame present" (blob only)
    density_window: int = 15
    grey_min: int = 22
    sat_min: int = 60
    sat_value_min: int = 40  # saturation only counts on pixels at least this bright (dark noise is "saturated")
    open_cells: int = 31
    close_cells: int = 41
    run: int = 40  # sustained feature run (cells) that marks the print edge
    search_out: int = 80
    search_in: int = 200
    segment: int = 240  # edge segment length (cells) whose rows/columns are averaged into one profile
    segment_step: int = 120
    plateau: int = 40  # cells inside the edge that must form a stable plateau
    before: int = 12  # cells outside the edge compared against the plateau
    contrast: float = 3.0  # plateau median >= contrast * outside level + contrast_offset
    contrast_offset: float = 0.02
    plateau_min: float = 0.02  # pixel share; sparse glitter frames sit around 0.03-0.3
    start_frac: float = 0.5  # the edge cell itself reaches this fraction of the plateau
    gap_frac: float = 0.4  # plateau cells below this fraction of its median count as gaps
    gap_max: float = 0.15  # allowed share of gap cells inside the plateau
    dip_cells: int = 30  # no cell this close to the edge may fall below dip_frac of the plateau
    dip_frac: float = 0.25
    spike_ratio: float = 4.0  # a sparse plateau may not start with a spike this much brighter (holder lip)
    spike_cells: int = 4
    extend_max: int = 10  # extend the edge outward over contiguous bright cells (edge highlight)
    # full-resolution refinement of each cell edge point: 50 % crossing between holder and print
    refine_out: int = 6  # cells outside the cell point that give the holder level (darkest value)
    refine_in: Tuple[int, int] = (3, 8)  # cells inside the cell point that give the print level (median)
    refine_frac: float = 0.5  # crossing level between holder and print
    refine_contrast: float = 20.0  # min print - holder level (max RGB, 8 bit); below: speck density profile
    refine_density: float = 0.02  # min density contrast (pixel share); below: keep the cell point
    ahead_window: int = 8  # a sparse plateau followed this soon by a bright one is a holder gap
    ahead_ratio: float = 6.0
    corner_skip: float = 0.08  # fraction of each side ignored near the corners
    end_zone: float = 0.025  # end zones span corner_skip down to this fraction; sampled to measure flare
    trim_keep: float = 0.85  # share of consensus inliers kept for the final line fit
    line_tol_cells: float = 4.0  # consensus tolerance in cells
    rim_cells: int = 30  # an inner edge candidate lies at most this far inside the outer one (holder rim band)
    parallel_tol_deg: float = 0.25  # opposite sides of a print are parallel within this
    border_margin: float = 0.001  # quad must stay this far (fraction) from the scan border


@dataclass
class EdgeStats:
    n_points: int
    n_inliers: int
    mad_px: float  # median absolute residual of inlier edge points to the fitted line
    bow_px: float  # middle minus ends of inliers; positive = edge bulges outward
    max_px: float  # largest inlier residual
    # a second consistent line behind the edge (holder rim parallel to the print), if any:
    inner_px: Optional[float] = None  # its distance inside the fitted edge
    inner_share: float = 0.0  # share of segments that support it
    # where the print runs relative to the line near its two ends (top/left end first),
    # outward positive; a curled or bent corner shows up here
    flare_start_px: Optional[float] = None
    flare_end_px: Optional[float] = None
    # how far a run of edge points (3 consecutive segments, outliers included) lies inside
    # the outline; holder shows in the crop there when this exceeds the crop margin
    inside_px: Optional[float] = None

    def to_dict(self) -> dict:
        return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in asdict(self).items()}


@dataclass
class Detection:
    quad: np.ndarray  # (4, 2) float64, order tl, tr, br, bl
    scan_size: Tuple[int, int]  # (width, height)
    size_px: Tuple[float, float]  # mean (width, height) of the quad
    skew_deg: float
    edges: Dict[str, EdgeStats]
    speck_threshold: int
    holder_noise: Optional[float]
    factor: int
    time_s: float
    source: str = "auto"  # "auto" or "manual"
    repaired: Tuple[str, ...] = ()  # sides refit on inner candidates (holder rim touching the print)

    @property
    def aspect(self) -> float:
        return self.size_px[0] / self.size_px[1]

    def to_dict(self) -> dict:
        return {
            "quad": [[round(float(x), 2), round(float(y), 2)] for x, y in self.quad],
            "scan_size": list(self.scan_size),
            "size_px": [round(self.size_px[0], 1), round(self.size_px[1], 1)],
            "aspect": round(self.aspect, 4),
            "skew_deg": round(self.skew_deg, 3),
            "edges": {k: v.to_dict() for k, v in self.edges.items()},
            "speck_threshold": self.speck_threshold,
            "holder_noise": None if self.holder_noise is None else round(self.holder_noise, 1),
            "factor": self.factor,
            "time_s": round(self.time_s, 2),
            "source": self.source,
            "repaired": list(self.repaired),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Detection":
        return cls(
            quad=np.array(d["quad"], dtype=np.float64),
            scan_size=tuple(d["scan_size"]),
            size_px=tuple(d["size_px"]),
            skew_deg=d["skew_deg"],
            edges={k: EdgeStats(**v) for k, v in d.get("edges", {}).items()},
            speck_threshold=d.get("speck_threshold", 0),
            holder_noise=d.get("holder_noise"),
            factor=d.get("factor", 0),
            time_s=d.get("time_s", 0.0),
            source=d.get("source", "auto"),
            repaired=tuple(d.get("repaired", ())),
        )


@dataclass
class DetectionDebug:
    """Analysis-scale images and edge points, for overlays."""

    rgb_small: np.ndarray
    feature: np.ndarray
    points: Dict[str, np.ndarray] = field(default_factory=dict)  # full-res coordinates


class DetectionError(RuntimeError):
    pass


def _kernel(n: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))


def _max_rgb(rgb: np.ndarray) -> np.ndarray:
    m = np.maximum(np.maximum(rgb[..., 0], rgb[..., 1]), rgb[..., 2])
    if m.dtype == np.uint16:
        m = (m >> 8).astype(np.uint8)
    return m


def _pool(img: np.ndarray, f: int, how: str) -> np.ndarray:
    """Pool ``f`` x ``f`` cells: "max" for single-channel uint8, otherwise area mean."""
    h, w = img.shape[:2]
    hh, ww = h // f, w // f
    v = img[: hh * f, : ww * f]
    if how == "max":
        if f == 1:
            return v.copy()
        # dilate then pick one pixel per cell: same result as a block max, much faster
        d = cv2.dilate(v, np.ones((f, f), np.uint8), anchor=(0, 0))
        return np.ascontiguousarray(d[::f, ::f][:hh, :ww])
    return cv2.resize(v, (ww, hh), interpolation=cv2.INTER_AREA)


def _first_run(profile: np.ndarray, run: int, need: Optional[int] = None) -> int:
    """First index whose window of ``run`` cells holds at least ``need`` set cells (default: all)."""
    p = (profile > 0).astype(np.int32)
    if len(p) < run:
        return -1
    cs = np.concatenate(([0], np.cumsum(p)))
    sums = cs[run:] - cs[:-run]
    idx = np.flatnonzero(sums >= (run if need is None else need))
    return int(idx[0]) if idx.size else -1


def _fit_line(pts: np.ndarray):
    c = pts.mean(axis=0)
    d = np.linalg.svd(pts - c, full_matrices=False)[2][0]
    return c, d


def _robust_line(pts: np.ndarray, tol: float, keep: float):
    """Consensus line: the point pair with the most inliers within ``tol``, refit on inliers,
    then a trimmed least-squares pass. Returns (centre, direction, inlier mask)."""
    pts = pts.astype(np.float64)
    n = len(pts)
    best = None
    for i in range(n):
        for j in range(i + 1, n):
            d = pts[j] - pts[i]
            norm = np.hypot(*d)
            if norm < 1e-9:
                continue
            nrm = np.array([-d[1], d[0]]) / norm
            inl = np.abs((pts - pts[i]) @ nrm) <= tol
            cnt = int(inl.sum())
            if best is None or cnt > best[0]:
                best = (cnt, inl)
    inl = best[1] if best is not None else np.ones(n, bool)
    c, d = _fit_line(pts[inl])
    nrm = np.array([-d[1], d[0]])
    inl = np.abs((pts - c) @ nrm) <= tol
    sub = pts[inl]
    res = np.abs((sub - c) @ nrm)
    keep_idx = np.argsort(res)[: max(2, int(round(len(sub) * keep)))]
    c, d = _fit_line(sub[keep_idx])
    nrm = np.array([-d[1], d[0]])
    return c, d, np.abs((pts - c) @ nrm) <= tol


def _intersect(l1, l2) -> np.ndarray:
    (c1, d1), (c2, d2) = l1, l2
    a = np.array([d1, -d2]).T
    t = np.linalg.solve(a, c2 - c1)
    return c1 + t[0] * d1


def _corners(lines) -> np.ndarray:
    """Quad (tl, tr, br, bl) from the four side lines."""
    return np.array([
        _intersect(lines["top"], lines["left"]),
        _intersect(lines["top"], lines["right"]),
        _intersect(lines["bottom"], lines["right"]),
        _intersect(lines["bottom"], lines["left"]),
    ])


def _angle_diff_deg(l1, l2) -> float:
    """Unsigned angle between two lines in degrees, in [0, 90]."""
    (_, d1), (_, d2) = l1, l2
    c = abs(float(np.dot(d1, d2))) / (np.hypot(*d1) * np.hypot(*d2))
    return float(np.degrees(np.arccos(min(1.0, c))))


def _consensus_offset(pts: np.ndarray, d: np.ndarray, tol: float) -> np.ndarray:
    """Inlier mask of the densest band of width ``2 * tol`` along the normal of direction ``d``."""
    nrm = np.array([-d[1], d[0]]) / np.hypot(*d)
    off = pts @ nrm
    so = np.sort(off)
    hi = np.searchsorted(so, so + 2 * tol, side="right")
    i = int(np.argmax(hi - np.arange(len(so))))
    centre = np.median(so[i: hi[i]])
    return np.abs(off - centre) <= tol


class _Analysis:
    def __init__(self, rgb: np.ndarray, params: DetectParams):
        self.p = params
        h, w = rgb.shape[:2]
        self.scan_size = (w, h)
        self.f = max(1, int(round(max(h, w) / params.analysis_long_edge)))
        self.mx = _max_rgb(rgb)
        self.mx_cells = _pool(self.mx, self.f, "max")
        self._density: Dict[int, np.ndarray] = {}
        small = _pool(rgb, self.f, "area")
        self.rgb_small = small if small.dtype == np.uint8 else (small >> 8).astype(np.uint8)
        self.grey = cv2.cvtColor(self.rgb_small, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(self.rgb_small, cv2.COLOR_RGB2HSV)
        self.colour = (hsv[..., 1] > params.sat_min) & (hsv[..., 2] >= params.sat_value_min)

    def feature(self, thr: int):
        """Coarse "anything but holder" map for the blob: blurred pixel density or solid frame."""
        dens = cv2.blur(self.density(thr), (self.p.density_window,) * 2)
        feat = (dens > self.p.density) | (self.grey > self.p.grey_min) | self.colour
        return feat.astype(np.uint8) * 255, None

    def blob(self, feat: np.ndarray) -> Optional[np.ndarray]:
        m = cv2.morphologyEx(feat, cv2.MORPH_OPEN, _kernel(self.p.open_cells))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
        if n < 2:
            return None
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        blob = (lab == idx).astype(np.uint8) * 255
        blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, _kernel(self.p.close_cells))
        cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(blob)
        cv2.drawContours(filled, cnts, -1, 255, cv2.FILLED)
        return filled

    def density(self, thr: int) -> np.ndarray:
        """Share of full-resolution pixels per cell brighter than ``thr`` (not max-pooled).

        Max-pooling would turn 1.6 % dust pixels into 23 % "occupied" cells; the true
        pixel share keeps dust (~0.02) and glitter frames (~0.1) apart.
        """
        if thr not in self._density:
            f = self.f
            h, w = self.mx_cells.shape
            _, m = cv2.threshold(self.mx[: h * f, : w * f], thr, 255, cv2.THRESH_BINARY)
            d = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA) if f > 1 else m
            self._density[thr] = d.astype(np.float32) / 255.0
        return self._density[thr]

    def raw(self, thr: int) -> np.ndarray:
        solid = ((self.grey > self.p.grey_min) | self.colour).astype(np.float32)
        return np.maximum(self.density(thr), solid)

    def _brighter_ahead(self, prof: np.ndarray, k: int, med: float) -> bool:
        p = self.p
        ahead = prof[k: k + p.plateau + p.ahead_window]
        if len(ahead) < p.ahead_window:
            return False
        win = np.lib.stride_tricks.sliding_window_view(ahead, p.ahead_window)
        return bool((np.median(win, axis=1) >= max(0.5, p.ahead_ratio * med)).any())

    def inner_edge(self, prof: np.ndarray, k1: int) -> Optional[int]:
        """Second edge candidate behind a dark seam within ``rim_cells`` of the outer one, or None.

        A holder rim band can touch the print with only a thin seam between them; the outer
        candidate then sits on the rim. Used only when the outer line is not parallel to its
        opposite side.
        """
        p = self.p
        med = float(np.median(prof[k1: k1 + p.plateau]))
        win = prof[k1 + 1: k1 + p.rim_cells]
        if len(win) < 3:
            return None
        start = k1 + 1 + int(np.argmin(win))  # the seam can be a single cell wide
        if prof[start] >= p.start_frac * med:
            return None
        inner = prof[start: start + p.plateau]
        med2 = float(np.median(inner))
        if med2 < p.plateau_min:
            return None
        rise = np.nonzero(inner >= p.start_frac * med2)[0]
        if not rise.size:
            return None
        k2 = start + int(rise[0])
        if k2 > k1 + p.rim_cells or prof[k2: k2 + p.dip_cells].min() < p.dip_frac * med2:
            return None
        return k2

    def profile_edge(self, prof: np.ndarray) -> Optional[int]:
        """Index of the print edge in a profile ordered from outside to inside, or None.

        The edge is the outermost step into a stable plateau that is clearly brighter
        than what lies just outside. Rejected candidates:

        - holder rim lines: a dark gap follows them (gap share / dip test);
        - dusty or flare-lit ramps: the step is not sharp (start test, contrast test);
        - the bright holder lip in front of a dim plateau (spike test).
        """
        p = self.p
        n = len(prof)
        if n < p.before + p.plateau:
            return None
        for k in range(p.before, n - p.plateau):
            v = prof[k]
            if v < p.plateau_min * p.start_frac:
                continue
            seg = prof[k: k + p.plateau]
            med = float(np.median(seg))
            if med < p.plateau_min or v < p.start_frac * med:
                continue
            out = prof[k - p.before: k]
            outside = float(np.convolve(out, np.ones(2) / 2, mode="valid").min())  # darkest seam just outside
            if med < p.contrast * outside + p.contrast_offset:
                continue
            if np.mean(seg < p.gap_frac * med) > p.gap_max:
                continue
            if seg[: p.dip_cells].min() < p.dip_frac * med:
                continue
            if med < 0.5 and seg[: p.spike_cells].max() > p.spike_ratio * med:
                continue
            if med < 0.5 and self._brighter_ahead(prof, k, med):
                continue  # a dim strip right in front of a bright print is a holder gap, not a frame
            j = k
            while j > k - p.extend_max and j - 1 >= 0 and prof[j - 1] >= p.start_frac * med:
                j -= 1
            return j
        return None

    def refine(self, pos: int, a: int, b: int, rows: bool, near: bool, thr: int) -> float:
        """Full-resolution edge position near the cell boundary ``pos`` (full-res pixels).

        Cell points sit on the outer boundary of the first bright cell, and the speck threshold
        fires early in the soft print edge, so they lie outward of the visible edge. This
        averages ``max(R, G, B)`` across the edge over the segment's full-res rows (or columns)
        ``a * f .. b * f`` and returns the ``refine_frac`` crossing between the holder level just
        outside and the print level just inside, interpolated between pixel centres. A sparse
        glitter frame gives too little mean contrast; the share of pixels above ``thr`` is used
        instead, and ``pos`` itself when that is flat too.
        """
        p, f = self.p, self.f
        wo, ia, ib = p.refine_out * f, p.refine_in[0] * f, p.refine_in[1] * f
        h, w = self.mx.shape
        lim = w if rows else h
        lo, hi = (pos - wo, pos + ib) if near else (pos - ib, pos + wo)
        if lo < 0 or hi > lim:
            return float(pos)
        win = self.mx[a * f: b * f, lo:hi] if rows else self.mx[lo:hi, a * f: b * f].T
        if not near:
            win = win[:, ::-1]  # outside first

        def crossing(prof: np.ndarray, min_contrast: float) -> Optional[float]:
            level0 = float(prof[:wo].min())
            level1 = float(np.median(prof[wo + ia: wo + ib]))
            if level1 - level0 < min_contrast:
                return None
            half = level0 + p.refine_frac * (level1 - level0)
            below = np.flatnonzero(prof[: wo + ia] < half)
            if not below.size:
                return None
            j = int(below[-1]) + 1  # prof[j - 1] < half <= prof[j], all >= half from j to wo + ia
            return j - 0.5 + (half - prof[j - 1]) / max(1e-9, float(prof[j] - prof[j - 1]))

        t = crossing(win.mean(axis=0), p.refine_contrast)
        if t is None:
            t = crossing((win > thr).mean(axis=0), p.refine_density)
        if t is None:
            return float(pos)
        return lo + t if near else hi - t

    def edge_points(self, blob: np.ndarray, thr: int):
        """Edge points per side, in full-res coordinates.

        Each segment averages the raw feature over ``segment`` rows (or columns) along the
        edge, which makes sparse glitter frames, dust and the holder rim separable.
        Each point is then refined at full resolution (:meth:`refine`). Returns ``(points, inner, ends)``:
        ``inner`` holds each point's inner candidate (NaN where there is none); ``ends`` holds
        points from the corner zones, not used for fitting, as ``(x, y, end, x_inner, y_inner)``
        with end 0 for the top/left end of the side and 1 for the other.
        """
        p, f = self.p, self.f
        raw = self.raw(thr)
        h, w = blob.shape
        ys, xs = np.nonzero(blob)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        my, mx = int((y1 - y0) * p.corner_skip), int((x1 - x0) * p.corner_skip)
        ey, ex = int((y1 - y0) * p.end_zone), int((x1 - x0) * p.end_zone)
        pts: Dict[str, List[Tuple[float, float]]] = {s: [] for s in SIDES}
        alt: Dict[str, List[Tuple[float, float]]] = {s: [] for s in SIDES}
        ends: Dict[str, List[Tuple[float, ...]]] = {s: [] for s in SIDES}
        seg_len = min(p.segment, max(8, (y1 - y0 - 2 * my) // 4), max(8, (x1 - x0 - 2 * mx) // 4))
        step = max(4, min(p.segment_step, seg_len // 2))

        def spans(a: int, b: int):
            for s0 in range(a, max(a + 1, b - seg_len + 1), step):
                yield s0, min(b, s0 + seg_len)

        def halves(a: int, b: int):
            m = (a + b) // 2
            return [(a, m), (m, b)] if m - a >= 4 else []

        def scan(a: int, b: int, rows: bool, end: Optional[int] = None):
            """Edge points of the two sides crossing rows (or columns) a..b."""
            strip = blob[a:b] if rows else blob[:, a:b].T
            has = strip.any(axis=1)
            if not has.any():
                return
            n = strip.shape[1]
            first = np.argmax(strip > 0, axis=1)[has]
            last = n - 1 - np.argmax(strip[:, ::-1] > 0, axis=1)[has]
            prof = raw[a:b].mean(axis=0) if rows else raw[:, a:b].mean(axis=1)
            c = (a + b) / 2 * f
            near, far = ("left", "right") if rows else ("top", "bottom")
            pos = (lambda v: (v, c)) if rows else (lambda v: (c, v))
            lo = max(0, int(first.min()) - p.search_out)
            hi = min(n, int(last.max()) + p.search_out + 1)
            for side, pr, to_px, is_near in (
                (near, prof[lo: int(first.max()) + p.search_in], lambda k: (lo + k) * f, True),
                (far, prof[max(0, int(last.min()) - p.search_in): hi][::-1], lambda k: (hi - k) * f, False),
            ):
                k = self.profile_edge(pr)
                if k is None:
                    continue
                k2 = self.inner_edge(pr, k)
                point = pos(self.refine(to_px(k), a, b, rows, is_near, thr))
                second = (np.nan, np.nan) if k2 is None else pos(self.refine(to_px(k2), a, b, rows, is_near, thr))
                if end is not None:
                    ends[side].append((*point, end, *second))
                else:
                    pts[side].append(point)
                    alt[side].append(second)

        for ra, rb in spans(y0 + my, y1 - my):
            scan(ra, rb, True)
        for ca, cb in spans(x0 + mx, x1 - mx):
            scan(ca, cb, False)
        for (a, b), e in [((y0 + ey, y0 + my), 0), ((y1 - my, y1 - ey), 1)]:
            for ra, rb in halves(a, b):
                scan(ra, rb, True, e)
        for (a, b), e in [((x0 + ex, x0 + mx), 0), ((x1 - mx, x1 - ex), 1)]:
            for ca, cb in halves(a, b):
                scan(ca, cb, False, e)

        def as_arrays(d, cols):
            return {k: np.array(v, dtype=np.float64).reshape(-1, cols) for k, v in d.items()}

        return as_arrays(pts, 2), as_arrays(alt, 2), as_arrays(ends, 5)

    def fit(self, pts: Dict[str, np.ndarray], inner: Optional[Dict[str, np.ndarray]] = None,
            end_pts: Optional[Dict[str, np.ndarray]] = None):
        """Fit four lines and intersect them. Returns ``(quad, edges, points, repaired)``.

        Each side's outline is its robust line moved to the median of the middle third of its
        inlier points; edge statistics are relative to the robust line.

        When opposite sides are not parallel and one of them has inner candidates, that
        side is refit on the candidates that line up parallel to its opposite side.
        """
        tol = self.p.line_tol_cells * self.f
        pts = dict(pts)
        lines, inliers = {}, {}
        for s in SIDES:
            if len(pts[s]) < 4:
                raise DetectionError(f"too few edge points on the {s} side ({len(pts[s])})")
            c, d, inl = _robust_line(pts[s], tol, self.p.trim_keep)
            if inl.sum() < max(3, 0.4 * len(pts[s])):
                raise DetectionError(f"no consistent edge on the {s} side")
            lines[s], inliers[s] = (c, d), inl
        repaired = []
        for pair in (("top", "bottom"), ("left", "right")) if inner else ():
            diff = _angle_diff_deg(lines[pair[0]], lines[pair[1]])
            if diff <= self.p.parallel_tol_deg:
                continue
            best = None
            for s, o in (pair, pair[::-1]):
                alt = inner[s]
                have = ~np.isnan(alt[:, 0])
                if have.sum() < 3:
                    continue
                union = np.concatenate([pts[s], alt[have]])
                band = _consensus_offset(union, lines[o][1], tol)
                if band.sum() < 3:
                    continue
                c, d, _ = _robust_line(union[band], tol, self.p.trim_keep)
                nrm = np.array([-d[1], d[0]])
                alt0 = np.where(have[:, None], alt, 0.0)
                choose = have & (np.abs((alt0 - c) @ nrm) < np.abs((pts[s] - c) @ nrm))
                chosen = np.where(choose[:, None], alt0, pts[s])
                inl = np.abs((chosen - c) @ nrm) <= tol
                nd = _angle_diff_deg((c, d), lines[o])
                if inl.sum() >= max(3, 0.4 * len(chosen)) and nd < diff and (best is None or nd < best[0]):
                    best = (nd, s, (c, d), inl, chosen)
            if best is not None and (best[0] <= self.p.parallel_tol_deg or best[0] < 0.5 * diff):
                _, s, line, inl, chosen = best
                lines[s], inliers[s], pts[s] = line, inl, chosen
                repaired.append(s)
        centre = _corners(lines).mean(axis=0)
        edges, outline = {}, {}
        for s in SIDES:
            c, d = lines[s]
            n = np.array([-d[1], d[0]])
            if np.dot(c - centre, n) < 0:
                n = -n  # outward normal
            res = (pts[s][inliers[s]] - c) @ n
            k = len(res)
            third, sixth = max(1, k // 3), max(1, k // 6)
            mid = np.median(res[third: max(third + 1, 2 * k // 3)])
            ends = np.median(np.concatenate([res[:sixth], res[-sixth:]]))
            st = EdgeStats(len(pts[s]), k, float(np.median(np.abs(res))), float(mid - ends),
                           float(np.abs(res).max()))
            # the outline runs through the middle third of the side: prints are often concave,
            # and the fitted line then sits outward of the edge middle (statistics stay on it)
            outline[s] = (c + mid * n, d)
            inward = mid - (pts[s] - c) @ n  # all points, in side order; positive = inside the outline
            if len(inward) >= 3:
                runs = np.lib.stride_tricks.sliding_window_view(inward, 3).min(axis=1)
                st.inside_px = float(max(0.0, runs.max()))
            alt = inner[s][~np.isnan(inner[s][:, 0])] if inner else np.zeros((0, 2))
            if len(alt) >= 3:
                band = _consensus_offset(alt, d, tol)
                depth = -float(np.median((alt[band] - c) @ n))
                if band.sum() >= 0.4 * len(pts[s]) and depth > tol:
                    st.inner_px, st.inner_share = depth, float(band.sum() / len(pts[s]))
            e = end_pts[s] if end_pts else np.zeros((0, 5))
            for which, attr in ((0, "flare_start_px"), (1, "flare_end_px")):
                sel = e[e[:, 2] == which]
                if len(sel):
                    # outer or inner candidate, whichever is closer: the holder rim curves close
                    # to the print at the corners
                    res = (sel[:, :2] - c) @ n
                    res2 = np.where(np.isnan(sel[:, 3]), np.inf, (np.nan_to_num(sel[:, 3:5]) - c) @ n)
                    res = np.where(np.abs(res2) < np.abs(res), res2, res)
                    setattr(st, attr, float(res[np.argmax(np.abs(res))]))
            edges[s] = st
        return _corners(outline), edges, pts, tuple(repaired)

    def plausible(self, quad: np.ndarray) -> bool:
        w, h = self.scan_size
        m = self.p.border_margin * min(w, h)
        if quad[:, 0].min() < m or quad[:, 1].min() < m or quad[:, 0].max() > w - m or quad[:, 1].max() > h - m:
            return False
        wq, hq = quad_size(quad)
        return 0.5 < wq / hq < 2.0

    def attempt(self, thr: int):
        feat, _ = self.feature(thr)
        blob = self.blob(feat)
        if blob is None:
            return None
        try:
            pts, inner, ends = self.edge_points(blob, thr)
            quad, edges, pts, repaired = self.fit(pts, inner, ends)
        except DetectionError:
            return None
        if not self.plausible(quad):
            return None
        return quad, edges, pts, feat, repaired

    def holder_noise(self, quad: np.ndarray) -> Optional[float]:
        """99.5th percentile of max(R,G,B) on the holder, sampled beyond ``noise_gap`` of the outline.

        The gap skips the holder lip, which sits within ~1.2 % of the print size.
        """
        h, w = self.mx_cells.shape
        outside = np.ones((h, w), np.uint8)
        cv2.fillPoly(outside, [np.round(quad / self.f).astype(np.int32).reshape(-1, 1, 2)], 0)
        grow = max(1, int(self.p.noise_gap * min(h, w)))
        outside = cv2.erode(outside, _kernel(2 * grow + 1))
        vals = self.mx_cells[outside > 0]
        return float(np.percentile(vals, 99.5)) if vals.size >= 500 else None


def quad_size(quad: np.ndarray) -> Tuple[float, float]:
    w = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    h = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
    return float(w), float(h)


def quad_skew_deg(quad: np.ndarray) -> float:
    v = quad[1] - quad[0]
    return float(np.degrees(np.arctan2(v[1], v[0])))


def detect(rgb: np.ndarray, params: DetectParams = DetectParams(), debug: bool = False):
    """Detect the print outline in a full-resolution RGB scan.

    Returns a :class:`Detection`, or ``(Detection, DetectionDebug)`` when ``debug`` is set.
    Raises :class:`DetectionError` when no plausible outline is found.
    """
    t0 = time.perf_counter()
    a = _Analysis(rgb, params)
    found, thr = None, None
    for t in params.thresholds:
        found = a.attempt(t)
        if found is not None:
            thr = t
            break
    if found is None:
        raise DetectionError("no plausible print outline at any threshold")
    noise = a.holder_noise(found[0])
    if noise is not None:
        thr2 = int(noise + params.noise_margin)
        if thr2 > thr:
            again = a.attempt(thr2)
            # a much smaller outline means the raised threshold ate into a dark frame: keep pass one
            if again is not None and np.prod(quad_size(again[0])) >= 0.9 * np.prod(quad_size(found[0])):
                found, thr = again, thr2
    quad, edges, pts, feat, repaired = found
    det = Detection(
        quad=quad,
        scan_size=a.scan_size,
        size_px=quad_size(quad),
        skew_deg=quad_skew_deg(quad),
        edges=edges,
        speck_threshold=int(thr),
        holder_noise=noise,
        factor=a.f,
        time_s=time.perf_counter() - t0,
        repaired=repaired,
    )
    if debug:
        return det, DetectionDebug(a.rgb_small, feat, pts)
    return det


def manual_detection(quad: np.ndarray, scan_size: Tuple[int, int]) -> Detection:
    """Detection object for a hand-set quad (no edge statistics)."""
    quad = np.asarray(quad, dtype=np.float64)
    return Detection(quad, tuple(scan_size), quad_size(quad), quad_skew_deg(quad), {}, 0, None, 0, 0.0, "manual")
