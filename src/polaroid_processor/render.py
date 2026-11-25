"""Visual checks: detection overlay and the zoomed inspect sheet for corner review."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .detection import quad_size
from .imageio import to_uint8

CORNERS = ("tl", "tr", "br", "bl")
GREEN = (40, 230, 40)
CYAN = (0, 220, 255)
MAGENTA = (255, 0, 255)
YELLOW = (255, 220, 0)
WHITE = (255, 255, 255)


def circle_of(quad: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """Centre, semi axes and angle (deg) of the ellipse inscribed in a (round window's) bounding quad."""
    q = np.asarray(quad, np.float64)
    top, left = q[1] - q[0], q[3] - q[0]
    return tuple(q.mean(axis=0)), (float(np.hypot(*top)) / 2, float(np.hypot(*left)) / 2), \
        float(np.degrees(np.arctan2(top[1], top[0])))


def _draw_circle(img: np.ndarray, quad: np.ndarray, scale: float, shift: Tuple[float, float], color, thickness: int) -> None:
    (cx, cy), (a, b), ang = circle_of(quad)
    cv2.ellipse(img, (int(round((cx - shift[0]) * scale)), int(round((cy - shift[1]) * scale))),
                (max(1, int(round(a * scale))), max(1, int(round(b * scale)))), ang, 0, 360, color, thickness,
                cv2.LINE_AA)


def overlay(rgb: np.ndarray, quad: np.ndarray, long_edge: int = 1200,
            points: Optional[Dict[str, np.ndarray]] = None, color=GREEN, ref: Optional[np.ndarray] = None,
            circle: bool = False) -> np.ndarray:
    """Downscaled scan with the quad (and optional edge points) drawn. ``ref``: a second quad (the print
    outline when ``quad`` is a photo window), drawn in green; ``circle``: also the circle inscribed in ``quad``."""
    h, w = rgb.shape[:2]
    f = long_edge / max(h, w)
    small = cv2.resize(to_uint8(rgb), (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
    small = np.ascontiguousarray(small)
    if points:
        for pts in points.values():
            for x, y in pts[::6]:
                cv2.circle(small, (int(x * f), int(y * f)), 1, YELLOW, -1)
    if ref is not None:
        cv2.polylines(small, [np.round(ref * f).astype(np.int32).reshape(-1, 1, 2)], True, GREEN, 2, cv2.LINE_AA)
    q = np.round(quad * f).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(small, [q], True, color, 2, cv2.LINE_AA)
    if circle:
        _draw_circle(small, quad, f, (0.0, 0.0), color, 2)
    for name, (x, y) in zip(CORNERS, quad * f):
        cv2.putText(small, name, (int(x) + 6, int(y) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, MAGENTA, 1, cv2.LINE_AA)
    return small


def _stretch(tile: np.ndarray) -> np.ndarray:
    """Levels and gamma so that a dark frame edge on the dark holder becomes visible."""
    lo, hi = np.percentile(tile, (1, 99.5))
    v = np.clip((tile.astype(np.float32) - lo) / max(1.0, hi - lo), 0, 1) ** 0.5
    return (v * 255).astype(np.uint8)


def is_dark(tile: np.ndarray) -> bool:
    """True when the tile is dark throughout (90th percentile of max RGB below 100)."""
    return bool(np.percentile(tile.max(axis=2), 90) < 100)


def raw_tile(rgb: np.ndarray, centre: Tuple[float, float], window: int,
             stretch: str = "auto") -> Tuple[np.ndarray, int, int, bool]:
    """Full-resolution crop of ``window`` px around ``centre``, black outside the scan.

    Returns (uint8 tile, x0, y0, stretched) where (x0, y0) is the scan coordinate of the tile's
    top-left pixel. ``stretch`` is "auto" (only tiles that are dark throughout), "on" or "off".
    """
    h, w = rgb.shape[:2]
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    x0, y0 = cx - window // 2, cy - window // 2
    tile = np.zeros((window, window, 3), np.uint8)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(w, x0 + window), min(h, y0 + window)
    if xb > xa and yb > ya:
        tile[ya - y0: yb - y0, xa - x0: xb - x0] = to_uint8(rgb[ya:yb, xa:xb])
    stretched = stretch == "on" or (stretch == "auto" and is_dark(tile))
    if stretched:
        tile = _stretch(tile)
    return tile, x0, y0, stretched


def _tile(rgb: np.ndarray, centre: Tuple[float, float], window: int, zoom: int, quad: np.ndarray,
          title: str, grid: int, stretch: str = "auto", outline: Optional[np.ndarray] = None, color=GREEN,
          ref: Optional[np.ndarray] = None, circle: bool = False) -> np.ndarray:
    """Full-resolution crop around ``centre`` with a coordinate grid and the crop edge drawn.

    The crop edge is ``outline`` (a curved, closed polyline) when given, else the quad; with an
    outline the straight quad is drawn faintly for reference. ``ref`` is a second quad drawn in green (the print
    outline, when ``quad`` is a photo window in ``color``); ``circle`` adds the circle inscribed in ``quad``."""
    tile, x0, y0, stretched = raw_tile(rgb, centre, window, stretch)
    if stretched:
        title += " stretched"
    tile = cv2.resize(tile, (window * zoom, window * zoom), interpolation=cv2.INTER_NEAREST)
    # grid in absolute scan coordinates
    for gx in range((x0 // grid + 1) * grid, x0 + window, grid):
        px = (gx - x0) * zoom
        cv2.line(tile, (px, 0), (px, tile.shape[0] - 1), (90, 90, 160), 1)
        cv2.putText(tile, str(gx), (px + 2, 12), cv2.FONT_HERSHEY_PLAIN, 0.8, (160, 160, 255), 1, cv2.LINE_AA)
    for gy in range((y0 // grid + 1) * grid, y0 + window, grid):
        py = (gy - y0) * zoom
        cv2.line(tile, (0, py), (tile.shape[1] - 1, py), (90, 90, 160), 1)
        cv2.putText(tile, str(gy), (2, py - 2), cv2.FONT_HERSHEY_PLAIN, 0.8, (160, 160, 255), 1, cv2.LINE_AA)
    q = np.round((quad - [x0, y0]) * zoom).astype(np.int32).reshape(-1, 1, 2)
    if ref is not None:
        cv2.polylines(tile, [np.round((ref - [x0, y0]) * zoom).astype(np.int32).reshape(-1, 1, 2)], True, GREEN, 1,
                      cv2.LINE_AA)
    if outline is not None:
        cv2.polylines(tile, [q], True, (110, 110, 110), 1, cv2.LINE_AA)
        o = np.round((outline - [x0, y0]) * zoom).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(tile, [o], True, GREEN, 1, cv2.LINE_AA)
    else:
        cv2.polylines(tile, [q], True, color, 1, cv2.LINE_AA)
    if circle:
        _draw_circle(tile, quad, zoom, (x0, y0), color, 1)
    for (x, y) in quad:
        px, py = int(round((x - x0) * zoom)), int(round((y - y0) * zoom))
        cv2.drawMarker(tile, (px, py), MAGENTA, cv2.MARKER_CROSS, 14, 1)
    cv2.rectangle(tile, (0, tile.shape[0] - 18), (tile.shape[1] - 1, tile.shape[0] - 1), (0, 0, 0), -1)
    cv2.putText(tile, title, (4, tile.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
    return tile


def inspect_points(quad: np.ndarray) -> List[Tuple[str, Tuple[float, float]]]:
    tl, tr, br, bl = quad
    mid = lambda a, b: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)  # noqa: E731
    return [
        ("tl", tuple(tl)), ("top", mid(tl, tr)), ("tr", tuple(tr)), ("right", mid(tr, br)),
        ("br", tuple(br)), ("bottom", mid(br, bl)), ("bl", tuple(bl)), ("left", mid(bl, tl)),
    ]


def inspect_sheet(rgb: np.ndarray, quad: np.ndarray, window: int = 360, zoom: int = 1, grid: int = 50,
                  header: str = "", stretch: str = "auto", outline: Optional[np.ndarray] = None, color=GREEN,
                  ref: Optional[np.ndarray] = None, circle: bool = False) -> np.ndarray:
    """Overview plus 8 full-resolution tiles (corners and edge midpoints), 4 per row.

    ``outline``: the curved crop edge of a flattened print (drawn instead of the quad). For a photo window
    ``quad`` is the window in ``color``, ``ref`` the print outline (green) and ``circle`` a round window."""
    tiles = []
    for name, p in inspect_points(quad):
        title = f"{name} ({p[0]:.0f}, {p[1]:.0f})"
        tiles.append(_tile(rgb, p, window, zoom, quad, title, grid, stretch, outline, color, ref, circle))
    ts = window * zoom
    rows = [np.concatenate(tiles[:4], axis=1), np.concatenate(tiles[4:], axis=1)]
    grid_img = np.concatenate([rows[0], np.full((4, ts * 4, 3), 40, np.uint8), rows[1]], axis=0)
    ov = overlay(rgb, quad, long_edge=grid_img.shape[0], points={"outline": outline} if outline is not None else None,
                 color=color, ref=ref, circle=circle)
    sheet_h = grid_img.shape[0]
    pad = np.full((sheet_h, ov.shape[1] + 8, 3), 40, np.uint8)
    pad[: ov.shape[0], 8:] = ov
    sheet = np.concatenate([grid_img, pad], axis=1)
    if header:
        bar = np.full((26, sheet.shape[1], 3), 20, np.uint8)
        cv2.putText(bar, header, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
        sheet = np.concatenate([bar, sheet], axis=0)
    return sheet


def nice_step(raw: float) -> int:
    """A round grid spacing (1, 2 or 5 times a power of ten, at least 1) close to ``raw``."""
    raw = max(1.0, float(raw))
    base = 10 ** int(np.floor(np.log10(raw)))
    for m in (1, 2, 5, 10):
        if m * base >= raw:
            return int(m * base)
    return int(10 * base)


def scan_sheet(rgb: np.ndarray, rect: np.ndarray, grid: int, header: str = "", long_edge: int = 1600) -> np.ndarray:
    """The whole scan with a pixel grid in absolute scan coordinates (a line every ``grid`` px, labelled) and
    ``rect`` (a start rectangle: cyan outline, magenta corner names) drawn: what an agent needs to place a crop on a
    scan whose detection failed. Dark scans are contrast-stretched like the review overview."""
    h, w = rgb.shape[:2]
    f = long_edge / max(h, w)
    small = cv2.resize(to_uint8(rgb), (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
    if is_dark(small):
        small = _stretch(small)
    small = np.ascontiguousarray(small)
    step_px = grid * f
    skip = max(1, int(np.ceil(46 / step_px))) if step_px > 0 else 1      # a label at least every ~46 screen px
    for i, gx in enumerate(range(grid, w, grid), 1):
        px = int(round(gx * f))
        cv2.line(small, (px, 0), (px, small.shape[0] - 1), (90, 90, 160), 1)
        if i % skip == 0:
            cv2.putText(small, str(gx), (px + 2, 12), cv2.FONT_HERSHEY_PLAIN, 0.8, (160, 160, 255), 1, cv2.LINE_AA)
    for i, gy in enumerate(range(grid, h, grid), 1):
        py = int(round(gy * f))
        cv2.line(small, (0, py), (small.shape[1] - 1, py), (90, 90, 160), 1)
        if i % skip == 0:
            cv2.putText(small, str(gy), (2, py - 2), cv2.FONT_HERSHEY_PLAIN, 0.8, (160, 160, 255), 1, cv2.LINE_AA)
    q = np.asarray(rect, np.float64)
    cv2.polylines(small, [np.round(q * f).astype(np.int32).reshape(-1, 1, 2)], True, CYAN, 2, cv2.LINE_AA)
    for name, (x, y) in zip(CORNERS, q * f):
        cv2.putText(small, name, (int(x) + 6, int(y) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, MAGENTA, 1, cv2.LINE_AA)
    if header:
        bar = np.full((26, small.shape[1], 3), 20, np.uint8)
        cv2.putText(bar, header, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
        small = np.concatenate([bar, small], axis=0)
    return small


def quad_summary(quad: np.ndarray) -> Dict[str, list]:
    w, h = quad_size(quad)
    return {"corners": {n: [round(float(x), 1), round(float(y), 1)] for n, (x, y) in zip(CORNERS, quad)},
            "size_px": [round(w, 1), round(h, 1)]}
