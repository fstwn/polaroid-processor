"""The item kind of a scan: what a hand crop (or a detected outline) shows.

A scan whose detection failed can be cropped by hand, and any scan can be told what it holds: ``600`` or ``go``
(an instant print of that format, which replaces the shape guess everywhere) or ``other`` (any flat item: exported
exactly as drawn, no margin, no flattening, no shadow removal, no photo crop, no format). The choice lives in
``entry["item_fix"]``, which only ``fix`` writes (the rule of ``window_fix``; the batch drops it only when the scan
file's content changed)::

    {"kind": "600" | "go" | "other", "scan_ppi": float or None (kind other only: the scan's own resolution),
     "author", "at", "note": who chose it and when}

An agent's crop on a scan without a detected outline carries its own mark, ``entry["agent_crop"]`` (see
:func:`agent_crop`), which is not part of the item.

A scan without ``item_fix`` is "as detected": its results, records and export keys are those of before this feature.
This module is pure (no image, no manifest): the checks of a hand-drawn quad and the start rectangle of the
place step, which the page, the CLI, the API and the MCP server share.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from .formats import FORMAT_OF_ITEM, PrintFormat

KINDS = ("600", "go", "other")
AUTO = "auto"                      # "as detected": no choice stored
AGENT_WARNING = "cropped by agent"
SCALE_UNVERIFIED = "scale_unverified"
MIN_SIDE_PX = 50                   # every side of a hand-drawn crop, and every corner from the lines of the sides it is not on
CORNER_NAMES = ("tl", "tr", "br", "bl")
MAX_ROTATION_DEG = 45.0            # the top side against the scan's x axis; the export keeps the scan's orientation
START_FILL = 0.8                   # the start rectangle fills at most this much of the scan
MAX_SCAN_PPI = 50000.0


def valid_item(it: Optional[dict]) -> Optional[dict]:
    """``it`` when it is a stored item choice (a dict with a known kind), else None."""
    return it if isinstance(it, dict) and it.get("kind") in KINDS else None


def item_of(entry: Optional[dict]) -> Optional[dict]:
    """The stored item choice of a manifest entry, or None ("as detected")."""
    return valid_item((entry or {}).get("item_fix"))


def kind_of(entry: Optional[dict]) -> Optional[str]:
    it = item_of(entry)
    return it["kind"] if it else None


def is_other(entry: Optional[dict]) -> bool:
    return kind_of(entry) == "other"


def forced_format(entry: Optional[dict]) -> Optional[PrintFormat]:
    """The format the user named (kind 600 or Go), else None: the shape decides."""
    return FORMAT_OF_ITEM.get(kind_of(entry) or "")


def scan_ppi_of(entry: Optional[dict]) -> Optional[float]:
    """The scan's own resolution (kind other only)."""
    it = item_of(entry)
    v = it.get("scan_ppi") if it and it["kind"] == "other" else None
    return float(v) if v else None


def agent_crop(entry: Optional[dict]) -> Optional[dict]:
    """The mark of a crop an agent placed on a scan without a detected outline (it needs a human's confirm). It is a
    field of its own (``entry["agent_crop"]``: author, at, note), independent of the item choice: a plain hand-fixed
    print can carry it too. Only ``fix`` writes it; a changed scan file drops it with the crop."""
    m = (entry or {}).get("agent_crop")
    return m if isinstance(m, dict) and m else None


def held_by_hand(entry: Optional[dict]) -> bool:
    """A crop on a scan without a detection, with or without an item kind (also a plain hand-fixed print whose later
    detection failed): later runs do not detect it again while its file is unchanged (only ``redetect`` does), and a failed
    detection leaves it ``ok``."""
    return bool((entry or {}).get("override")) and not (entry or {}).get("detection")


def check_scan_ppi(value) -> float:
    """The scan's own resolution as a number (ppi); ``ValueError`` when it is not one above 0."""
    if isinstance(value, bool):
        raise ValueError(f"scan_ppi must be a number, not {value!r}")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"scan_ppi must be a number, not {value!r}") from None
    if not math.isfinite(v) or not 0 < v <= MAX_SCAN_PPI:
        raise ValueError(f"scan_ppi must be a number above 0 and at most {MAX_SCAN_PPI:g}, not {value!r}")
    return v


def _segments_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return o1 * o2 < 0 and o3 * o4 < 0


def check_crop_quad(quad, width: float, height: float) -> np.ndarray:
    """The corners (tl, tr, br, bl) of a hand-drawn crop, as the four callers share them: finite numbers inside the
    scan (``width`` x ``height`` px), running clockwise on the scan, no crossing sides, convex, every side at least
    :data:`MIN_SIDE_PX` px, every corner at least :data:`MIN_SIDE_PX` px from the lines of the two sides it is not on
    (no sliver), the top side rotated by at most :data:`MAX_ROTATION_DEG` against the scan. Returns the quad as
    floats; ``ValueError`` with the reason otherwise."""
    q = np.asarray(quad, dtype=np.float64)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        raise ValueError("crop corners must be four finite points in the order tl, tr, br, bl")
    if (q[:, 0] < 0).any() or (q[:, 0] > width).any() or (q[:, 1] < 0).any() or (q[:, 1] > height).any():
        raise ValueError(f"crop corners must lie inside the scan (0..{width:g} x 0..{height:g} px)")
    x, y = q[:, 0], q[:, 1]
    if _segments_cross(q[0], q[1], q[2], q[3]) or _segments_cross(q[1], q[2], q[3], q[0]):
        raise ValueError("crop sides must not cross (corners in the order tl, tr, br, bl)")
    if float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) <= 0:      # image coordinates: y points down
        raise ValueError("crop corners must run clockwise on the scan in the order tl, tr, br, bl")
    for i in range(4):      # convex: every turn goes the same way (clockwise, as the area said)
        a, b, c = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
        if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) <= 0:
            raise ValueError("crop must be convex: the corner "
                             f"{CORNER_NAMES[(i + 1) % 4]} bends the wrong way")
    for i, name in enumerate(("top", "right", "bottom", "left")):
        side = float(np.linalg.norm(q[(i + 1) % 4] - q[i]))
        if side < MIN_SIDE_PX:
            raise ValueError(f"crop side {name} is {side:.0f} px, at least {MIN_SIDE_PX} px are needed")
    for i, name in enumerate(CORNER_NAMES):      # high enough in both directions: no sliver
        for j in ((i + 1) % 4, (i + 2) % 4):      # the two sides the corner is not on: (i+1, i+2) and (i+2, i+3)
            a, b = q[j], q[(j + 1) % 4]
            d = abs((b[0] - a[0]) * (q[i][1] - a[1]) - (b[1] - a[1]) * (q[i][0] - a[0])) / np.linalg.norm(b - a)
            if d < MIN_SIDE_PX:
                raise ValueError(f"crop is too thin: corner {name} is {d:.0f} px from the line of the side "
                                 f"{('top', 'right', 'bottom', 'left')[j]}, at least {MIN_SIDE_PX} px are needed")
    top = q[1] - q[0]
    angle = float(np.degrees(np.arctan2(top[1], top[0])))
    if abs(angle) > MAX_ROTATION_DEG + 1e-6:
        raise ValueError(f"crop is rotated by {angle:.1f} deg, at most {MAX_ROTATION_DEG:g} deg are allowed")
    return q


def start_rectangle(width: float, height: float, kind: Optional[str], px_per_mm: Optional[float] = None) -> np.ndarray:
    """The rectangle the place step starts from (tl, tr, br, bl, scan px): ``other`` (and an unknown kind) centred at
    :data:`START_FILL` of the scan; ``600`` / ``go`` the format's nominal size centred when a scale is known
    (``px_per_mm``; a print fills most of a flatbed scan, so it is not cut to :data:`START_FILL`, only to the scan
    itself when it would not fit), else the largest rectangle of the format's aspect ratio within :data:`START_FILL`
    of the scan."""
    fmt = FORMAT_OF_ITEM.get(kind or "")
    if fmt is None:
        w, h = START_FILL * width, START_FILL * height
    else:
        if px_per_mm and px_per_mm > 0:
            s = min(float(px_per_mm), width / fmt.width_mm, height / fmt.height_mm)
        else:
            s = min(START_FILL * width / fmt.width_mm, START_FILL * height / fmt.height_mm)
        w, h = fmt.width_mm * s, fmt.height_mm * s
    cx, cy = width / 2, height / 2
    return np.round(np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                              [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]]), 2)

