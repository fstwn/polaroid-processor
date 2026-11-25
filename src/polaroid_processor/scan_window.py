"""Photo window of a scan, in scan coordinates, for the manifest.

``window.py`` finds the window in a straightened print; this module cuts the print out of the scan,
runs the finder on a reduced copy, maps the result back to the scan, and tells whether a stored
window still belongs to the print outline.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .detection import quad_size
from .formats import FORMATS, PrintFormat, classify, get_format
from .geometry import homography, rectify
from .item import forced_format, is_other
from .manifest import Manifest
from .window import NOMINAL, WindowParams, find_window_for_formats


def outline_id(quad: np.ndarray) -> str:
    """Identity of a print outline: a stored window is current only while this matches the outline."""
    return hashlib.sha1(json.dumps(np.round(np.asarray(quad, float), 1).tolist()).encode()).hexdigest()[:12]


def is_manual(w: Optional[dict]) -> bool:
    return bool(w) and w.get("source") == "manual"


def window_state(w: Optional[dict]) -> str:
    """"found" (a window with a quad), "none" (the user chose no photo crop) or "missing" (no window found)."""
    if w and w.get("found") and w.get("quad"):
        return "found"
    return "none" if is_manual(w) else "missing"


def inside_outline(points: np.ndarray, quad: np.ndarray) -> bool:
    """True when every point lies inside (or on) the print outline."""
    q = np.asarray(quad, np.float32).reshape(-1, 1, 2)
    return all(cv2.pointPolygonTest(q, (float(x), float(y)), False) >= 0 for x, y in np.asarray(points, float))


def auto_valid(w: Optional[dict], quad: Optional[np.ndarray], fmt: Optional[PrintFormat] = None) -> bool:
    """True when the automatic window ``w`` was found for the print outline ``quad`` (its ``outline``) in the way the
    print's format is known now: measured with the format the user named (``fmt``, item kind 600 or Go: the window then
    carries ``named``), or, without such a format, with the format the shape gave."""
    if not w or quad is None or w.get("outline") != outline_id(quad):
        return False
    if fmt is None:
        return not w.get("named")
    return bool(w.get("named")) and w.get("format") == fmt.name


# The user's window choices live in ``entry["window_fix"]``, which only ``fix`` writes (like ``override``):
# ``{"window": hand-set window or no-photo-crop choice or None, "accepted": accept mark or None}``. The batch
# owns ``entry["window"]``, the automatic window, and never writes the other.
def fix_window(entry: dict) -> Optional[dict]:
    """The hand-set window or the no-photo-crop choice of an entry (None when the user made none)."""
    return (entry.get("window_fix") or {}).get("window")


def fix_in_force(entry: dict, quad: Optional[np.ndarray] = None) -> Optional[dict]:
    """The user's window choice when it holds for the outline: a hand-set window lying inside the outline, or the
    no-photo-crop choice (which holds for any outline). A hand-set window outside the outline is not in force."""
    w = fix_window(entry)
    if not w or is_other(entry):      # an item of kind Other has no photo window at all
        return None
    if window_state(w) == "none":
        return w
    quad = Manifest.effective_quad(entry) if quad is None else quad
    return w if quad is not None and inside_outline(np.asarray(w["quad"], float), quad) else None


def fix_outside(entry: dict, quad: Optional[np.ndarray] = None) -> bool:
    """True when the entry has a hand-set window that does not lie inside its outline (a fix of the outline,
    a refit or a new detection moved the outline away from it)."""
    w = fix_window(entry)
    return bool(w) and not is_other(entry) and window_state(w) == "found" and fix_in_force(entry, quad) is None


def effective_window(entry: dict, quad: Optional[np.ndarray] = None) -> Optional[dict]:
    """The window the export and the status use: the user's choice when it holds for the outline, else the
    automatic window (None when there is none yet; it can also be one found for an earlier outline)."""
    if is_other(entry):
        return None
    return fix_in_force(entry, quad) or entry.get("window")


def window_format(entry: dict) -> Optional[str]:
    """The format name a nominal window is placed for: the one the user named (item kind 600 or Go), else the one the
    automatic window was measured with (the shape decides when there is none)."""
    named = forced_format(entry)
    return named.name if named else (entry.get("window") or {}).get("format")


def window_current(entry: dict) -> bool:
    """True when the window in force holds for the entry's present outline: a hand-set window or the
    no-photo-crop choice, else an automatic window found for this outline."""
    quad = Manifest.effective_quad(entry)
    return fix_in_force(entry, quad) is not None or auto_valid(entry.get("window"), quad, forced_format(entry))


def window_shape(w: Optional[dict]) -> Optional[list]:
    """What an export cuts from a window: ``[kind, corners]``, or None when it cuts no photo crop (the identity of
    the photo part of an export; see ``Settings.export_key``)."""
    if w and w.get("found") and w.get("quad"):
        return [w.get("kind"), np.round(np.asarray(w["quad"], float), 1).tolist()]
    return None


def accept_valid(entry: dict, quad: Optional[np.ndarray] = None) -> bool:
    """True when the user accepted the scan for its present outline (``fix --accept``, the app's Confirm): the
    mark belongs to the outline it was given for and lapses when the outline changes."""
    acc = (entry.get("window_fix") or {}).get("accepted")
    quad = Manifest.effective_quad(entry) if quad is None else quad
    return bool(acc) and quad is not None and acc.get("outline") == outline_id(quad)


def nominal_window(quad: np.ndarray, kind: str = "square", format_name: Optional[str] = None) -> dict:
    """The nominal window of the print ``quad`` (tl, tr, br, bl in scan px): where a window of its format sits
    on a print of this size, as a window dict without provenance (``found``, ``kind``, ``quad``, ``size_mm``,
    ``format``). Square for every format, round for the 600 only (``ValueError`` otherwise). ``format_name`` is
    the format the window finder picked for the scan, when known (a 600 shot that ``classify`` takes for a Go);
    else the format the print's size gives."""
    quad = np.asarray(quad, np.float64)
    w, h = quad_size(quad)
    if w > h:
        raise ValueError("print outline is landscape")
    try:
        fmt = get_format(format_name) if format_name else classify((w, h))
    except (KeyError, ValueError):
        fmt = classify((w, h))
    spec = NOMINAL.get((fmt.name, kind))
    if spec is None:
        raise ValueError(f"no {kind} window known for {fmt.name}")
    px = w / fmt.width_mm
    box = np.array([[spec.left, spec.top], [spec.left + spec.width, spec.top],
                    [spec.left + spec.width, spec.top + spec.height], [spec.left, spec.top + spec.height]]) * px
    Hm, _ = homography(quad, 0.0)
    pts = cv2.perspectiveTransform((box - 0.5).reshape(-1, 1, 2), np.linalg.inv(Hm)).reshape(-1, 2) + 0.5
    return {"found": True, "kind": kind, "quad": np.round(pts, 2).tolist(), "format": fmt.name,
            "size_mm": [spec.width, spec.height]}


def window_size_mm(window_quad: np.ndarray, quad: np.ndarray, format_name: Optional[str] = None) -> list:
    """Width and height of a window in mm, on the basis the finder measures (print width over format width)."""
    quad = np.asarray(quad, np.float64)
    pw, ph = quad_size(quad)
    try:
        fmt = get_format(format_name) if format_name else classify((pw, ph))
    except (KeyError, ValueError):
        fmt = classify((pw, ph))
    px = pw / fmt.width_mm
    q = np.asarray(window_quad, np.float64)
    wd = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2 / px
    ht = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2 / px
    return [round(float(wd), 3), round(float(ht), 3)]


def window_for_scan(rgb: np.ndarray, quad: np.ndarray, scale: float = 0.25,
                    params: WindowParams = WindowParams(), fmt: Optional[PrintFormat] = None) -> Tuple[dict, float]:
    """The photo window of a scan, in scan coordinates, for the manifest.

    The print is cut out of the scan and shrunk by ``scale``, straightened by the plain homography of
    ``quad`` (on a flattened print the window lies where the unwarp equals it) and handed to
    :func:`find_window_for_formats`; the window quad is mapped back through the homography. A round window is
    stored as its bounding square, which a hand-set quad can replace; the circle is the one inscribed in it.
    Returns ``(window dict, seconds)``. Keys: ``found``, ``reason``, ``kind``, ``quad`` (tl, tr, br, bl, scan
    px), ``size_mm``, ``format``, ``notes``, ``outline``, ``source`` ("auto"), ``accept_needed`` (True: a window
    not found stays a review item until the scan is accepted; windows made before this key existed were
    accepted by any fix of the outline). ``fmt``: the format the user named for the print (item kind 600 or Go);
    only that format is tried, whatever the print's shape says.
    """
    t0 = time.perf_counter()
    quad = np.asarray(quad, np.float64)
    w, h = quad_size(quad)
    out = {"found": False, "reason": "", "kind": None, "format": None, "notes": [], "outline": outline_id(quad),
           "source": "auto", "accept_needed": True}
    if fmt is not None:      # only that format was tried; the window is not current any more when the choice is taken away
        out["named"] = True
    if w > h:
        out["reason"] = "print outline is landscape"
        return out, time.perf_counter() - t0
    named = fmt is not None
    fmt = fmt or classify((w, h))
    H, W = rgb.shape[:2]
    pad = 8
    x0, y0 = max(0, int(quad[:, 0].min()) - pad), max(0, int(quad[:, 1].min()) - pad)
    x1, y1 = min(W, int(quad[:, 0].max()) + pad + 1), min(H, int(quad[:, 1].max()) + pad + 1)
    cut = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    size = (max(1, round(cut.shape[1] * scale)), max(1, round(cut.shape[0] * scale)))
    small = cv2.resize(cut, size, interpolation=cv2.INTER_AREA)
    fx, fy = size[0] / cut.shape[1], size[1] / cut.shape[0]
    q_small = (quad - [x0, y0]) * [fx, fy]
    img = rectify(small, q_small, 0.0, "cubic")
    res, won = find_window_for_formats(img, fmt, formats=(fmt,) if named else FORMATS, params=params)
    out.update(found=res.found, reason=res.reason, kind=res.kind, format=won.name, notes=list(res.notes))
    if res.size_mm is not None:
        out["size_mm"] = [round(float(v), 3) for v in res.size_mm]
    if res.quad is not None:
        Hm, _ = homography(q_small, 0.0)
        pts = cv2.perspectiveTransform((res.quad - 0.5).reshape(-1, 1, 2), np.linalg.inv(Hm)).reshape(-1, 2) + 0.5
        out["quad"] = np.round(pts / [fx, fy] + [x0, y0], 2).tolist()
    return out, time.perf_counter() - t0
