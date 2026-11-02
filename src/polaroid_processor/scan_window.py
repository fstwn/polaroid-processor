"""Photo window of a scan, in scan coordinates, for the manifest.

``window.py`` finds the window in a straightened print; this module cuts the print out of the scan,
runs the finder on a reduced copy, maps the result back to the scan, and tells whether a stored
window still belongs to the print outline.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Tuple

import cv2
import numpy as np

from .detection import quad_size
from .formats import classify
from .geometry import homography, rectify
from .manifest import Manifest
from .window import WindowParams, find_window_for_formats


def outline_id(quad: np.ndarray) -> str:
    """Identity of a print outline: a stored window is current only while this matches the outline."""
    return hashlib.sha1(json.dumps(np.round(np.asarray(quad, float), 1).tolist()).encode()).hexdigest()[:12]


def window_current(entry: dict) -> bool:
    """True when the entry's stored window was computed from the entry's present outline."""
    w, quad = entry.get("window"), Manifest.effective_quad(entry)
    return bool(w) and quad is not None and w.get("outline") == outline_id(quad)


def window_for_scan(rgb: np.ndarray, quad: np.ndarray, scale: float = 0.25,
                    params: WindowParams = WindowParams()) -> Tuple[dict, float]:
    """The photo window of a scan, in scan coordinates, for the manifest.

    The print is cut out of the scan and shrunk by ``scale``, straightened by the plain homography of
    ``quad`` (on a flattened print the window lies where the unwarp equals it) and handed to
    :func:`find_window_for_formats`; the window quad is mapped back through the homography. A round window is
    stored as its bounding square, which a hand-set quad can replace; the circle is the one inscribed in it.
    Returns ``(window dict, seconds)``. Keys: ``found``, ``reason``, ``kind``, ``quad`` (tl, tr, br, bl, scan
    px), ``size_mm``, ``format``, ``notes``, ``outline``, ``source`` ("auto").
    """
    t0 = time.perf_counter()
    quad = np.asarray(quad, np.float64)
    w, h = quad_size(quad)
    out = {"found": False, "reason": "", "kind": None, "format": None, "notes": [], "outline": outline_id(quad),
           "source": "auto"}
    if w > h:
        out["reason"] = "print outline is landscape"
        return out, time.perf_counter() - t0
    fmt = classify((w, h))
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
    res, won = find_window_for_formats(img, fmt, params=params)
    out.update(found=res.found, reason=res.reason, kind=res.kind, format=won.name, notes=list(res.notes))
    if res.size_mm is not None:
        out["size_mm"] = [round(float(v), 3) for v in res.size_mm]
    if res.quad is not None:
        Hm, _ = homography(q_small, 0.0)
        pts = cv2.perspectiveTransform((res.quad - 0.5).reshape(-1, 1, 2), np.linalg.inv(Hm)).reshape(-1, 2) + 0.5
        out["quad"] = np.round(pts / [fx, fy] + [x0, y0], 2).tolist()
    return out, time.perf_counter() - t0
