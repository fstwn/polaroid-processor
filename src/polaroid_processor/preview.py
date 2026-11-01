"""Small straightened previews of the crop, for review grids.

A preview is the export's rectification (or unwarp along traced edges) at low resolution:
the scan is cut to the print and shrunk first, so a 500 MB scan costs a fraction of a second.
Edge-shadow correction is left out. The key of a preview says which outline, curves and margin
it shows; when the key of a manifest entry no longer matches, the preview is stale.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from . import flatten as flat
from .detection import quad_size
from .geometry import rectify
from .imageio import write_derivative

PREVIEW_DIR = "_previews"
LONG_EDGE = 600
QUALITY = 82


def preview_key(quad: np.ndarray, curves: Optional[dict], manual: bool, margin_mm: float, flatten: str,
                long_edge: int = LONG_EDGE) -> str:
    """Identity of a preview: the outline, the curves it follows (none for a moved outline or with
    flatten off), the margin and the size."""
    payload = {"quad": np.round(np.asarray(quad, float), 2).tolist(), "margin_mm": margin_mm,
               "long_edge": long_edge, "flatten": flatten}
    if flatten != "off" and curves and not manual:
        payload["curves"] = {s: [round(v, 3) for v in c["coef"]] for s, c in sorted(curves.items())}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def _scaled(curves: Dict[str, flat.SideCurve], f: float, dx: float, dy: float) -> Dict[str, flat.SideCurve]:
    """The side curves in the coordinates of a cut-out (shift) that was shrunk by ``f``."""
    shift = np.array([dx, dy])
    return {s: replace(c, p0=(c.p0 - shift) * f, length=c.length * f, coef=c.coef * f, stats={})
            for s, c in curves.items()}


def render_preview(rgb: np.ndarray, quad: np.ndarray, px_per_mm: float, curves: Optional[dict], manual: bool,
                   margin_mm: float = 0.1, flatten: str = "auto",
                   params: flat.FlattenParams = flat.FlattenParams(), long_edge: int = LONG_EDGE) -> np.ndarray:
    """The crop of ``quad`` straightened, its long edge about ``long_edge`` px (RGB)."""
    quad = np.asarray(quad, np.float64)
    w, h = quad_size(quad)
    f = min(1.0, long_edge / max(w, h))
    pad = 4 * px_per_mm  # the unwarp samples up to ~2 mm outside the outline
    H, W = rgb.shape[:2]
    x0, y0 = max(0, int(quad[:, 0].min() - pad)), max(0, int(quad[:, 1].min() - pad))
    x1, y1 = min(W, int(quad[:, 0].max() + pad) + 1), min(H, int(quad[:, 1].max() + pad) + 1)
    cut = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    size = (max(1, round(cut.shape[1] * f)), max(1, round(cut.shape[0] * f)))
    small = cv2.resize(cut, size, interpolation=cv2.INTER_AREA) if f < 1.0 else cut
    fx, fy = size[0] / cut.shape[1], size[1] / cut.shape[0]
    f = (fx + fy) / 2
    q = (quad - [x0, y0]) * [fx, fy]
    margin = margin_mm * px_per_mm * f
    sides = None
    if flatten != "off" and not manual:
        fplan = flat.plan(flat.curves_from_dict(curves), px_per_mm, params)
        sides = flat.curves_from_dict(curves) if fplan.apply else None
    if sides:
        warp = params.warp_px(px_per_mm)
        return flat.unwarp(small, _scaled(sides, f, x0, y0), q, margin, "linear",
                           fade_px=warp["fade_px"] * f, hold_px=warp["hold_px"] * f)
    return rectify(small, q, margin, "linear")


def write_preview(rgb: np.ndarray, quad: np.ndarray, px_per_mm: float, curves: Optional[dict], manual: bool,
                  settings: dict, path: Path) -> dict:
    """Render and write a preview JPEG. ``settings``: ``margin_mm``, ``flatten``, ``flatten_params``,
    ``long_edge``. Returns ``{"file", "key", "size"}``."""
    long_edge = settings.get("long_edge", LONG_EDGE)
    img = render_preview(rgb, quad, px_per_mm, curves, manual, settings["margin_mm"], settings["flatten"],
                         settings.get("flatten_params", flat.FlattenParams()), long_edge)
    write_derivative(str(path), img, "jpg", QUALITY, None)
    key = preview_key(quad, curves, manual, settings["margin_mm"], settings["flatten"], long_edge)
    return {"file": str(path), "key": key, "size": [int(img.shape[1]), int(img.shape[0])]}
