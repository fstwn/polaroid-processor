"""The settings panel's view of a project's settings.

The panel edits a flat set of fields; the project stores them as the manifest's ``settings`` (the dict of
``pipeline.settings_to_dict``), so the CLI, MCP and the app read the same thing. Derivatives other than
``high`` and ``web`` (made on the command line) are kept as they are.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .. import api
from ..pipeline import DEFAULT_DERIVATIVES, PHOTO_CROP, Derivative, settings_from_dict, settings_to_dict
from ..scale import explicit_scale

HIGH_WEB = {d.name: d for d in DEFAULT_DERIVATIVES}
CHOICES = {"flatten": ("auto", "off"), "shadows": ("auto", "off"), "tiff_compression": ("zlib", "none"),
           "interpolation": ("linear", "cubic", "lanczos"), "photo_crop": PHOTO_CROP}
FIELDS = ("margin_mm", "photo_crop", "photo_bleed_mm", "flatten", "shadows", "high_on", "high_edge", "high_quality", "web_on", "web_edge",
          "web_quality", "ppi", "profile", "tiff_compression", "interpolation", "overlay")


def to_panel(d: Optional[dict]) -> Dict[str, Any]:
    """The flat fields of the panel for a manifest settings dict (missing keys: the tool's defaults)."""
    s = settings_from_dict(d or {})
    by_name = {x.name: x for x in s.derivatives}
    out: Dict[str, Any] = {"margin_mm": s.margin_mm, "photo_crop": s.photo_crop, "photo_bleed_mm": s.photo_bleed_mm,
                           "flatten": s.flatten, "shadows": s.shadows, "ppi": s.ppi,
                           "profile": s.profile, "tiff_compression": s.tiff_compression,
                           "interpolation": s.interpolation, "overlay": s.overlay,
                           "other_derivatives": sum(1 for x in s.derivatives if x.name not in HIGH_WEB)}
    for name, default in HIGH_WEB.items():
        x = by_name.get(name)
        out[f"{name}_on"] = x is not None
        out[f"{name}_edge"] = (x or default).long_edge
        out[f"{name}_quality"] = (x or default).quality
    return out


def _number(v: Any, what: str, lo: float, hi: float, integer: bool = False):
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise api.UsageError(f"{what} must be a number") from None
    if not lo <= x <= hi:
        raise api.UsageError(f"{what} must be between {lo:g} and {hi:g}")
    return int(round(x)) if integer else x


def _flag(v: Any, what: str) -> bool:
    if not isinstance(v, bool):
        raise api.UsageError(f"{what} must be true or false")
    return v


def from_panel(current: Optional[dict], update: Dict[str, Any]) -> dict:
    """``current`` (a manifest settings dict) with the panel fields in ``update`` applied and checked; the
    new settings dict. Unknown fields are refused."""
    unknown = set(update) - set(FIELDS)
    if unknown:
        raise api.UsageError(f"unknown setting: {', '.join(sorted(unknown))}")
    s = settings_from_dict(current or {})
    if "margin_mm" in update:
        s.margin_mm = _number(update["margin_mm"], "margin", 0.0, 5.0)
    if "photo_bleed_mm" in update:
        s.photo_bleed_mm = _number(update["photo_bleed_mm"], "photo bleed", 0.0, 3.0)
    for key, choices in CHOICES.items():
        if key in update:
            if update[key] not in choices:
                raise api.UsageError(f"{key} must be {' or '.join(choices)}")
            setattr(s, key, update[key])
    if "overlay" in update:
        s.overlay = _flag(update["overlay"], "overlay")
    if "ppi" in update:
        v = update["ppi"]
        s.ppi = None if v in (None, "") else _number(v, "ppi", 10.0, 20000.0)
    if "profile" in update:
        v = str(update["profile"] or "").strip()
        s.profile = v or None
    if s.ppi is not None and s.profile:
        raise api.UsageError("set either ppi or a scanner profile, not both")
    if s.profile:       # an unknown profile would fail every run, also an agent's export after a fix
        try:
            explicit_scale(None, s.profile, s.profiles_file)
        except (KeyError, OSError, ValueError) as e:
            raise api.UsageError(str(e.args[0]) if isinstance(e, KeyError) and e.args else str(e)) from None
    for name in HIGH_WEB:
        if f"{name}_on" in update:
            _flag(update[f"{name}_on"], f"{name}_on")
    out = []
    current_by = {x.name: x for x in s.derivatives}
    for x in s.derivatives:                      # in their order; one the panel did not touch stays as it is
        touched = x.name in HIGH_WEB and any(f"{x.name}_{k}" in update for k in ("on", "edge", "quality"))
        if not touched:
            out.append(x)
        elif update.get(f"{x.name}_on", True):
            out.append(_derivative(x.name, x, update))
    for name in HIGH_WEB:
        if name not in current_by and update.get(f"{name}_on", False):
            out.append(_derivative(name, HIGH_WEB[name], update))
    s.derivatives = tuple(out)
    return settings_to_dict(s)


def _derivative(name: str, old: Derivative, update: Dict[str, Any]) -> Derivative:
    edge = _number(update.get(f"{name}_edge", old.long_edge or HIGH_WEB[name].long_edge), f"{name} long edge",
                   100, 30000, True)
    quality = _number(update.get(f"{name}_quality", old.quality), f"{name} quality", 1, 100, True)
    return Derivative(name, edge, None, "jpg", quality)


def defaults_of(d: dict) -> Dict[str, Any]:
    """The panel fields of ``d`` as the defaults for new projects (the scale is not carried over: it belongs to a
    scanner setting, not to a user's habit)."""
    p = to_panel(d)
    p.pop("other_derivatives")
    p["ppi"], p["profile"] = None, None
    return p
