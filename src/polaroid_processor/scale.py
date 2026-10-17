"""Physical scale (pixels per millimetre) of a scan.

Precedence:
1. explicit value (``--ppi`` or a named scanner profile),
2. the TIFF resolution tag, when it yields a plausible print size,
3. inference from the batch median, assuming the default format.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, Optional, Tuple

import yaml

from .formats import DEFAULT_FORMAT, FORMATS, PrintFormat, match_format

MM_PER_INCH = 25.4
TAG_TOLERANCE = 0.05  # tag accepted when the print matches a known format within 5 %

PROFILE_ENV = "POLAROID_PROFILES"
DEFAULT_PROFILE_PATH = Path.home() / ".config" / "polaroid-processor" / "profiles.yaml"


@dataclass(frozen=True)
class Scale:
    px_per_mm: float
    source: str  # "ppi", "profile:<name>", "tiff-tag", "batch-median", "single-scan"

    @property
    def ppi(self) -> float:
        return self.px_per_mm * MM_PER_INCH

    def to_dict(self) -> dict:
        return {"px_per_mm": round(self.px_per_mm, 3), "ppi": round(self.ppi, 1), "source": self.source}


def profiles_path(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get(PROFILE_ENV)
    return Path(env) if env else DEFAULT_PROFILE_PATH


def load_profiles(path: Optional[str] = None) -> Dict[str, float]:
    """Read ``{name: ppi}`` from a YAML file shaped like ``profiles: {epson-3200: 3200}``."""
    p = profiles_path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    profs = data.get("profiles", data)
    out = {}
    for name, val in profs.items():
        ppi = val.get("ppi") if isinstance(val, dict) else val
        out[str(name)] = float(ppi)
    return out


def explicit_scale(ppi: Optional[float] = None, profile: Optional[str] = None,
                   profiles_file: Optional[str] = None) -> Optional[Scale]:
    if ppi:
        return Scale(float(ppi) / MM_PER_INCH, "ppi")
    if profile:
        profs = load_profiles(profiles_file)
        if profile not in profs:
            where = profiles_path(profiles_file)
            raise KeyError(f"scanner profile {profile!r} not found in {where}")
        return Scale(profs[profile] / MM_PER_INCH, f"profile:{profile}")
    return None


def scale_from_tag(dpi_tag: Optional[Tuple[float, float]], size_px: Tuple[float, float],
                   formats: Iterable[PrintFormat] = FORMATS) -> Optional[Scale]:
    """Use the TIFF tag only when the detected print then has a known size."""
    if not dpi_tag or not dpi_tag[0] or dpi_tag[0] <= 1:
        return None
    s = Scale(float(dpi_tag[0]) / MM_PER_INCH, "tiff-tag")
    m = match_format(size_px, s.px_per_mm, tuple(formats))
    if m and m.error <= TAG_TOLERANCE:
        return s
    return None


def scale_from_size(size_px: Tuple[float, float], fmt: PrintFormat = DEFAULT_FORMAT,
                    source: str = "single-scan") -> Scale:
    short_px, long_px = sorted(size_px)
    px_per_mm = (short_px / fmt.width_mm + long_px / fmt.height_mm) / 2
    return Scale(px_per_mm, source)


def batch_median_scale(sizes_px: Iterable[Tuple[float, float]], fmt: PrintFormat = DEFAULT_FORMAT) -> Optional[Scale]:
    vals = [scale_from_size(s, fmt).px_per_mm for s in sizes_px]
    if not vals:
        return None
    return Scale(median(vals), "batch-median")
