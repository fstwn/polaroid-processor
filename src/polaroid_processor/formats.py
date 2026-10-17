"""Known instant print formats and matching of measured sizes against them."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class PrintFormat:
    """Outer size of a print in portrait orientation (width < height)."""

    name: str
    width_mm: float
    height_mm: float

    @property
    def aspect(self) -> float:
        return self.width_mm / self.height_mm


POLAROID_600 = PrintFormat("polaroid-600", 88.0, 107.0)  # also i-Type, SX-70
POLAROID_GO = PrintFormat("polaroid-go", 53.9, 66.6)

FORMATS: Tuple[PrintFormat, ...] = (POLAROID_600, POLAROID_GO)
DEFAULT_FORMAT = POLAROID_600


@dataclass(frozen=True)
class FormatMatch:
    format: PrintFormat
    size_mm: Tuple[float, float]  # measured (width, height) in mm
    error: float  # largest relative size deviation of the two sides


def get_format(name: str) -> PrintFormat:
    for f in FORMATS:
        if f.name == name:
            return f
    raise KeyError(f"unknown format {name!r}; known: {', '.join(f.name for f in FORMATS)}")


def match_format(
    size_px: Tuple[float, float],
    px_per_mm: float,
    formats: Sequence[PrintFormat] = FORMATS,
) -> Optional[FormatMatch]:
    """Return the closest known format for a measured size, or None if px_per_mm is unknown.

    Orientation is ignored: a landscape scan matches the portrait format.
    """
    if not px_per_mm or px_per_mm <= 0:
        return None
    w_mm, h_mm = size_px[0] / px_per_mm, size_px[1] / px_per_mm
    short, long_ = sorted((w_mm, h_mm))
    best = None
    for f in formats:
        err = max(abs(short - f.width_mm) / f.width_mm, abs(long_ - f.height_mm) / f.height_mm)
        if best is None or err < best.error:
            best = FormatMatch(f, (round(w_mm, 2), round(h_mm, 2)), round(err, 4))
    return best
