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


# Heights exclude the grey band at the top and bottom edge (the rounded print edge seen from
# above), as the reviewer labels them since 2026-11-01; measured on 32 600 and 22 Go labels,
# scaled by the width. With the band, the 600 measures the quoted 107 mm.
POLAROID_600 = PrintFormat("polaroid-600", 88.0, 106.85)  # also i-Type, SX-70
# measured on 20 Go prints scanned with 600 prints in the same batch (sd 0.01 / 0.05 mm);
# the often quoted 53.9 x 66.6 mm is about 1 % too large
POLAROID_GO = PrintFormat("polaroid-go", 53.5, 65.8)

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


def classify(size_px: Tuple[float, float], formats: Sequence[PrintFormat] = FORMATS) -> PrintFormat:
    """The format whose aspect ratio is closest to the measured outline; needs no scale.

    600 prints measure 0.819-0.824 (short / long side), Go prints 0.808-0.812, so the shape
    alone tells them apart even when the scan resolution is unknown.
    """
    short, long_ = sorted(size_px)
    return min(formats, key=lambda f: abs(short / long_ - f.aspect))


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
