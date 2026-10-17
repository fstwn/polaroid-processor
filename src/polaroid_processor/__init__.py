"""Detect, crop and rectify high-resolution flatbed scans of instant prints."""

__version__ = "0.1.0"

from .detection import Detection, DetectionError, DetectParams, detect  # noqa: E402,F401
from .formats import FORMATS, PrintFormat, match_format  # noqa: E402,F401
from .geometry import rectify  # noqa: E402,F401
from .imageio import load_rgb, read_meta  # noqa: E402,F401
