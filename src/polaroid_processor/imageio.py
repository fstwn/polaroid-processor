"""Reading scans and writing masters / derivatives.

Images are handled as RGB numpy arrays (uint8 or uint16, H x W x 3).
"""
from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import tifffile
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SCAN_SUFFIXES = (".tif", ".tiff")


@dataclass
class ScanMeta:
    """Metadata readable without decoding pixels."""

    path: str
    width: int
    height: int
    dpi_tag: Optional[Tuple[float, float]]
    icc_profile: Optional[bytes]
    bits: int


def read_meta(path: str) -> ScanMeta:
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        h, w = page.shape[:2]
        bits = int(page.bitspersample)
        icc = page.tags.get(34675)
        icc_bytes = bytes(icc.value) if icc is not None else None
        dpi = None
        xr = page.tags.get("XResolution")
        unit = page.tags.get("ResolutionUnit")
        if xr is not None:
            num, den = xr.value
            val = num / den if den else 0.0
            u = int(unit.value) if unit is not None else 2
            if u == 3:  # centimetre
                val *= 2.54
            if val > 0:
                dpi = (val, val)
    return ScanMeta(str(path), int(w), int(h), dpi, icc_bytes, bits)


def load_rgb(path: str) -> np.ndarray:
    """Decode a scan to RGB (uint8 or uint16). Alpha is dropped, grey is expanded."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        img = tifffile.imread(str(path))
        if img.ndim == 3 and img.shape[2] >= 3:
            img = img[..., :3]
        elif img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return np.ascontiguousarray(img)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img >> 8).astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8)


def file_fingerprint(path: str, samples: int = 32, chunk: int = 1 << 20) -> str:
    """SHA-256 over the file size and ``samples`` evenly spaced 1 MiB chunks.

    A full hash of a 500 MB scan costs ~1.2 s; re-scans and edits change bytes
    everywhere, so sampled chunks detect them at a fraction of the cost.
    """
    size = os.path.getsize(path)
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        if size <= samples * chunk:
            h.update(f.read())
        else:
            step = (size - chunk) / (samples - 1)
            for i in range(samples):
                f.seek(int(i * step))
                h.update(f.read(chunk))
    return "s32:" + h.hexdigest()


def _atomic_target(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(path.name + ".part")


def write_tiff(path: str, rgb: np.ndarray, px_per_mm: Optional[float], icc: Optional[bytes] = None,
               compression: Optional[str] = "zlib", compression_level: int = 1, threads: int = 8) -> None:
    """Lossless TIFF master with resolution tag (px/cm) and ICC profile."""
    target = Path(path)
    tmp = _atomic_target(target)
    extratags = []
    if icc:
        extratags.append((34675, "B", len(icc), icc, True))
    kwargs = {}
    if px_per_mm:
        kwargs["resolution"] = (px_per_mm * 10.0, px_per_mm * 10.0)
        kwargs["resolutionunit"] = "CENTIMETER"
    if compression and compression != "none":
        kwargs["compression"] = compression
        kwargs["predictor"] = True
        if compression in ("zlib", "deflate"):
            kwargs["compressionargs"] = {"level": compression_level}
        kwargs["rowsperstrip"] = 512
        kwargs["maxworkers"] = threads
    tifffile.imwrite(str(tmp), rgb, photometric="rgb", extratags=extratags, **kwargs)
    os.replace(tmp, target)


def resize_long_edge(rgb: np.ndarray, long_edge: Optional[int] = None, scale: Optional[float] = None) -> np.ndarray:
    h, w = rgb.shape[:2]
    if long_edge:
        f = long_edge / max(h, w)
    elif scale:
        f = scale
    else:
        return rgb
    if f >= 1.0:
        return rgb
    size = (max(1, round(w * f)), max(1, round(h * f)))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def write_derivative(path: str, rgb: np.ndarray, fmt: str, quality: int, dpi: Optional[float],
                     icc: Optional[bytes] = None, alpha: Optional[np.ndarray] = None) -> None:
    target = Path(path)
    tmp = _atomic_target(target)
    img8 = to_uint8(rgb)
    pil = Image.fromarray(np.ascontiguousarray(img8))
    params = {}
    if icc:
        params["icc_profile"] = icc
    if dpi:
        params["dpi"] = (dpi, dpi)
    fmt = fmt.lower()
    if fmt in ("jpg", "jpeg"):
        pil.save(tmp, "JPEG", quality=int(quality), subsampling=0 if quality >= 90 else 2, optimize=True, **params)
    elif fmt == "png":
        if alpha is not None:
            pil.putalpha(Image.fromarray(alpha))
        pil.save(tmp, "PNG", compress_level=6, **params)
    else:
        raise ValueError(f"unsupported derivative format {fmt!r}")
    os.replace(tmp, target)


def encode_png(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(to_uint8(rgb))).save(buf, "PNG", compress_level=3)
    return buf.getvalue()


def set_resolution(path: str, px_per_mm: float) -> None:
    """Patch the resolution metadata of an already written TIFF, JPEG or PNG in place.

    Lets the batch fix the scale after all scans are measured without re-encoding.
    """
    import struct
    import zlib

    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".tif", ".tiff"):
        with tifffile.TiffFile(str(p), mode="r+b") as tf:
            page = tf.pages[0]
            for name in ("XResolution", "YResolution"):
                page.tags[name].overwrite(tifffile.rational(px_per_mm * 10.0) if hasattr(tifffile, "rational")
                                          else (int(round(px_per_mm * 10.0 * 1000)), 1000))
            page.tags["ResolutionUnit"].overwrite(3)
        return
    dpi = px_per_mm * 25.4
    with open(p, "r+b") as f:
        data = f.read(1 << 16)
        if suffix in (".jpg", ".jpeg"):
            i = data.find(b"JFIF\x00")
            if i < 0:
                return
            f.seek(i + 7)  # version (2) then units
            f.write(struct.pack(">BHH", 1, int(round(dpi)), int(round(dpi))))
        elif suffix == ".png":
            i = data.find(b"pHYs")
            if i < 0:
                return
            ppm = int(round(px_per_mm * 1000))
            payload = struct.pack(">IIB", ppm, ppm, 1)
            f.seek(i + 4)
            f.write(payload)
            f.write(struct.pack(">I", zlib.crc32(b"pHYs" + payload) & 0xFFFFFFFF))
