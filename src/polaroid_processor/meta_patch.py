"""Write and rewrite the text headers of TIFF, JPEG and PNG files without touching the pixel data.

One function per format, :func:`set_tiff`, :func:`set_jpeg`, :func:`set_png`, puts a :class:`~.metadata.FileMeta`
into a file: the XMP packet plus the legacy fields (TIFF / EXIF ``ImageDescription``, ``Make``, ``Model``,
``Software``, ``Artist``, ``Copyright``, ``DateTimeOriginal`` for full dates, the Windows ``XPTitle`` / ``XPComment`` /
``XPAuthor`` / ``XPKeywords``). They serve the first write (the file just encoded, still a ``.part``) and a later
rewrite alike, so a rewritten file equals a freshly written one. A file that does not look as expected raises
:class:`PatchError`; the caller exports it again.

* TIFF: the image file directory is written again at the end of the file and the header points to it; strips,
  resolution values and the ICC profile stay where they are. The old directory stays behind as unused bytes.
* JPEG: the APP1 segments (Exif, XMP) are replaced, the rest of the file is copied byte for byte.
* PNG: the ``eXIf`` chunk and the XMP ``iTXt`` chunk are replaced, every other chunk is copied byte for byte.
"""
from __future__ import annotations

import os
import shutil
import struct
import zlib
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple

from .metadata import FileMeta, strip_bad, xmp_packet

ASCII, BYTE, SHORT, LONG, UNDEFINED, IFD, LONG8, IFD8 = 2, 1, 3, 4, 7, 13, 16, 18
_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4, 16: 8, 17: 8, 18: 8}

TAG_DESCRIPTION, TAG_MAKE, TAG_MODEL, TAG_SOFTWARE, TAG_ARTIST = 270, 271, 272, 305, 315
TAG_COPYRIGHT, TAG_XMP, TAG_EXIF_IFD = 33432, 700, 34665
TAG_XP_TITLE, TAG_XP_COMMENT, TAG_XP_AUTHOR, TAG_XP_KEYWORDS, TAG_XP_SUBJECT = 40091, 40092, 40093, 40094, 40095
TAG_EXIF_VERSION, TAG_DATE_ORIGINAL = 36864, 36867
# every tag the tool sets: removed from a file's directory before the new ones are added
MANAGED = {TAG_DESCRIPTION, TAG_MAKE, TAG_MODEL, TAG_SOFTWARE, TAG_ARTIST, TAG_COPYRIGHT, TAG_XMP, TAG_EXIF_IFD,
           TAG_XP_TITLE, TAG_XP_COMMENT, TAG_XP_AUTHOR, TAG_XP_KEYWORDS, TAG_XP_SUBJECT}

XMP_JPEG_ID = b"http://ns.adobe.com/xap/1.0/\x00"
XMP_PNG_KEY = b"XML:com.adobe.xmp"


class PatchError(Exception):
    """The file cannot be patched safely (not the structure the tool writes)."""


# --------------------------------------------------------------------------- directory entries
class Entry:
    """One directory entry. ``data`` is a new value (bytes in the file's byte order); ``raw`` the field of an entry
    that is carried over (its value, or the offset of it, exactly as the file has it)."""

    def __init__(self, tag: int, typ: int, count: int, data: Optional[bytes] = None, raw: Optional[bytes] = None):
        self.tag, self.typ, self.count, self.data, self.raw = tag, typ, count, data, raw


def _ascii(text: str) -> Entry:
    b = strip_bad(text).encode("utf-8") + b"\x00"
    return Entry(0, ASCII, len(b), b)


def _xp(text: str) -> Entry:
    b = strip_bad(text).encode("utf-16-le") + b"\x00\x00"
    return Entry(0, BYTE, len(b), b)


def legacy_entries(fm: FileMeta) -> Tuple[Dict[int, Entry], Dict[int, Entry]]:
    """The legacy fields of ``fm`` as (main directory, Exif directory) entries; empty fields write nothing."""
    main: Dict[int, Entry] = {}
    for tag, text in ((TAG_DESCRIPTION, fm.description or fm.title), (TAG_MAKE, fm.make), (TAG_MODEL, fm.model),
                      (TAG_SOFTWARE, fm.software), (TAG_ARTIST, fm.author), (TAG_COPYRIGHT, fm.copyright)):
        if text:
            main[tag] = _ascii(text)
    for tag, text in ((TAG_XP_TITLE, fm.title), (TAG_XP_COMMENT, fm.description), (TAG_XP_AUTHOR, fm.author),
                      (TAG_XP_KEYWORDS, ";".join(k.replace(";", ",") for k in fm.keywords))):
        if text:
            main[tag] = _xp(text)
    exif: Dict[int, Entry] = {}
    if len(fm.date_taken) == 10:      # EXIF wants a full date and time
        exif[TAG_EXIF_VERSION] = Entry(0, UNDEFINED, 4, b"0231")
        exif[TAG_DATE_ORIGINAL] = _ascii(fm.date_taken.replace("-", ":") + " 00:00:00")
    for tag, e in list(main.items()) + list(exif.items()):
        e.tag = tag
    return main, exif


def _pack_value(e: Entry, big: bool, bo: str, offset: int) -> bytes:
    field = 8 if big else 4
    if e.raw is not None:
        return e.raw
    if len(e.data) <= field:
        return e.data.ljust(field, b"\x00")
    return struct.pack(bo + ("Q" if big else "I"), offset)


def _layout(entries: List[Entry], start: int, big: bool, bo: str) -> bytes:
    """One directory plus the out-of-line values of its new entries, laid out to begin at file offset ``start``."""
    entries = sorted(entries, key=lambda e: e.tag)
    n = len(entries)
    head = (8 if big else 2) + n * (20 if big else 12) + (8 if big else 4)
    pos = start + head
    chunks: List[bytes] = []
    out = bytearray(struct.pack(bo + ("Q" if big else "H"), n))
    field = 8 if big else 4
    for e in entries:
        value_offset = 0
        if e.raw is None and len(e.data) > field:
            if pos % 2:
                chunks.append(b"\x00")
                pos += 1
            value_offset = pos
            chunks.append(e.data)
            pos += len(e.data)
        if not big and pos >= 1 << 32:
            raise PatchError("the file is too large for classic TIFF offsets")
        out += struct.pack(bo + ("HHQ" if big else "HHI"), e.tag, e.typ, e.count) + _pack_value(e, big, bo, value_offset)
    out += struct.pack(bo + ("Q" if big else "I"), 0)
    return bytes(out) + b"".join(chunks)


def _build(main: List[Entry], exif: Dict[int, Entry], start: int, big: bool, bo: str) -> bytes:
    """The main directory, followed by the Exif directory when there is one (the main one points to it)."""
    if not exif:
        return _layout(main, start, big, bo)
    ptr_type = IFD8 if big else LONG
    placeholder = Entry(TAG_EXIF_IFD, ptr_type, 1, struct.pack(bo + ("Q" if big else "I"), 0))
    size = len(_layout(main + [placeholder], start, big, bo))
    exif_at = start + size + (size % 2)
    ptr = Entry(TAG_EXIF_IFD, ptr_type, 1, struct.pack(bo + ("Q" if big else "I"), exif_at))
    head = _layout(main + [ptr], start, big, bo)
    return head + b"\x00" * (exif_at - start - len(head)) + _layout(list(exif.values()), exif_at, big, bo)


def exif_block(fm: FileMeta) -> Optional[bytes]:
    """A little TIFF structure (header and directories) with the legacy fields: the body of a JPEG Exif segment or
    a PNG ``eXIf`` chunk. None when there is nothing to put in."""
    main, exif = legacy_entries(fm)
    if not main and not exif:
        return None
    return b"II*\x00\x08\x00\x00\x00" + _build(list(main.values()), exif, 8, False, "<")


# --------------------------------------------------------------------------- TIFF
def set_tiff(path: str, fm: FileMeta) -> None:
    """Replace the text tags of a TIFF in place (see the module text)."""
    xmp = xmp_packet(fm)
    with open(path, "r+b") as f:
        head = f.read(16)
        if head[:2] == b"II":
            bo = "<"
        elif head[:2] == b"MM":
            bo = ">"
        else:
            raise PatchError("not a TIFF file")
        magic = struct.unpack(bo + "H", head[2:4])[0]
        if magic == 42:
            big, ifd_at, ptr_at = False, struct.unpack(bo + "I", head[4:8])[0], 4
        elif magic == 43:
            big, ifd_at, ptr_at = True, struct.unpack(bo + "Q", head[8:16])[0], 8
        else:
            raise PatchError("not a TIFF file")
        size = os.fstat(f.fileno()).st_size
        if not 8 <= ifd_at < size:
            raise PatchError("bad first directory offset")
        f.seek(ifd_at)
        cw, ew, nw = (8, 20, 8) if big else (2, 12, 4)
        raw = f.read(cw)
        if len(raw) < cw:
            raise PatchError("truncated directory")
        n = struct.unpack(bo + ("Q" if big else "H"), raw)[0]
        body = f.read(n * ew + nw)
        if n == 0 or n > 4096 or len(body) < n * ew + nw:
            raise PatchError("unexpected directory")
        if struct.unpack(bo + ("Q" if big else "I"), body[n * ew:])[0] != 0:
            raise PatchError("more than one image in the file")
        kept: List[Entry] = []
        for i in range(n):
            b = body[i * ew:(i + 1) * ew]
            tag, typ = struct.unpack(bo + "HH", b[:4])
            count = struct.unpack(bo + ("Q" if big else "I"), b[4:4 + nw])[0]
            if typ not in _SIZE:
                raise PatchError(f"unknown field type {typ} in tag {tag}")
            if tag not in MANAGED:
                kept.append(Entry(tag, typ, count, raw=b[4 + nw:]))
        main, exif = legacy_entries(fm)
        new = list(main.values())
        if xmp:
            new.append(Entry(TAG_XMP, BYTE, len(xmp), xmp))
        start = size + (size % 2)
        block = _build(kept + new, exif, start, big, bo)
        f.seek(size)
        f.write(b"\x00" * (start - size) + block)
        f.flush()
        if _durable(path):
            os.fsync(f.fileno())
        f.seek(ptr_at)      # the switch: until here the old directory is the file's
        f.write(struct.pack(bo + ("Q" if big else "I"), start))
        f.flush()
        if _durable(path):
            os.fsync(f.fileno())


# --------------------------------------------------------------------------- JPEG
def _durable(path) -> bool:
    """Whether a write is flushed to disk before it counts: a rewrite of a finished export, not a file that is still
    a ``.part`` (the rename that finishes it is not flushed either, and a master is hundreds of MB)."""
    return not str(path).endswith(".part")


def _swap(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".patch")      # a killed run leaves this: the next run removes it (_clean_partial)
    try:
        with open(path, "rb") as src, open(tmp, "wb") as dst:
            write(src, dst)
            if _durable(path):
                dst.flush()
                os.fsync(dst.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def set_jpeg(path: str, fm: FileMeta) -> None:
    """Replace the Exif and XMP APP1 segments of a JPEG; the rest is copied byte for byte."""
    exif = exif_block(fm)
    xmp = xmp_packet(fm)
    new: List[bytes] = []
    for payload in ((b"Exif\x00\x00" + exif) if exif else None, (XMP_JPEG_ID + xmp) if xmp else None):
        if payload is None:
            continue
        if len(payload) + 2 > 0xFFFF:
            raise PatchError("the metadata does not fit in a JPEG segment")
        new.append(b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload)

    def write(src: BinaryIO, dst: BinaryIO) -> None:
        if src.read(2) != b"\xff\xd8":
            raise PatchError("not a JPEG file")
        dst.write(b"\xff\xd8")
        inserted = False
        while True:
            m = src.read(2)
            if len(m) < 2 or m[0] != 0xFF:
                raise PatchError("bad JPEG segment")
            if m[1] == 0xFF:      # fill byte
                src.seek(-1, 1)
                continue
            if m[1] in (0x01, *range(0xD0, 0xD9)):      # markers without a length
                dst.write(m)
                continue
            ln = src.read(2)
            if len(ln) < 2:
                raise PatchError("truncated JPEG")
            size = struct.unpack(">H", ln)[0]
            if size < 2:
                raise PatchError("bad JPEG segment length")
            data = src.read(size - 2)
            if len(data) < size - 2:
                raise PatchError("truncated JPEG")
            old = m[1] == 0xE1 and (data.startswith(b"Exif\x00\x00") or data.startswith(XMP_JPEG_ID))
            if m[1] != 0xE0 and not inserted:      # after the JFIF segment, before everything else
                dst.write(b"".join(new))
                inserted = True
            if not old:
                dst.write(m + ln + data)
            if m[1] == 0xDA:      # start of scan: the entropy-coded data follows, no more segments
                shutil.copyfileobj(src, dst, 1 << 20)
                return

    _swap(Path(path), write)


# --------------------------------------------------------------------------- PNG
_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def set_png(path: str, fm: FileMeta) -> None:
    """Replace the ``eXIf`` and XMP ``iTXt`` chunks of a PNG; every other chunk is copied byte for byte."""
    exif = exif_block(fm)
    xmp = xmp_packet(fm)
    new = b""
    if exif:
        new += _chunk(b"eXIf", exif)
    if xmp:
        new += _chunk(b"iTXt", XMP_PNG_KEY + b"\x00\x00\x00\x00\x00" + xmp)

    def write(src: BinaryIO, dst: BinaryIO) -> None:
        if src.read(8) != _PNG_SIG:
            raise PatchError("not a PNG file")
        dst.write(_PNG_SIG)
        inserted = False
        while True:
            head = src.read(8)
            if len(head) < 8:
                raise PatchError("truncated PNG")
            ln, kind = struct.unpack(">I", head[:4])[0], head[4:]
            if kind in (b"IDAT", b"IEND") and not inserted:
                dst.write(new)
                inserted = True
            if kind == b"eXIf":
                src.seek(ln + 4, 1)
                continue
            if kind == b"iTXt" and ln < (1 << 20):
                body = src.read(ln + 4)
                if not body.startswith(XMP_PNG_KEY + b"\x00"):
                    dst.write(head + body)
                continue
            dst.write(head)
            left = ln + 4
            while left:
                buf = src.read(min(left, 1 << 20))
                if not buf:
                    raise PatchError("truncated PNG")
                dst.write(buf)
                left -= len(buf)
            if kind == b"IEND":
                return

    _swap(Path(path), write)


def set_metadata(path: str, fm: FileMeta) -> None:
    """Put ``fm`` into the file at ``path`` (TIFF, JPEG or PNG by its suffix), in place."""
    suffix = Path(path).suffix.lower()
    if suffix == ".part":
        suffix = Path(path[:-5]).suffix.lower()
    if suffix in (".tif", ".tiff"):
        set_tiff(path, fm)
    elif suffix in (".jpg", ".jpeg"):
        set_jpeg(path, fm)
    elif suffix == ".png":
        set_png(path, fm)
    else:
        raise PatchError(f"no metadata writer for {suffix!r} files")
