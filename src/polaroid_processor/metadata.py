"""Export metadata: the project's fields, placeholders, and what a file carries.

A project's metadata is a small dict (``title``, ``author``, ``copyright``, ``description``, ``keywords``,
``date_taken``); the text fields may hold placeholders that are filled per scan. :func:`fill` turns the project's
dict into a :class:`FileMeta` for one scan (with the scanner's Make / Model and our Software string), which
:mod:`.meta_patch` writes into TIFF, JPEG and PNG files.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from xml.sax.saxutils import escape

log = logging.getLogger("polaroid_processor")

TEXT_FIELDS = ("title", "author", "copyright", "description")
FIELDS = TEXT_FIELDS + ("keywords", "date_taken")
PLACEHOLDERS = ("stem", "n", "project", "format")

MAX_TEXT = 500            # characters per title / author / copyright / keyword
MAX_DESCRIPTION = 4000
MAX_KEYWORDS = 100
MAX_TOTAL = 8000          # all texts together (one JPEG APP1 segment holds 64 KB, UTF-8 and UTF-16 copies included)
_WORST_VALUE = "x" * 260  # a file name, the longest a placeholder can expand to

# Bump when what the writers put into a file changes (a new tag, another mapping): it is part of `FileMeta.key`, so the
# next run rewrites the headers of the exports that have a key. 1: first release (ImageDescription = description or
# title, XMP with tiff:Make / tiff:Model).
WRITER_VERSION = 1

FORMAT_LABELS ={"polaroid-600": "600", "polaroid-go": "Go"}

_DATE = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")


# --------------------------------------------------------------------------- placeholders
def fill_text(text: str, values: Mapping[str, str]) -> str:
    """``text`` with ``{name}`` replaced from ``values``; ``{{`` and ``}}`` give literal braces.
    An unknown name or a stray brace raises ``ValueError``."""
    out: List[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "{":
            if text[i + 1:i + 2] == "{":
                out.append("{")
                i += 2
                continue
            j = text.find("}", i + 1)
            if j < 0:
                raise ValueError(f"a '{{' without '}}' in {text!r} (write '{{{{' for a literal brace)")
            name = text[i + 1:j]
            if name not in values:
                raise ValueError(f"unknown placeholder {{{name}}} in {text!r}; known: "
                                 + ", ".join("{" + p + "}" for p in PLACEHOLDERS))
            out.append(values[name])
            i = j + 1
        elif c == "}":
            if text[i + 1:i + 2] == "}":
                out.append("}")
                i += 2
                continue
            raise ValueError(f"a '}}' without '{{' in {text!r} (write '}}}}' for a literal brace)")
        else:
            out.append(c)
            i += 1
    return "".join(out)


def placeholder_values(stem: str, project: str, format_label: str) -> Dict[str, str]:
    """``{stem}``, ``{n}`` (the digits the stem ends in, as written; empty when it ends in none), ``{project}``
    (the input folder's name) and ``{format}`` (``600`` or ``Go``)."""
    m = re.search(r"(\d+)$", stem)
    return {"stem": stem, "n": m.group(1) if m else "", "project": project, "format": format_label}


def format_label(fmt: Optional[Mapping[str, Any]], size_px: Optional[Tuple[float, float]] = None) -> str:
    """``600`` / ``Go`` of a manifest format record (``{"name": "polaroid-go", ...}``); without one, of the shape
    of the outline (``formats.classify``)."""
    name = (fmt or {}).get("name")
    if name in FORMAT_LABELS:
        return FORMAT_LABELS[name]
    if size_px is not None:
        from .formats import classify
        return FORMAT_LABELS.get(classify(size_px).name, "")
    return ""


# --------------------------------------------------------------------------- the project's fields
def check_date(value: str) -> str:
    """``YYYY``, ``YYYY-MM`` or ``YYYY-MM-DD`` (a real date, year 1000 or later); returns it."""
    m = _DATE.match(value)
    if not m:
        raise ValueError(f"date taken {value!r}: use YYYY, YYYY-MM or YYYY-MM-DD")
    year, month, day = int(m.group(1)), int(m.group(2) or 1), int(m.group(3) or 1)
    try:
        _dt.date(year, month, day)
    except ValueError:
        raise ValueError(f"date taken {value!r} is not a date") from None
    if year < 1000:
        raise ValueError(f"date taken {value!r}: the year needs four digits, 1000 or later")
    return value


_BAD_TEXT = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff￾￿]")      # no place in a header text


def _clean(s: Any, what: str) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        raise ValueError(f"metadata {what} must be text, not {type(s).__name__}")
    if _BAD_TEXT.search(s):
        raise ValueError(f"metadata {what} holds a control character or an unpaired surrogate")
    return s.strip()


def strip_bad(s: str) -> str:
    """``s`` without the characters that cannot go into a header (control characters except tab and line breaks,
    unpaired surrogates)."""
    return _BAD_TEXT.sub("", s)


def normalize(d: Optional[Mapping[str, Any]], keep_empty: bool = False) -> dict:
    """The checked form of a metadata dict: known fields only, texts stripped, keywords a list of non-empty
    strings, the date validated and every placeholder checked. Empty fields are left out, or with
    ``keep_empty`` kept (``""`` / ``[]``) so that a merge can clear them. Raises ``ValueError``."""
    if d is None:
        return {}
    if not isinstance(d, Mapping):
        raise ValueError("metadata must be an object with title, author, copyright, description, keywords, date_taken")
    unknown = set(d) - set(FIELDS)
    if unknown:
        raise ValueError(f"unknown metadata field: {', '.join(sorted(unknown))}; known: {', '.join(FIELDS)}")
    out: Dict[str, Any] = {}
    for f in TEXT_FIELDS:
        if d.get(f) is not None:      # None is a field left out, "" clears it
            v = _clean(d[f], f)
            if v or keep_empty:
                out[f] = v
    if d.get("keywords") is not None:
        kw = d["keywords"]
        if isinstance(kw, str) or not isinstance(kw, Iterable):
            raise ValueError("metadata keywords must be a list of texts")
        words = list(dict.fromkeys(w for w in (_clean(x, "keyword") for x in kw) if w))      # no duplicates
        if words or keep_empty:
            out["keywords"] = words
    if d.get("date_taken") is not None:
        v = _clean(d["date_taken"], "date_taken")
        if v:
            out["date_taken"] = check_date(v)
        elif keep_empty:
            out["date_taken"] = ""
    _check(out)
    return out


def _check(d: Mapping[str, Any]) -> None:
    worst = placeholder_values(_WORST_VALUE, _WORST_VALUE, "Go")
    total = 0
    for f in TEXT_FIELDS:
        v = d.get(f)
        if not v:
            continue
        filled = fill_text(v, worst)      # raises on an unknown placeholder or a stray brace
        limit = MAX_DESCRIPTION if f == "description" else MAX_TEXT
        if len(v) > limit:
            raise ValueError(f"metadata {f} is longer than {limit} characters")
        total += len(filled)
    kws = d.get("keywords") or []
    if len(kws) > MAX_KEYWORDS:
        raise ValueError(f"metadata keywords: at most {MAX_KEYWORDS}")
    for w in kws:
        if len(w) > MAX_TEXT:
            raise ValueError(f"a keyword is longer than {MAX_TEXT} characters")
        total += len(fill_text(w, worst))
    if total > MAX_TOTAL:
        raise ValueError(f"metadata texts add up to more than {MAX_TOTAL} characters")


def normalize_stored(d: Any, warn: bool = True) -> dict:
    """The tolerant form of a *saved* metadata dict (a manifest may have been edited by hand): never raises. Text
    fields and keywords that are text are kept as they are (placeholders that cannot be filled stay literal; the
    checks of :func:`normalize` are not applied), anything that cannot be text and unknown fields are dropped with
    a warning."""
    if d is None or d == {}:
        return {}
    out: Dict[str, Any] = {}
    if not isinstance(d, Mapping):
        if warn:
            log.warning("saved metadata is not an object (%s): ignored", type(d).__name__)
        return out
    for f, v in d.items():
        if f in TEXT_FIELDS or f == "date_taken":
            if isinstance(v, str):
                if v.strip():
                    out[f] = v.strip()
            elif v is not None and warn:
                log.warning("saved metadata %s is not text (%s): ignored", f, type(v).__name__)
        elif f == "keywords":
            if isinstance(v, (list, tuple)):
                words = list(dict.fromkeys(w.strip() for w in v if isinstance(w, str) and w.strip()))
                if len(words) != len(v) and warn:
                    log.warning("saved metadata keywords: entries that are not text or empty were ignored")
                if words:
                    out[f] = words
            elif v is not None and warn:
                log.warning("saved metadata keywords is not a list (%s): ignored", type(v).__name__)
        elif warn:
            log.warning("saved metadata has an unknown field %r: ignored", f)
    return out


def merge(stored: Optional[Mapping[str, Any]], given: Optional[Mapping[str, Any]]) -> dict:
    """The project's metadata after a run that gave ``given`` (checked, raises ``ValueError``): a field in ``given``
    replaces the stored one, an empty one (``""``, ``[]``) clears it, a field left out (or None) keeps the stored
    value. ``given`` None keeps all. The stored side is read tolerantly (:func:`normalize_stored`)."""
    out = normalize_stored(stored)
    if given is None:
        return out
    for f, v in normalize(given, keep_empty=True).items():
        if v in ("", []):
            out.pop(f, None)
        else:
            out[f] = v
    return out


# --------------------------------------------------------------------------- what one file carries
@dataclass(frozen=True)
class FileMeta:
    """The filled metadata of one exported file plus the tags the tool sets itself."""

    title: str = ""
    author: str = ""
    copyright: str = ""
    description: str = ""
    keywords: Tuple[str, ...] = ()
    date_taken: str = ""
    make: str = ""
    model: str = ""
    software: str = ""

    def user_fields(self) -> dict:
        """The filled fields of the project (not Make / Model / Software), empty ones left out."""
        d = {f: getattr(self, f) for f in TEXT_FIELDS + ("date_taken",) if getattr(self, f)}
        if self.keywords:
            d["keywords"] = list(self.keywords)
        return d

    def key(self) -> str:
        """Identity of the filled fields; ``""`` when there are none. Stored per export to see when the headers
        of a file are out of date."""
        d = self.user_fields()
        if not d:      # no writer version here: exports without metadata have no key, whatever the writers do
            return ""
        payload = json.dumps({"fields": d, "writer": WRITER_VERSION}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(payload.encode()).hexdigest()[:16]


def fill(meta: Optional[Mapping[str, Any]], *, stem: str, project: str, format_name: str,
         make: str = "", model: str = "", software: str = "") -> FileMeta:
    """The metadata of one scan: the project's dict with the placeholders filled. A stored dict that does not
    pass the checks (a hand-edited manifest) never stops an export: a text that cannot be filled stays as it is."""
    values = placeholder_values(stem, project, format_name)

    def one(t: str) -> str:
        try:
            t = fill_text(t, values)
        except ValueError:
            pass
        return strip_bad(t).strip()

    meta = normalize_stored(meta, warn=False)      # a value that is not text is not there
    kws = tuple(dict.fromkeys(w for w in (one(k) for k in meta.get("keywords") or []) if w))
    date = meta.get("date_taken") or ""
    try:
        date = check_date(date) if date else ""
    except ValueError:
        date = ""
    kw = {f: one(meta.get(f) or "") for f in TEXT_FIELDS}
    return FileMeta(keywords=kws, date_taken=date, make=make, model=model, software=software, **kw)


# --------------------------------------------------------------------------- XMP
def _x(s: str) -> str:
    return escape(strip_bad(s))


def xmp_packet(fm: FileMeta) -> Optional[bytes]:
    """The XMP packet (UTF-8) with the project's fields, or None when there are none."""
    if not fm.user_fields():
        return None
    props: List[str] = []
    if fm.title:
        props.append(f'<dc:title><rdf:Alt><rdf:li xml:lang="x-default">{_x(fm.title)}</rdf:li></rdf:Alt></dc:title>')
    if fm.author:
        props.append(f"<dc:creator><rdf:Seq><rdf:li>{_x(fm.author)}</rdf:li></rdf:Seq></dc:creator>")
    if fm.copyright:
        props.append(f'<dc:rights><rdf:Alt><rdf:li xml:lang="x-default">{_x(fm.copyright)}</rdf:li></rdf:Alt></dc:rights>')
    if fm.description:
        props.append('<dc:description><rdf:Alt><rdf:li xml:lang="x-default">'
                     f"{_x(fm.description)}</rdf:li></rdf:Alt></dc:description>")
    if fm.keywords:
        props.append("<dc:subject><rdf:Bag>" + "".join(f"<rdf:li>{_x(k)}</rdf:li>" for k in fm.keywords)
                     + "</rdf:Bag></dc:subject>")
    if fm.date_taken:
        props.append(f"<photoshop:DateCreated>{fm.date_taken}</photoshop:DateCreated>")
        props.append(f"<exif:DateTimeOriginal>{fm.date_taken}</exif:DateTimeOriginal>")
    if fm.software:
        props.append(f"<xmp:CreatorTool>{_x(fm.software)}</xmp:CreatorTool>")
    if fm.make:
        props.append(f"<tiff:Make>{_x(fm.make)}</tiff:Make>")
    if fm.model:
        props.append(f"<tiff:Model>{_x(fm.model)}</tiff:Model>")
    body = "\n   ".join(props)
    xml = ('<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
           f'<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="{_x(fm.software or "polaroid-processor")}">\n'
           ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
           '  <rdf:Description rdf:about=""\n'
           '    xmlns:dc="http://purl.org/dc/elements/1.1/"\n'
           '    xmlns:xmp="http://ns.adobe.com/xap/1.0/"\n'
           '    xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/"\n'
           '    xmlns:exif="http://ns.adobe.com/exif/1.0/"\n'
           '    xmlns:tiff="http://ns.adobe.com/tiff/1.0/">\n'
           f"   {body}\n"
           "  </rdf:Description>\n"
           " </rdf:RDF>\n"
           "</x:xmpmeta>\n"
           '<?xpacket end="w"?>')
    return xml.encode("utf-8")


def read_xmp_fields(packet: bytes) -> dict:
    """Read back the project's fields from a packet made by :func:`xmp_packet` (for checks and tests)."""
    import xml.etree.ElementTree as ET
    ns = {"rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#", "dc": "http://purl.org/dc/elements/1.1/",
          "photoshop": "http://ns.adobe.com/photoshop/1.0/", "exif": "http://ns.adobe.com/exif/1.0/",
          "xmp": "http://ns.adobe.com/xap/1.0/", "tiff": "http://ns.adobe.com/tiff/1.0/"}
    text = packet.decode("utf-8")
    start = text.find("<x:xmpmeta")
    root = ET.fromstring(text[start:text.rfind("</x:xmpmeta>") + len("</x:xmpmeta>")])
    desc = root.find("rdf:RDF/rdf:Description", ns)
    out: Dict[str, Any] = {}

    def first(path: str) -> Optional[str]:
        li = desc.find(path + "//rdf:li", ns)
        return None if li is None else li.text or ""

    for key, path in (("title", "dc:title"), ("author", "dc:creator"), ("copyright", "dc:rights"),
                      ("description", "dc:description")):
        v = first(path)
        if v is not None:
            out[key] = v
    kw = [li.text or "" for li in desc.findall("dc:subject/rdf:Bag/rdf:li", ns)]
    if kw:
        out["keywords"] = kw
    d = desc.find("photoshop:DateCreated", ns)
    if d is not None:
        out["date_taken"] = d.text
    tool = desc.find("xmp:CreatorTool", ns)
    if tool is not None:
        out["software"] = tool.text
    for key, tag in (("make", "tiff:Make"), ("model", "tiff:Model")):
        el = desc.find(tag, ns)
        if el is not None:
            out[key] = el.text
    return out
