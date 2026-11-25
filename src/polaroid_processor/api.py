"""Command operations shared by the CLI and the MCP server.

Every function returns plain JSON-serialisable records and never prints. Bad
arguments raise :class:`UsageError`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import flatten as flat
from .detection import DetectionError, detect, quad_size
from .geometry import shift_side
from . import metadata as meta_mod
from . import item as item_mod
from .imageio import file_fingerprint, load_rgb, read_meta, write_derivative
from .item import AUTO, KINDS as ITEM_KINDS, agent_crop, item_of
from .lock import RunLocked
from .manifest import Manifest, now
from .pipeline import (PHOTO_CROP, Derivative, Settings, collect_inputs, flatten_plan, item_record, manual_outline,
                       print_median_scale, reevaluate, result_record, run_batch, settings_from_dict, window_record)
from .scale import explicit_scale, scale_from_size
from .scan_window import (effective_window, fix_outside, fix_window, inside_outline, nominal_window, outline_id,
                          window_format, window_size_mm, window_state)
from .render import CORNERS, CYAN, inspect_sheet, nice_step, quad_summary, scan_sheet

SIDES = ("top", "right", "bottom", "left")
KINDS = ("square", "round")
INSPECT_HINT = ("Tiles show full-resolution pixels; grid labels are absolute scan coordinates. "
                "The green line is the current crop edge (curved when the print is flattened; the grey "
                "line is then the straight outline); magenta crosses are corners. "
                "Tiles marked 'stretched' are contrast-enhanced. With of=window the tiles sit on the corners "
                "of the photo window: the cyan line is the window border (a round window: its bounding square and "
                "the circle), the green line stays the print outline.")
SCAN_HINT = ("The scan has no outline yet (detection failed). The sheet shows the whole scan with a pixel grid in "
             "absolute scan coordinates and a start rectangle in cyan (the corners are in the record). Place the crop with "
             "fix: set all four corners (tl, tr, br, bl, clockwise) and an item kind (600, go or other; default other); "
             "then inspect the scan again to check the crop at full resolution.")


class UsageError(ValueError):
    """Invalid arguments or a request that cannot be served (exit code 2)."""


Emit = Callable[[dict], None]


def _ignore(_: dict) -> None:
    pass


# --------------------------------------------------------------------------- process / detect
def make_settings(*, margin_mm: float = 0.1, derivatives: Optional[Sequence[str]] = None,
                  no_derivatives: bool = False, interpolation: str = "cubic", tiff_compression: str = "zlib",
                  overlay: bool = False, ppi: Optional[float] = None, profile: Optional[str] = None,
                  profiles: Optional[str] = None, strict: bool = False, force: bool = False,
                  redetect: bool = False, dry_run: bool = False, workers: Optional[int] = None,
                  flatten: str = "auto", shadows: str = "auto", export: bool = True,
                  previews: bool = False, photo_crop: Optional[str] = None,
                  photo_bleed_mm: Optional[float] = None, metadata: Optional[dict] = None) -> Settings:
    """``photo_crop`` / ``photo_bleed_mm`` left at None keep the value stored in the output folder's
    manifest (off / 0.3 for a new folder); ``run_batch`` resolves them.

    ``metadata`` (title, author, copyright, description, keywords (a list), date_taken (YYYY, YYYY-MM or
    YYYY-MM-DD)) goes into every exported file; the texts may hold the placeholders {stem}, {n}, {project},
    {format} (``{{`` and ``}}`` for literal braces). A field left out keeps the output folder's value, an empty
    one (``""``, ``[]``) clears it; ``None`` keeps all. Bad values raise :class:`UsageError` here, never at export."""
    s = Settings()
    s.margin_mm = float(margin_mm)
    if derivatives is not None:
        s.derivatives = tuple(Derivative.parse(x) for x in derivatives)
    if no_derivatives:
        s.derivatives = ()
    if interpolation not in ("linear", "cubic", "lanczos"):
        raise UsageError(f"interpolation must be linear, cubic or lanczos, not {interpolation!r}")
    if tiff_compression not in ("zlib", "none"):
        raise UsageError(f"tiff_compression must be zlib or none, not {tiff_compression!r}")
    s.interpolation, s.tiff_compression, s.overlay = interpolation, tiff_compression, overlay
    s.ppi, s.profile, s.profiles_file = ppi, profile, profiles
    s.strict, s.force, s.redetect, s.dry_run = strict, force, redetect, dry_run
    for name, value in (("flatten", flatten), ("shadows", shadows)):
        if value not in ("auto", "off"):
            raise UsageError(f"{name} must be auto or off, not {value!r}")
    s.flatten, s.shadows = flatten, shadows
    s.workers = workers
    s.export, s.previews = export, previews
    if photo_crop is not None and photo_crop not in PHOTO_CROP:
        raise UsageError(f"photo_crop must be {', '.join(PHOTO_CROP)}, not {photo_crop!r}")
    if photo_bleed_mm is not None and not 0 <= float(photo_bleed_mm) <= 3:
        raise UsageError("photo_bleed_mm must be between 0 and 3")
    s.photo_crop = photo_crop
    s.photo_bleed_mm = None if photo_bleed_mm is None else float(photo_bleed_mm)
    try:
        s.metadata = None if metadata is None else meta_mod.normalize(metadata, keep_empty=True)
    except ValueError as e:
        raise UsageError(str(e)) from None
    return s


def process(inputs: Sequence[str], output: Optional[str], settings: Settings, *, recursive: bool = False,
            emit: Emit = _ignore, progress: Optional[Callable[[str, str], None]] = None,
            cancel: Optional[Callable[[], bool]] = None, on_jobs: Optional[Callable[[int], None]] = None,
            on_phase: Optional[Callable[[str], None]] = None) -> dict:
    """Run a batch. ``emit`` gets one result record per scan at the end, ``progress`` one call
    per finished job, ``on_jobs`` the number of scans to detect before the first one starts;
    ``cancel()`` turning true stops the scans not yet started, ``on_phase`` the step after detection.
    Returns the summary record.

    With ``settings.export`` off, the detections are stored in the manifest and nothing is exported.
    """
    if output is None and not settings.dry_run:
        raise UsageError("process needs an output folder (or dry_run to only detect)")
    return run_batch(list(inputs), output, settings, emit=emit, recursive=recursive, progress=progress,
                     cancel=cancel, on_jobs=on_jobs, on_phase=on_phase)


def detect_scans(inputs: Sequence[str], *, recursive: bool = False, emit: Emit = _ignore) -> List[dict]:
    """Detect outlines without writing anything."""
    records = []
    for f in collect_inputs(list(inputs), recursive):
        rec = {"type": "detection", "file": f.name, "input": str(f.resolve())}
        try:
            rec.update(status="ok", detection=detect(load_rgb(str(f))).to_dict())
        except DetectionError as e:
            rec.update(status="failed", error=str(e))
        emit(rec)
        records.append(rec)
    return records


# --------------------------------------------------------------------------- inspect
def resolve_scan(output: Optional[str], scan: str) -> Tuple[Optional[Manifest], Optional[str], str]:
    """Return (manifest or None, manifest key or None, input path)."""
    if output:
        m = Manifest(output)
        if not m.path.exists():
            raise UsageError(f"no manifest in {output}")
        key = m.find(scan)
        if key is None:
            raise UsageError(f"{scan!r} is not in the manifest of {output}")
        if not m.get(key).get("input"):
            raise UsageError(f"{scan!r} has not been detected yet; run detect first")
        return m, key, m.get(key)["input"]
    if not Path(scan).is_file():
        raise UsageError(f"{scan!r} is not a file (give the output folder to look it up in a manifest)")
    return None, None, scan


def flatten_manual(entry: dict) -> bool:
    return manual_outline((entry.get("override") or {}).get("quad"), entry.get("detection"))


def scan_size_px(e: Optional[dict], path: str) -> Tuple[int, int]:
    """Width and height of a scan: from the manifest entry, else read from the file."""
    meta = (e or {}).get("meta") or {}
    if meta.get("width") and meta.get("height"):
        return int(meta["width"]), int(meta["height"])
    disk = read_meta(path)
    return disk.width, disk.height


def start_px_per_mm(m: Optional[Manifest], key: Optional[str]) -> Optional[float]:
    """The scale the start rectangle of a 600 or Go crop is sized by: the project's ppi or profile, else the
    median of its prints with the same scan size (at least 3), else None (the rectangle then fills the scan to its
    limit with the format's aspect ratio). The TIFF tag is not used: nothing checks it."""
    e = m.get(key) if m and key else None
    if not m or not e:
        return None
    settings = settings_from_dict(m.data.get("settings") or {})
    try:
        explicit = explicit_scale(settings.ppi, settings.profile, settings.profiles_file)
    except Exception:  # noqa: BLE001  (a profile the file lacks: no scale, the rectangle is sized by the scan)
        explicit = None
    sc = explicit or print_median_scale(m, e)
    return sc.px_per_mm if sc else None


def start_rectangle(output: Optional[str], scan: str, item: str = "other") -> dict:
    """The rectangle the place step starts from for scan ``scan`` and item kind ``item`` (600, go or other): its corners
    (tl, tr, br, bl, scan px), the scan size and the scale it was sized by (None: none known). See ``item.start_rectangle``."""
    kind = _check_item(item or "other")
    if kind == AUTO:
        raise UsageError("the start rectangle needs an item kind: 600, go or other")
    m, key, path = resolve_scan(output, scan)
    w, h = scan_size_px(m.get(key) if m else None, path)
    px = start_px_per_mm(m, key) if kind != "other" else None
    rect = item_mod.start_rectangle(w, h, kind, px)
    return {"item": kind, "scan_px": [w, h], "px_per_mm": px, **quad_summary(rect)}


def inspect_scan(scan: str, output: Optional[str] = None, *, image: Optional[str] = None, write: bool = True,
                 window: int = 360, zoom: int = 1, grid: Optional[int] = None, stretch: str = "auto",
                 of: str = "print", item: Optional[str] = None) -> Tuple[dict, np.ndarray]:
    """Render the review sheet of one scan. Returns (record, sheet RGB).

    The PNG is written to ``image``, else to ``<output>/_inspect/<stem>.png`` when the scan
    comes from a manifest; with ``write=False`` nothing is written. Never writes into the
    input folder. ``window`` is the tile size in scan pixels.

    ``of="window"`` centres the tiles on the corners of the photo window (cyan; the print outline stays green,
    a round window shows its circle) instead of the print's corners; needs ``output`` with the photo crop on.
    A scan without a window shows the nominal window of its format, which ``fix`` can start from.

    A scan without an outline (its detection failed and nobody cropped it) gets an overview sheet instead: the whole
    scan with a pixel grid (``grid`` px, default about 20 lines along the long edge) and the start rectangle of item
    kind ``item`` (600, go or other; default other) in cyan; the record has ``failed: true`` and the rectangle's corners.
    ``grid`` is the tiles' grid (default 50 px) for a scan with an outline.
    """
    if stretch not in ("auto", "on", "off"):
        raise UsageError(f"stretch must be auto, on or off, not {stretch!r}")
    if of not in ("print", "window"):
        raise UsageError(f"of must be print or window, not {of!r}")
    if item is not None and _check_item(item) == AUTO:
        raise UsageError("item for inspect must be 600, go or other")
    m, key, path = resolve_scan(output, scan)
    rgb = load_rgb(path)
    quad = Manifest.effective_quad(m.get(key)) if m else None
    source = "override" if quad is not None and m.get(key).get("override") else "auto"
    overview = False
    if quad is None:
        if m is not None:      # failed in the manifest: detection would fail again
            if not _scan_is_open(m.get(key)):
                raise UsageError(f"{scan!r} has not been detected yet; run detect first")
            overview = True
        else:
            try:
                quad = detect(rgb).quad
            except DetectionError:
                overview = True
    stem = Path(path).stem
    img_path = None
    if image:
        img_path = Path(image)
    elif write and m:
        img_path = m.out_dir / "_inspect" / f"{stem}.png"
    elif write:
        img_path = Path.cwd() / f"{stem}.inspect.png"
    if img_path is not None and Path(path).resolve().parent == img_path.resolve().parent:
        raise UsageError("refusing to write the inspect image into the input folder; give another image path")
    e = m.get(key) if m else None
    if overview:
        return _inspect_failed(rgb, m, key, path, img_path, grid, item or "other")
    outline = None
    if e and not flatten_manual(e) and not item_mod.is_other(e):      # Other is exported straight: no curve to follow
        px = (e.get("scale") or {}).get("px_per_mm") or scale_from_size(quad_size(quad)).px_per_mm
        mode = (m.data.get("settings") or {}).get("flatten", "auto")
        fplan = flatten_plan((e.get("detection") or {}).get("curves"), px, mode, False)
        if fplan.apply:
            outline = flat.outline_points(flat.curves_from_dict(e["detection"]["curves"]), quad,
                                          **flat.FlattenParams().warp_px(px))
    grid = grid or 50
    header = f"{stem}  quad={source}{'  curved edges' if outline is not None else ''}  grid={grid}px  window={window}px"
    wq, wkind, wnote = None, None, ""
    if of == "window":
        wq, wkind, wnote = _window_to_inspect(m, key, quad)
        header = f"{stem}  photo window ({wkind}, {wnote})  grid={grid}px  window={window}px"
        sheet = inspect_sheet(rgb, wq, window, zoom, grid, header, stretch, None, color=CYAN, ref=quad,
                              circle=wkind == "round")
    else:
        sheet = inspect_sheet(rgb, quad, window, zoom, grid, header, stretch, outline)
    if img_path is not None:
        write_derivative(str(img_path), sheet, "png", 0, None)
    rec = {"type": "inspect", "file": Path(path).name, "image": str(img_path.resolve()) if img_path else None,
           "quad_source": source, **quad_summary(quad), "hint": INSPECT_HINT}
    if of == "window":
        rec.update(of="window", window_corners=quad_summary(wq)["corners"], window_kind=wkind, window_note=wnote)
    if m:
        e = m.get(key)
        rec.update(status=e.get("status"), warnings=e.get("warnings", []))
        if (e.get("window") or fix_window(e)) and (m.data.get("settings") or {}).get("photo_crop", "off") != "off" \
                and not item_mod.is_other(e):
            rec["window"] = window_record(e)
        if item_of(e):
            rec["item"] = item_record(e)
    return rec, sheet


def _inspect_failed(rgb: np.ndarray, m: Optional[Manifest], key: Optional[str], path: str, img_path: Optional[Path],
                    grid: Optional[int], kind: str) -> Tuple[dict, np.ndarray]:
    """The review sheet of a scan without an outline: the whole scan, a pixel grid and the start rectangle."""
    h, w = rgb.shape[:2]
    px = start_px_per_mm(m, key) if kind != "other" else None
    rect = item_mod.start_rectangle(w, h, kind, px)
    step = int(grid) if grid else nice_step(max(h, w) / 20)
    step = max(step, nice_step(max(h, w) / 200)) if step < max(h, w) / 200 else step      # at most ~200 lines
    stem = Path(path).stem
    header = f"{stem}  no outline yet  grid={step}px  start rectangle for item {kind} (cyan)"
    sheet = scan_sheet(rgb, rect, step, header)
    if img_path is not None:
        write_derivative(str(img_path), sheet, "png", 0, None)
    rec = {"type": "inspect", "file": Path(path).name, "image": str(img_path.resolve()) if img_path else None,
           "failed": True, "quad_source": None, "scan_px": [w, h], "grid": step, "item": kind, "px_per_mm": px,
           "start_rectangle": True, **quad_summary(rect), "hint": SCAN_HINT}
    if m:
        e = m.get(key)
        rec.update(status=e.get("status"), warnings=e.get("warnings", []))
        if e.get("error"):
            rec["error"] = e["error"]
    return rec, sheet


def batch_window_kind(m: Manifest) -> str:
    """The kind of window most scans of the batch have (round only when more than half do), else square."""
    kinds = []
    for _, e in m.items():
        w = effective_window(e)
        if window_state(w) == "found":
            kinds.append(w["kind"])
    return "round" if kinds and kinds.count("round") * 2 > len(kinds) else "square"


def _window_to_inspect(m: Optional[Manifest], key: Optional[str], quad: np.ndarray) -> Tuple[np.ndarray, str, str]:
    """Window quad, kind and a note for ``inspect_scan(of="window")``: the window in force, else the nominal one."""
    if m is None or (m.data.get("settings") or {}).get("photo_crop", "off") == "off":
        raise UsageError("of=window needs an output folder whose photo_crop is on")
    e = m.get(key)
    fmt_name = window_format(e)
    if item_mod.is_other(e):
        raise UsageError("an item of kind other has no photo window")
    if fix_outside(e, quad):      # what a window fix edits (see _hand_window), though the automatic window is cut
        w = fix_window(e)
        cut = "the automatic window is cut" if window_state(e.get("window")) == "found" else "no photo crop is cut"
        return np.array(w["quad"], float), w["kind"], f"manual, outside the print outline, not in force ({cut})"
    w = effective_window(e, quad)
    if window_state(w) == "found":
        return np.array(w["quad"], float), w["kind"], f"{w.get('source', 'auto')}, as stored"
    kind, fmt = batch_window_kind(m), fmt_name
    try:
        try:
            nom = nominal_window(quad, kind, fmt)
        except ValueError:
            kind = "square"
            nom = nominal_window(quad, kind, fmt)
    except ValueError as ex:
        raise UsageError(f"no nominal window for this scan: {ex}") from None
    why = "no photo crop by choice" if window_state(w) == "none" else "no window found"
    return np.array(nom["quad"], float), kind, f"{why}: nominal placement, not stored"


# --------------------------------------------------------------------------- fix
def apply_fix(quad: np.ndarray, sets: Optional[Mapping[str, Sequence[float]]] = None,
              nudges: Optional[Mapping[str, Sequence[float]]] = None,
              shifts: Optional[Mapping[str, Sequence[float]]] = None) -> np.ndarray:
    """Corner sets, corner nudges and side shifts (positive = outward), in scan pixels."""
    q = quad.astype(np.float64).copy()
    idx = {n: i for i, n in enumerate(CORNERS)}
    for n, (x, y) in (sets or {}).items():
        q[idx[n]] = (x, y)
    for n, (dx, dy) in (nudges or {}).items():
        q[idx[n]] += (dx, dy)
    for side, (d,) in (shifts or {}).items():
        q = shift_side(q, side, d)
    return q


def check_fix_values(values: Optional[Mapping[str, Sequence[float]]], names: Sequence[str], n_vals: int,
                     what: str) -> Dict[str, List[float]]:
    out = {}
    for k, v in (values or {}).items():
        k = str(k).strip().lower()
        if k not in names:
            raise UsageError(f"{what}: bad name {k!r}, expected one of {', '.join(names)}")
        vals = [float(x) for x in (v if isinstance(v, (list, tuple)) else [v])]
        if len(vals) != n_vals:
            raise UsageError(f"{what}: {k} needs {n_vals} number(s), got {len(vals)}")
        out[k] = vals
    return out


def check_window_quad(new: np.ndarray, quad: np.ndarray) -> None:
    """A hand-set window must be four finite corners in the order tl, tr, br, bl (clockwise on the scan, so the
    photo is not mirrored), convex, and inside the print outline."""
    pts = np.asarray(new, float)
    if pts.shape != (4, 2) or not np.isfinite(pts).all() or not cv2.isContourConvex(pts.astype(np.float32).reshape(-1, 1, 2)):
        raise UsageError("window corners must be finite and form a convex quadrilateral in the order tl, tr, br, bl")
    x, y = pts[:, 0], pts[:, 1]
    if float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) <= 0:      # image coordinates: y points down
        raise UsageError("window corners must run clockwise on the scan in the order tl, tr, br, bl "
                         "(the photo would be mirrored)")
    if not inside_outline(pts, quad):
        raise UsageError("window corners must lie inside the print outline")


def _hand_window(m: Manifest, key: str, wsets: dict, wnudges: dict, kind: Optional[str], author: str,
                 note: str) -> dict:
    """The window dict for a hand edit: the hand-set window, else the automatic one (or, when none was found, the
    nominal one) with the corner sets and nudges applied, validated against the outline. ``kind`` changes the
    window's kind."""
    e = m.get(key)
    quad = Manifest.effective_quad(e)
    if quad is None:
        raise UsageError("scan has no outline yet")
    w = fix_window(e)
    if window_state(w) != "found":
        w = e.get("window")
    auto_fmt = window_format(e)
    if window_state(w) == "found":
        base, fmt, chosen = np.array(w["quad"], float), w.get("format"), kind or w["kind"]
    else:        # no window to start from: the nominal one (kind: asked for, else the batch's)
        chosen = kind or batch_window_kind(m)
        try:
            nom = nominal_window(quad, chosen, auto_fmt)
        except ValueError as ex:
            raise UsageError(f"cannot place a {chosen} window: {ex}") from None
        base, fmt = np.array(nom["quad"], float), nom["format"]
    new = apply_fix(base, wsets, wnudges)
    check_window_quad(new, quad)
    return {"found": True, "reason": "", "kind": chosen, "quad": np.round(new, 2).tolist(),
            "size_mm": window_size_mm(new, quad, fmt), "format": fmt, "notes": [], "outline": outline_id(quad),
            "source": "manual", "author": author, "at": now(), "note": note}


def _check_item(item: Optional[str]) -> Optional[str]:
    """The item kind given to ``fix`` / ``inspect`` (600, go, other, or auto for "as detected"), as stored."""
    if item is None:
        return None
    k = str(item).strip().lower()
    if k not in (*ITEM_KINDS, AUTO):
        raise UsageError(f"item must be one of {', '.join((*ITEM_KINDS, AUTO))}, not {item!r}")
    return k


def check_crop_quad(quad, width: float, height: float) -> np.ndarray:
    """The checks of a hand-drawn crop, the same for the page, the CLI, the API and the MCP server (see
    ``item.check_crop_quad``): inside the scan, clockwise, no crossing sides, every side at least 50 px, rotated by at
    most 45 deg. ``UsageError`` with the reason; returns the quad."""
    try:
        return item_mod.check_crop_quad(quad, width, height)
    except ValueError as ex:
        raise UsageError(str(ex)) from None


def _is_agent(author: str, agent: Optional[bool]) -> bool:
    """Every MCP fix is an agent's (``agent=True``), whatever author it gives; in the CLI and API the author ``agent`` is
    one too (``agent=False`` does not undo that: the name decides when it says agent)."""
    return bool(agent) or str(author).strip().lower() == "agent"


def _check_scan_fresh(e: dict) -> None:
    """A hand crop or an item choice is made only on the scan file the manifest entry describes: ``UsageError`` when the
    file is missing or its size or time differs from the entry (a time that differs while the content fingerprint is the
    same is accepted). A run takes in a changed file (and drops the crop and the choices made on the old one)."""
    path = e.get("input")
    try:
        st = os.stat(path)
    except (OSError, TypeError):
        raise UsageError("the scan file is missing; run process first") from None
    if e.get("size_bytes") is None or e.get("mtime") is None:
        return
    if st.st_size != e["size_bytes"]:
        raise UsageError("the scan file changed since the last run; run process first")
    if abs(st.st_mtime - e["mtime"]) >= 1e-3:
        fp = e.get("fingerprint")
        if not fp or file_fingerprint(path) != fp:
            raise UsageError("the scan file changed since the last run; run process first")


def _scan_is_open(e: dict) -> bool:
    """A scan that was run and has an outline, a crop or a failure to show: only these can be cropped or inspected. One
    that was never detected (pending) stays closed (decision 3)."""
    return bool(e.get("detection") or e.get("override") or e.get("status") == "failed")


def fix_scan(output: str, scan: str, *, sets=None, nudges=None, shifts=None, accept: bool = False,
             reset: bool = False, window_sets=None, window_nudges=None, window_kind: Optional[str] = None,
             no_photo_crop: bool = False, reset_window: bool = False, item: Optional[str] = None,
             scan_ppi: Optional[float] = None, agent: Optional[bool] = None, author: str = "human", note: str = "",
             export: bool = False, emit: Emit = _ignore) -> dict:
    """Store a corrected (or accepted) outline, a hand-set photo window and/or an item choice in the manifest,
    optionally re-export.

    Outline: ``sets`` / ``nudges`` / ``shifts`` / ``accept`` / ``reset`` as before. Photo window (only with the
    photo crop on): ``window_sets`` and ``window_nudges`` move corners of the window (a scan without a window
    starts from the nominal one for its format; ``window_kind`` picks square or round, default the batch's
    majority), ``window_kind`` alone changes the kind, ``no_photo_crop`` stores the choice of no photo crop for
    this scan, ``reset_window`` takes the hand-set window (or that choice) away. These are stored in the entry's
    ``window_fix``, which only fixes write; the run keeps writing ``window``, the automatic window. A hand-set
    window stays through outline fixes while it lies inside the outline; when it does not, the scan is
    ``needs_review`` and the automatic window is used. ``accept`` also accepts a missing window for the outline
    the scan has then: the print is exported without a photo crop. The mark lapses when the outline changes.

    Item: ``item`` = ``600`` / ``go`` (a print of that format: it replaces the shape guess everywhere) / ``other``
    (any flat item: exported exactly as drawn, no margin, flattening, shadow removal or photo crop, no format) /
    ``auto`` (back to "as detected"); ``scan_ppi`` = the scan's own resolution (kind other only; 0 takes it away).
    They are stored in the entry's ``item_fix``, which only fixes write. A scan whose detection failed is cropped by
    setting all four corners (``sets``), which turns it ``ok`` at once; this first crop has the item kind ``other``
    unless ``item`` says another (it needs a kind, so ``auto`` is refused there). A scan that was never detected
    (pending) cannot be cropped. A moved outline is checked as a hand crop (inside the scan, clockwise, no crossing
    sides, convex, every side and every corner-to-opposite-line distance at least 50 px, rotated by at most 45 deg)
    when the scan has an item kind or no detected outline;
    a plain fix of a detected print, an accept and an item alone are not checked.

    A crop an agent places or moves on a scan without a detected outline (``agent=True``, or ``author="agent"``)
    gets the mark ``entry["agent_crop"]``: the scan is ``needs_review`` with the warning ``cropped by agent`` until a
    human confirms (``accept``) or saves its corners (also together with an item); an item or ppi change alone keeps
    it, and an agent's accept, re-save or item change does not clear it. A crop that carries the mark counts in no
    batch scale and no reference size. Every change of the mark is a history entry. ``reset`` takes the crop, the item kind, the own ppi and the
    mark away: back to the automatic outline, or ``failed`` with its first error without one. On a scan without a
    detection an agent's ``reset`` is allowed only on a crop that still carries the mark (it goes with the crop), else
    ``UsageError`` and nothing is stored; a human's reset is always allowed; a reset on a scan that was never detected
    (pending) is refused like ``set`` and an item.

    An agent may not change or remove an item choice or an own ppi that a human set (``item_fix["by_human"]``; one without
    the record counts as a human's): ``item``, ``item="auto"``, ``scan_ppi`` and ``reset`` are refused with ``UsageError`` and
    nothing is stored, with or without a detected outline, also on an agent's own unconfirmed crop whose item or ppi a human
    changed. The same item given again, and corners together with an unchanged item, stay allowed; a human's actions are
    always allowed.

    Returns the result record of the scan (type ``fix``), with the last history entries and,
    after an export, the batch summary under ``summary``.
    """
    m = Manifest(output)
    if not m.path.exists():
        raise UsageError(f"no manifest in {output}")
    key = m.find(scan)
    if key is None:
        raise UsageError(f"{scan!r} is not in the manifest of {output}")
    e = m.get(key)
    current = Manifest.effective_quad(e)
    crop_on = (m.data.get("settings") or {}).get("photo_crop", "off") != "off"
    by_agent = _is_agent(author, agent)
    wsets = check_fix_values(window_sets, CORNERS, 2, "window_set")
    wnudges = check_fix_values(window_nudges, CORNERS, 2, "window_nudge")
    if window_kind is not None and window_kind not in KINDS:
        raise UsageError(f"window_kind must be square or round, not {window_kind!r}")
    window_ops = bool(wsets or wnudges or window_kind or no_photo_crop or reset_window)
    if window_ops and not crop_on:
        raise UsageError("photo window options need photo_crop on in this output folder (process with photo_crop)")
    if no_photo_crop and (wsets or wnudges or window_kind or reset_window):
        raise UsageError("no_photo_crop cannot be combined with other window options")
    if reset_window and (wsets or wnudges or window_kind):
        raise UsageError("reset_window cannot be combined with other window options")
    if reset_window and not fix_window(m.get(key)):
        raise UsageError("reset_window: this scan has no hand-set window and no no-photo-crop choice")
    item = _check_item(item)
    ppi_given = scan_ppi is not None
    ppi = None
    if ppi_given:
        try:
            ppi = None if _is_zero(scan_ppi) else item_mod.check_scan_ppi(scan_ppi)
        except ValueError as ex:
            raise UsageError(str(ex)) from None
    old_item = item_of(e)
    old_kind = old_item["kind"] if old_item else None
    mark_old = agent_crop(e)
    detected = bool(e.get("detection"))
    judge = crop_on      # the status follows at once (a run would give the same); with the crop off only for item work
    if reset:
        if item is not None or ppi_given:
            raise UsageError("reset cannot be combined with item or scan_ppi")
        if not _scan_is_open(e):
            raise UsageError("this scan has not been detected yet; run detect first")
        if by_agent and not detected and (e.get("override") or old_item) and not (mark_old and mark_old.get("from_start") is True):
            # a crop on a scan without a detection is the agent's to reset only when it was an agent's from the start: no
            # human saved corners or confirmed it since (the mark says so; one without the information is read as a human's)
            raise UsageError("an agent can reset only a crop that was an agent's from the start and still carries the agent "
                             "mark; a human's crop (or one a human saved or confirmed since) is reset by a human")
        if by_agent and old_item and old_item.get("by_human", True):
            # a human set the kind or the ppi (an item without a record of who is read as a human's): not the agent's to take
            # away, with or without a detected outline (also on an agent's own crop whose item or ppi a human changed)
            raise UsageError("an agent cannot take away an item choice or an own ppi that a human set; a human resets it")
        had_crop = bool(old_item or mark_old or (not detected and e.get("override")))
        m.set_override(key, None, author, note or "reset to automatic detection", "reset")
        if (m.get(key).get("window_fix") or {}).get("accepted"):      # an accept stores the detected outline as the
            m.set_window_accept(key, None, author)                    # override: its id would still match after the reset
        if old_item:
            m.set_item_fix(key, None, author, note or "reset with the crop", "item-reset")
        if mark_old:
            m.set_agent_crop(key, None, author, note or "reset with the crop", "agent-crop-clear")
        judge = judge or bool(old_item) or bool(mark_old)      # an item or a mark taken away changes the judgement
    else:
        sets = check_fix_values(sets, CORNERS, 2, "set")
        nudges = check_fix_values(nudges, CORNERS, 2, "nudge")
        shifts = check_fix_values(shifts, SIDES, 1, "shift")
        if not (sets or nudges or shifts or accept or window_ops or item is not None or ppi_given):
            raise UsageError("fix needs set, nudge, shift, accept, reset, an item option or a window option")
        geometry = bool(sets or nudges or shifts)      # a corner moves (an accept and an item change none)
        new = None
        if sets or nudges or shifts or accept:
            if current is None:      # a scan without an outline: all four corners alone make the first crop
                if not _scan_is_open(e):
                    raise UsageError("this scan has not been detected yet; run detect first (only a scan whose "
                                     "detection failed can be cropped by hand)")
                if len(sets) != len(CORNERS):
                    raise UsageError("scan has no outline yet; set all four corners")
                if nudges or shifts or accept:
                    raise UsageError("scan has no outline yet; give all four corners with set alone, without nudge, "
                                     "shift or accept")
                new = apply_fix(np.zeros((4, 2)), sets)
            else:
                new = apply_fix(current, sets, nudges, shifts)
        elif (item is not None or ppi_given) and current is None:
            if not _scan_is_open(e):
                raise UsageError("this scan has not been detected yet; run detect first")
            raise UsageError("scan has no outline yet; set all four corners together with the item")
        first_crop = current is None and new is not None
        if (geometry and not detected) or item is not None or ppi_given:
            _check_scan_fresh(e)      # a hand crop or an item is made on the picture the manifest knows
        kind = old_kind
        if item == AUTO:
            if first_crop:
                raise UsageError("a first crop needs an item kind (600, go or other); reset takes a crop away")
            kind = None
        elif item:
            kind = item
        elif first_crop and kind is None:
            kind = "other"      # the first choice on a failed scan
        if window_ops and kind == "other":
            raise UsageError("window options do not apply to an item of kind other (it has no photo crop)")
        if ppi_given and ppi is not None and kind != "other":
            raise UsageError("scan_ppi is the resolution of an item of kind other; it does not apply to "
                             f"{'600 or go' if kind else 'a scan as detected'}")
        if geometry and (kind is not None or not detected):      # hand-crop rules; a plain fix of a print is as before
            w, h = scan_size_px(e, e["input"])
            check_crop_quad(new, w, h)
        old_ppi = old_item.get("scan_ppi") if old_item else None
        new_ppi = (ppi if ppi_given else old_ppi) if kind == "other" else None
        if by_agent and old_item and old_item.get("by_human", True) and (kind, new_ppi) != (old_kind, old_ppi):
            # a human set some part of the choice (an item without a record of who is read as a human's): an agent may not
            # replace it, take it away (--item auto) or change its ppi; the same item given again changes nothing
            raise UsageError("an agent cannot change or take away an item choice or an own ppi that a human set; "
                             "a human changes it")
        mark = mark_old
        if by_agent:
            if geometry and not detected:
                # from_start: the crop was an agent's from its first corners on (a first crop, or a mark that already says
                # so); a crop a human saved or confirmed since has no mark, so it starts a new one that does not
                mark = {"author": author, "at": now(), "note": note,
                        "from_start": current is None or bool(mark_old and mark_old.get("from_start") is True)}
        elif sets or nudges or shifts or accept:
            mark = None      # a human confirmed it or saved corners (also with an item); an item or ppi alone keeps it
        if new is not None:
            action = "accept" if accept and not geometry else "fix"
            m.set_override(key, new, author, note or "", action)
        if kind is None:
            if old_item:
                m.set_item_fix(key, None, author, note or "back to as detected", "item-reset")
        elif (kind, new_ppi) != (old_kind, old_ppi):
            # by_human: a human set some part of the choice (sticky while it exists; one without the record counts as human)
            by_human = bool(old_item and old_item.get("by_human", True)) or not by_agent
            m.set_item_fix(key, {"kind": kind, "scan_ppi": new_ppi, "author": author, "at": now(), "note": note,
                                 "by_human": by_human}, author, note)
        if mark != mark_old:
            m.set_agent_crop(key, mark, author, note, "agent-crop" if mark else "agent-crop-clear")
        judge = judge or bool(old_item) or kind is not None or item is not None or ppi_given or (
            not detected and new is not None) or mark != mark_old
    if accept and crop_on:
        # The mark answers what the reviewer sees: a window not found, or a hand-set window outside the outline. With a
        # window found it stores nothing, so a window that is missing at the next run is shown, not covered.
        e = m.get(key)
        quad = Manifest.effective_quad(e)
        if not item_mod.is_other(e) and (fix_outside(e, quad) or (
                e.get("window") and window_state(effective_window(e, quad)) == "missing")):
            m.set_window_accept(key, outline_id(quad), author, note)
    if window_ops:
        if no_photo_crop:
            m.set_window_fix(key, {"found": False, "reason": "no photo crop by choice", "kind": None, "format": None,
                                   "notes": [], "source": "manual", "author": author, "at": now(), "note": note},
                             author, note, "window-none")
        elif reset_window:
            m.set_window_fix(key, None, author, note or "back to the automatic window", "window-reset")
        else:
            m.set_window_fix(key, _hand_window(m, key, wsets, wnudges, window_kind, author, note), author, note)
    e = m.get(key)
    if judge and (e.get("detection") or e.get("override")):
        reevaluate(m, [key], settings_from_dict(m.data.get("settings", {})))
    if reset and had_crop and not e.get("detection") and not e.get("override"):
        # a crop was taken away from a scan nothing was detected on: back to where detection left it, failed with its
        # first error, and nothing of the crop's judgement stays
        e["status"], e["warnings"] = "failed", []
        e["error"] = e.pop("detection_error", None) or e.get("error") or "detection failed"
        e.pop("scale", None)
        e.pop("format", None)
    m.save()
    summary = None
    if export:
        settings = settings_from_dict(m.data.get("settings", {}))
        settings.workers = 1
        try:
            summary = run_batch([m.get(key)["input"]], output, settings, emit=emit)
        except RunLocked as ex:       # the fix is stored; a retry would only add a second history entry
            m2 = Manifest(output)
            rec = result_record(m2, key)
            rec["type"] = "fix"
            rec["history"] = m2.history(key)[-3:]
            rec["note"] = f"fix saved; export skipped: {ex}"
            return rec
    m2 = Manifest(output)
    rec = result_record(m2, key, exported=bool(summary and summary.get("exported")))
    rec["type"] = "fix"
    rec["history"] = m2.history(key)[-3:]
    if summary is not None:
        rec["summary"] = summary
    else:
        rec["note"] = "manifest updated; run process again or fix with export to write files"
    return rec


def _is_zero(v) -> bool:
    """``scan_ppi`` given as the number 0 (not a bool): take the own ppi away."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0


# --------------------------------------------------------------------------- status
def status(output: str, *, needs_review: bool = False, emit: Emit = _ignore) -> dict:
    """Emit one record per scan (only flagged ones with ``needs_review``); return the summary."""
    m = Manifest(output)
    if not m.path.exists():
        raise UsageError(f"no manifest in {output}")
    counts: Dict[str, int] = {}
    for key, e in sorted(m.items()):
        st = e.get("status", "pending")
        counts[st] = counts.get(st, 0) + 1
        if needs_review and st != "needs_review":
            continue
        rec = result_record(m, key)
        rec["type"] = "status"
        emit(rec)
    return {"type": "summary", "scans": len(m.scans), "counts": counts, "manifest": str(m.path)}
