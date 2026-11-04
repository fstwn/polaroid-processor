"""Command operations shared by the CLI and the MCP server.

Every function returns plain JSON-serialisable records and never prints. Bad
arguments raise :class:`UsageError`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import flatten as flat
from .detection import DetectionError, detect, quad_size
from .geometry import shift_side
from .imageio import load_rgb, write_derivative
from .lock import RunLocked
from .manifest import Manifest, now
from .pipeline import (PHOTO_CROP, Derivative, Settings, collect_inputs, flatten_plan, manual_outline, reevaluate,
                       result_record, run_batch, settings_from_dict, window_record)
from .scale import scale_from_size
from .scan_window import (effective_window, fix_outside, fix_window, inside_outline, nominal_window, outline_id,
                          window_size_mm, window_state)
from .render import CORNERS, CYAN, inspect_sheet, quad_summary

SIDES = ("top", "right", "bottom", "left")
KINDS = ("square", "round")
INSPECT_HINT = ("Tiles show full-resolution pixels; grid labels are absolute scan coordinates. "
                "The green line is the current crop edge (curved when the print is flattened; the grey "
                "line is then the straight outline); magenta crosses are corners. "
                "Tiles marked 'stretched' are contrast-enhanced. With of=window the tiles sit on the corners "
                "of the photo window: the cyan line is the window border (a round window: its bounding square and "
                "the circle), the green line stays the print outline.")


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
                  photo_bleed_mm: Optional[float] = None) -> Settings:
    """``photo_crop`` / ``photo_bleed_mm`` left at None keep the value stored in the output folder's
    manifest (off / 0.3 for a new folder); ``run_batch`` resolves them."""
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
        return m, key, m.get(key)["input"]
    if not Path(scan).is_file():
        raise UsageError(f"{scan!r} is not a file (give the output folder to look it up in a manifest)")
    return None, None, scan


def flatten_manual(entry: dict) -> bool:
    return manual_outline((entry.get("override") or {}).get("quad"), entry.get("detection"))


def inspect_scan(scan: str, output: Optional[str] = None, *, image: Optional[str] = None, write: bool = True,
                 window: int = 360, zoom: int = 1, grid: int = 50, stretch: str = "auto",
                 of: str = "print") -> Tuple[dict, np.ndarray]:
    """Render the review sheet of one scan. Returns (record, sheet RGB).

    The PNG is written to ``image``, else to ``<output>/_inspect/<stem>.png`` when the scan
    comes from a manifest; with ``write=False`` nothing is written. Never writes into the
    input folder. ``window`` is the tile size in scan pixels.

    ``of="window"`` centres the tiles on the corners of the photo window (cyan; the print outline stays green,
    a round window shows its circle) instead of the print's corners; needs ``output`` with the photo crop on.
    A scan without a window shows the nominal window of its format, which ``fix`` can start from.
    """
    if stretch not in ("auto", "on", "off"):
        raise UsageError(f"stretch must be auto, on or off, not {stretch!r}")
    if of not in ("print", "window"):
        raise UsageError(f"of must be print or window, not {of!r}")
    m, key, path = resolve_scan(output, scan)
    rgb = load_rgb(path)
    quad = Manifest.effective_quad(m.get(key)) if m else None
    source = "override" if quad is not None and m.get(key).get("override") else "auto"
    if quad is None:
        quad = detect(rgb).quad
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
    outline = None
    e = m.get(key) if m else None
    if e and not flatten_manual(e):
        px = (e.get("scale") or {}).get("px_per_mm") or scale_from_size(quad_size(quad)).px_per_mm
        mode = (m.data.get("settings") or {}).get("flatten", "auto")
        fplan = flatten_plan((e.get("detection") or {}).get("curves"), px, mode, False)
        if fplan.apply:
            outline = flat.outline_points(flat.curves_from_dict(e["detection"]["curves"]), quad,
                                          **flat.FlattenParams().warp_px(px))
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
        if (e.get("window") or fix_window(e)) and (m.data.get("settings") or {}).get("photo_crop", "off") != "off":
            rec["window"] = window_record(e)
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
    if fix_outside(e, quad):      # what a window fix edits (see _hand_window), though the automatic window is cut
        w = fix_window(e)
        cut = "the automatic window is cut" if window_state(e.get("window")) == "found" else "no photo crop is cut"
        return np.array(w["quad"], float), w["kind"], f"manual, outside the print outline, not in force ({cut})"
    w = effective_window(e, quad)
    if window_state(w) == "found":
        return np.array(w["quad"], float), w["kind"], f"{w.get('source', 'auto')}, as stored"
    kind, fmt = batch_window_kind(m), (e.get("window") or {}).get("format")
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
    auto_fmt = (e.get("window") or {}).get("format")
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


def fix_scan(output: str, scan: str, *, sets=None, nudges=None, shifts=None, accept: bool = False,
             reset: bool = False, window_sets=None, window_nudges=None, window_kind: Optional[str] = None,
             no_photo_crop: bool = False, reset_window: bool = False, author: str = "human", note: str = "",
             export: bool = False, emit: Emit = _ignore) -> dict:
    """Store a corrected (or accepted) outline and/or a hand-set photo window in the manifest, optionally re-export.

    Outline: ``sets`` / ``nudges`` / ``shifts`` / ``accept`` / ``reset`` as before. Photo window (only with the
    photo crop on): ``window_sets`` and ``window_nudges`` move corners of the window (a scan without a window
    starts from the nominal one for its format; ``window_kind`` picks square or round, default the batch's
    majority), ``window_kind`` alone changes the kind, ``no_photo_crop`` stores the choice of no photo crop for
    this scan, ``reset_window`` takes the hand-set window (or that choice) away. These are stored in the entry's
    ``window_fix``, which only fixes write; the run keeps writing ``window``, the automatic window. A hand-set
    window stays through outline fixes while it lies inside the outline; when it does not, the scan is
    ``needs_review`` and the automatic window is used. ``accept`` also accepts a missing window for the outline
    the scan has then: the print is exported without a photo crop. The mark lapses when the outline changes.

    Returns the result record of the scan (type ``fix``), with the last history entries and,
    after an export, the batch summary under ``summary``.
    """
    m = Manifest(output)
    if not m.path.exists():
        raise UsageError(f"no manifest in {output}")
    key = m.find(scan)
    if key is None:
        raise UsageError(f"{scan!r} is not in the manifest of {output}")
    current = Manifest.effective_quad(m.get(key))
    crop_on = (m.data.get("settings") or {}).get("photo_crop", "off") != "off"
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
    if reset:
        m.set_override(key, None, author, note or "reset to automatic detection", "reset")
        if (m.get(key).get("window_fix") or {}).get("accepted"):      # an accept stores the detected outline as the
            m.set_window_accept(key, None, author)                    # override: its id would still match after the reset
    else:
        sets = check_fix_values(sets, CORNERS, 2, "set")
        nudges = check_fix_values(nudges, CORNERS, 2, "nudge")
        shifts = check_fix_values(shifts, SIDES, 1, "shift")
        if not (sets or nudges or shifts or accept or window_ops):
            raise UsageError("fix needs set, nudge, shift, accept, reset or a window option")
        if sets or nudges or shifts or accept:
            if current is None:
                raise UsageError("scan has no outline yet; set all four corners")
            new = apply_fix(current, sets, nudges, shifts)
            action = "accept" if accept and not (sets or nudges or shifts) else "fix"
            m.set_override(key, new, author, note or "", action)
    if accept and crop_on:
        # The mark answers what the reviewer sees: a window not found, or a hand-set window outside the outline. With a
        # window found it stores nothing, so a window that is missing at the next run is shown, not covered.
        e = m.get(key)
        quad = Manifest.effective_quad(e)
        if fix_outside(e, quad) or (e.get("window") and window_state(effective_window(e, quad)) == "missing"):
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
    if crop_on:      # the status follows at once (a run would give the same)
        reevaluate(m, [key], settings_from_dict(m.data.get("settings", {})))
    m.save()
    summary = None
    if export:
        settings = settings_from_dict(m.data.get("settings", {}))
        settings.workers = 1
        try:
            summary = run_batch([m.get(key)["input"]], output, settings, emit=emit)
        except RunLocked as e:       # the fix is stored; a retry would only add a second history entry
            m2 = Manifest(output)
            rec = result_record(m2, key)
            rec["type"] = "fix"
            rec["history"] = m2.history(key)[-3:]
            rec["note"] = f"fix saved; export skipped: {e}"
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
