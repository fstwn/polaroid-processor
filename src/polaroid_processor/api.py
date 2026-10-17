"""Command operations shared by the CLI and the MCP server.

Every function returns plain JSON-serialisable records and never prints. Bad
arguments raise :class:`UsageError`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .detection import DetectionError, detect
from .imageio import load_rgb, write_derivative
from .manifest import Manifest
from .pipeline import Derivative, Settings, collect_inputs, result_record, run_batch, settings_from_dict
from .render import CORNERS, inspect_sheet, quad_summary

SIDES = ("top", "right", "bottom", "left")
INSPECT_HINT = ("Tiles show full-resolution pixels; grid labels are absolute scan coordinates. "
                "The green line is the current crop edge; magenta crosses are corners. "
                "Tiles marked 'stretched' are contrast-enhanced.")


class UsageError(ValueError):
    """Invalid arguments or a request that cannot be served (exit code 2)."""


Emit = Callable[[dict], None]


def _ignore(_: dict) -> None:
    pass


# --------------------------------------------------------------------------- process / detect
def make_settings(*, margin_mm: float = 0.2, derivatives: Optional[Sequence[str]] = None,
                  no_derivatives: bool = False, interpolation: str = "cubic", tiff_compression: str = "zlib",
                  overlay: bool = False, ppi: Optional[float] = None, profile: Optional[str] = None,
                  profiles: Optional[str] = None, strict: bool = False, force: bool = False,
                  redetect: bool = False, dry_run: bool = False, workers: Optional[int] = None) -> Settings:
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
    s.workers = workers
    return s


def process(inputs: Sequence[str], output: Optional[str], settings: Settings, *, recursive: bool = False,
            emit: Emit = _ignore, progress: Optional[Callable[[str, str], None]] = None) -> dict:
    """Run a batch. ``emit`` gets one result record per scan at the end, ``progress`` one call
    per finished job; returns the summary record."""
    if output is None and not settings.dry_run:
        raise UsageError("process needs an output folder (or dry_run to only detect)")
    return run_batch(list(inputs), output, settings, emit=emit, recursive=recursive, progress=progress)


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


def inspect_scan(scan: str, output: Optional[str] = None, *, image: Optional[str] = None, write: bool = True,
                 window: int = 360, zoom: int = 1, grid: int = 50, stretch: str = "auto") -> Tuple[dict, np.ndarray]:
    """Render the review sheet of one scan. Returns (record, sheet RGB).

    The PNG is written to ``image``, else to ``<output>/_inspect/<stem>.png`` when the scan
    comes from a manifest; with ``write=False`` nothing is written. Never writes into the
    input folder.
    """
    if stretch not in ("auto", "on", "off"):
        raise UsageError(f"stretch must be auto, on or off, not {stretch!r}")
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
    header = f"{stem}  quad={source}  grid={grid}px  window={window}px"
    sheet = inspect_sheet(rgb, quad, window, zoom, grid, header, stretch)
    if img_path is not None:
        write_derivative(str(img_path), sheet, "png", 0, None)
    rec = {"type": "inspect", "file": Path(path).name, "image": str(img_path.resolve()) if img_path else None,
           "quad_source": source, **quad_summary(quad), "hint": INSPECT_HINT}
    if m:
        e = m.get(key)
        rec.update(status=e.get("status"), warnings=e.get("warnings", []))
    return rec, sheet


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
    side_corners = {"top": ("tl", "tr"), "right": ("tr", "br"), "bottom": ("br", "bl"), "left": ("bl", "tl")}
    centre = q.mean(axis=0)
    for side, (d,) in (shifts or {}).items():
        a, b = (idx[c] for c in side_corners[side])
        t = q[b] - q[a]
        n = np.array([-t[1], t[0]]) / np.linalg.norm(t)
        if np.dot((q[a] + q[b]) / 2 - centre, n) < 0:
            n = -n
        q[a] += d * n
        q[b] += d * n
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


def fix_scan(output: str, scan: str, *, sets=None, nudges=None, shifts=None, accept: bool = False,
             reset: bool = False, author: str = "human", note: str = "", export: bool = False,
             emit: Emit = _ignore) -> dict:
    """Store a corrected (or accepted) outline in the manifest, optionally re-export.

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
    if reset:
        m.set_override(key, None, author, note or "reset to automatic detection", "reset")
    else:
        sets = check_fix_values(sets, CORNERS, 2, "set")
        nudges = check_fix_values(nudges, CORNERS, 2, "nudge")
        shifts = check_fix_values(shifts, SIDES, 1, "shift")
        if not (sets or nudges or shifts or accept):
            raise UsageError("fix needs set, nudge, shift, accept or reset")
        if current is None:
            raise UsageError("scan has no outline yet; set all four corners")
        new = apply_fix(current, sets, nudges, shifts)
        action = "accept" if accept and not (sets or nudges or shifts) else "fix"
        m.set_override(key, new, author, note or "", action)
    m.save()
    summary = None
    if export:
        settings = settings_from_dict(m.data.get("settings", {}))
        settings.workers = 1
        summary = run_batch([m.get(key)["input"]], output, settings, emit=emit)
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
