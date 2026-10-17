"""Command line interface.

Machine-readable mode (``--json``): one JSON object per line on stdout, logs on stderr.
Exit codes: 0 success (scans may still need review), 1 at least one scan failed,
2 usage error.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from . import __version__
from .detection import DetectionError, detect, manual_detection
from .imageio import load_rgb, write_derivative
from .manifest import Manifest
from .pipeline import (DEFAULT_DERIVATIVES, Derivative, Settings, collect_inputs, result_record, run_batch,
                       settings_from_dict)
from .render import CORNERS, inspect_sheet, quad_summary

log = logging.getLogger("polaroid_processor")


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _human(rec: dict) -> None:
    if rec.get("type") == "summary":
        c = rec["counts"]
        print(f"\n{rec['scans']} scan(s): " + ", ".join(f"{v} {k}" for k, v in sorted(c.items()))
              + f" | exported {rec['exported']} | {rec['time_s']} s")
        for k in rec.get("needs_review", []):
            print(f"  needs review: {k}")
        for k in rec.get("failed", []):
            print(f"  FAILED: {k}")
        if rec.get("manifest"):
            print(f"manifest: {rec['manifest']}")
        return
    size = rec.get("size_px")
    fmt = rec.get("format") or {}
    line = f"{rec['status']:<13} {rec['file']}"
    if size:
        line += f"  {size[0]:.0f}x{size[1]:.0f}px"
    if fmt:
        line += f"  {fmt['name']} {fmt['size_mm'][0]:.1f}x{fmt['size_mm'][1]:.1f}mm"
    if rec.get("error"):
        line += f"  ERROR {rec['error']}"
    elif rec.get("warnings"):
        line += "  [" + ", ".join(w for w in rec["warnings"] if not w.startswith("hint:")) + "]"
    print(line)


def _setup_logging(verbose: int, quiet: bool) -> None:
    level = logging.WARNING if quiet else (logging.DEBUG if verbose > 1 else logging.INFO if verbose else logging.WARNING)
    logging.basicConfig(stream=sys.stderr, level=level, format="%(levelname)s %(message)s")


# --------------------------------------------------------------------------- process / detect
def _settings_from_args(a) -> Settings:
    s = Settings()
    s.margin_mm = a.margin_mm
    if a.derivative is not None:
        s.derivatives = tuple(Derivative.parse(x) for x in a.derivative)
    if a.no_derivatives:
        s.derivatives = ()
    s.interpolation = a.interpolation
    s.tiff_compression = a.tiff_compression
    s.overlay = a.overlay
    s.ppi, s.profile, s.profiles_file = a.ppi, a.profile, a.profiles
    s.strict, s.force, s.redetect = a.strict, a.force, a.redetect
    s.dry_run = a.dry_run
    s.workers = a.workers
    return s


def cmd_process(a) -> int:
    out = a.output
    if out is None and not a.dry_run:
        raise SystemExit("process: --output is required (use --dry-run to only detect)")
    settings = _settings_from_args(a)
    emit = _emit if a.json else _human
    summary = run_batch(a.inputs, out, settings, emit=emit, recursive=a.recursive)
    emit(summary)
    return 1 if summary["failed"] else 0


def cmd_detect(a) -> int:
    failed = False
    for f in collect_inputs(a.inputs, a.recursive):
        rec = {"type": "detection", "file": f.name, "input": str(f.resolve())}
        try:
            det = detect(load_rgb(str(f)))
            rec.update(status="ok", detection=det.to_dict())
        except DetectionError as e:
            failed = True
            rec.update(status="failed", error=str(e))
        if a.json:
            _emit(rec)
        else:
            if rec["status"] == "ok":
                d = rec["detection"]
                print(f"ok      {f.name}  {d['size_px'][0]:.0f}x{d['size_px'][1]:.0f}px  aspect {d['aspect']}  "
                      f"skew {d['skew_deg']}deg  thr {d['speck_threshold']}  {d['time_s']}s")
            else:
                print(f"FAILED  {f.name}  {rec['error']}")
    return 1 if failed else 0


# --------------------------------------------------------------------------- inspect
def _resolve_scan(manifest_dir: Optional[str], scan: str):
    """Return (manifest or None, key or None, input path)."""
    if manifest_dir:
        m = Manifest(manifest_dir)
        key = m.find(scan)
        if key is None:
            raise SystemExit(f"{scan!r} is not in the manifest of {manifest_dir}")
        return m, key, m.get(key)["input"]
    if not Path(scan).is_file():
        raise SystemExit(f"{scan!r} is not a file (pass --manifest/-m to look it up in an output folder)")
    return None, None, scan


def cmd_inspect(a) -> int:
    m, key, path = _resolve_scan(a.manifest, a.scan)
    rgb = load_rgb(path)
    quad = Manifest.effective_quad(m.get(key)) if m else None
    source = ("override" if m.get(key).get("override") else "auto") if quad is not None else "auto"
    if quad is None:
        quad = detect(rgb).quad
    stem = Path(path).stem
    if a.image:
        img_path = Path(a.image)
    elif m:
        img_path = m.out_dir / "_inspect" / f"{stem}.png"
    else:
        img_path = Path.cwd() / f"{stem}.inspect.png"
    if Path(path).resolve().parent == img_path.resolve().parent:
        raise SystemExit("refusing to write the inspect image into the input folder; pass --image")
    header = f"{stem}  quad={source}  grid={a.grid}px  window={a.window}px"
    sheet = inspect_sheet(rgb, quad, a.window, a.zoom, a.grid, header, a.stretch)
    write_derivative(str(img_path), sheet, "png", 0, None)
    rec = {"type": "inspect", "file": Path(path).name, "image": str(img_path.resolve()), "quad_source": source,
           **quad_summary(quad),
           "hint": "Tiles show full-resolution pixels; grid labels are absolute scan coordinates. "
                   "The green line is the current crop edge; magenta crosses are corners."}
    if a.json:
        _emit(rec)
    else:
        print(f"inspect image: {rec['image']}")
        for n, c in rec["corners"].items():
            print(f"  {n}: {c[0]:.1f}, {c[1]:.1f}")
    return 0


# --------------------------------------------------------------------------- fix
def _parse_pairs(items: Optional[List[str]], names, n_vals: int):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"bad value {it!r}: expected NAME=...")
        k, v = it.split("=", 1)
        k = k.strip().lower()
        if k not in names:
            raise SystemExit(f"bad name {k!r}: expected one of {', '.join(names)}")
        vals = [float(x) for x in v.split(",")]
        if len(vals) != n_vals:
            raise SystemExit(f"bad value {it!r}: expected {n_vals} number(s)")
        out[k] = vals
    return out


def apply_fix(quad: np.ndarray, sets=None, nudges=None, shifts=None) -> np.ndarray:
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


def cmd_fix(a) -> int:
    m = Manifest(a.output)
    key = m.find(a.scan)
    if key is None:
        raise SystemExit(f"{a.scan!r} is not in the manifest of {a.output}")
    e = m.get(key)
    current = Manifest.effective_quad(e)
    if a.reset:
        m.set_override(key, None, a.author, a.note or "reset to automatic detection", "reset")
    else:
        sets = _parse_pairs(a.set, CORNERS, 2)
        nudges = _parse_pairs(a.nudge, CORNERS, 2)
        shifts = _parse_pairs(a.shift, ("top", "right", "bottom", "left"), 1)
        if not (sets or nudges or shifts or a.accept):
            raise SystemExit("fix: give --set, --nudge, --shift, --accept or --reset")
        if current is None:
            raise SystemExit("fix: scan has no quad yet; set all four corners with --set")
        new = apply_fix(current, sets, nudges, shifts)
        action = "accept" if a.accept and not (sets or nudges or shifts) else "fix"
        m.set_override(key, new, a.author, a.note or "", action)
    m.save()
    rc = 0
    if a.export:
        settings = settings_from_dict(m.data.get("settings", {}))
        settings.workers = 1
        summary = run_batch([m.get(key)["input"]], a.output, settings,
                            emit=_emit if a.json else _human)
        rc = 1 if summary["failed"] else 0
    else:
        m2 = Manifest(a.output)
        rec = result_record(m2, key)
        rec["type"] = "fix"
        rec["history"] = m2.history(key)[-3:]
        rec["note"] = "manifest updated; run 'polaroid process' again or pass --export to write files"
        (_emit if a.json else lambda r: print(json.dumps(r, indent=1)))(rec)
    return rc


# --------------------------------------------------------------------------- status
def cmd_status(a) -> int:
    m = Manifest(a.output)
    if not m.path.exists():
        raise SystemExit(f"no manifest in {a.output}")
    counts = {}
    for key, e in sorted(m.items()):
        st = e.get("status", "pending")
        counts[st] = counts.get(st, 0) + 1
        if a.needs_review and st != "needs_review":
            continue
        rec = result_record(m, key)
        rec["type"] = "status"
        (_emit if a.json else _human)(rec)
    summary = {"type": "summary", "scans": len(m.scans), "counts": counts, "manifest": str(m.path)}
    if a.json:
        _emit(summary)
    else:
        print("\n" + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    return 0


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="polaroid", description="Detect, crop and rectify scans of instant prints.")
    p.add_argument("--version", action="version", version=f"polaroid-processor {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--json", action="store_true", help="one JSON object per line on stdout")
        sp.add_argument("-v", "--verbose", action="count", default=0, help="log progress to stderr")
        sp.add_argument("-q", "--quiet", action="store_true")

    sp = sub.add_parser("process", help="detect, rectify and export scans (resumable)")
    sp.add_argument("inputs", nargs="+", help="scan files or folders")
    sp.add_argument("-o", "--output", help="output folder (holds the manifest)")
    sp.add_argument("-r", "--recursive", action="store_true")
    sp.add_argument("--margin-mm", type=float, default=0.2, help="crop this far inside the print edge (default 0.2)")
    sp.add_argument("--derivative", action="append", metavar="NAME=SIZE[:FMT[:Q]]",
                    help="derivative preset, repeatable; SIZE = long edge px or xSCALE; FMT jpg|png. "
                         f"Default: {', '.join(f'{d.name}={d.long_edge}:{d.fmt}:{d.quality}' for d in DEFAULT_DERIVATIVES)}")
    sp.add_argument("--no-derivatives", action="store_true", help="write only the TIFF master")
    sp.add_argument("--ppi", type=float, help="true scan resolution (pixels per inch)")
    sp.add_argument("--profile", help="named scanner profile from the profiles file")
    sp.add_argument("--profiles", help="profiles YAML (default ~/.config/polaroid-processor/profiles.yaml)")
    sp.add_argument("--interpolation", choices=("linear", "cubic", "lanczos"), default="cubic")
    sp.add_argument("--tiff-compression", choices=("zlib", "none"), default="zlib")
    sp.add_argument("--overlay", action="store_true", help="also write a detection overlay JPG per scan")
    sp.add_argument("--strict", action="store_true", help="do not export scans that need review")
    sp.add_argument("--force", action="store_true", help="re-export even if up to date")
    sp.add_argument("--redetect", action="store_true", help="re-run detection even if the scan is unchanged")
    sp.add_argument("--dry-run", action="store_true", help="detect and evaluate only; write nothing")
    sp.add_argument("-j", "--workers", type=int, help="parallel workers (default: from free RAM and CPUs)")
    common(sp)
    sp.set_defaults(fn=cmd_process)

    sp = sub.add_parser("detect", help="detect print outlines and print them (writes nothing)")
    sp.add_argument("inputs", nargs="+")
    sp.add_argument("-r", "--recursive", action="store_true")
    common(sp)
    sp.set_defaults(fn=cmd_detect)

    sp = sub.add_parser("inspect", help="write a zoomed corner/edge review image")
    sp.add_argument("scan", help="scan path, or file name when used with --manifest")
    sp.add_argument("-m", "--manifest", metavar="OUTPUT", help="output folder whose manifest holds the scan")
    sp.add_argument("--image", help="where to write the PNG")
    sp.add_argument("--window", type=int, default=360, help="tile size in scan pixels (default 360)")
    sp.add_argument("--zoom", type=int, default=1, help="integer upscaling of tiles")
    sp.add_argument("--grid", type=int, default=50, help="grid spacing in scan pixels")
    sp.add_argument("--stretch", choices=("auto", "on", "off"), default="auto",
                    help="contrast stretch for dark tiles (auto: tiles that are dark throughout)")
    common(sp)
    sp.set_defaults(fn=cmd_inspect)

    sp = sub.add_parser("fix", help="correct corners of a scan in the manifest")
    sp.add_argument("output", help="output folder holding the manifest")
    sp.add_argument("scan", help="scan file name or stem")
    sp.add_argument("--set", action="append", metavar="CORNER=X,Y", help="set a corner (tl,tr,br,bl)")
    sp.add_argument("--nudge", action="append", metavar="CORNER=DX,DY", help="move a corner by pixels")
    sp.add_argument("--shift", action="append", metavar="SIDE=D", help="move a side outward (+) or inward (-)")
    sp.add_argument("--accept", action="store_true", help="accept the current quad as reviewed")
    sp.add_argument("--reset", action="store_true", help="drop the manual fix, back to automatic detection")
    sp.add_argument("--author", default="human", help="who made the fix (human, agent, or a name)")
    sp.add_argument("--note", default="")
    sp.add_argument("--export", action="store_true", help="re-export this scan right away")
    common(sp)
    sp.set_defaults(fn=cmd_fix)

    sp = sub.add_parser("status", help="show manifest state of an output folder")
    sp.add_argument("output")
    sp.add_argument("--needs-review", action="store_true", help="only scans that need review")
    common(sp)
    sp.set_defaults(fn=cmd_status)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    _setup_logging(a.verbose, a.quiet)
    try:
        return a.fn(a)
    except (FileNotFoundError, ValueError, KeyError) as e:
        if getattr(a, "json", False):
            _emit({"type": "error", "error": str(e)})
        else:
            print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
