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
from typing import List, Optional

from . import __version__, api
from .api import apply_fix  # noqa: F401  (public, used by tests)
from .pipeline import DEFAULT_DERIVATIVES, Settings

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
    return api.make_settings(
        margin_mm=a.margin_mm, derivatives=a.derivative, no_derivatives=a.no_derivatives,
        interpolation=a.interpolation, tiff_compression=a.tiff_compression, overlay=a.overlay,
        ppi=a.ppi, profile=a.profile, profiles=a.profiles, strict=a.strict, force=a.force,
        redetect=a.redetect, dry_run=a.dry_run, workers=a.workers, flatten=a.flatten, shadows=a.shadows)


def cmd_process(a) -> int:
    emit = _emit if a.json else _human
    summary = api.process(a.inputs, a.output, _settings_from_args(a), recursive=a.recursive, emit=emit)
    emit(summary)
    return 1 if summary["failed"] else 0


def _human_detection(rec: dict) -> None:
    if rec["status"] == "ok":
        d = rec["detection"]
        print(f"ok      {rec['file']}  {d['size_px'][0]:.0f}x{d['size_px'][1]:.0f}px  aspect {d['aspect']}  "
              f"skew {d['skew_deg']}deg  thr {d['speck_threshold']}  {d['time_s']}s")
    else:
        print(f"FAILED  {rec['file']}  {rec['error']}")


def cmd_detect(a) -> int:
    records = api.detect_scans(a.inputs, recursive=a.recursive, emit=_emit if a.json else _human_detection)
    return 1 if any(r["status"] == "failed" for r in records) else 0


# --------------------------------------------------------------------------- inspect
def cmd_inspect(a) -> int:
    rec, _ = api.inspect_scan(a.scan, a.manifest, image=a.image, window=a.window, zoom=a.zoom, grid=a.grid,
                              stretch=a.stretch)
    if a.json:
        _emit(rec)
    else:
        print(f"inspect image: {rec['image']}")
        for n, c in rec["corners"].items():
            print(f"  {n}: {c[0]:.1f}, {c[1]:.1f}")
    return 0


# --------------------------------------------------------------------------- fix
def _parse_pairs(items: Optional[List[str]], what: str) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise api.UsageError(f"{what}: bad value {it!r}, expected NAME=...")
        k, v = it.split("=", 1)
        try:
            out[k] = [float(x) for x in v.split(",")]
        except ValueError:
            raise api.UsageError(f"{what}: bad number in {it!r}") from None
    return out


def cmd_fix(a) -> int:
    emit = _emit if a.json else _human
    rec = api.fix_scan(a.output, a.scan, sets=_parse_pairs(a.set, "--set"),
                       nudges=_parse_pairs(a.nudge, "--nudge"), shifts=_parse_pairs(a.shift, "--shift"),
                       accept=a.accept, reset=a.reset, author=a.author, note=a.note, export=a.export, emit=emit)
    summary = rec.pop("summary", None)
    if summary is not None:
        emit(summary)
        return 1 if summary["failed"] else 0
    (_emit if a.json else lambda r: print(json.dumps(r, indent=1)))(rec)
    return 0


# --------------------------------------------------------------------------- status
def cmd_status(a) -> int:
    summary = api.status(a.output, needs_review=a.needs_review, emit=_emit if a.json else _human)
    if a.json:
        _emit(summary)
    else:
        print("\n" + ", ".join(f"{v} {k}" for k, v in sorted(summary["counts"].items())))
    return 0


# --------------------------------------------------------------------------- review
def cmd_review(a) -> int:
    from .review import ReviewApp, serve

    app = ReviewApp(a.output, show_all=a.all, author=a.author, export=a.export)
    n = len(app.list_scans()["scans"])
    if not n:
        print("no scans need review (use --all to look at every scan)" if not a.all else "the manifest has no scans")
        return 0
    print(f"{n} scan(s) to review in {a.output}", file=sys.stderr)
    serve(app, a.port, open_browser=not a.no_browser, announce=lambda m: print(m, file=sys.stderr))
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
    sp.add_argument("--flatten", choices=("auto", "off"), default="auto",
                    help="auto: unwarp curled prints along their traced edges (default); off: straight sides")
    sp.add_argument("--shadows", choices=("auto", "off"), default="auto",
                    help="auto: remove edge shadows on plain frames (default); off: leave tones untouched")
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

    sp = sub.add_parser("review", help="local web page to drag outlines onto the print edge")
    sp.add_argument("output", help="output folder holding the manifest")
    sp.add_argument("--all", action="store_true", help="every scan, not only those that need review")
    sp.add_argument("--port", type=int, default=0, help="port on 127.0.0.1 (default: any free port)")
    sp.add_argument("--author", default="human", help="author recorded with each fix (default human)")
    sp.add_argument("--export", action="store_true", help="re-export each scan right after it is saved")
    sp.add_argument("--no-browser", action="store_true", help="do not open the page in the browser")
    common(sp)
    sp.set_defaults(fn=cmd_review)
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
