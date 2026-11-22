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
from .pipeline import DEFAULT_DERIVATIVES, PHOTO_BLEED_MM, PHOTO_CROP, Settings

log = logging.getLogger("polaroid_processor")


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _human(rec: dict) -> None:
    if rec.get("type") == "summary":
        c = rec["counts"]
        print(f"\n{rec['scans']} scan(s): " + ", ".join(f"{v} {k}" for k, v in sorted(c.items()))
              + f" | exported {rec['exported']}"
              + (f" | metadata updated {rec['metadata_updated']}" if rec.get("metadata_updated") else "")
              + f" | {rec['time_s']} s")
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
        redetect=a.redetect, dry_run=a.dry_run, workers=a.workers, flatten=a.flatten, shadows=a.shadows,
        photo_crop=a.photo_crop, photo_bleed_mm=a.photo_bleed_mm, metadata=_metadata_from_args(a))


def _metadata_from_args(a) -> Optional[dict]:
    """The metadata fields given on the command line (None when none: the output folder's stay)."""
    given = {f: getattr(a, f) for f in ("title", "author", "copyright", "description") if getattr(a, f) is not None}
    if a.keyword is not None:
        given["keywords"] = a.keyword
    if a.date_taken is not None:
        given["date_taken"] = a.date_taken
    return given or None


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
                              stretch=a.stretch, of=a.of)
    if a.json:
        _emit(rec)
    else:
        print(f"inspect image: {rec['image']}")
        for n, c in (rec.get("window_corners") or rec["corners"]).items():
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
                       accept=a.accept, reset=a.reset, window_sets=_parse_pairs(a.window_set, "--window-set"),
                       window_nudges=_parse_pairs(a.window_nudge, "--window-nudge"), window_kind=a.window_kind,
                       no_photo_crop=a.no_photo_crop, reset_window=a.reset_window,
                       author=a.author, note=a.note, export=a.export, emit=emit)
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


# --------------------------------------------------------------------------- app, review
def _announce(m: str) -> None:
    print(m, file=sys.stderr)


def cmd_app(a) -> int:
    from .app import launch

    return launch(a.folders, port=a.port, open_browser=not a.no_browser, config=a.config, announce=_announce)


def cmd_review(a) -> int:
    """Opens the app on the output folder (the standalone review page is gone as a command; the app's
    drag-to-fix view is the same page)."""
    from .app import launch

    if a.export:
        print("note: --export is ignored here; the app has an Export button", file=sys.stderr)
    return launch(output=a.output, port=a.port, open_browser=not a.no_browser, author=a.author,
                  filter_name=None if a.all else "needs_review", announce=_announce)


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
    sp.add_argument("--margin-mm", type=float, default=0.1, help="crop this far inside the print edge (default 0.1)")
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
    sp.add_argument("--photo-crop", choices=PHOTO_CROP,
                    help="also crop the photo window of each print into <output>/photo/: off, "
                         "derivatives (JPGs as for the full print), tiff (and a TIFF master; round windows get an "
                         "alpha channel). A window not found marks the scan needs_review; the full print is "
                         "always exported. Left out: keep the output folder's setting (off for a new folder)")
    sp.add_argument("--photo-bleed-mm", type=float,
                    help=f"cut the photo crop this far inside the window border (default {PHOTO_BLEED_MM}; "
                         "left out: keep the output folder's setting)")
    sp.add_argument("--title", help="metadata written into every exported file; the texts may hold {stem}, {n} "
                    "(the number the file name ends in), {project} (input folder name), {format} (600 or Go); "
                    "{{ and }} give literal braces. A field left out keeps the output folder's value, an empty "
                    "one (\"\") clears it")
    sp.add_argument("--author", help="metadata: author (Artist)")
    sp.add_argument("--copyright", help="metadata: copyright notice")
    sp.add_argument("--description", help="metadata: description (ImageDescription)")
    sp.add_argument("--keyword", action="append", help="metadata: a keyword, repeatable (replaces the output "
                    "folder's keywords; --keyword \"\" clears them)")
    sp.add_argument("--date-taken", metavar="YYYY[-MM[-DD]]", help="metadata: when the picture was taken, to the year, "
                    "month or day")
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
    sp.add_argument("--of", choices=("print", "window"), default="print",
                    help="window: tiles on the corners of the photo window (cyan), needs --manifest with the "
                         "photo crop on; a scan without a window shows the nominal one")
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
    sp.add_argument("--window-set", action="append", metavar="CORNER=X,Y",
                    help="set a corner of the photo window (photo crop on); a scan without a window starts "
                         "from the nominal one")
    sp.add_argument("--window-nudge", action="append", metavar="CORNER=DX,DY", help="move a window corner by pixels")
    sp.add_argument("--window-kind", choices=api.KINDS,
                    help="kind of the window: alone it changes the kind; where none was found it picks "
                         "the nominal window to start from (default: what most scans of the batch have)")
    sp.add_argument("--no-photo-crop", action="store_true", help="export this scan without a photo crop, by choice")
    sp.add_argument("--reset-window", action="store_true", help="take the hand-set window away, back to the automatic one")
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

    sp = sub.add_parser("app", help="local web app: projects, runs, review grid, drag-to-fix, export")
    sp.add_argument("folders", nargs="*", help="scan folders to make projects of; an output folder opens as it is")
    sp.add_argument("--port", type=int, help="port on 127.0.0.1 (default 8780, or a free one when that is busy)")
    sp.add_argument("--no-browser", action="store_true", help="do not open the page in the browser")
    sp.add_argument("--config", help="user data file (default %%APPDATA%%/polaroid-processor/app.json)")
    common(sp)
    sp.set_defaults(fn=cmd_app)

    sp = sub.add_parser("review", help="open the app on an output folder (drag outlines onto the print edge)")
    sp.add_argument("output", help="output folder holding the manifest")
    sp.add_argument("--all", action="store_true", help="show every scan (default: the grid starts on 'needs review')")
    sp.add_argument("--port", type=int, help="port on 127.0.0.1 (default 8780, or a free one when that is busy)")
    sp.add_argument("--author", default="human", help="author recorded with each fix (default human)")
    sp.add_argument("--export", action="store_true", help="ignored; the app has an Export button")
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
