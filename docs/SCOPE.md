# Polaroid processor: v1 scope

Decided in a scope review on 2026-10-17. Background and measurements: `.dev/REVIEW.md`. Prototype detector: `.dev/proto/proto_detect.py`.

## Goal

A robust tool that crops and rectifies high-resolution flatbed scans of instant prints. It must work unattended for agents (Claude and others) and from the command line, and produce lossless masters plus share-ready copies.

## Input

- Exactly one print per scan.
- v1: print lies in the black scanning holder (near-black, flat background). This is the precision path.
- v2: "light background" mode (white/grey scanner lid). Auto-detected from the scan border, overridable by flag. Designed once sample scans exist.
- Input files are never written, moved or renamed.

## Formats

- Polaroid 600 / i-Type / SX-70: 88 x 107 mm.
- Polaroid Go: 53.9 x 66.6 mm.
- Frame types: white, black (incl. glitter), gold/metallic, printed/coloured special editions. No per-frame-type configuration.

## Scale (px per mm)

Order of precedence:

1. Scanner profile set by the user once (named, e.g. "Epson 3200 ppi").
2. TIFF resolution tag, if it is plausible for the detected print size.
3. Assume the default format and infer px/mm from the batch median, with a warning.

The current scans are tagged 600 dpi but measure ~128 px/mm (~3250 ppi), so the tag alone is not trusted.

## Geometry

- Detection: frame-colour-agnostic (bright-speck density, per-side robust line fit, adaptive holder-noise threshold), see prototype.
- Rectification: one homography from the four fitted corners. Single resampling step.
- Edge curvature is measured per edge and reported. Above a threshold the result carries a warning suggesting flattening.
- Edge flattening: opt-in flag.
- Removed: global radial (k1) distortion correction, percentage safety crop. Replaced by a margin in mm (small default, e.g. 0.2 mm).

## Output

- Default: full print, frame included, at measured size and native resolution. No forced aspect ratio. DPI tag written from the true px/mm.
- Opt-in: photo area only. Rectangular windows cropped directly. Round windows: tight square crop with an alpha mask outside the circle (TIFF/PNG); JPG keeps the frame pixels in the corners.
- Files:
  - Full-resolution TIFF master, lossless, same bit depth as input, ICC profile carried over. Always written.
  - Derivatives as a list of named size presets (e.g. `high`, `web`), each sized by long edge or scale, each JPG or PNG.
- Tone: pixel-faithful, no colour or brightness changes by default. Heuristic shadow correction removed. Opt-in flat-field correction from a one-time calibration scan (white sheet in the same holder).

## State and failure handling

- One JSON manifest per output folder. One entry per scan, keyed by file name and content hash. Holds detection, metrics, settings used, manual fixes, status.
- Fixes keep provenance: author (human or agent), timestamp, previous value. Revertible.
- Re-runs reuse manual fixes and skip unchanged scans.
- Low-confidence detections (size off the format prior, high edge residuals, curvature warning): exported anyway, marked `needs_review` in manifest and JSON output, listed in the batch summary. `--strict` skips export instead.

## Interfaces (v1)

- Python library: typed functions (`detect`, `rectify`, `process`, ...) and result dataclasses. All logic lives here.
- CLI:
  - `--json` prints one result object per file on stdout; logs go to stderr; non-zero exit code when any file failed.
  - `--dry-run` detects only.
  - `inspect` writes zoomed corner and edge crops with a pixel grid (for humans and agents).
  - `fix` sets or nudges corners and stores them in the manifest; re-export follows.
  - Overlay preview of the detection on request.
- MCP server: thin wrapper over the library. Tools for detect, process, inspect (crops returned inline), fix, batch status.
- Claude Code skill: how to run a batch, read the JSON, review flagged prints via `inspect`, apply `fix`.
- Agent autonomy: agents may apply corner fixes and re-export without asking. Every fix is logged in the manifest. Inputs stay read-only.

## Not in v1

- GUI (planned for v2: local web UI on localhost with batch run, review grid and corner dragging).
- Light-background mode (v2).
- Multiple prints per scan, phone photos, Instax formats.
- Automatic colour grading.

## Quality targets

- Ground truth: hand-labelled corners on ~20 scans covering every frame type, made with `inspect` + `fix`.
- Every labelled corner within 0.2 mm (~25 px at current scan resolution).
- No holder pixels visible at the default margin.
- Zero crashes on all 88 sample scans.
- Regression test against the labelled set runs on every change.

## Performance

- Typical batch: 10-200 scans.
- Target: at most 6 s per scan per worker.
- Parallel workers chosen from free RAM (about 2 GB per worker), overridable.
- Resumable via manifest hashes.

## Distribution

- Installable Python package (pyproject, src layout), usable in conda or pipx.
- Console commands for CLI and MCP server.
- Windows start script for non-terminal users.

## Code base

- Commit the current state as a baseline first.
- Build the new package next to the old code; port only the useful warp and I/O ideas.
- Retire `main.py`, `shadow_corrector`, radial distortion, `border_processor` and `edge_detector` once the new CLI passes the tests (end of M1).

## Milestones

1. **M1 core**: package skeleton, detector, rectify, TIFF master + derivative presets, manifest, CLI with `--json`, `inspect` / `fix`, labelled set and regression tests. Old code retired.
2. **M2 agents and packaging**: Claude Code skill, MCP server, pip packaging, Windows launcher.
3. **M3 opt-ins**: photo-area crop incl. round windows with alpha, flat-field correction, edge flattening.
