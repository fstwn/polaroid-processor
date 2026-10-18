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
- Retire `main.py`, `shadow_corrector`, radial distortion, `border_processor` and `edge_detector` once the new CLI passes the tests (end of M1). Done: removed together with `config/processing_params.yaml` and `requirements.txt`; recoverable from the baseline commit.

## Milestones

1. **M1 core**: package skeleton, detector, rectify, TIFF master + derivative presets, manifest, CLI with `--json`, `inspect` / `fix`, labelled set and regression tests. Old code retired.
2. **M2 agents and packaging**: Claude Code skill, MCP server, pip packaging, Windows launcher.
3. **M3 opt-ins**: photo-area crop incl. round windows with alpha, flat-field correction, edge flattening.

## M1 status (2026-10-17)

Measured on the 88 sample scans (12 folders, one output folder per input folder):

- 88/88 detected, 0 crashes. 85 `ok`, 3 `needs_review`, all three genuine:
  - `260315_Polaroids_Samsoe_02`: right edge physically curved (S-shape, about 0.5 mm); no straight line fits.
  - `260328_RollerRoids_tttacho_14-2`: black glitter frame on the black holder; left and right edges uncertain.
  - `250928_SICKalps_L2_D3_29 (2)`: holder rim parallel to the bottom edge, outline 0.46 mm too tall. The review output names the fix (`fix --shift bottom=-59`); applied and verified.
- Holder rim touching the print at a slight angle (8 scans) is now repaired automatically (`edge_repaired:<side>`).
- Labelled set: 19 scans (white, gold, printed, black glitter, round window, both scan resolutions) in `tests/data/labels.json`, status `proposed` (agent-reviewed, awaiting human confirmation). 18/19 within 0.2 mm; the remaining one is the manual fix above (xfail).
- Print size spread at ~127 px/mm: width 11125-11196 px, height 13551-13576 px after the rim fix (about +-0.3 mm).
- Throughput: 423 s for 88 scans on this machine (4.8 s per scan, auto workers). A single job takes about 8 s alone; with 11 parallel workers each job takes about 38 s, so worker scaling is poor (likely disk and memory bandwidth). Worker count needs tuning.

Open findings for M3 (edge flattening):

- Many prints have slightly concave sides: the corners lie 20-60 px (0.15-0.45 mm) outside the fitted line while the middle sits on it, probably print cupping. The crop then trims a sliver of print at the corners but never shows holder.
- Some corners flare outward by up to 0.6 mm (`flare_start_px` / `flare_end_px` per edge in the manifest). The measurement is still noisy near the holder rim and on printed frames, so it is diagnostic only and sets no warning.

## Edge bias fix (2026-10-18)

The 19 labels are now human-confirmed (outline on the middle of the soft print-holder transition). Against them the detector sat systematically outward. Two causes, both fixed in `detection.py`:

- Edge points came from the analysis cells: outer cell boundary, extended outward over bright cells, and the speck threshold fires early in the soft edge. Each point is now refined at full resolution: `max(R,G,B)` averaged over the segment, 50 % crossing between the darkest holder level up to 6 cells outside and the median print level 3-8 cells inside, sub-pixel interpolated. Sparse glitter frames (contrast < 20 levels) use the speck-density profile instead, and keep the cell point when that is flat too.
- Concave sides: the robust line through all points runs outward of the side's middle, where the labeller placed the outline. Each outline is now moved to the median of the middle third of its inlier points. Edge statistics (residuals, bow, flare, inner candidates) stay relative to the robust line, so review rules are unchanged.

Detected minus label, outward positive, mm, 18 labels (known failure excluded). Script: `.dev/bias/measure.py`.

| side | before: median (range) | after: median (range) |
| --- | --- | --- |
| top | +0.026 (-0.000..+0.100) | -0.005 (-0.040..+0.050) |
| right | +0.072 (+0.009..+0.115) | +0.012 (-0.021..+0.029) |
| bottom | +0.038 (+0.000..+0.086) | -0.019 (-0.043..+0.012) |
| left | +0.080 (-0.022..+0.122) | +0.013 (-0.030..+0.030) |
| worst corner | 0.152 | 0.095 |

Regression tolerance tightened from 0.2 mm to 0.1 mm. The worst corner is the black glitter frame `tttacho_13-2` (density fallback). The remaining spread (about +-0.04 mm per side) is close to the label noise: on long, soft ramps (bottom edges, cupped top edges) the labeller's choice varies between scans. The parameters were chosen on the same 18 labels, so new labels should be checked before tuning further.

88-scan dry run (one output folder per input folder, `.dev/bias/run/`): 88/88 detected, 0 crashes, 87 `ok`, 1 `needs_review` (`250928_SICKalps_L2_D3_29 (2)`, same rim case and hint as before). No new review cases, but two of the three M1 review cases now pass by a hair, although both edges are still not straight:

- `260315_Polaroids_Samsoe_02`: the right edge is S-shaped. The refit line is more vertical now, so `not_parallel:left-right` dropped from 0.32 deg to 0.27 deg (rule: 0.3). The bottom-right corner sits about 40 px outside the print, the old outline was worse there.
- `260328_RollerRoids_tttacho_14-2`: black glitter frame with a concave left edge. Left inliers went from 13/23 to 14/23 (rule: share 0.6). The five middle points that sit 21-36 px inside the line are outliers, so the outline in the middle of the side runs about 0.25 mm outside the print, more than the 0.2 mm margin.

Both show that the review rules missed a curved edge whose middle points are rejected as outliers. New rule `edge_inside:<side>:<mm>`: per side, the deepest run of three consecutive edge points (outliers included) inside the outline; above the crop margin, holder would show in the export, so the scan needs review. On the 88 scans every good scan stays at or below 0.07 mm. Flagged: `tttacho_14-2` (right 1.23 mm, left 0.24 mm), `tttacho_16-2` (left 1.75 mm, right 0.60 mm; black glitter, previously `ok`), and `D3_29 (2)` (bottom 0.55 mm, alongside its rim hint). Result: 85 `ok`, 3 `needs_review`. `Samsoe_02` stays `ok`: its S-shaped right edge bulges outward at the bottom, so the crop trims print there but shows no holder.

Detection time is unchanged: 3.3 s per labelled scan with 4 parallel jobs (3.5 s before). The refinement averages about 100 small full-res windows per attempt.

## M2 status (2026-10-17)

- `polaroid_processor.api`: the commands as functions returning JSON-ready records; CLI and MCP server share it.
- MCP server `polaroid-mcp`: dependency-free stdio implementation, because the official `mcp` SDK needs Python 3.10+ and the target environment is 3.9. Five tools (process, detect, inspect with inline image, fix, status), a progress notification per finished scan, tool errors as `isError` results. Tested end to end over stdio, including parallel workers on real scans.
- Claude Code skill in `skills/polaroid-processor/`.
- Packaging: version 0.2.0, console scripts `polaroid` and `polaroid-mcp`; wheel builds offline and installs into a clean venv.
- Windows launcher `launcher/polaroid-drop.bat`: drag folders onto it; finds `polaroid.exe` on the PATH or in a named conda environment.
- Not done: publishing to PyPI, a Claude Code plugin bundle (skill plus MCP registration in one install).

## Held-out check (2026-10-18)

The user offered five more folders from the scan library (`D:/11_pictures/Pictures`). Of 119 files, 36 hold the same pixels as files in `input/`; 83 are new.

- Detection only (dry run) on the 83: 72 `ok`, 10 `needs_review`, 1 failed. The failure is an emulsion lift on white paper, outside v1 scope (light-background mode, v2). All 10 flags point at real problems: holder rim on the bottom edge (D2_13, D3_10, Misc_22), curled prints whose sides bow up to 1 mm (D2_18, 27, 28, 29), black glitter frames (Samsoe_06, 09), and two size outliers.
- 16 of the `ok` scans (olive, yellow, printed, gold, round window, white) were labelled by the user in a second review round and stored in `tests/data/labels_holdout.json`. They are used to measure, never to tune.
- Detected minus label, outward positive: median 0.000 mm on every side, worst corner 0.12 mm (top edge of a printed frame, which sits 0.03-0.12 mm outside the print). 7 of 16 outlines were confirmed unchanged; 13 of 16 are within 0.1 mm, all within the 0.2 mm target. `test_corners_match_holdout_labels` checks them against 0.2 mm.
- So the edge bias fix carries over to unseen prints. The remaining failure modes are the holder rim at the bottom edge and curled prints (M3 edge flattening).
