# polaroid-processor

Detects, crops and rectifies high-resolution flatbed scans of instant prints (Polaroid 600 / i-Type / SX-70 and Polaroid Go) lying in a black scanning holder. Works with white, black (incl. glitter), gold and printed special-edition frames. Built to run unattended, from the command line or by an agent.

Scope and roadmap: [docs/SCOPE.md](docs/SCOPE.md).

## Install

Python 3.9 or newer.

```bash
pip install -e .
```

Inside an existing conda environment that already provides OpenCV, NumPy and Pillow, skip dependency resolution:

```bash
pip install -e . --no-deps
```

This installs the `polaroid` command. `python -m polaroid_processor` works too.

## Quick start

```bash
polaroid process scans/2026-03-28 -o processed/2026-03-28
```

Per scan this writes:

| File | Content |
|---|---|
| `<out>/<name>.tif` | Lossless master, full resolution, ICC profile kept, true resolution in the TIFF tag |
| `<out>/high/<name>.jpg` | Long edge 6000 px, quality 95 |
| `<out>/web/<name>.jpg` | Long edge 2048 px, quality 88 |
| `<out>/polaroid-manifest.json` | Detection, scale, status, manual fixes and their history |

Re-running the same command only processes new or changed scans and scans whose settings changed.

## How it works

1. **Detect** the print outline at reduced resolution. The holder is near-black and flat; every frame type has pixels brighter than the holder noise. Each edge is scanned from outside inward, averaged over long segments, and the edge is the first sharp step into a stable plateau. Holder rim lines, dust and flare-lit gaps are rejected. A consensus line fit per side gives the four corners. Where the holder rim touches the print, the side is refit on the print edge behind the rim so that it runs parallel to the opposite side.
2. **Scale.** Pixels per millimetre come from `--ppi` / `--profile`, else from the TIFF tag if it gives a plausible print size, else from the median of all scans of the same scan size (assuming 88 x 107 mm).
3. **Review rules** flag a scan `needs_review` when its size does not match a known format or the other scans of the batch, opposite sides are not parallel, or an edge is noisy, curved or inconsistent.
4. **Export** with one perspective warp at full resolution, cropped by a margin in millimetres (default 0.2 mm). No other resampling, no colour changes.

## Commands

### `process`

```bash
polaroid process INPUT... -o OUTPUT [options]
```

| Option | Effect |
|---|---|
| `--margin-mm 0.2` | Crop this far inside the detected print edge |
| `--derivative NAME=SIZE[:FMT[:Q]]` | Derivative preset, repeatable. SIZE is a long edge in px or `x0.5`; FMT `jpg` or `png` |
| `--no-derivatives` | Only the TIFF master |
| `--ppi N` / `--profile NAME` | True scan resolution, directly or from a scanner profile |
| `--overlay` | Also write `_overlays/<name>.jpg` with the detected outline |
| `--strict` | Do not export scans that need review |
| `--force` / `--redetect` | Re-export / re-detect even if up to date |
| `--dry-run` | Detect and evaluate only, write nothing |
| `-j N` | Worker processes (default: from free RAM, about 1.5 GB per worker) |
| `--json` | Machine-readable output (see below) |

Scanner profiles live in `~/.config/polaroid-processor/profiles.yaml` (or `--profiles`, or `$POLAROID_PROFILES`):

```yaml
profiles:
  epson-v850-3200: 3200
```

Scan file names must be unique within one output folder. Use one output folder per input folder.

### `detect`

```bash
polaroid detect SCAN... [--json]
```

Prints the detected outline. Writes nothing.

### `inspect`

```bash
polaroid inspect NAME -m OUTPUT [--json]
polaroid inspect path/to/scan.tif --image review.png
```

Writes a PNG with eight full-resolution tiles (four corners, four edge midpoints) and an overview. The grid is labelled in absolute scan pixels; the green line is the current crop edge. Tiles that are dark throughout (black, glitter or gold frames on the black holder) are contrast-stretched and marked `stretched`; `--stretch on|off` overrides. Use it to check a flagged scan or to read off exact coordinates for `fix`. It never writes into the input folder.

### `fix`

```bash
polaroid fix OUTPUT NAME --nudge tl=4,-2 --shift right=-12 --export
polaroid fix OUTPUT NAME --set bl=265,13868
polaroid fix OUTPUT NAME --accept
polaroid fix OUTPUT NAME --reset
```

Corners are `tl`, `tr`, `br`, `bl`; sides are `top`, `right`, `bottom`, `left`. `--shift` moves a side outward (positive) or inward (negative). `--accept` marks the current outline as reviewed. Every change is stored in the manifest with author (`--author`, default `human`), time, note and previous value. `--export` re-exports that scan right away; otherwise the next `process` run does it.

### `status`

```bash
polaroid status OUTPUT [--needs-review] [--json]
```

## Machine-readable output

With `--json`, `process` prints one JSON object per scan, then a summary:

```json
{"type": "result", "file": "scan_01.tif", "status": "needs_review",
 "warnings": ["edge_curved:right:-0.51mm"], "quad": [[223.0, 324.0], ...],
 "quad_source": "auto", "size_px": [11161.0, 13574.0],
 "scale": {"px_per_mm": 126.9, "ppi": 3223.0, "source": "batch-median"},
 "format": {"name": "polaroid-600", "size_mm": [87.97, 106.98], "error": 0.0003},
 "edges": {"top": {"n_points": 18, "n_inliers": 18, "mad_px": 1.0, "bow_px": -0.2, "max_px": 3.1}, ...},
 "outputs": {"tiff": "...", "high": "...", "web": "..."}, "exported": true, "error": null}
{"type": "summary", "scans": 14, "counts": {"ok": 13, "needs_review": 1}, "exported": 14,
 "needs_review": ["scan_01.tif"], "failed": [], "manifest": "...", "time_s": 59.0}
```

| Status | Meaning |
|---|---|
| `ok` | Exported, no warnings that need a look |
| `needs_review` | Exported (unless `--strict`), check with `inspect` and correct with `fix` or `fix --accept` |
| `failed` | No outline found or export error; see `error` |

Warnings that set `needs_review`:

| Warning | Meaning |
|---|---|
| `size_mismatch:<pct>` | Size differs from the nearest known print format by more than 1.5 % |
| `size_outlier:<axis>:<mm>` | Width or height differs from the median print of the same scan size by more than 0.35 mm (5+ scans) |
| `not_parallel:<pair>:<deg>` | Opposite sides differ in angle by more than 0.3 degrees |
| `edge_noisy:<side>` | Edge points scatter more than 0.3 mm around the fitted line |
| `edge_curved:<side>:<mm>` | Edge bows by more than 0.4 mm |
| `edge_uncertain:<side>:<inliers>/<points>` | Less than 60 % of the edge points agree on one line |

Informational: `scale_inferred:<source>`, `size_unverified` (single scan, no scale reference), `edge_repaired:<side>` (the holder rim touched that side; the side was refit on the print edge behind it).

Exit codes: `0` success (scans may still need review), `1` at least one scan failed, `2` usage error. Logs go to stderr (`-v` for progress).

### Review loop for agents

1. `polaroid process IN -o OUT --json` and collect scans with `status: needs_review`.
2. For each: `polaroid inspect NAME -m OUT --json`, open the PNG, compare the green line with the print edge.
3. Correct with `polaroid fix OUT NAME --nudge/--shift/--set ... --author agent --note "why"`, or confirm with `--accept --author agent`.
4. `polaroid process IN -o OUT --json` again, or pass `--export` to `fix`.

Agents never need to touch the input folder; all state lives in the output folder's manifest.

## Python API

```python
from polaroid_processor import load_rgb, detect, rectify

rgb = load_rgb("scan.tif")
det = detect(rgb)              # Detection: quad, size_px, skew_deg, edges, ...
img = rectify(rgb, det.quad, margin_px=25)
```

Batch processing with manifest and exports: `polaroid_processor.pipeline.run_batch`.

## Development

```bash
pip install -e ".[dev]"
python -m pytest              # synthetic tests; real-scan regression runs when input/ is present
python tools/export_labels.py OUTPUT...   # collect reviewed corners into tests/data/labels.json
```

`input/` and `output/` hold real scans and are git-ignored. Experiments go to `.dev/`.

## License

See [LICENSE](LICENSE).
