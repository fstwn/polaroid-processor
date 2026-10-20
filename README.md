# polaroid-processor

Detects, crops and rectifies high-resolution flatbed scans of instant prints (Polaroid 600 / i-Type / SX-70 and Polaroid Go) lying in a black scanning holder, and camera shots of white-framed prints on light paper. Works with white, black (incl. glitter), gold and printed special-edition frames. Built to run unattended, from the command line or by an agent.

Scope and roadmap: [docs/SCOPE.md](docs/SCOPE.md).

## Install

Python 3.9 or newer. As a standalone tool (puts `polaroid` and `polaroid-mcp` on the PATH):

```bash
pipx install git+https://github.com/fstwn/polaroid-processor.git
```

Into the current environment, or from a checkout for development:

```bash
pip install git+https://github.com/fstwn/polaroid-processor.git
pip install -e .
```

Inside a conda environment that already provides OpenCV, NumPy and Pillow, skip dependency resolution (`pip install -e . --no-deps`) and add the rest yourself (`pip install tifffile pyyaml imagecodecs`); otherwise pip adds `opencv-python-headless` next to conda's OpenCV.

`imagecodecs` gives `tifffile` a faster zlib encoder: TIFF masters write about 3x faster with identical pixels. Without it everything still works, only slower.

This installs two commands: `polaroid` (CLI) and `polaroid-mcp` (MCP server for agents). `python -m polaroid_processor` works too.

### Windows: drag and drop

[`launcher/polaroid-drop.bat`](launcher/polaroid-drop.bat) needs no terminal. Drag one or more scan folders onto it (or double-click it and type a folder path). Each folder is processed into a sibling folder `<folder>_processed`, which opens in Explorer when done. If `polaroid.exe` is not on the PATH, set `POLAROID_CONDA_ENV` (or `POLAROID_EXE`) at the top of the file. Extra options go into `POLAROID_ARGS`, e.g. `--overlay`.

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
   On a light background (the image border is bright: a print photographed on paper) a separate detector runs: the photo window is the strongest line in a texture map, and walking outward from it through the flat white frame, the print edge is where the brightness leaves the frame level. Plain white frames only. The holder path is unchanged.
2. **Scale.** Pixels per millimetre come from `--ppi` / `--profile`, else from the TIFF tag if it gives a plausible print size, else from the median of all scans of the same scan size. Each outline's format is told by its shape (600: short/long side 0.819-0.824, Go: 0.808-0.812), so mixed batches of 600 and Go prints get one consistent scale.
3. **Review rules** flag a scan `needs_review` when its size does not match a known format or the other scans of the batch, opposite sides are not parallel, or an edge is noisy, curved or inconsistent.
4. **Flatten** curled prints. Each side is traced every 0.5 mm at full resolution and fitted with a smooth curve. When a reliable side bends by 0.1 mm or more, the export maps the curved outline onto a rectangle, so all four sides come out straight. Only the edge band is corrected (fading out within 5 mm); the inside and the output size are exactly those of the straight export, so the photo window keeps its shape; sides whose trace is not trustworthy (printed graphics, black glitter) stay straight. Manually fixed outlines are always straight. `--flatten off` keeps straight sides everywhere.
5. **Export** with one warp at full resolution, cropped by a margin in millimetres (default 0.2 mm).
6. **Remove edge shadows.** Near its edges a print darkens (3-18 % at 0.25 mm on white frames, reaching about 1.5 mm in). On plain frames the darkening is measured from the frame border itself and divided out. Printed, dark and glossy-gradient frames are left untouched. `--shadows off` keeps the tones exactly as scanned; no other colour changes are made.

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
| `--flatten auto\|off` | Unwarp curled prints along their traced edges (default `auto`) |
| `--shadows auto\|off` | Remove edge shadows on plain frames (default `auto`) |
| `--overlay` | Also write `_overlays/<name>.jpg` with the detected outline (and the curved edges when flattened) |
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

### `review`

```bash
polaroid review OUTPUT [--all] [--port N] [--author NAME] [--export] [--no-browser]
```

Serves a local page (stdlib server on `127.0.0.1`, no network access, no extra dependencies) and opens it in the browser. By default it lists the scans of the manifest with status `needs_review`; `--all` lists every scan. Each scan is a 3x3 grid: the four corners and the four edge midpoints as 480 px tiles of the original pixels (contrast-stretched when dark), and an overview in the middle. The green line is the current outline.

- Drag a magenta corner, or the green grip on an edge tile (moves the whole side), onto the print edge.
- Click a tile to enlarge it; there the arrow keys move 1 px and `Shift` + arrow 10 px. `Esc` closes.
- **Confirm** stores the current outline as reviewed (`fix --accept`) and goes to the next scan. **Save correction** stores the moved corners and cuts the tiles again at the new outline. **Reset** drops edits that are not saved yet. **Skip** goes on without writing.

Saves go through the same code as `fix` (history with author, time, note and previous outline; author is `--author`, default `human`). With `--export`, each saved scan is re-exported right away; otherwise the next `process` run does it. Tiles are cut from the input scans, which are only read. `--port` defaults to any free port; the URL is printed.

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
| `edge_inside:<side>:<mm>` | A stretch of the print edge lies further inside the outline than the crop margin, so holder would show in the export there (curved or concave edge) |

Informational: `scale_inferred:<source>`, `size_unverified` (single scan, no scale reference), `edge_repaired:<side>` (the holder rim touched that side; the side was refit on the print edge behind it), `flattened:<sides>:<mm>` (the export follows these sides' curves; largest bend), `flatten_straight:<side>` (in a flattened print, this side's trace was not reliable and stays straight).

For sides that follow a curve, the straight-line checks `not_parallel`, `edge_inside` and `edge_curved` do not apply. Each result also carries `flatten` (applied, sides, bend per side) and `shadows` (corrected sides, strongest darkening, share of each side that could be measured).

Exit codes: `0` success (scans may still need review), `1` at least one scan failed, `2` usage error. Logs go to stderr (`-v` for progress).

### Review loop for agents

1. `polaroid process IN -o OUT --json` and collect scans with `status: needs_review`.
2. For each: `polaroid inspect NAME -m OUT --json`, open the PNG, compare the green line with the print edge.
3. Correct with `polaroid fix OUT NAME --nudge/--shift/--set ... --author agent --note "why"`, or confirm with `--accept --author agent`.
4. `polaroid process IN -o OUT --json` again, or pass `--export` to `fix`.

Agents never need to touch the input folder; all state lives in the output folder's manifest.

## Agents

### MCP server

`polaroid-mcp` speaks the Model Context Protocol over stdio (no extra dependencies). Register it in Claude Code:

```bash
claude mcp add polaroid -- polaroid-mcp
```

Use the full path to `polaroid-mcp` (for example `.../envs/<env>/Scripts/polaroid-mcp.exe`) if it is not on the PATH. Other MCP clients take the same command.

| Tool | Does |
|---|---|
| `polaroid_process` | `process` a batch; returns the summary and per-scan results (full records for flagged scans). Sends a progress notification per finished scan. |
| `polaroid_detect` | Detect outlines, write nothing |
| `polaroid_inspect` | Review sheet as an inline image plus corner coordinates; also saved to `<output>/_inspect/` |
| `polaroid_fix` | `set` / `nudge` / `shift` / `accept` / `reset` as JSON objects, author `agent` by default, re-exports by default |
| `polaroid_status` | Counts and the scans that need review |

Tool errors (bad arguments, missing manifest) come back as tool results with `isError`, so the agent can correct itself.

### Claude Code skill

[`skills/polaroid-processor/SKILL.md`](skills/polaroid-processor/SKILL.md) teaches Claude the batch, review and fix workflow, how to read the review sheet, and when to hand a scan to a human. Install it for all projects by copying the folder:

```bash
cp -r skills/polaroid-processor ~/.claude/skills/
```

It works with the MCP tools when they are registered and falls back to the CLI otherwise.

## Python API

```python
from polaroid_processor import load_rgb, detect, rectify

rgb = load_rgb("scan.tif")
det = detect(rgb)              # Detection: quad, size_px, skew_deg, edges, ...
img = rectify(rgb, det.quad, margin_px=25)
```

Batch processing with manifest and exports: `polaroid_processor.pipeline.run_batch`. The commands as functions that return JSON-ready records (used by the CLI and the MCP server): `polaroid_processor.api`.

## Development

```bash
pip install -e ".[dev]"
python -m pytest              # synthetic tests; real-scan regression runs when input/ is present
python tools/export_labels.py OUTPUT...   # collect reviewed corners into tests/data/labels.json
```

`input/` and `output/` hold real scans and are git-ignored. Experiments go to `.dev/`.

## License

See [LICENSE](LICENSE).
