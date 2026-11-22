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

Two launchers need no terminal. [`launcher/polaroid-app.bat`](launcher/polaroid-app.bat) starts the [web app](#web-app): double-click it, or drag scan folders onto it to make them projects (each is processed into a sibling folder `<folder>_processed`); an app that is already running is reused. It reads `POLAROID_EXE` / `POLAROID_CONDA_ENV` like the other launcher, and `POLAROID_APP_ARGS` adds options such as `--port`. Close its window to stop the app. A folder whose name contains `&` is cut at the `&` if Explorer does not quote it when dragged (the same limit as `polaroid-drop.bat`); start such a folder from a terminal with `polaroid app "C:\R&D\scans"`.

[`launcher/polaroid-drop.bat`](launcher/polaroid-drop.bat) processes without the app. Drag one or more scan folders onto it (or double-click it and type a folder path). Each folder is processed into a sibling folder `<folder>_processed`, which opens in Explorer when done. If `polaroid.exe` is not on the PATH (as when started from Explorer), both launchers look for it in every conda environment under `miniconda3` / `anaconda3` in the user profile, `ProgramData` or `LocalAppData` and use the first one that has it; set `POLAROID_CONDA_ENV` (or `POLAROID_EXE`) at the top of the file to pick one. Extra options go into `POLAROID_ARGS`, e.g. `--overlay`.

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

With `--photo-crop` the photo area is cropped as an extra output next to the full print, which is always written (see [Photo area crop](#photo-area-crop)). With `--title`, `--author` and friends every file carries your text metadata (see [Export metadata](#export-metadata)).

Re-running the same command only processes new or changed scans and scans whose settings changed. When only the batch's inferred scale moved (a fix changed the median print size), the resolution tags of the finished exports are patched in place and nothing is encoded again, as long as the scale it was cut at is within 0.5 % of the new one. A scale set on purpose (`--ppi`, a profile) is a new export.

## Web app

```bash
polaroid app                       # starts the app and opens the browser (http://127.0.0.1:8780)
polaroid app D:\scans\roll_01      # ... and makes that folder a project
python -m polaroid_processor.app   # the same
```

A local app (stdlib server on `127.0.0.1`, plain JavaScript, no extra dependencies) for the whole loop: pick a scan folder, run, review, fix, export.

- **Projects.** The home page lists recent projects (input folder, output folder, counts of ok / needs review / failed / exported). A new project takes an input folder (Browse opens the native folder dialog, or type a path; recent folders are offered) and an output folder, by default `<input>_processed` next to it. Any existing output folder with a `polaroid-manifest.json` can be opened too, also one made on the command line. The list is kept per user in `%APPDATA%/polaroid-processor/app.json` (`~/.polaroid-processor/` elsewhere).
- **Detect.** Finds the print in every scan and stores the outlines in the manifest, without exporting. A bar shows done/total and time left; the grid fills as scans finish, each card with a small straightened preview of the crop (cached in `<out>/_previews/`), its status and warnings. Cancel stops the run at once; the scans in flight are redone by the next run. The grid filters by status.
- **Fix.** A click on a card opens the 9-tile drag view (the same page the old `review` command served): drag a corner or an edge grip onto the print edge, then Confirm or Save correction. The card and its preview update. Fixes can be made at any time, also during a run.
- **Export.** Writes the TIFF master and the JPG derivatives of every detected scan that has no up to date export; `Export (N)` says how many. It has the same progress and Cancel. `Detect + export` does both in one go, for batches you trust. A scale-only change shows as `Update tags (T)`: the resolution tags are patched, nothing is encoded. A change of the metadata counts in `Export (N)` too (the card says "metadata to update"); that run only rewrites the text headers of the finished files (the bar says "updating metadata") and encodes nothing.
- **Settings.** Per project, kept in the manifest: crop margin, the photo area crop and its bleed, the JPG derivatives `high` and `web`, flatten and shadow correction, the Metadata group (see [Export metadata](#export-metadata)); under Advanced the scale (a ppi, or a scanner profile picked from a dropdown of the profiles file; see [Scanner profiles](#scanner-profiles)), TIFF compression, interpolation and the overlay. Your last choices are the start for new projects (the scale is not carried over; of the metadata only author and copyright are, title, description, keywords and date taken belong to a project). Settings apply to the next run and cannot be changed while one goes; a change shows in the Export count.
- **With the command line and agents.** The app watches the manifest: a fix made by `polaroid fix`, an agent or another window shows up in the grid and in an open drag view without a reload. Only one run can work on an output folder at a time (see [Run lock](#run-lock-and-concurrent-writers)); the app shows when another process holds it.

`polaroid review OUTPUT [--all] [--port N] [--no-browser]` opens the app on that project; the grid starts on "needs review" unless `--all` is given. `--author NAME` sets the author recorded with fixes made in the app (only when this command starts the app; a running app keeps its own); `--export` is ignored (use the Export button). `polaroid app` and `polaroid review` reuse an app that already runs on the port (default 8780, or a free one when that is busy and no `--port` is given): the browser opens on it and on the project, and no second server starts. A run in the running app is never stopped for that; the app stays on its project.

## How it works

1. **Detect** the print outline at reduced resolution. The holder is near-black and flat; every frame type has pixels brighter than the holder noise. Each edge is scanned from outside inward, averaged over long segments, and the edge is the first sharp step into a stable plateau. Holder rim lines, dust and flare-lit gaps are rejected. A consensus line fit per side gives the four corners. Where the holder rim touches the print, the side is refit on the print edge behind the rim so that it runs parallel to the opposite side.
   On a light background (the image border is bright: a print photographed on paper) a separate detector runs: the photo window is the strongest line in a texture map, and walking outward from it through the flat white frame, the print edge is where the brightness leaves the frame level. Plain white frames only. The holder path is unchanged.
2. **Scale.** Pixels per millimetre come from `--ppi` / `--profile`, else from the TIFF tag if it gives a plausible print size, else from the median of all scans of the same scan size. Each outline's format is told by its shape (600: short/long side 0.819-0.824, Go: 0.808-0.812), so mixed batches of 600 and Go prints get one consistent scale.
3. **Review rules** flag a scan `needs_review` when its size does not match a known format or the other scans of the batch, opposite sides are not parallel, or an edge is noisy, curved or inconsistent.
4. **Flatten** curled prints. Each side is traced every 0.5 mm at full resolution and fitted with a smooth curve. When a reliable side bends by 0.1 mm or more, the export maps the curved outline onto a rectangle, so all four sides come out straight. Only the edge band is corrected (fading out within 5 mm); the inside and the output size are exactly those of the straight export, so the photo window keeps its shape; sides whose trace is not trustworthy (printed graphics, black glitter) stay straight. Manually fixed outlines are always straight. `--flatten off` keeps straight sides everywhere.
5. **Export** with one warp at full resolution, cropped by a margin in millimetres (default 0.1 mm).
6. **Remove edge shadows.** Near its edges a print darkens (3-18 % at 0.25 mm on white frames, reaching about 1.5 mm in). On plain frames the darkening is measured from the frame border itself and divided out. Printed, dark and glossy-gradient frames are left untouched. `--shadows off` keeps the tones exactly as scanned; no other colour changes are made.

## Commands

### `process`

```bash
polaroid process INPUT... -o OUTPUT [options]
```

| Option | Effect |
|---|---|
| `--margin-mm 0.1` | Crop this far inside the detected print edge |
| `--derivative NAME=SIZE[:FMT[:Q]]` | Derivative preset, repeatable. SIZE is a long edge in px or `x0.5`; FMT `jpg` or `png` |
| `--no-derivatives` | Only the TIFF master |
| `--photo-crop off\|derivatives\|tiff` | Also crop the photo window into `<out>/photo/`; left out, the output folder keeps its setting (`off` for a new one); see [Photo area crop](#photo-area-crop) |
| `--photo-bleed-mm 0.3` | Cut the photo crop this far inside the window border |
| `--title`, `--author`, `--copyright`, `--description`, `--keyword` (repeatable), `--date-taken` | Metadata written into every exported file; left out, the output folder keeps its value; see [Export metadata](#export-metadata) |
| `--ppi N` / `--profile NAME` | True scan resolution, directly or from a scanner profile |
| `--flatten auto\|off` | Unwarp curled prints along their traced edges (default `auto`) |
| `--shadows auto\|off` | Remove edge shadows on plain frames (default `auto`) |
| `--overlay` | Also write `_overlays/<name>.jpg` with the detected outline (and the curved edges when flattened) |
| `--strict` | Do not export scans that need review |
| `--force` / `--redetect` | Re-export / re-detect even if up to date |
| `--dry-run` | Detect and evaluate only, write nothing |
| `-j N` | Worker processes (default: from free RAM, about 1.5 GB per worker) |
| `--json` | Machine-readable output (see below) |

<a id="scanner-profiles"></a>

Scanner profiles live in `~/.config/polaroid-processor/profiles.yaml` (or `--profiles`, or `$POLAROID_PROFILES`):

```yaml
profiles:
  epson-v850-3200: 3200
```

In the app, Settings > Advanced has a dropdown with "none" and every profile of the file the project uses, shown as `name (3200 ppi)` and sorted by name; the page does not read the file, the app does. Pick one instead of typing a ppi (a project takes either, not both). A project whose saved profile is not in the file keeps it as an extra option, `name (not in profiles file)`; other settings still save, but a run fails with a clear message until you add it to the file, choose another profile, or choose "none" and type the profile's ppi in the same save (that keeps the scale and exports nothing again; "none" alone infers the scale and may export again). Without a profiles file the dropdown has only "none" and a hint with the file's path. Profiles are edited in the file; the app does not write it. A changed scale or profile exports an exported project again.
Scan file names must be unique within one output folder, and so must their stems: `a.tif` next to `a.tiff` is refused because both would write `a.tif`, `a.jpg`... Use one output folder per input folder.

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

With the photo crop on, `--of window` puts the tiles on the corners of the photo window instead: its border is cyan (a round window: its bounding square and the circle in it), the print outline stays green. A scan without a window shows the nominal window of its format, which `fix --window-set` / `--window-nudge` start from.

### `fix`

```bash
polaroid fix OUTPUT NAME --nudge tl=4,-2 --shift right=-12 --export
polaroid fix OUTPUT NAME --set bl=265,13868
polaroid fix OUTPUT NAME --accept
polaroid fix OUTPUT NAME --reset
polaroid fix OUTPUT NAME --window-nudge br=-4,-3 --window-set tl=610,905
polaroid fix OUTPUT NAME --window-kind round
polaroid fix OUTPUT NAME --no-photo-crop
polaroid fix OUTPUT NAME --reset-window
```

Corners are `tl`, `tr`, `br`, `bl`; sides are `top`, `right`, `bottom`, `left`. `--shift` moves a side outward (positive) or inward (negative). `--accept` marks the current outline as reviewed. Every change is stored in the manifest with author (`--author`, default `human`), time, note and previous value. `--export` re-exports that scan right away; otherwise the next `process` run does it.

The `--window-*` options fix the photo window and need the photo crop on in that output folder (see [Photo area crop](#photo-area-crop)): `--window-set` / `--window-nudge` move its corners (a scan without a window starts from the nominal one for its format), `--window-kind square|round` changes the kind, or picks the nominal window to start from (default: what most scans of the batch have), `--no-photo-crop` exports this scan without a photo crop by choice, `--reset-window` takes a hand-set window away. The scan's status is judged again at once.

### `app` and `review`

```bash
polaroid app [FOLDER...] [--port N] [--no-browser]
polaroid review OUTPUT [--all] [--port N] [--author NAME] [--no-browser]
```

`app` starts the [web app](#web-app); each `FOLDER` becomes a project (an output folder with a manifest opens as it is). `review` opens the app on one output folder, with the grid on "needs review" unless `--all` is given. Both start a local stdlib server on `127.0.0.1` (no network access, no extra dependencies) and open the browser, or reuse the app that already runs. When they start the server they block until it is stopped (Ctrl+C); when an app already runs they open the browser on it and return at once. The fix page of the app shows each scan as a 3x3 grid: the four corners and the four edge midpoints as 480 px tiles of the original pixels (contrast-stretched when dark), and an overview in the middle. The green line is the current outline.

- Drag a magenta corner, or the green grip on an edge tile (moves the whole side), onto the print edge.
- Click a tile to enlarge it; there the arrow keys move 1 px and `Shift` + arrow 10 px. `Esc` closes.
- **Confirm** stores the current outline as reviewed (`fix --accept`) and goes to the next scan. **Save correction** stores the moved corners and cuts the tiles again at the new outline. **Reset** drops edits that are not saved yet. **Skip** goes on without writing.
- **Photo window.** With the photo crop on, the window is drawn in cyan (a round one as its circle) in every tile and the overview, and the side panel has a "Photo window" section. **Photo window** switches the tiles to the window's corners and edges: drag the cyan handles onto the picture's edge, then Save correction (a hand-set window is stored with author, time and the previous window). **Print outline** switches back. Where no window was found, **Place square window** / **Place round window** start from the nominal window of the format, placed from the print outline; then drag its corners. **Square** / **Round** change the kind (a round window is edited through its bounding square), **No photo crop** exports this print without a photo crop by choice (the scan is ok), **Back to automatic** takes a hand-set window away. Window corners must run clockwise (tl, tr, br, bl) and lie inside the print outline; a round window is cut as the ellipse inscribed in its four corners, the shape drawn in the page. Save correction stores a moved outline and a moved window in one fix: both or neither. Without the photo crop none of this is shown. The grid cards show the window as a cyan outline over the crop preview, and tag a scan with a hand-set window "window by hand" and one with the choice of no photo crop "no photo crop".

Saves go through the same code as `fix` (history with author, time, note and previous outline; author is `--author`, default `human`); the scan is judged again on the new outline and its preview renewed. Export from the app's Export button, or run `process` again. Tiles are cut from the input scans, which are only read. The URL is printed.

### `status`

```bash
polaroid status OUTPUT [--needs-review] [--json]
```

## Photo area crop

An extra output, off by default: next to the full print (frame included, always written exactly as before) each export also writes the picture area inside the frame. Settings: `--photo-crop {off,derivatives,tiff}` and `--photo-bleed-mm` (CLI, `api.make_settings`, the MCP `polaroid_process`, the app's settings panel: "Photo area crop" and "Photo crop bleed"); both are part of the export key, so changing them exports again, while a project with the crop `off` keeps its old keys. A run that leaves both options out keeps the values stored in the output folder (a new folder starts with `off` and 0.3 mm), so re-running a project without the flag does not switch its photo crop off.

| File | Content |
|---|---|
| `<out>/photo/high/<name>.jpg`, `<out>/photo/web/<name>.jpg` | The same derivative list and sizes as the full print |
| `<out>/photo/<name>.tif` | Only with `tiff`: lossless master, same ICC profile, compression and resolution tag |

- **The window** is measured per scan at Detect, on a quarter-scale straightened copy (about 0.4 s a scan), and its four corners are stored in the manifest in scan pixels (`window` of the scan's entry, with the outline it was found from). A hand fix of the outline, a rim refit or a re-detect makes an automatic window stale and finds it again; turning the crop on for an already detected project runs only this step. Square windows are cut along their own border, so a window that sits a fraction of a degree rotated or sheared against the frame comes out square. Round windows are a tight square around the circle.
- **Bleed.** The crop is cut `--photo-bleed-mm` (default 0.3) inside the measured border on every side, which hides the soft step and the thin dark line at the window edge. The photo is warped straight from the original scan (one resampling, as the full print); edge shadow correction is not applied, as it only touches the frame.
- **Round windows.** The `tiff` master has an alpha channel, transparent outside the circle; the JPG derivatives keep the frame pixels in the four corners.
- **Window not found.** The full print is exported as always, no `photo/` files are written for that scan, the manifest gets the warning `photo window not found: <reason>` and the scan is `needs_review`, until you do one of three things: set a window by hand (`fix --window-set` / `--window-nudge`, or the drag view's Place window), choose no photo crop (`fix --no-photo-crop`, the drag view's No photo crop: the scan is ok and gets no photo files), or accept the scan (`fix --accept`, the app's Confirm: the full print is exported without a photo crop, the warning stays). A fix of the print outline alone does not clear it. An export that makes no photo crop for a scan (window lost after a fix, crop turned off) removes that scan's older files in `photo/`.
- **Hand-set window.** Your window choices (a hand-set window, no photo crop, an accept) are stored in `window_fix` of the scan's entry, which only `fix` writes; the run keeps writing `window`, the automatic window, and never touches your choice, also when a fix is saved while a run works. A hand-set window has `source: "manual"`, author, time and note; the history keeps the previous choice, and `fix --reset-window` (the drag view's Back to automatic) takes it away. It is a place on the scan, independent of the print outline: a later fix of the outline keeps it, as long as it still lies inside the new outline. If it does not (or a re-detect or refit moved the outline), the scan is `needs_review` with the warning `photo window set by hand lies outside the print outline` and the automatic window is cut meanwhile (no photo crop when that is missing too), also after you accept the scan, until you move the window inside or take it away. An accept is stored only while there is something to accept (a window not found, or a hand-set window outside the outline), and `fix --reset` of the outline revokes it. An automatic window is found again whenever the outline changes. An accept (`fix --accept`, the app's Confirm) belongs to the outline it was given for and lapses when the outline changes; it survives a new detection that leaves the outline as it is. A hand-set window and a no-photo-crop choice are dropped (history `window-drop`) only when the scan file itself changed. Manifests made before hand-set windows existed: a missing window counted as accepted by any outline fix; such a window (no `accept_needed` key) still does, until it is found again.
- **Results.** With the crop on, `process`/`status` records carry `window` (`state` found, missing or none, kind, format, size, corners, reason, source, and for a hand-set window who set it; `accepted` when the scan is accepted for its outline; `outside_outline` with the hand-set window's kind, corners and author when the outline moved away from it, in which case `window` describes the automatic window that is cut meanwhile); `outputs` has `photo_tiff`, `photo_high`, `photo_web`.

## Export metadata

Text metadata in every exported file, per project: **title**, **author**, **copyright**, **description**, **keywords** (a list) and **date taken** (`YYYY`, `YYYY-MM` or `YYYY-MM-DD`; Polaroids are often known to the month only). All are optional; with none set, a file carries only the scanner's maker and model, the software (`polaroid-processor <version>`), the resolution and the ICC profile, and no description (tifffile's own `{"shape": ...}` text is gone in every case).

```bash
polaroid process scans/260328_Roll -o processed/260328_Roll \
  --title "Roll 4, {n}" --author "Jörg Müller" --copyright "© 2026 Jörg Müller" \
  --keyword polaroid --keyword "{project}" --date-taken 1987-07
```

- **Placeholders** in title, author, copyright, description and each keyword, filled per scan: `{stem}` (the file name without extension), `{n}` (the number the stem ends in, as written: `Polaroid_06` gives `06`; empty when it ends in no digit), `{project}` (the input folder's name) and `{format}` (`600` or `Go`). `{{` and `}}` give literal braces. An unknown placeholder, a stray brace or a bad date is refused when the setting is saved (CLI exit 2, API `UsageError`, the app's panel names the field), never in the middle of an export. Limits: 500 characters per text and keyword, 4000 for the description, 100 keywords, 8000 characters over all texts.
- **Where to set it.** CLI: `process --title/--author/--copyright/--description/--keyword/--date-taken`. Python: `api.make_settings(metadata={...})`. MCP: the `metadata` object of `polaroid_process`. App: the "Metadata" group of the settings panel, with a live preview of the filled title and description for the project's first scan (and the placeholders listed under it). A run that leaves the metadata out keeps the project's values (stored in the manifest's `settings`); a field set to `""` (keywords: `--keyword ""` or `[]`) clears it. New app projects start with the author and copyright of your last panel choice.
- **What a file carries.** One XMP packet (title, creator, rights, description, keywords, date, the scan's maker and model) in the TIFF (tag 700), the JPG (APP1) and the PNG (`iTXt`), plus the legacy fields common viewers read: `ImageDescription` (the description, else the title), `Artist`, `Copyright`, the EXIF `DateTimeOriginal` (full dates only; a year or month is in the XMP only) and the Windows `XPTitle`, `XPAuthor`, `XPKeywords` and `XPComment` tags. Text is UTF-8, so umlauts survive. It goes into the full print's TIFF and derivatives and into the photo crops, not into the cached previews or overlays.
- **Where you see it.** Windows Explorer's Details tab and the Properties dialog show Title, Authors, Copyright, Tags, Comments and Date taken (PNG: no Subject and Comments). Lightroom and other XMP readers read the XMP packet: title, caption (the description), creator, copyright, keywords and date created (checked against the XMP itself, not yet in Lightroom). The EXIF date needs a full date; with only a year or month the date lives in the XMP.
- **A change after the export.** Metadata is not part of the export key. Each export remembers the metadata it was written with; when the project's metadata changes, the next run rewrites the headers of the existing files in place, without touching a pixel, and says `metadata_updated` in the summary (`metadata updated N` on the console). The app counts such scans in `Export (N)` until a run has done it. A file whose structure is not as expected is exported again instead. The run needs the files free: if a program holds one open (a viewer, a sync client), the file keeps its old headers, the scan gets the warning `metadata_not_updated: <file>: <reason>` and the summary counts it in `metadata_skipped` (console: `metadata NOT updated N`); it does not change the scan's status, and the next run tries again.

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
{"type": "summary", "scans": 14, "counts": {"ok": 13, "needs_review": 1}, "exported": 14, "retagged": 0,
 "needs_review": ["scan_01.tif"], "failed": [], "manifest": "...", "cancelled": false, "time_s": 59.0}
```

Summary keys beyond the counts: `exported` (scans written in this run), `retagged` (finished exports whose resolution tags were patched because only the inferred scale moved; nothing was encoded again), `metadata_updated` (finished exports whose text headers were rewritten in place because the project's metadata changed), `metadata_skipped` (those that could not be, a file in use: see [Export metadata](#export-metadata)), `cancelled` (the run was stopped before every scan was started: run it again to resume).

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

| `photo window not found: <reason>` | Only with the photo crop on: no photo window was found in the print (see [Photo area crop](#photo-area-crop)) |

Informational: `metadata_not_updated: <file>: <reason>` (a file's text headers could not be rewritten, the next run tries again; the status is not changed), `scale_inferred:<source>`, `size_unverified` (single scan, no scale reference), `edge_repaired:<side>` (the holder rim touched that side; the side was refit on the print edge behind it), `flattened:<sides>:<mm>` (the export follows these sides' curves; largest bend), `flatten_straight:<side>` (in a flattened print, this side's trace was not reliable and stays straight), `rim_refit:<side>:<mm>` (the outline sat on the holder rim along that side; the batch showed the print oversized by exactly the distance to the print edge behind the rim, so the side was moved there; the original outline stays in the manifest as `quad_detected`).

For sides that follow a curve, the straight-line checks `not_parallel`, `edge_inside` and `edge_curved` do not apply. Each result also carries `flatten` (applied, sides, bend per side) and `shadows` (corrected sides, strongest darkening, share of each side that could be measured).

Exit codes: `0` success (scans may still need review), `1` at least one scan failed, `2` usage error, which includes a refused run (see below). Logs go to stderr (`-v` for progress).

### Run lock and concurrent writers

Only one run (`process`, the app's Detect / Export, MCP `polaroid_process`, `fix --export`) works on an output folder at a time. A run takes an operating-system lock on `<out>/polaroid-run.lock`, which also holds its pid and start time. The system drops the lock when the process ends, so a crashed run leaves nothing behind. A second run fails with `RunLocked` ("another run is working on ...") and exit code 2. A fix without `--export` never needs the lock, and `fix --export` during a run stores the fix and returns the record with `note: "fix saved; export skipped: ..."` (exit code 0): run `process` or Export afterwards.

Writers merge before they save: the manifest is re-read under a short lock and merged per scan, so a fix made by the app, the CLI or an agent while a run goes is kept, and the run judges the fixed scan on its fixed outline.

### Review loop for agents

1. `polaroid process IN -o OUT --json` and collect scans with `status: needs_review`.
2. For each: `polaroid inspect NAME -m OUT --json`, open the PNG, compare the green line with the print edge.
3. Correct with `polaroid fix OUT NAME --nudge/--shift/--set ... --author agent --note "why"`, or confirm with `--accept --author agent`. A scan flagged `photo window not found` (photo crop on): `polaroid inspect NAME -m OUT --of window` shows the nominal window and the print; place it with `--window-set` / `--window-nudge`, or use `--no-photo-crop`, or `--accept` to export the full print without the crop.
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
| `polaroid_process` | `process` a batch; returns the summary and per-scan results (full records for flagged scans and for scans whose metadata headers could not be rewritten); takes a `metadata` object for the export metadata. Sends a progress notification per finished scan. |
| `polaroid_detect` | Detect outlines, write nothing |
| `polaroid_inspect` | Review sheet as an inline image plus corner coordinates; also saved to `<output>/_inspect/`; `of: "window"` centres it on the photo window |
| `polaroid_fix` | `set` / `nudge` / `shift` / `accept` / `reset` as JSON objects, author `agent` by default, re-exports by default; with the photo crop on also `window_set` / `window_nudge` / `window_kind` / `no_photo_crop` / `reset_window` |
| `polaroid_status` | Counts and the scans that need review |

Tool errors (bad arguments, missing manifest, another run working on the output folder) come back as tool results with `isError`, so the agent can correct itself. A refused run is not a failure of the scans: wait and try again. `polaroid_fix` with `export` while another run holds the folder still saves the fix; its result says the export was skipped.

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
python -m pytest              # synthetic tests; real-scan regression runs when the scans are present
python tools/export_labels.py OUTPUT...   # collect reviewed corners into tests/data/labels.json
```

The real scans live outside the repo: the regression tests read them from `$POLAROID_SAMPLES` (default `E:/POLAROID_PROCESSOR_SAMPLES`) and `$POLAROID_LIBRARY`, and skip what is missing. Experiments go to `.dev/`.

## License

See [LICENSE](LICENSE).
