---
name: polaroid-processor
description: Crop and rectify flatbed scans of instant prints (Polaroid 600, i-Type, SX-70, Go) lying in a black scanning holder, then review and fix flagged scans. Use when the user asks to process, crop, straighten or export Polaroid/instant-print scans, or to review scans that need review in a polaroid-manifest.json output folder.
---

# Polaroid processor

Turns TIFF scans of one instant print in a black holder into a cropped, perspective-corrected TIFF master plus JPG/PNG copies. All state lives in `polaroid-manifest.json` in the output folder.

## Tools

Use the MCP tools when they are available (`polaroid_process`, `polaroid_inspect`, `polaroid_fix`, `polaroid_status`, `polaroid_detect`). Otherwise use the CLI with `--json`; it prints one JSON object per line on stdout:

```bash
polaroid process INPUT_FOLDER -o OUTPUT_FOLDER --json
polaroid status OUTPUT_FOLDER --needs-review --json
polaroid inspect SCAN_NAME -m OUTPUT_FOLDER --json        # writes OUTPUT_FOLDER/_inspect/<stem>.png; open it
polaroid fix OUTPUT_FOLDER SCAN_NAME --shift bottom=-59 --author agent --note "why" --export --json
```

If `polaroid` is not found, it is installed in a Python environment that is not active; ask the user which one.

## Rules

- Never write into, rename or delete anything in the input folders. Outputs go to a separate output folder, one per input folder (scan file names must be unique per output folder).
- Every fix gets `author` = `agent` and a `note` saying what you saw. Fixes are logged and revertible (`reset`).
- Do not change `margin_mm`, derivative presets or scale (`ppi`, `profile`) unless the user asks.

## Workflow

1. Process: `polaroid_process` with `inputs` and `output`. Read `summary.counts`, `needs_review` and `failed`.
2. For each scan with status `needs_review`, read its `warnings`:
   - `hint:fix --shift <side>=-<px> ...`: the review rules found the likely fix. Apply it, then verify with inspect.
   - `size_outlier`, `not_parallel`, `edge_uncertain`, `edge_noisy`, `edge_curved`, `size_mismatch`: inspect first.
   - `scale_inferred`, `size_unverified`, `edge_repaired`: informational, no action.
3. Inspect: `polaroid_inspect` with `scan` and `output`. The image has eight full-resolution tiles (corners `tl tr br bl`, edge midpoints `top right bottom left`) and an overview.
   - Grid labels are absolute scan pixel coordinates; the green line is the crop edge; magenta crosses are corners.
   - The print edge is where the print (white, gold, coloured or glittery area) meets the black holder. The grey fibrous band around the print is the holder rim, not the print.
   - Tiles marked `stretched` are contrast-enhanced: glitter specks show the print area.
4. Decide per side:
   - Green line on the print edge everywhere: `accept`.
   - Whole side off by a constant amount: `shift` that side (positive moves it outward, negative inward) by the pixel distance read off the grid.
   - Only one corner off: `nudge` that corner by `[dx, dy]`, or `set` it to `[x, y]`.
   - Then inspect again and check the tiles you changed.
5. Escalate to the user instead of guessing when: the edge is invisible even in stretched tiles (black frame on black holder), an edge is visibly curved so no straight line fits, or a fix would move a side by more than about 1 mm (~130 px at 128 px/mm).
6. Report: counts, what you fixed and why, what you left for the user, and the output folder.

## Notes

- Scale: about 128 px/mm on the usual scanner setting, so 0.2 mm is about 25 px.
- Many prints have slightly concave sides: corners may sit 20-60 px outside the line while the middle is on it. That is normal; do not chase it with corner nudges unless holder becomes visible inside the crop.
- `fix` with export re-exports only that scan; `process` again re-exports anything whose outline or settings changed.
