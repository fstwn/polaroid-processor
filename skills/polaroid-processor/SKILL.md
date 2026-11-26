---
name: polaroid-processor
description: Crop and rectify flatbed scans of instant prints (Polaroid 600, i-Type, SX-70, Go) lying in a black scanning holder, or photographed on white paper, then review and fix flagged scans and crop failed ones by hand (also tickets, emulsion lifts and other flat items). Use when the user asks to process, crop, straighten or export Polaroid/instant-print scans, or to handle scans that failed or need review in a polaroid-manifest.json output folder.
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
polaroid inspect SCAN_NAME -m OUTPUT_FOLDER --of window --json     # photo crop on: tiles on the photo window's corners
polaroid fix OUTPUT_FOLDER SCAN_NAME --window-nudge br=-4,-3 --author agent --note "why" --export --json
polaroid fix OUTPUT_FOLDER SCAN_NAME --set tl=378,1620 --set tr=6988,1626 --set br=6962,10880 --set bl=386,10882 --item other --author agent --note "why" --export --json   # crop a failed scan
```

If `polaroid` is not found, it is installed in a Python environment that is not active; ask the user which one.

## Rules

- Never write into, rename or delete anything in the input folders. Outputs go to a separate output folder, one per input folder (scan file names must be unique per output folder).
- Every fix gets `author` = `agent` and a `note` saying what you saw. Fixes are logged and revertible (`reset`, within the limits in [Failed scans](#failed-scans-hand-crop)).
- You cannot change or remove an `item` or a `scan_ppi` that a human set, and you cannot clear `cropped by agent`. Do not try with another author name; leave both to the user.
- Do not change `margin_mm`, derivative presets or scale (`ppi`, `profile`) unless the user asks.
- `photo_crop` (`off` default, `derivatives`, `tiff`) and `photo_bleed_mm` (0.3) add a crop of the photo window next to the full print (`<out>/photo/high|web/<name>.jpg`, with `tiff` also `<out>/photo/<name>.tif`, round windows with alpha). Turn it on only when the user asks. Leave both options out to keep the output folder's setting (a new folder starts with `off`, 0.3), so a later run does not switch a project's photo crop off. The full print is always exported as before.
- `metadata` (an object with `title`, `author`, `copyright`, `description`, `keywords` (a list), `date_taken` `YYYY`, `YYYY-MM` or `YYYY-MM-DD`; CLI `--title --author --copyright --description --keyword --date-taken`) writes text metadata into the headers of every exported file (full print, derivatives, photo crops): Explorer's Title, Authors, Copyright, Tags, Comments, Date taken and the XMP that Lightroom reads. Set it only when the user gives you the values; never invent an author, copyright or date. The texts take placeholders filled per scan: `{stem}` (file name without extension), `{n}` (the number it ends in, digits as written), `{project}` (input folder name), `{format}` (`600` or `Go`); `{{` `}}` are literal braces. An unknown placeholder or a bad date makes `polaroid_process` fail before anything is written: fix the text and call again. Fields you leave out keep the output folder's values, so a later run without `metadata` keeps what a project has; an empty string (empty list for keywords) clears a field. Changing it on an output folder that was already exported only rewrites the headers of the finished files in place (no re-encoding, no pixel change): `summary.metadata_updated` counts them. It is not part of the export key.
- One run per output folder. When `process`, `polaroid_process` or an export fails with "another run is working on ... (pid ...)" (exit code 2), the user's app or another agent has the folder: wait and try again; never delete `polaroid-run.lock` unless the user says no run is going. A `fix` is still saved while a run goes; with `--export` it returns a `note` saying the export was skipped, so run `process` again afterwards instead of repeating the fix.

## Workflow

1. Process: `polaroid_process` with `inputs` and `output`. Read `summary.counts`, `needs_review` and `failed` (failed scans: see [Failed scans](#failed-scans-hand-crop); the call then reports an error or exit code 1, but the other results are valid). `summary.cancelled` true means the run was stopped early (by the app's Cancel): run it again to finish. `summary.retagged` counts exports whose resolution tags were patched because only the inferred scale moved; nothing was encoded again, no action. `summary.metadata_updated` counts exports whose text headers were rewritten because the metadata changed, no action. `summary.metadata_skipped` above 0 means a file's headers could not be rewritten (a viewer, Explorer's preview or a sync client holds it): those scans carry the warning `metadata_not_updated: <file>: <reason>` (status unchanged, listed with their warnings even when ok); ask the user to close the program that holds the file and run `polaroid_process` again (the same call, it retries by itself).
2. For each scan with status `needs_review`, read its `warnings`:
   - `hint:fix --shift <side>=-<px> ...`: the review rules found the likely fix. Apply it, then verify with inspect.
   - `size_outlier`, `not_parallel`, `edge_uncertain`, `edge_noisy`, `edge_curved`, `size_mismatch`: inspect first.
   - `edge_inside:<side>:<mm>`: part of that side's print edge lies inside the crop line, so holder would show. Shift the side inward by the reported amount minus the 0.1 mm margin (`shift` negative, in px; about 128 px per mm) if the rest of the side allows it; if the edge is visibly curved, escalate.
   - `photo window not found: <reason>` (only with `photo_crop` on): no photo window was found in this print; the full print is exported but no photo crop. `inspect` with `of: "window"` (CLI `--of window`) shows the print and the nominal window for its format. Then pick one: (a) the picture is there: place the window with `polaroid_fix` `window_set` / `window_nudge` (corners `tl tr br bl` in scan pixels, read off the grid; `window_kind` `round` for a round window, whose corners are its bounding square), then `inspect` again with `of: "window"` and check that the cyan line lies on the picture's edge; (b) the print has no usable window or the user does not want a crop for it: `no_photo_crop` (the scan becomes ok, no photo files, by choice); (c) the user just wants the full print through: `accept` (exports the full print without a photo crop, the warning stays). A fix of the print outline alone does not clear the warning. Escalate to the user when you cannot see where the picture's edge is.
   - Check a found window the same way when a photo crop looks wrong (`of: "window"`): the cyan border should lie on the picture's edge, a round window shows its circle. Correct it with `window_nudge`. A hand-set window (`source: "manual"` in the status record) stays through later outline fixes and runs while it lies inside the outline; if the outline moves away from it, the scan is flagged `photo window set by hand lies outside the print outline` (the status record's `window.outside_outline` then holds the hand-set corners, which `inspect` with `of: "window"` shows and `window_nudge` edits, while `window` itself is the automatic window that is cut; window corners must also run clockwise, tl tr br bl). `reset_window` takes the hand-set window or the no-photo-crop choice away. An accept belongs to the outline it was given for.
   - `metadata_not_updated: <file>: <reason>`: see step 1; not a scan problem, do not fix or accept anything, rerun once the file is free.
   - `cropped by agent`: your own (or another agent's) hand crop, waiting for a human. Check it with inspect; then leave it for the user.
   - `scale_unverified`: informational, only on item `other`: the resolution tag is the scanner's tag unchecked, or missing. Tell the user; set `scan_ppi` only when the user gives the value.
   - `scale_inferred`, `size_unverified`, `edge_repaired`, `flattened`, `flatten_straight`, `rim_refit`: informational, no action. `rim_refit` means a side was moved off the holder rim because the batch size proved it; check it only if the user asks. `flattened` means the export unwarps the curled print along its traced edges; in inspect the green crop line is then curved and the grey line is the straight outline.
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
6. Hand-off: for the scans you escalated, tell the user to run `polaroid review OUTPUT_FOLDER` (the grid starts on "To review": needs review and failed; `--all` starts on "All"). It opens the local web app on that project: they click a scan, drag the outline onto the print edge and press Confirm or Save correction; fixes go into the manifest with `author` `human`. Do not run it yourself (it blocks until stopped). The user can also start `polaroid app` (or `launcher/polaroid-app.bat`) and pick the folder. Afterwards run `process` again, or let the user press Export in the app. The app shows your own fixes live while it is open, and a fix by the user while you work is kept (writers merge); read `polaroid_status` again before you decide what is left.
7. Report: counts, what you fixed and why, what you left for the user (name the scans and the `review` command), and the output folder.

## Failed scans (hand crop)

A scan with status `failed` has no outline. It may be any flat item (a ticket, a paper sheet with an emulsion lift) or a print detection could not read. Crop it by hand when the user wants it exported.

1. Find it: `summary.failed` of `polaroid_process`, or `polaroid_status` with `needs_review: false` (CLI `status --json`). A scan that was never detected cannot be cropped: process it first. A `failed` scan can also be an export error on a scan that has an outline: read `error`.
2. `polaroid_inspect` it (`scan`, `output`): the whole scan with a pixel grid in scan coordinates and a cyan start rectangle (`item` `600`, `go` or `other` picks it). The record has `failed: true`, `scan_px` and the rectangle's corners. The overview is downscaled, so corners read from it are rough; step 4 refines them. A smaller `grid` gives more lines.
3. `polaroid_fix` with `set` = all four corners (`tl tr br bl`, clockwise) and `item`:
   - `600` / `go`: an instant print of that format; exported like a hand-fixed print.
   - `other` (default on a failed scan): any flat item, exported exactly as drawn: no margin, flattening, shadow removal, photo crop or format. Draw the corners where the export should end.
   - The corners lie inside the scan, form a convex quadrilateral, have sides of at least 50 px (every corner also 50 px from the lines of the other two sides) and are turned at most 45 deg.
4. `polaroid_inspect` again: it now shows the full-resolution tiles. Correct with `nudge`, `shift` or `set`; pass a larger `window` when an edge lies outside a tile. The 1 mm escalation limit of the workflow does not apply while you place your own crop.
5. Report it. The scan is `needs_review` with `cropped by agent` until a human confirms or saves it in the app (`polaroid review OUTPUT_FOLDER` shows it under "To review"). It is exported meanwhile unless `strict`.

Escalate instead of cropping when you cannot see where the item ends, or when it is not clear what the user wants in the picture (for an emulsion lift: the whole sheet or only the lift). The user crops in `polaroid review OUTPUT_FOLDER`: its grid starts on "To review", which shows scans that need review and failed scans (`--all` starts on "All"); a click on a failed scan opens the place step.

Details:

- `scan_ppi` (item `other` only, `0` takes it away) is the scan's own resolution and sets only the resolution tag. Without it the tag comes from the project's `ppi` / `profile`, else the median of at least 3 prints with the same scan size in pixels, else the TIFF tag as it is (`scale_unverified`), else there is none. A small item with its own scan area gets the scanner's tag, which can be wrong. Never guess the value.
- `item` also works alone on a detected scan (`auto` = as detected). Change it only when the user asks or the scan clearly shows something else.
- Later runs keep the crop and do not detect the scan again; `redetect` does, and the crop stays in force. A changed scan file drops crop, item and ppi. A crop or item is refused while the file changed since the last run: process first.
- `reset` takes crop, item and `scan_ppi` away. On a failed scan you may reset only a crop that was an agent's from the start (no human saved or confirmed it since) and still carries the mark. Files already exported stay.

## Notes

- Scale: about 128 px/mm on the usual scanner setting, so 0.2 mm is about 25 px.
- Curled prints have bowed sides. With `flatten` on (the default) the export follows the traced edges, so the corners are not trimmed. A manual fix (`set`, `nudge`, `shift`) turns flattening off for that scan, because the corrected outline has straight sides; `accept` on an automatic outline keeps it.
- `fix` with export re-exports only that scan; `process` again re-exports anything whose outline or settings changed.
