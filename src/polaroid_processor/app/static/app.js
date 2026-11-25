"use strict";
const $ = (id) => document.getElementById(id);
// ?filter=needs_review (from `polaroid review`) applies to the first project only, then the address is clean
let startFilter = new URLSearchParams(location.search).get("filter");
if (startFilter) history.replaceState(null, "", location.pathname);
const SETTING_FIELDS = ["margin_mm", "photo_crop", "photo_bleed_mm", "flatten", "shadows", "high_on", "high_edge", "high_quality", "web_on", "web_edge",
  "web_quality", "ppi", "profile", "tiff_compression", "interpolation", "overlay"];
const META_FIELDS = ["title", "author", "copyright", "description", "keywords", "date_taken"];
const S = { shownMeta: {}, view: "home", project: null, rev: null, filter: "all", cards: new Map(), keys: [], scans: [], prog: null, timer: null };

// ---------------------------------------------------------------- helpers
function el(tag, cls, text, attrs) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = text;
  if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}

function showMsg(text, info) {
  const m = $("msg");
  m.textContent = text || "";
  m.className = info ? "info" : "";
  m.style.display = text ? "block" : "none";
}

async function api(path, body) {
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  let data = null;
  try { data = await r.json(); } catch (e) { /* not JSON */ }
  if (!r.ok) {
    const err = new Error((data && data.error) || `${r.status} ${r.statusText}`);
    if (data && data.fields) err.fields = data.fields;      // a settings field that failed its check, by name
    throw err;
  }
  return data;
}

async function guarded(fn) {
  try { return await fn(); } catch (e) { showMsg(e.message); return undefined; }
}

function fmtTime(s) {
  s = Math.max(0, Math.round(s));
  const m = Math.floor(s / 60), r = s % 60;
  return m >= 60 ? `${Math.floor(m / 60)}:${String(m % 60).padStart(2, "0")}:${String(r).padStart(2, "0")}` : `${m}:${String(r).padStart(2, "0")}`;
}

function plural(n, word) { return `${n} ${word}${n === 1 ? "" : "s"}`; }

function defaultOutput(input) {
  const p = input.trim().replace(/[\\/]+$/, "");
  return p ? p + "_processed" : "";
}

function countLine(c) {
  const parts = [];
  const add = (cls, n, label) => { if (n) { const s = el("span", cls); s.append(el("b", null, n), " " + label); parts.push(s); } };
  const wrap = el("div", "counts");
  const s = el("span"); s.append(el("b", null, c.total), " " + (c.total === 1 ? "scan" : "scans")); parts.push(s);
  add("ok", c.ok, "ok");
  add("needs_review", c.needs_review, "needs review");
  add("failed", c.failed, "failed");
  add("exported", c.exported, "exported");
  add("todo", c.pending + c.detected, "to do");
  parts.forEach((p) => wrap.append(p));
  return wrap;
}

// ---------------------------------------------------------------- views
function setView(v) {
  S.view = v;
  $("home").hidden = v !== "home";
  $("project").hidden = v !== "project";
  clearTimeout(S.timer);
  const actions = $("actions");
  actions.textContent = "";
  showMsg("");
  if (v === "home") {
    $("title").textContent = "Polaroid";
    $("where").textContent = "";
    document.title = "Polaroid";
  }
}

// ---------------------------------------------------------------- home
let outEdited = false;

function syncOutput() {
  if (!outEdited) $("out-path").value = defaultOutput($("in-path").value);
}

async function browse(target, edited) {
  const r = await guarded(() => api("/api/browse", { initial: $(target).value }));
  if (r && r.path) {
    $(target).value = r.path;
    if (target === "in-path") syncOutput();
    if (edited) outEdited = true;
  }
}

async function openProject(body) {
  const p = await guarded(() => api("/api/project/open", body));
  if (p) enterProject(p);
}

function renderProjects(home) {
  const box = $("projects");
  box.textContent = "";
  if (!home.projects.length) { box.append(el("div", "dim", "No projects yet.")); }
  for (const p of home.projects) {
    const row = el("div", "proj" + (p.output_exists ? "" : " gone"));
    const main = el("div", "main");
    main.append(el("div", "nm", p.name));
    main.append(el("div", "paths", (p.input || "input folder unknown") + "  →  " + p.output, { title: `${p.input}\n${p.output}` }));
    main.append(countLine(p.counts));
    main.onclick = () => openProject(p.has_manifest ? { output: p.output } : { input: p.input, output: p.output });
    const forget = el("button", "link", "Remove from list");
    forget.onclick = async (ev) => { ev.stopPropagation(); await guarded(() => api("/api/project/forget", { output: p.output })); loadHome(); };
    row.append(main, forget);
    box.append(row);
  }
  const chips = $("recent-inputs");
  chips.textContent = "";
  for (const path of home.recent_inputs) {
    const c = el("button", "chip", path, { title: path });
    c.onclick = () => { $("in-path").value = path; syncOutput(); };
    chips.append(c);
  }
}

async function loadHome() {
  setView("home");
  const home = await guarded(() => api("/api/home"));
  if (home) renderProjects(home);
}

$("in-path").addEventListener("input", syncOutput);
$("out-path").addEventListener("input", () => { outEdited = true; });
$("in-browse").onclick = () => browse("in-path");
$("out-browse").onclick = () => browse("out-path", true);
$("open-browse").onclick = () => browse("open-path");
$("create").onclick = () => openProject({ input: $("in-path").value, output: $("out-path").value });
$("open-go").onclick = () => openProject({ output: $("open-path").value });
for (const [id, btn] of [["in-path", "create"], ["out-path", "create"], ["open-path", "open-go"]]) {
  $(id).addEventListener("keydown", (e) => { if (e.key === "Enter") $(btn).click(); });
}

// ---------------------------------------------------------------- project
const FILTERS = [["all", "All"], ["ok", "OK"], ["needs_review", "Needs review"], ["failed", "Failed"], ["todo", "Not checked yet"]];

function matches(scan, filter) {
  if (filter === "all") return true;
  if (filter === "todo") return scan.status === "pending" || scan.status === "detected";
  return scan.status === filter;
}

function filterCount(c, f) {
  if (f === "all") return c.total;
  if (f === "todo") return c.pending + c.detected;
  return c[f];
}

function renderFilters() {
  const box = $("filters");
  box.textContent = "";
  const c = S.prog ? S.prog.counts : { total: 0, ok: 0, needs_review: 0, failed: 0, pending: 0, detected: 0 };
  for (const [f, label] of FILTERS) {
    const chip = el("button", `chip ${f}${S.filter === f ? " on" : ""}`);
    chip.append(label + " ", el("span", "n", filterCount(c, f)));
    chip.onclick = () => { S.filter = f; renderFilters(); applyFilter(); };
    box.append(chip);
  }
}

function applyFilter() {
  let shown = 0;
  for (const s of S.scans) {
    const card = S.cards.get(s.key);
    const on = matches(s, S.filter);
    card.hidden = !on;
    shown += on;
  }
  const empty = $("empty");
  empty.hidden = shown > 0;
  if (!shown) {
    empty.textContent = !S.scans.length ? "This project has no scans yet."
      : S.filter === "all" ? "" : "No scans with this status.";
    if (S.filter === "todo" && S.scans.length) empty.textContent = "Every scan has been checked.";
  }
}

function warnItems(scan) {
  if (scan.error && scan.status === "failed") return [scan.error];
  // "cropped by agent" is a tag of its own on the card; the stored detection error of a hand crop is not in the record
  return (scan.warnings || []).filter((w) => !w.startsWith("hint:") && !w.startsWith("scale_inferred") && !w.startsWith("flatten") && w !== "cropped by agent");
}

const ITEM_LABEL = { "600": "600", go: "Go", other: "Other" };

function sig(scan) {
  return JSON.stringify([scan.status, scan.preview && scan.preview.url, scan.stale, scan.warnings, scan.error, scan.export, scan.quad_source, scan.window, scan.window_fix, scan.item, scan.agent_crop, scan.hand_crop]);
}

// The photo window (cyan) over a card's crop preview, placed on the image as it is shown in the box.
function windowOverlay(box, im, w) {
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("class", "win");
  svg.setAttribute("viewBox", "0 0 1 1");
  svg.setAttribute("preserveAspectRatio", "none");
  const q = w.box;
  let pts = q;
  if (w.kind === "round") {      // the circle in the bounding square, as a polygon (the mapping to the card is affine)
    const c = [0, 1].map((k) => (q[0][k] + q[1][k] + q[2][k] + q[3][k]) / 4);
    const t = [0, 1].map((k) => (q[1][k] - q[0][k]) / 2), l = [0, 1].map((k) => (q[3][k] - q[0][k]) / 2);
    pts = Array.from({ length: 48 }, (_, i) => { const a = (i / 48) * 2 * Math.PI; return [c[0] + Math.cos(a) * t[0] + Math.sin(a) * l[0], c[1] + Math.cos(a) * t[1] + Math.sin(a) * l[1]]; });
  }
  const poly = document.createElementNS(NS, "polygon");
  poly.setAttribute("points", pts.map((p) => p.join(",")).join(" "));
  svg.append(poly);
  const place = () => {
    svg.style.left = im.offsetLeft + "px"; svg.style.top = im.offsetTop + "px";
    svg.style.width = im.clientWidth + "px"; svg.style.height = im.clientHeight + "px";
  };
  im.addEventListener("load", place);
  if (im.complete) place();
  if (window.ResizeObserver) new ResizeObserver(place).observe(box);
  return svg;
}

function buildCard(scan) {
  const card = el("div", "card " + scan.status);
  card._sig = sig(scan);
  const img = el("div", "img");
  if (scan.preview) {
    const im = el("img", null, null, { alt: scan.key, loading: "lazy" });
    im.src = scan.preview.url;
    img.append(im);
    if (scan.window && !scan.stale) img.append(windowOverlay(img, im, scan.window));
    if (scan.stale) img.append(el("span", "stale", "updating…"));
  } else {
    img.append(el("span", "none", scan.status === "failed" ? "no outline: click to crop by hand" : scan.status === "pending" ? "waiting for detection" : "preview pending"));
  }
  const meta = el("div", "meta");
  meta.append(el("div", "nm", scan.key, { title: scan.key }));
  const tags = el("div", "tags");
  tags.append(el("span", "badge " + scan.status, scan.status.replace("_", " ")));
  if (scan.hand_crop) tags.append(el("span", "x", "hand crop"));      // a crop on a scan nothing was found on
  else if (scan.quad_source === "override") tags.append(el("span", "x", "fixed"));
  if (scan.item) tags.append(el("span", "x", "item: " + ITEM_LABEL[scan.item]));
  if (scan.agent_crop) tags.append(el("span", "x warnx", "cropped by agent"));
  if (scan.window_fix === "hand") tags.append(el("span", "x", "window by hand"));
  else if (scan.window_fix === "none") tags.append(el("span", "x", "no photo crop"));
  if (scan.export === "done") tags.append(el("span", "x", "exported"));
  else if (scan.export === "tags") tags.append(el("span", "x", "exported, tags to update"));
  else if (scan.export === "headers") tags.append(el("span", "x", "exported, metadata to update"));
  else if (scan.export === "outdated") tags.append(el("span", "x warnx", "export outdated"));
  meta.append(tags);
  const w = warnItems(scan);
  if (w.length) {
    const ul = el("ul", "warns", null, { title: (scan.warnings || []).join("\n") || scan.error });
    for (const t of w.slice(0, 2)) ul.append(el("li", t.startsWith("hint:") ? "hint" : "", t));
    if (w.length > 2) ul.append(el("li", null, `+${w.length - 2} more`));
    meta.append(ul);
  }
  card.append(img, meta);
  card.onclick = () => openScan(scan);
  return card;
}

function openScan(scan) {
  // a failed scan opens too (its fix page starts at the place step); a scan not detected yet stays closed
  if (!scan.size_px && scan.status !== "failed") { showMsg("This scan has not been detected yet.", true); return; }
  showMsg("");
  // location.replace: opening and closing the page adds no history entries to step through with Back
  $("fixframe").contentWindow.location.replace("/fix#" + encodeURIComponent(scan.key));
  $("fixview").hidden = false;
  document.body.style.overflow = "hidden";
  history.pushState({ fix: true }, "");          // Back closes the page (asking first when edits are unsaved)
}

let skipPop = false;
function closeFix() {
  $("fixview").hidden = true;
  $("fixframe").contentWindow.location.replace("about:blank");
  document.body.style.overflow = "";
  if (history.state && history.state.fix) { skipPop = true; history.back(); }
  tick(true);
}

// Browser Back with the fix page open: the page decides (it asks about unsaved drags); a page that does not
// answer (blank, broken) is closed, so the overlay is never left stuck.
let closeTimer = null;
window.addEventListener("popstate", () => {
  if (skipPop) { skipPop = false; return; }
  if ($("fixview").hidden) return;
  clearTimeout(closeTimer);
  closeTimer = setTimeout(closeFix, 1500);
  try { $("fixframe").contentWindow.postMessage({ type: "request-close" }, location.origin); } catch (e) { closeFix(); }
});

window.addEventListener("message", (e) => {
  if (e.origin !== location.origin || !e.data) return;
  clearTimeout(closeTimer);
  if (e.data.type === "fix-saved" || e.data.type === "fix-close") closeFix();
  else if (e.data.type === "fix-stay") history.pushState({ fix: true }, "");     // Back was refused: the page stays
});

function renderScans(data) {
  S.rev = data.rev;
  S.scans = data.scans;
  const keys = data.scans.map((s) => s.key);
  const sameKeys = keys.length === S.keys.length && keys.every((k, i) => k === S.keys[i]);
  const grid = $("grid");
  const next = new Map();
  for (const s of data.scans) {
    const old = S.cards.get(s.key);
    if (old && old._sig === sig(s)) { next.set(s.key, old); continue; }
    const card = buildCard(s);
    if (old && sameKeys) old.replaceWith(card);
    next.set(s.key, card);
  }
  if (!sameKeys) {
    grid.textContent = "";
    for (const k of keys) grid.append(next.get(k));
  }
  S.cards = next;
  S.keys = keys;
  applyFilter();
}

function renderProgress(p) {
  S.prog = p;
  const run = p.run;
  const bar = $("runbar");
  let text = "", frac = 0, cls = "";
  const verb = { detect: "Detecting", export: "Exporting", all: "Detecting and exporting" };
  if (run) {
    const total = run.total;
    if (run.state === "running" || run.state === "cancelling") {
      if (total === null) text = "Looking for scans…";
      else if (total === 0) { text = "Nothing to do"; frac = 1; }
      else {
        frac = run.done / total;
        text = `${verb[run.mode] || "Working"} · ${run.done} / ${total}` + (run.eta !== null ? ` · ${fmtTime(run.eta)} left` : "");
        if (run.state === "cancelling") text = `Cancelling\u2026 (${run.done} of ${total} done)`;
      }
      if (run.phase) text = (total ? `${run.done} / ${total} · ` : "") + `${run.phase}…`;
    } else if (run.state === "finished") {
      cls = "done"; frac = 1;
      const did = { detect: "Detected", export: "Exported", all: "Detected and exported" }[run.mode] || "Processed";
      text = total ? `${did} ${plural(run.done, "scan")} in ${fmtTime(run.elapsed)}` + (run.errors ? ` · ${run.errors} failed` : "") : "Nothing new to do";
      if (run.metadata_updated) text = (total ? text + " · " : "") + `Updated the text headers of ${plural(run.metadata_updated, "scan")} (metadata)`;
      if (run.metadata_skipped) {
        text = (text ? text + " · " : "") + `${plural(run.metadata_skipped, "scan")} could not get new metadata (file in use?); the next run tries again`;
        cls = "locked";
      }
    } else if (run.state === "cancelled") {
      frac = total ? run.done / total : 0;
      text = `Cancelled: ${run.done} of ${total} done. Start it again to continue with the rest.`;
    } else if (run.state === "error") {
      cls = "error"; frac = 1; text = run.error;
    }
  }
  if (p.previews_pending && !(run && (run.state === "running" || run.state === "cancelling"))) {
    text = (text ? text + " · " : "") + `making previews, ${p.previews_pending} left`;
  }
  if (p.scale_error && !text.includes(p.scale_error) && !(run && (run.state === "running" || run.state === "cancelling"))) {      // export states are left out then
    text = (text ? text + " · " : "") + `The scale cannot be worked out: ${p.scale_error}. Runs fail until that is fixed (Settings > Advanced).`;
    cls = "locked";
  }
  if (p.lock && !(run && (run.state === "running" || run.state === "cancelling"))) {
    text = `Another process is working on this output folder (pid ${p.lock.pid}, ${p.lock.who}); runs wait until it ends.`;
    cls = "locked";
  }
  bar.hidden = !text;
  bar.className = "panel runbar " + cls;
  $("bar-fill").style.width = Math.round(frac * 100) + "%";
  $("bar-text").textContent = text;
  renderActions(p);
  renderFilters();
}

function renderActions(p) {
  const run = p.run, running = run && (run.state === "running" || run.state === "cancelling");
  const actions = $("actions");
  actions.textContent = "";
  const c = p.counts;
  const started = c.ok + c.needs_review + c.failed + c.detected > 0;
  const start = (mode) => async () => { await guarded(() => api(`/api/project/${mode}`, {})); tick(true); };
  if (running) {
    const b = el("button", "warn", "Cancel");
    b.disabled = run.state === "cancelling";
    b.onclick = () => guarded(() => api("/api/project/cancel", {}));
    actions.append(b);
  } else {
    const locked = !!p.lock;
    const todo = c.pending > 0;
    const d = el("button", started ? "" : "primary", !started ? "Detect" : todo ? "Resume detect" : "Detect again");
    d.disabled = !p.can_run || locked;
    d.title = p.can_run ? "Find the print in every scan and store the outlines; nothing is exported" : "The input folder of this project is not known or not found";
    d.onclick = start("detect");
    const e = c.to_export, h = c.to_headers || 0, t = c.to_retag;
    const n = e + h;      // scans whose files the run touches: a new export, or only new text headers (metadata)
    const x = el("button", started ? "primary" : "", n ? `Export (${n})` : t ? `Update tags (${t})` : "Export");
    x.disabled = !(n || t) || locked;
    x.title = n ? (e ? `Export the detected scans that have no up to date export (${e})` : "")
        + (e && h ? "; " : "") + (h ? `${plural(h, "scan")} only get new text headers (metadata), nothing is encoded` : "")
        + (t ? `; ${t} more only get new resolution tags` : "")
      : t ? "Only the scale changed: the resolution tags of these exports are written again, nothing is encoded" : "Nothing to export";
    x.onclick = start("export");
    const all = el("button", null, "Detect + export");
    all.disabled = !p.can_run || locked;
    all.title = "Detect new scans and export everything in one go; for batches you trust";
    all.onclick = start("all");
    actions.append(d, x, all);
  }
  const set = el("button", null, "Settings");
  set.onclick = () => toggleSettings();
  actions.append(set);
  const back = el("button", null, "← Projects");
  back.onclick = leaveProject;
  actions.append(back);
}

// ---------------------------------------------------------------- settings
async function loadSettings() {
  const r = await guarded(() => api("/api/project/settings"));
  if (!r) return;
  const v = r.settings;
  S.loadedSettings = v;
  fillProfiles(r.profiles, v.profile);      // before the field loop: it sets the select's value, which needs the options
  for (const f of SETTING_FIELDS) {
    const e = $("s-" + f);
    if (e.type === "checkbox") e.checked = !!v[f]; else e.value = v[f] === null || v[f] === undefined ? "" : v[f];
    e.disabled = r.running || !!r.locked_by;
  }
  for (const f of META_FIELDS) {
    const e = $("m-" + f), saved = (v.metadata && v.metadata[f]) || (f === "keywords" ? [] : "");
    e.value = f === "keywords" ? saved.join(", ") : saved;
    S.shownMeta[f] = e.value;       // the text as the field holds it (a textarea turns \r\n into \n): the base of "changed"
    e.disabled = r.running || !!r.locked_by;
  }
  showMetaErrors({});
  refreshMetaPreview(true);
  $("s-save").disabled = r.running || !!r.locked_by;
  $("s-note").textContent = r.running ? "A run is going: settings apply to the next one."
    : r.locked_by ? `Another process (${r.locked_by}) is working here.`
    : (v.other_derivatives ? `${v.other_derivatives} more derivative(s) from the command line are kept.` : "");
}

// The scanner profile dropdown: "none", then the profiles of the project's profiles file (the server lists them) as
// "name (3200 ppi)"; a saved profile that is not in the file stays as an extra, selected option so a save does not drop it.
function fillProfiles(p, current) {
  p = p || { path: "", items: [], error: null };
  const sel = $("s-profile"), hint = $("s-profile-hint");
  sel.textContent = "";
  const add = (value, text) => { const o = el("option", null, text); o.value = value; sel.append(o); return o; };
  add("", "none");
  for (const it of p.items) add(it.name, `${it.name} (${Number(it.ppi)} ppi)`);
  const missing = !!current && !p.items.some((it) => it.name === current);
  if (missing) add(current, `${current} (not in profiles file)`);
  let text = "";
  if (p.error) text = `Problem with the profiles file ${p.path}: ${p.error}. It should look like: profiles: {name: ppi}`;
  else if (!p.items.length) text = `No scanner profiles yet. They live in ${p.path}, written as: profiles: {name: ppi}  (for example profiles: {epson-v850: 3200})`;
  else text = `Profiles from ${p.path}`;
  const warn = `The saved profile "${current}" is not in ${p.path}: runs fail until it is replaced or added to the file (other settings still save). To keep the same scale without exporting again, choose "none" and type the profile's ppi in the same save ("none" alone infers the scale, which may export again); or choose another profile, or add this one to the file.`;
  if (missing) text = p.items.length && !p.error ? warn : warn + " " + text;      // the path is in the warning already
  hint.textContent = text;
  hint.classList.toggle("warn", missing || !!p.error);
}

async function saveSettings() {
  const body = {};                 // only what the user changed: the rest (a derivative made on the command line) stays as it is
  for (const f of SETTING_FIELDS) {
    const e = $("s-" + f);
    const value = e.type === "checkbox" ? e.checked : (e.type === "number" ? (e.value === "" ? null : Number(e.value)) : e.value);
    const old = S.loadedSettings ? S.loadedSettings[f] : undefined;
    if (value !== (old === undefined ? null : old) && !(value === "" && (old === null || old === undefined))) body[f] = value;
  }
  const meta = {};                 // metadata: the fields whose text was edited since it was shown ("" or [] clears one)
  const typed = metaValues();      // a field left as shown is never sent, whatever it holds (a keyword with a comma, \r\n)
  for (const f of META_FIELDS) {
    if ($("m-" + f).value !== S.shownMeta[f]) meta[f] = typed[f];
  }
  if (Object.keys(meta).length) body.metadata = meta;
  let r;
  try { r = await api("/api/project/settings", body); } catch (e) {
    if (e.fields) {                // a metadata field failed its check: say so next to the field, nothing was saved
      showMetaErrors(e.fields);
      showMsg("Settings not saved: " + e.message);
    } else showMsg(e.message);
    return;
  }
  showMetaErrors({});
  showMsg("Settings saved. They apply to the next run.", true);
  await loadSettings();
  tick(true);
}

// ---- metadata group: values as typed, messages next to the fields, preview of the first scan
function metaValues() {
  const v = {};
  for (const f of META_FIELDS) {
    const t = $("m-" + f).value;
    v[f] = f === "keywords" ? t.split(",").map((w) => w.trim()).filter(Boolean) : t.trim();
  }
  return v;
}

function showMetaErrors(fields) {
  for (const f of META_FIELDS) {
    $("e-" + f).textContent = fields[f] || "";
    $("m-" + f).classList.toggle("bad", !!fields[f]);
  }
  const rest = Object.keys(fields).filter((k) => !META_FIELDS.includes(k)).map((k) => fields[k]);
  $("m-all-err").textContent = rest.join(" ");
}

let metaTimer = null, metaSeq = 0;
function refreshMetaPreview(now) {
  clearTimeout(metaTimer);
  metaTimer = setTimeout(async () => {
    const seq = ++metaSeq;
    let r;
    try { r = await api("/api/project/metadata/preview", { metadata: metaValues() }); } catch (e) { return; }
    if (seq !== metaSeq) return;          // a newer request is on its way
    showMetaErrors(r.errors || {});
    const box = $("m-preview");
    box.textContent = "";
    if (!r.scan) { box.textContent = "No scan in this project yet: the preview needs one."; return; }
    const line = (label, text) => { const d = el("div"); d.append(el("b", null, label + " "), text); box.append(d); };
    const any = r.title || r.description;
    if (!any) { box.textContent = `Preview for ${r.scan}: no title or description set.`; return; }
    box.append(el("div", "dim small", `Preview for ${r.scan}` + (r.format_known ? "" : " (not detected yet: {format} shows ?)")));
    if (r.title) line("Title:", r.title);
    if (r.description) line("Description:", r.description);
  }, now ? 0 : 250);
}
for (const f of META_FIELDS) $("m-" + f).addEventListener("input", () => refreshMetaPreview(false));

function toggleSettings(force) {
  const box = $("settings");
  box.hidden = force === undefined ? !box.hidden : !force;
  if (!box.hidden) loadSettings();
}
$("s-save").onclick = saveSettings;
$("s-close").onclick = () => toggleSettings(false);

async function enterProject(p) {
  S.project = p;
  S.rev = null; S.keys = []; S.cards = new Map(); S.scans = []; S.prog = null;
  S.filter = startFilter || "all";
  startFilter = null;
  if (!FILTERS.some((f) => f[0] === S.filter)) S.filter = "all";
  $("settings").hidden = true;
  setView("project");
  $("grid").textContent = "";
  $("title").textContent = p.name;
  $("where").textContent = `${p.input || "?"}  →  ${p.output}`;
  $("where").title = `${p.input}\n${p.output}`;
  document.title = `${p.name} – Polaroid`;
  renderFilters();
  tick(true);
}

async function leaveProject() {
  if (!$("fixview").hidden) closeFix();
  const r = await guarded(() => api("/api/project/close", {}));
  S.project = null;
  await loadHome();
  if (r && r.stopping) showMsg("The run is being cancelled. This project can be opened again in a moment.", true);
}

async function tick(now) {
  clearTimeout(S.timer);
  if (S.view !== "project") return;
  try {
    const p = await api("/api/project/progress");
    if (S.view !== "project") return;
    if (p.rev !== S.rev || now) {
      const data = await api("/api/project/scans");
      if (S.view !== "project") return;
      renderScans(data);
      if (!$("settings").hidden) refreshMetaPreview(false);      // a run may have detected the first scan: {format} is known now
    }
    renderProgress(p);
    showMsg("");
  } catch (e) { showMsg(e.message); }
  const busy = S.prog && ((S.prog.run && (S.prog.run.state === "running" || S.prog.run.state === "cancelling")) || S.prog.previews_pending);
  S.timer = setTimeout(tick, busy ? 1000 : 2500);
}

// ---------------------------------------------------------------- start
(async function init() {
  const cur = await guarded(() => api("/api/project"));
  if (cur && cur.project) enterProject(cur.project);
  else loadHome();
})();
