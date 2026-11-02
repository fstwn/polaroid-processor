"use strict";
const $ = (id) => document.getElementById(id);
// ?filter=needs_review (from `polaroid review`) applies to the first project only, then the address is clean
let startFilter = new URLSearchParams(location.search).get("filter");
if (startFilter) history.replaceState(null, "", location.pathname);
const SETTING_FIELDS = ["margin_mm", "photo_crop", "photo_bleed_mm", "flatten", "shadows", "high_on", "high_edge", "high_quality", "web_on", "web_edge",
  "web_quality", "ppi", "profile", "tiff_compression", "interpolation", "overlay"];
const S = { view: "home", project: null, rev: null, filter: "all", cards: new Map(), keys: [], scans: [], prog: null, timer: null };

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
  if (!r.ok) throw new Error((data && data.error) || `${r.status} ${r.statusText}`);
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
  return (scan.warnings || []).filter((w) => !w.startsWith("hint:") && !w.startsWith("scale_inferred") && !w.startsWith("flatten"));
}

function sig(scan) {
  return JSON.stringify([scan.status, scan.preview && scan.preview.url, scan.stale, scan.warnings, scan.error, scan.export, scan.quad_source]);
}

function buildCard(scan) {
  const card = el("div", "card " + scan.status);
  card._sig = sig(scan);
  const img = el("div", "img");
  if (scan.preview) {
    const im = el("img", null, null, { alt: scan.key, loading: "lazy" });
    im.src = scan.preview.url;
    img.append(im);
    if (scan.stale) img.append(el("span", "stale", "updating…"));
  } else {
    img.append(el("span", "none", scan.status === "failed" ? "no outline" : scan.status === "pending" ? "waiting for detection" : "preview pending"));
  }
  const meta = el("div", "meta");
  meta.append(el("div", "nm", scan.key, { title: scan.key }));
  const tags = el("div", "tags");
  tags.append(el("span", "badge " + scan.status, scan.status.replace("_", " ")));
  if (scan.quad_source === "override") tags.append(el("span", "x", "fixed"));
  if (scan.export === "done") tags.append(el("span", "x", "exported"));
  else if (scan.export === "tags") tags.append(el("span", "x", "exported, tags to update"));
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
  if (!scan.size_px) { showMsg(scan.status === "failed" ? "No outline was found for this scan." : "This scan has not been detected yet.", true); return; }
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
    const n = c.to_export, t = c.to_retag;
    const x = el("button", started ? "primary" : "", n ? `Export (${n})` : t ? `Update tags (${t})` : "Export");
    x.disabled = !(n || t) || locked;
    x.title = n ? "Export the detected scans that have no up to date export" + (t ? `; ${t} more only get new resolution tags` : "")
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
  for (const f of SETTING_FIELDS) {
    const e = $("s-" + f);
    if (e.type === "checkbox") e.checked = !!v[f]; else e.value = v[f] === null || v[f] === undefined ? "" : v[f];
    e.disabled = r.running || !!r.locked_by;
  }
  $("s-save").disabled = r.running || !!r.locked_by;
  $("s-note").textContent = r.running ? "A run is going: settings apply to the next one."
    : r.locked_by ? `Another process (${r.locked_by}) is working here.`
    : (v.other_derivatives ? `${v.other_derivatives} more derivative(s) from the command line are kept.` : "");
}

async function saveSettings() {
  const body = {};                 // only what the user changed: the rest (a derivative made on the command line) stays as it is
  for (const f of SETTING_FIELDS) {
    const e = $("s-" + f);
    const value = e.type === "checkbox" ? e.checked : (e.type === "number" ? (e.value === "" ? null : Number(e.value)) : e.value);
    const old = S.loadedSettings ? S.loadedSettings[f] : undefined;
    if (value !== (old === undefined ? null : old) && !(value === "" && (old === null || old === undefined))) body[f] = value;
  }
  const r = await guarded(() => api("/api/project/settings", body));
  if (!r) return;
  showMsg("Settings saved. They apply to the next run.", true);
  await loadSettings();
  tick(true);
}

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
