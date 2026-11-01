"use strict";
const $ = (id) => document.getElementById(id);
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
  return JSON.stringify([scan.status, scan.preview && scan.preview.url, scan.stale, scan.warnings, scan.error, scan.exported, scan.quad_source]);
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
  if (scan.exported) tags.append(el("span", "x", "exported"));
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

async function openScan(scan) {
  if (!scan.size_px) { showMsg(scan.status === "failed" ? "No outline was found for this scan." : "This scan has not been detected yet.", true); return; }
  const w = window.open("", "_blank");        // opened now, so the popup blocker lets it through
  const r = await guarded(() => api("/api/project/review", { key: scan.key }));
  if (r && w) w.location.href = r.url; else if (w) w.close();
}

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
  if (run) {
    const total = run.total;
    if (run.state === "running" || run.state === "cancelling") {
      if (total === null) text = "Looking for scans to detect…";
      else if (total === 0) { text = "Nothing to detect"; frac = 1; }
      else {
        frac = run.done / total;
        text = `${run.done} / ${total}` + (run.eta !== null ? ` · ${fmtTime(run.eta)} left` : "") + (run.state === "cancelling" ? " · stopping after the scans in flight…" : "");
      }
      if (run.phase) text = (total ? `${run.done} / ${total} · ` : "") + `${run.phase}…`;
    } else if (run.state === "finished") {
      cls = "done"; frac = 1;
      text = total ? `Detected ${plural(run.done, "scan")} in ${fmtTime(run.elapsed)}` + (run.errors ? ` · ${run.errors} failed` : "") : "Nothing new to detect";
    } else if (run.state === "cancelled") {
      frac = total ? run.done / total : 0;
      text = `Stopped at ${run.done} / ${total}. Detect again to continue.`;
    } else if (run.state === "error") {
      cls = "error"; frac = 1; text = run.error;
    }
  }
  if (p.previews_pending && !(run && (run.state === "running" || run.state === "cancelling"))) {
    text = (text ? text + " · " : "") + `making previews, ${p.previews_pending} left`;
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
  if (running) {
    const b = el("button", "warn", "Cancel");
    b.disabled = run.state === "cancelling";
    b.onclick = () => guarded(() => api("/api/project/cancel", {}));
    actions.append(b);
  } else {
    const todo = c.pending > 0;
    const b = el("button", "primary", !started ? "Detect" : todo ? "Resume detect" : "Detect again");
    b.disabled = !p.can_run;
    b.title = p.can_run ? "Find the print in every scan and store the outlines; nothing is exported" : "The input folder of this project is not known or not found";
    b.onclick = async () => { await guarded(() => api("/api/project/detect", {})); tick(true); };
    actions.append(b);
  }
  const back = el("button", null, "← Projects");
  back.onclick = leaveProject;
  actions.append(back);
}

async function enterProject(p) {
  S.project = p;
  S.rev = null; S.keys = []; S.cards = new Map(); S.scans = []; S.prog = null; S.filter = "all";
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
  const r = await guarded(() => api("/api/project/close", {}));
  S.project = null;
  await loadHome();
  if (r && r.stopping) showMsg("The run is stopping after the scans in flight. This project can be opened again when it has stopped.", true);
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
