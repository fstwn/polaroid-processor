// Test harness for the app's page code (src/polaroid_processor/app/static/app.js): loads the real file in a vm
// with a small DOM stub, plays a scenario against canned server answers and prints the result as one JSON line.
// Usage: node page_harness.js <path to app.js> <scenario>
// The stub is only as big as the scenarios need: elements with value, children and a class list, a textarea that
// turns \r\n into \n as a browser does, a <select> (#s-profile) that keeps a value only when an option has it, and a
// fetch that records the settings POST bodies and answers from the scenario's `settings` / `profiles`.
"use strict";
const fs = require("fs");
const vm = require("vm");

const els = new Map();
const classes = new Map();
function fakeEl(id, tag) {
  const isText = id === "m-description";
  const isSelect = id === "s-profile";
  const cls = new Set();
  classes.set(id, cls);
  const e = {
    id, tag: isSelect ? "select" : (tag || ""),
    type: id === "s-overlay" || /_on$/.test(id) ? "checkbox" : (/edge|quality|margin|bleed|ppi/.test(id) ? "number" : "text"),
    checked: false, disabled: false, hidden: false, className: "", style: {}, title: "", children: [], attrs: {},
    classList: { toggle(c, on) { if (on === undefined ? !cls.has(c) : on) cls.add(c); else cls.delete(c); }, add(c) { cls.add(c); },
                 remove(c) { cls.delete(c); }, contains(c) { return cls.has(c); } },
    addEventListener() {}, appendChild(c) { e.children.push(c); }, setAttribute(k, v) { e.attrs[k] = v; },
    querySelector() { return null; }, querySelectorAll() { return []; }, focus() {}, remove() {},
    append(...cs) { for (const c of cs) e.children.push(c); },
  };
  let text = "", v = "";
  Object.defineProperty(e, "textContent", { get() { return text; }, set(x) { text = String(x); e.children = []; } });
  Object.defineProperty(e, "value", {
    get() { return v; },
    set(x) {
      x = String(x);
      if (isSelect) v = e.children.some((c) => c.value === x) ? x : "";      // a select shows no value that is not an option
      else v = isText ? x.replace(/\r\n?/g, "\n") : x;      // a textarea normalizes line breaks
    },
  });
  return e;
}
const document = {
  getElementById(id) {
    if (!els.has(id)) {
      const e = fakeEl(id);
      if (id === "fixframe") e.contentWindow = { location: { replace(u) { opened.push(u); } } };      // the fix page's frame
      els.set(id, e);
    }
    return els.get(id);
  },
  createElement(t) { return fakeEl("_" + t, t); }, querySelector() { return null; }, querySelectorAll() { return []; },
  addEventListener() {}, body: fakeEl("body"), title: "",
};

const BASE = {
  margin_mm: 0.1, photo_crop: "off", photo_bleed_mm: 0.3, flatten: "auto", shadows: "auto", high_on: true, high_edge: 6000,
  high_quality: 95, web_on: true, web_edge: 2048, web_quality: 88, ppi: null, profile: null, tiff_compression: "zlib",
  interpolation: "linear", overlay: false, other_derivatives: 0,
  metadata: { title: "", author: "", copyright: "", description: "", keywords: [], date_taken: "" },
};
const options = (sel) => sel.children.map((o) => ({ value: o.value, text: o.textContent }));

const SCENARIOS = {
  // saving only an unrelated setting must not send the metadata fields; editing one sends that one
  metadata_save: {
    settings: { ...BASE, metadata: { title: "T {n}", author: "A", copyright: "", description: "Zeile 1\r\nZeile 2",
                                     keywords: ["Hamburg, Germany", "Garten"], date_taken: "" } },
    async run(h) {
      const out = {};
      await h.call("loadSettings()");
      out.shown_keywords = h.el("m-keywords").value;
      h.el("s-margin_mm").value = "0.2";                                   // only the margin is edited
      await h.call("saveSettings()");
      out.margin_only = h.lastPost();
      await h.call("loadSettings()");
      h.el("m-title").value = "New {n}";                                    // one metadata field edited
      await h.call("saveSettings()");
      out.title_edit = h.lastPost();
      await h.call("loadSettings()");
      h.el("m-keywords").value = "Hamburg, Germany, Garten, Neu";          // the keyword field edited
      await h.call("saveSettings()");
      out.keyword_edit = h.lastPost();
      return out;
    },
  },
  // the scanner profile dropdown; the scenario's profiles come in through argv[4] as {profile, profiles}
  profile_select: {
    async run(h, input) {
      this.settings = { ...BASE, profile: input.profile };
      this.profiles = input.profiles;
      const out = {};
      const sel = h.el("s-profile"), hint = h.el("s-profile-hint");
      await h.call("loadSettings()");
      out.options = options(sel);
      out.value = sel.value;
      out.hint = hint.textContent;
      out.hint_warns = h.classes("s-profile-hint").has("warn");
      h.el("s-margin_mm").value = "0.2";                                   // an unrelated save keeps the profile as it is
      await h.call("saveSettings()");
      out.margin_only = h.lastPost();
      for (const [name, v] of Object.entries(input.picks || {})) {          // then pick another one and save
        await h.call("loadSettings()");
        sel.value = v;
        await h.call("saveSettings()");
        out[name] = h.lastPost();
      }
      return out;
    },
  },
};

// grid cards of hand crops, and which scans open the fix page
SCENARIOS.cards = {
  settings: BASE,
  async run(h, input) {
    const text = (e) => (e.children.length ? e.children.map(text).join("") : e.textContent);
    const find = (e, cls) => e.children.find((c) => c.className === cls);
    const out = { cards: {}, opened: {} };
    for (const scan of input.scans) {
      const card = await h.call(`buildCard(${JSON.stringify(scan)})`);
      const meta = find(card, "meta"), tags = find(meta, "tags"), warns = find(meta, "warns");
      out.cards[scan.key] = { cls: card.className, tags: tags.children.map(text), warns: warns ? warns.children.map(text) : [],
                              img: text(find(card, "img")) };
      opened.length = 0;
      h.el("msg").textContent = "";
      h.el("fixview").hidden = true;
      await h.call(`openScan(${JSON.stringify(scan)})`);
      out.opened[scan.key] = { url: opened[0] || null, shown: !h.el("fixview").hidden, msg: h.el("msg").textContent };
    }
    return out;
  },
};

// the grid's filter chips; `search` is the address the page is loaded with (`polaroid review` adds ?filter=review)
SCENARIOS.filters = {
  settings: BASE,
  search: "?filter=review",
  async run(h, input) {
    const counts = { ok: 0, needs_review: 0, failed: 0, detected: 0, pending: 0, exported: 0, total: input.scans.length,
                     to_export: 0, to_retag: 0, to_headers: 0 };
    for (const s of input.scans) counts[s.status] += 1;
    const label = (chip) => chip.textContent + chip.children.map((c) => (typeof c === "string" ? c : c.textContent)).join("");
    const view = () => ({
      chips: h.el("filters").children.map(label), on: h.el("filters").children.filter((c) => / on$/.test(c.className)).map(label),
      shown: input.scans.map((s) => s.key).filter((k) => !h.call(`S.cards.get(${JSON.stringify(k)}).hidden`)),
      empty: h.el("empty").hidden ? null : h.el("empty").textContent,
    });
    h.call("tick = async () => {}");      // no poller: the scenario hands the page its answers itself
    await h.call('enterProject({ name: "p", input: "in", output: "out" })');
    h.call(`renderScans(${JSON.stringify({ rev: "1", scans: input.scans })})`);
    h.call(`renderProgress(${JSON.stringify({ rev: "1", counts, run: null, previews_pending: 0, lock: null, can_run: true, scale_error: null })})`);
    const out = { start: view() };
    for (const [name, i] of Object.entries(input.clicks)) {
      h.el("filters").children[i].onclick();
      out[name] = view();
    }
    return out;
  },
};

SCENARIOS.sig = {      // the signature that decides when a card is built again
  settings: BASE,
  async run(h, input) {
    const out = {};
    for (const [k, scan] of Object.entries(input)) out[k] = await h.call(`sig(${JSON.stringify(scan)})`);
    return out;
  },
};

SCENARIOS.scale_error = {
  settings: BASE,
  async run(h) {      // the progress bar says so when the server could not work out the scale
    const counts = { ok: 1, needs_review: 0, failed: 0, detected: 0, pending: 0, exported: 1, total: 1, to_export: 0, to_retag: 0, to_headers: 0 };
    const prog = (extra) => ({ rev: "1", counts, run: null, previews_pending: 0, lock: null, can_run: true, scale_error: null, ...extra });
    const out = {};
    await h.call("renderProgress(" + JSON.stringify(prog({})) + ")");
    out.plain = { text: h.el("bar-text").textContent, hidden: h.el("runbar").hidden };
    await h.call("renderProgress(" + JSON.stringify(prog({ scale_error: "scanner profile 'x' not found in /p.yaml" })) + ")");
    out.error = { text: h.el("bar-text").textContent, cls: h.el("runbar").className, hidden: h.el("runbar").hidden };
    return out;
  },
};

const sent = [], opened = [];
let scenario = null;
async function fetch(path, opt) {
  if (opt && opt.method === "POST" && path === "/api/project/settings") sent.push(JSON.parse(opt.body));
  const body = path === "/api/project/settings" ? { settings: scenario.settings, profiles: scenario.profiles, running: false, locked_by: null }
    : path === "/api/project/metadata/preview" ? { scan: null, errors: {} }
    : path === "/api/project" ? { project: null } : path === "/api/home" ? { projects: [], recent_inputs: [], current: null } : {};
  return { ok: true, status: 200, json: async () => body };
}
const search = (SCENARIOS[process.argv[3]] || {}).search || "";
const ctx = vm.createContext({
  document, window: { addEventListener() {}, location: { search, pathname: "/" } },
  location: { search, pathname: "/" }, history: { replaceState() {}, pushState() {} },
  URLSearchParams, fetch, setTimeout, clearTimeout, setInterval() { return 0; }, clearInterval() {}, console,
  requestAnimationFrame() {}, getComputedStyle() { return {}; }, navigator: {}, EventSource: function () {},
});
vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), ctx, { filename: "app.js" });

scenario = SCENARIOS[process.argv[3]];
if (!scenario) { console.error("unknown scenario " + process.argv[3]); process.exit(2); }
const helpers = {
  call: (code) => vm.runInContext(code, ctx),
  el: (id) => document.getElementById(id),
  classes: (id) => classes.get(id),
  lastPost: () => sent[sent.length - 1],
};
const input = process.argv[4] ? JSON.parse(process.argv[4]) : undefined;
scenario.run(helpers, input).then((out) => console.log(JSON.stringify(out)), (e) => { console.error("harness error:", e); process.exit(1); });
