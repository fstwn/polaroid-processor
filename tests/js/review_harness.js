// Test harness for the fix page's script (src/polaroid_processor/review/static/index.html): loads the real script in a vm
// with a small DOM stub, answers its fetches from canned routes, runs steps against it and prints the result as one JSON line.
// Usage: node review_harness.js <index.html> '<{"routes": {...}, "steps": ["js code", ...], "hash": "key"}>'
//   routes: "METHOD /path?query" -> a response {status?, body} (used every time) or a list of responses (used in turn, the
//           last one stays); a route may also be matched by prefix: the longest key that the request starts with wins.
//   steps:  async function bodies run in the page's global scope in order (`S`, `commit`, `isDirty`, ... are visible); each
//           returns a value that is collected. The output is {results: [...], calls: [{method, path, body}...], posted: [...]}.
// The stub is only as big as the page needs: elements with value, children, a class list and listeners, a canvas whose 2d
// context accepts every call, Image whose onload fires on the next tick, matchMedia (wide), and a pointer/keyboard dispatch.
"use strict";
const fs = require("fs");
const vm = require("vm");

const html = fs.readFileSync(process.argv[2], "utf8").replace(/\r\n/g, "\n");
const spec = JSON.parse(process.argv[3] || "{}");
const script = /<script>\n([\s\S]*?)<\/script>/.exec(html)[1];

const els = new Map();
const docListeners = {};
const noopCtx = new Proxy({}, { get: (t, k) => (k in t ? t[k] : (k === "measureText" ? () => ({ width: 10 }) : () => undefined)), set: (t, k, v) => { t[k] = v; return true; } });
function fakeEl(id, tag) {
  const cls = new Set(), listeners = {};
  const e = {
    id, tag: tag || "", hidden: false, disabled: false, className: "", style: {}, dataset: {}, children: [], title: "", checked: false,
    clientWidth: 1000, clientHeight: 700, width: 800, height: 600, innerHTML: "", validity: { badInput: false },
    classList: { toggle(c, on) { if (on === undefined ? !cls.has(c) : on) cls.add(c); else cls.delete(c); }, add(c) { cls.add(c); }, remove(c) { cls.delete(c); }, contains(c) { return cls.has(c); } },
    addEventListener(type, fn) { (listeners[type] = listeners[type] || []).push(fn); },
    fire(type, ev) { for (const fn of listeners[type] || []) fn({ preventDefault() {}, ...ev }); if (e["on" + type]) e["on" + type]({ preventDefault() {}, ...ev }); },
    append(...cs) { for (const c of cs) e.children.push(c); }, appendChild(c) { e.children.push(c); }, setAttribute() {}, remove() {},
    focus() { document.activeElement = e; }, blur() { if (document.activeElement === e) document.activeElement = null; },
    getContext() { return noopCtx; }, getBoundingClientRect() { return { left: 0, top: 0, width: e.clientWidth, height: e.clientHeight, right: e.clientWidth, bottom: e.clientHeight }; },
    setPointerCapture() {}, releasePointerCapture() {}, hasPointerCapture() { return false; }, querySelector() { return null; }, querySelectorAll() { return []; },
    click() { if (e.onclick) return e.onclick(); }, classes: cls,
  };
  let text = "";
  Object.defineProperty(e, "textContent", { get() { return text; }, set(x) { text = String(x); e.children = []; } });
  e.value = "";
  return e;
}
const document = {
  getElementById(id) { if (!els.has(id)) els.set(id, fakeEl(id)); return els.get(id); },
  createElement(t) { return fakeEl("_" + t, t); }, querySelector(sel) { return document.getElementById(sel); }, querySelectorAll() { return []; },
  addEventListener(type, fn) { (docListeners[type] = docListeners[type] || []).push(fn); }, removeEventListener(type, fn) { docListeners[type] = (docListeners[type] || []).filter((f) => f !== fn); },
  fire(type, ev) { for (const fn of docListeners[type] || []) fn({ preventDefault() {}, ...ev }); }, body: fakeEl("body"), title: "", activeElement: null,
};

const calls = [], posted = [];
const routes = spec.routes || {};
const used = {};
async function fetch(path, opt) {
  const method = opt && opt.method ? opt.method : "GET";
  const body = opt && opt.body ? JSON.parse(opt.body) : undefined;
  calls.push({ method, path, body });
  const exact = `${method} ${path}`;
  let key = routes[exact] !== undefined ? exact : Object.keys(routes).filter((k) => exact.startsWith(k)).sort((a, b) => b.length - a.length)[0];
  if (key === undefined) return { ok: false, status: 404, statusText: "no route", json: async () => ({ error: "no route " + exact }) };
  let r = routes[key];
  if (Array.isArray(r)) { used[key] = (used[key] || 0) + 1; r = r[Math.min(used[key], r.length) - 1]; }
  const status = r.status || 200;
  return { ok: status < 400, status, statusText: "", json: async () => r.body };
}
class FakeImage {
  constructor() { this.complete = false; this.naturalWidth = 0; }
  set src(v) { this._src = v; setTimeout(() => { this.complete = true; this.naturalWidth = 10; if (this.onload) this.onload(); }, 0); }
  get src() { return this._src; }
}
const win = {
  addEventListener() {}, devicePixelRatio: 1, innerWidth: 1280, innerHeight: 800, matchMedia: () => ({ matches: !!spec.narrow }),
  POLAROID_API: spec.api || "/api", POLAROID_EMBED: !!spec.embed, parent: { postMessage(m) { posted.push(m); } },
};
const ctx = vm.createContext({
  document, window: win, location: { hash: spec.hash ? "#" + spec.hash : "", origin: "http://x" }, fetch, Image: FakeImage, setTimeout, clearTimeout,
  console, Math, JSON, Object, Array, Number, String, Promise, Set, Map, Error, encodeURIComponent, decodeURIComponent, isFinite, parseFloat, Date,
});
vm.runInContext(script, ctx, { filename: "review.js" });

(async () => {
  await new Promise((r) => setTimeout(r, 30));      // init(): the listing, the first scan's tiles and start rectangle
  const results = [];
  for (const code of spec.steps || []) {
    const value = await vm.runInContext(`(async () => { ${code} })()`, ctx);
    results.push(value === undefined ? null : value);
    await new Promise((r) => setTimeout(r, 15));
  }
  console.log(JSON.stringify({ results, calls, posted }));
  process.exit(0);
})().catch((e) => { console.error("harness error:", e); process.exit(1); });
