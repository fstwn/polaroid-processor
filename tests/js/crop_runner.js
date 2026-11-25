// Runs the pure crop-geometry block of the fix page (src/polaroid_processor/review/static/index.html, between the
// "==== crop geometry" and "==== end of crop geometry" lines) on its own and answers calls as JSON.
// Usage: node crop_runner.js <index.html> '<[["rectToQuad", {...}], ["moveRect", r, 5, 5, 4000, 3000], ...]>'
// Each call is [function name, ...arguments]; the output is the list of results. A function that is not in the block is an error.
"use strict";
const fs = require("fs");
const vm = require("vm");

const html = fs.readFileSync(process.argv[2], "utf8");
const a = html.indexOf("// ==== crop geometry");
const b = html.indexOf("// ==== end of crop geometry ====");
if (a < 0 || b < 0) { console.error("geometry block not found"); process.exit(2); }
const ctx = vm.createContext({ Math, Object, Number, Array, JSON });
vm.runInContext(html.slice(a, b), ctx, { filename: "crop-geometry" });
const calls = JSON.parse(process.argv[3] === "-" ? fs.readFileSync(0, "utf8") : process.argv[3]);      // "-": the calls come on stdin (long lists)
const out = calls.map(([name, ...args]) => {
  const fn = vm.runInContext(`typeof ${name} === "function" ? ${name} : null`, ctx);
  if (!fn) throw new Error("no function " + name);
  return fn(...args);
});
console.log(JSON.stringify(out));
