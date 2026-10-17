"""Collect reviewed corners from output manifests into tests/data/labels.json.

A scan counts as reviewed when its manifest entry has an override (from
``polaroid fix --accept`` or a corner fix). Usage::

    python tools/export_labels.py OUTPUT_DIR [OUTPUT_DIR ...] [--input-root input] [--out tests/data/labels.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from polaroid_processor.manifest import Manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outputs", nargs="+")
    ap.add_argument("--input-root", default="input")
    ap.add_argument("--out", default="tests/data/labels.json")
    a = ap.parse_args()
    root = Path(a.input_root).resolve()
    out = Path(a.out)
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"scans": {}}
    data.setdefault("scans", {})
    data["note"] = ("Reviewed print corners (tl, tr, br, bl) in full-resolution scan pixels, continuous coordinates. "
                    "status 'proposed' = set by an agent, awaiting human confirmation.")
    added = 0
    for d in a.outputs:
        m = Manifest(d)
        for key, e in m.items():
            ov = e.get("override")
            if not ov:
                continue
            rel = Path(e["input"]).resolve().relative_to(root).as_posix()
            hist = e.get("history", [])
            data["scans"][rel] = {
                "quad": ov["quad"],
                "author": ov["author"],
                "status": "confirmed" if ov["author"] not in ("agent", "claude") else "proposed",
                "note": ov.get("note", ""),
                "reviewed_at": ov["at"],
                "auto_quad": (e.get("detection") or {}).get("quad"),
                "fixes": sum(1 for h in hist if h["action"] == "fix"),
            }
            added += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"{added} label(s) written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
