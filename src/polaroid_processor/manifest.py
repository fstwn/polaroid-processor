"""Per-output-folder manifest: detection, manual fixes with provenance, status, outputs.

The manifest is a single JSON file. Only the main process writes it, atomically.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from . import __version__

MANIFEST_NAME = "polaroid-manifest.json"
SCHEMA_VERSION = 1


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def scan_key(path: str) -> str:
    """Manifest key of a scan: its file name."""
    return Path(path).name


class Manifest:
    def __init__(self, out_dir: Optional[str], persist: bool = True):
        """Load (or start) the manifest of ``out_dir``. With ``persist=False`` nothing is written."""
        self.persist = persist and out_dir is not None
        self.out_dir = Path(out_dir) if out_dir else Path.cwd()
        self.path = self.out_dir / MANIFEST_NAME
        self.data: Dict[str, Any] = {"schema": SCHEMA_VERSION, "tool": f"polaroid-processor {__version__}", "scans": {}}
        if out_dir and self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            if self.data.get("schema") != SCHEMA_VERSION:
                raise ValueError(f"{self.path}: unsupported manifest schema {self.data.get('schema')}")

    # ---- persistence -------------------------------------------------
    def save(self) -> None:
        if not self.persist:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.data["tool"] = f"polaroid-processor {__version__}"
        self.data["updated_at"] = now()
        tmp = self.path.with_name(self.path.name + ".part")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1, ensure_ascii=False)
        os.replace(tmp, self.path)

    # ---- access ------------------------------------------------------
    @property
    def scans(self) -> Dict[str, Dict[str, Any]]:
        return self.data.setdefault("scans", {})

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.scans.get(key)

    def entry(self, key: str) -> Dict[str, Any]:
        return self.scans.setdefault(key, {"history": []})

    def find(self, name_or_path: str) -> Optional[str]:
        """Resolve a scan given by file name, stem or path to its manifest key."""
        key = scan_key(name_or_path)
        if key in self.scans:
            return key
        stem = Path(name_or_path).stem
        hits = [k for k in self.scans if Path(k).stem == stem]
        return hits[0] if len(hits) == 1 else None

    def items(self) -> Iterator:
        return iter(self.scans.items())

    def rel(self, path: Optional[str]) -> Optional[str]:
        if path is None:
            return None
        try:
            return Path(path).resolve().relative_to(self.out_dir.resolve()).as_posix()
        except ValueError:
            return str(path)

    def abs(self, rel: Optional[str]) -> Optional[Path]:
        if rel is None:
            return None
        p = Path(rel)
        return p if p.is_absolute() else self.out_dir / p

    # ---- quads and fixes --------------------------------------------
    @staticmethod
    def effective_quad(entry: Dict[str, Any]) -> Optional[np.ndarray]:
        ov = entry.get("override")
        if ov and ov.get("quad"):
            return np.array(ov["quad"], dtype=np.float64)
        det = entry.get("detection")
        if det and det.get("quad"):
            return np.array(det["quad"], dtype=np.float64)
        return None

    def set_override(self, key: str, quad: Optional[np.ndarray], author: str, note: str = "", action: str = "fix") -> None:
        e = self.entry(key)
        before = self.effective_quad(e)
        rec = {
            "action": action,
            "author": author,
            "at": now(),
            "note": note,
            "quad_before": None if before is None else np.round(before, 2).tolist(),
            "quad_after": None if quad is None else np.round(quad, 2).tolist(),
        }
        e.setdefault("history", []).append(rec)
        if quad is None:
            e["override"] = None
        else:
            e["override"] = {"quad": np.round(quad, 2).tolist(), "author": author, "at": rec["at"], "note": note}

    def history(self, key: str) -> List[Dict[str, Any]]:
        return list(self.entry(key).get("history", []))
