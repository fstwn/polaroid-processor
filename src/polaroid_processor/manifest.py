"""Per-output-folder manifest: detection, manual fixes with provenance, status, outputs.

The manifest is a single JSON file, written atomically. Writers (the batch, a fix, the app) merge
with what is on disk before they write, so a fix made during a run is not lost.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

import numpy as np

from . import __version__
from .lock import file_mutex

MANIFEST_NAME = "polaroid-manifest.json"
SCHEMA_VERSION = 1


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def scan_key(path: str) -> str:
    """Manifest key of a scan: its file name."""
    return Path(path).name


def _dumps(v: Any) -> str:
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


def _merge_entry(mine: dict, base: dict, theirs: dict) -> dict:
    """Entry changed by both sides: our fields where we changed them, theirs elsewhere."""
    out = {}
    for f in dict.fromkeys([*theirs, *mine, *base]):
        m, b, t = mine.get(f), base.get(f), theirs.get(f)
        if f == "history" and isinstance(m, list) and isinstance(b, list) and isinstance(t, list):
            n = len(b)
            out[f] = t + m[n:] if _dumps(m[:n]) == _dumps(b) and _dumps(t[:n]) == _dumps(b) else m
        elif (f in mine) != (f in base) or (f in mine and _dumps(m) != _dumps(b)):   # we changed it
            if f in mine:
                out[f] = m
        elif f in theirs:
            out[f] = t
    return out


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
        self._fixed: Set[str] = set()
        self._base_top: Dict[str, str] = {}
        self._base_scans: Dict[str, str] = {}
        self._rebase()

    # ---- persistence -------------------------------------------------
    def _rebase(self) -> None:
        """Remember what the file held, as far as this instance knows: the base of the next merge."""
        if self.persist:
            self._base_top = {k: _dumps(v) for k, v in self.data.items() if k != "scans"}
            self._base_scans = {k: _dumps(e) for k, e in self.scans.items()}

    def _read_disk(self) -> Optional[Dict[str, Any]]:
        for attempt in range(40):  # Windows refuses to open the file while a writer swaps it
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data if data.get("schema") == SCHEMA_VERSION else None
            except FileNotFoundError:
                return None
            except (PermissionError, ValueError):
                if attempt == 39:
                    raise
                time.sleep(0.05)
        return None

    def _merge_disk(self, disk: Dict[str, Any]) -> List[str]:
        """Take over what another writer changed since this instance last read or saved.

        Per scan: if only the other writer changed it, theirs; if only this instance did, ours; if both
        did, field by field (a field this instance did not touch comes from the file; ``history`` keeps
        the entries of both). Scans and top-level fields are never dropped.

        Returns the scans whose ``override`` or ``history`` came from the other writer: a fix this
        instance has not judged yet.
        """
        scans = self.scans
        fixed: List[str] = []
        for key, theirs in (disk.get("scans") or {}).items():
            t_str = _dumps(theirs)
            mine = scans.get(key)
            base = self._base_scans.get(key)
            before = _dumps([mine.get("override"), mine.get("history")]) if mine is not None else None
            if mine is None:
                scans[key] = theirs
            elif base is not None:
                if _dumps(mine) == base:
                    mine.clear()
                    mine.update(theirs)
                elif t_str != base:
                    merged = _merge_entry(mine, json.loads(base), theirs)
                    mine.clear()
                    mine.update(merged)
            self._base_scans[key] = t_str     # the file's state, which this instance has now taken in
            now_ = scans[key]
            if before != _dumps([now_.get("override"), now_.get("history")]) and (now_.get("override") or now_.get("history")):
                fixed.append(key)
                self._fixed.add(key)
        for k, v in disk.items():
            if k != "scans":
                if k not in self.data or _dumps(self.data[k]) == self._base_top.get(k):
                    self.data[k] = v
                self._base_top[k] = _dumps(v)
        return fixed

    @property
    def fixed(self) -> Set[str]:
        """Scans with a fix from another writer that the caller has not taken yet (see :meth:`take_fixed`)."""
        return set(self._fixed)

    def take_fixed(self) -> List[str]:
        """The scans whose fix came from another writer in any merge since the last call; clears the list.
        A caller that judges scans (the batch) re-judges these: their status may predate the fix."""
        taken, self._fixed = sorted(self._fixed), set()
        return taken

    def refresh(self) -> List[str]:
        """Merge in what other writers saved, without writing. Returns the scans with a fix from the
        other writer (see :meth:`_merge_disk`)."""
        if not self.persist or not self.path.exists():
            return []
        disk = self._read_disk()
        return self._merge_disk(disk) if disk is not None else []

    def save(self) -> List[str]:
        """Write the manifest, merged with what another writer (the CLI, an agent, the app) saved since
        this instance last read or saved it, so that two writers do not lose each other's changes.
        Returns the scans with a fix from the other writer; the caller may judge them again and save."""
        if not self.persist:
            return []
        fixed: List[str] = []
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with file_mutex(self.path.with_name(self.path.name + ".lock")):
            disk = self._read_disk() if self.path.exists() else None
            if disk is not None:
                fixed = self._merge_disk(disk)
            self.data["tool"] = f"polaroid-processor {__version__}"
            self.data["updated_at"] = now()
            tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{threading.get_ident()}.part")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=1, ensure_ascii=False)
            for attempt in range(40):  # Windows refuses the swap while a reader (the app) has the file open
                try:
                    os.replace(tmp, self.path)
                    break
                except PermissionError:
                    if attempt == 39:
                        tmp.unlink(missing_ok=True)
                        raise
                    time.sleep(0.05)
        self._rebase()
        return fixed

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

    def set_window_fix(self, key: str, window: Optional[Dict[str, Any]], author: str, note: str = "",
                       action: str = "window") -> None:
        """Store the user's window choice (a hand-set window, or the no-photo-crop choice; ``None`` takes it
        away) in ``window_fix`` and log it like an outline fix: author, time, note and the previous choice.
        Only fixes write ``window_fix`` (the batch owns ``window``, the automatic window)."""
        e = self.entry(key)
        fix = dict(e.get("window_fix") or {})
        e.setdefault("history", []).append({
            "action": action, "author": author, "at": now(), "note": note,
            "window_before": fix.get("window"), "window_after": window,
        })
        fix["window"] = window
        e["window_fix"] = fix if fix.get("window") or fix.get("accepted") else None

    def set_window_accept(self, key: str, outline: Optional[str], author: str, note: str = "") -> None:
        """Mark the scan as accepted for the outline ``outline`` (an outline id; ``None`` clears the mark): a
        window not found is then no review item while the outline stays. The accept is an entry of the
        outline's own history (``accept`` / ``fix``), so nothing is logged here."""
        e = self.entry(key)
        fix = dict(e.get("window_fix") or {})
        fix["accepted"] = None if outline is None else {"outline": outline, "author": author, "at": now(), "note": note}
        e["window_fix"] = fix if fix.get("window") or fix.get("accepted") else None

    def set_item_fix(self, key: str, item: Optional[Dict[str, Any]], author: str, note: str = "",
                     action: str = "item") -> None:
        """Store the user's item choice (``item_fix``, see ``item.py``; ``None`` takes it away) and log it like an
        outline fix: author, time, note and the previous choice. Only fixes write ``item_fix`` (the batch drops it only
        when the scan's content changed)."""
        e = self.entry(key)
        e.setdefault("history", []).append({
            "action": action, "author": author, "at": now(), "note": note,
            "item_before": e.get("item_fix"), "item_after": item,
        })
        e["item_fix"] = item

    def set_agent_crop(self, key: str, mark: Optional[Dict[str, Any]], author: str, note: str = "",
                       action: str = "agent-crop") -> None:
        """Set (or, with ``None``, clear) the mark of a crop an agent placed on a scan without a detected outline
        (``entry["agent_crop"]``, a field of its own) and log it: every change of the mark is a history entry, so that
        another writer's merge sees it (``_merge_disk`` judges the scans whose history changed). Only fixes write it
        (the batch drops it with the crop when the scan file's content changed)."""
        e = self.entry(key)
        e.setdefault("history", []).append({
            "action": action, "author": author, "at": now(), "note": note,
            "agent_crop_before": e.get("agent_crop"), "agent_crop_after": mark,
        })
        e["agent_crop"] = mark

    def history(self, key: str) -> List[Dict[str, Any]]:
        return list(self.entry(key).get("history", []))
