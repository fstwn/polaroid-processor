"""One project of the app: an input folder and its output folder (with the manifest).

Reads the manifest for the grid and the counts, runs the detect pass in a background thread,
and keeps the crop previews up to date. No HTTP in here.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

from .. import api
from ..detection import quad_size
from ..imageio import load_rgb
from ..manifest import MANIFEST_NAME, Manifest
from ..pipeline import collect_inputs, manual_outline, settings_from_dict
from ..preview import LONG_EDGE, PREVIEW_DIR, preview_key, write_preview
from ..scale import scale_from_size

log = logging.getLogger("polaroid_processor.app")

STATUSES = ("ok", "needs_review", "failed", "detected", "pending")


def load_manifest(output: str) -> Optional[Manifest]:
    """The manifest of ``output``, or None when there is none. Retries while a writer swaps the file."""
    if not (Path(output) / MANIFEST_NAME).exists():
        return None
    for attempt in range(6):
        try:
            return Manifest(output)
        except (PermissionError, json.JSONDecodeError, FileNotFoundError):
            if attempt == 5:
                raise
            time.sleep(0.05)


def input_folder_of(m: Manifest) -> Optional[str]:
    """The folder the scans of a manifest came from (None when they come from several)."""
    parents = {str(Path(e["input"]).parent) for _, e in m.items() if e.get("input")}
    return parents.pop() if len(parents) == 1 else None


def list_inputs(input_dir: Optional[str]) -> List[str]:
    if not input_dir or not Path(input_dir).is_dir():
        return []
    return [p.name for p in collect_inputs([input_dir])]


def entry_status(e: Optional[dict]) -> str:
    if not e:
        return "pending"
    if e.get("status") in ("ok", "needs_review", "failed"):
        return e["status"]
    return "detected" if e.get("detection") else "pending"


def count_scans(m: Optional[Manifest], names: List[str]) -> Dict[str, int]:
    """Scans per status over the manifest and the input folder, plus ``exported`` and ``total``."""
    keys = set(names)
    if m:
        keys |= set(m.scans)
    counts = {s: 0 for s in STATUSES}
    exported = 0
    for k in keys:
        e = m.get(k) if m else None
        counts[entry_status(e)] += 1
        exported += bool(e and e.get("outputs"))
    return {**counts, "exported": exported, "total": len(keys)}


def _file_sig(path: Path) -> Optional[Tuple[int, int]]:
    try:
        st = path.stat()
        return st.st_mtime_ns, st.st_size
    except OSError:
        return None


_SUMMARY: Dict[str, tuple] = {}


def project_summary(input_dir: Optional[str], output: str) -> dict:
    """Counts of a project for the home page; cached by the manifest's and the input folder's stamps."""
    sig = (_file_sig(Path(output) / MANIFEST_NAME), _file_sig(Path(input_dir)) if input_dir else None)
    hit = _SUMMARY.get(output)
    if hit and hit[0] == sig:
        return hit[1]
    try:
        m = load_manifest(output)
    except (OSError, ValueError):
        m = None
    info = {"counts": count_scans(m, list_inputs(input_dir)), "has_manifest": m is not None,
            "output_exists": Path(output).is_dir(), "input_exists": bool(input_dir and Path(input_dir).is_dir())}
    _SUMMARY[output] = (sig, info)
    return info


# --------------------------------------------------------------------------- the run
class Run:
    """State of one detect pass (written by the run thread, read by the poller)."""

    def __init__(self):
        self.state = "running"   # running, cancelling, finished, cancelled, error
        self.total: Optional[int] = None
        self.done = 0
        self.errors = 0
        self.started = time.time()
        self.ended: Optional[float] = None
        self.error: Optional[str] = None
        self.current: Optional[str] = None
        self.phase: Optional[str] = None
        self.cancel = threading.Event()

    def record(self) -> dict:
        end = self.ended or time.time()
        elapsed = end - self.started
        eta = None
        if self.state in ("running", "cancelling") and self.done and self.total:
            eta = elapsed / self.done * max(0, self.total - self.done)
        return {"state": self.state, "total": self.total, "done": self.done, "errors": self.errors,
                "elapsed": round(elapsed, 1), "eta": None if eta is None else round(eta, 1), "error": self.error,
                "current": self.current, "phase": self.phase if self.state in ("running", "cancelling") else None}


class Project:
    def __init__(self, input_dir: Optional[str], output: str):
        self.input = str(input_dir) if input_dir else None
        self.output = str(output)
        self.name = Path(self.output).name
        self.run: Optional[Run] = None
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._keeper: Optional[threading.Thread] = None
        self._stop_keeper = threading.Event()
        self._closed = False
        self._bad: set = set()           # (key, preview key) that failed to render: not tried again
        self._sig = None
        self._snap: Optional[dict] = None

    # ---- manifest snapshot ---------------------------------------------
    def _settings(self, m: Optional[Manifest]) -> Tuple[float, str]:
        s = (m.data.get("settings") if m else None) or {}
        return float(s.get("margin_mm", 0.1)), s.get("flatten", "auto")

    @staticmethod
    def _quad_curves(e: dict):
        det = e.get("detection") or {}
        manual = manual_outline((e.get("override") or {}).get("quad"), det)
        return Manifest.effective_quad(e), det.get("curves"), manual

    def _record(self, key: str, e: Optional[dict], margin: float, flatten: str) -> dict:
        status = entry_status(e)
        quad = Manifest.effective_quad(e) if e else None
        rec = {"key": key, "status": status, "warnings": (e or {}).get("warnings", []),
               "error": (e or {}).get("error"), "exported": bool(e and e.get("outputs")),
               "quad_source": "override" if e and e.get("override") else ("auto" if quad is not None else None),
               "format": ((e or {}).get("format") or {}).get("name"), "preview": None, "stale": False}
        if quad is not None:
            w, h = quad_size(quad)
            rec["size_px"] = [round(w), round(h)]
            q, curves, manual = self._quad_curves(e)
            want = preview_key(q, curves, manual, margin, flatten)
            pv = e.get("preview")
            if pv and (self.out_dir / pv["file"]).exists():
                rec["preview"] = {"url": f"/api/project/preview/{quote(key)}.jpg?rev={pv['key']}",
                                  "size": pv.get("size")}
                rec["stale"] = pv.get("key") != want
            else:
                rec["stale"] = True
            rec["_want"] = want
        return rec

    @property
    def out_dir(self) -> Path:
        return Path(self.output)

    def snapshot(self) -> dict:
        """Scans (input folder and manifest merged), counts and stale-preview count; cached until the
        manifest or the input folder changes."""
        sig = (_file_sig(self.out_dir / MANIFEST_NAME), _file_sig(Path(self.input)) if self.input else None)
        with self._lock:
            if self._snap is not None and sig == self._sig:
                return self._snap
            m = load_manifest(self.output)
            names = list_inputs(self.input)
            margin, flatten = self._settings(m)
            keys = sorted(set(names) | set(m.scans if m else {}))
            scans = [self._record(k, m.get(k) if m else None, margin, flatten) for k in keys]
            self._sig = sig
            self._snap = {"rev": f"{sig[0][0] if sig[0] else 0}-{sig[1][0] if sig[1] else 0}", "scans": scans,
                          "counts": count_scans(m, names),
                          "stale": sum(1 for r in scans if r["stale"] and "_want" in r and (r["key"], r["_want"]) not in self._bad)}
            return self._snap

    def scans(self) -> dict:
        snap = self.snapshot()
        return {"rev": snap["rev"], "counts": snap["counts"],
                "scans": [{k: v for k, v in r.items() if k != "_want"} for r in snap["scans"]]}

    def progress(self) -> dict:
        with self._lock:        # the run first: a finished run must come with a manifest that is at least as new
            run = self.run.record() if self.run else None
        snap = self.snapshot()
        if run is None or run["state"] not in ("running", "cancelling"):
            self.kick_previews()
        return {"rev": snap["rev"], "counts": snap["counts"], "run": run, "previews_pending": snap["stale"],
                "can_run": bool(self.input and Path(self.input).is_dir())}

    def record(self) -> dict:
        return {"input": self.input, "output": self.output, "name": self.name}

    # ---- the detect run -------------------------------------------------
    @property
    def running(self) -> bool:
        return bool(self.run and self.run.state in ("running", "cancelling"))

    def start_detect(self) -> dict:
        if not self.input or not Path(self.input).is_dir():
            raise api.UsageError("the input folder of this project is not known or not found")
        with self._lock:
            if self._closed:
                raise api.UsageError("this project is closed")
            if self.running:
                raise api.UsageError("a run is already going")
            self.run = run = Run()       # first, so that the preview keeper does not start again
        self._pause_previews()
        with self._lock:
            self._thread = threading.Thread(target=self._detect, args=(run,), daemon=True, name="polaroid-detect")
            self._thread.start()
            return run.record()

    def cancel(self) -> dict:
        with self._lock:
            if not self.running:
                raise api.UsageError("no run is going")
            if self.run.state == "running":
                self.run.state = "cancelling"
            self.run.cancel.set()
            return self.run.record()

    def _detect(self, run: Run) -> None:
        def progress(key: str, what: str) -> None:
            run.done += 1
            run.current = key
            if what != "detected":
                run.errors += 1

        def on_jobs(n: int) -> None:
            run.total = n

        def on_phase(text: str) -> None:
            run.phase = text

        state, error = "finished", None
        try:
            m = load_manifest(self.output)
            settings = settings_from_dict((m.data.get("settings") if m else None) or {})
            settings.export, settings.previews = False, True
            summary = api.process([self.input], self.output, settings, progress=progress, cancel=run.cancel.is_set,
                                  on_jobs=on_jobs, on_phase=on_phase)
            if summary.get("cancelled") and run.done < (run.total or 0):
                state = "cancelled"
        except Exception as e:  # noqa: BLE001  (shown in the page)
            log.exception("detect run failed")
            state, error = "error", f"{type(e).__name__}: {e}"
        with self._lock:        # one step, so that a cancel arriving now cannot leave "cancelling" behind
            run.state, run.error, run.ended = state, error, time.time()
            self._sig = None
        self.kick_previews()

    # ---- previews ---------------------------------------------------------
    def _next_stale(self) -> Optional[Tuple[str, str]]:
        self._sig = None
        for r in self.snapshot()["scans"]:
            if r["stale"] and "_want" in r and (r["key"], r["_want"]) not in self._bad:
                return r["key"], r["_want"]
        return None

    def kick_previews(self) -> None:
        """Make the missing and stale previews, one at a time, when no run is going."""
        with self._lock:
            if (self._closed or self.running or (self._keeper and self._keeper.is_alive())
                    or self._snap and not self._snap["stale"]):
                return
            self._stop_keeper.clear()
            self._keeper = threading.Thread(target=self._keep, daemon=True, name="polaroid-previews")
            self._keeper.start()

    def _pause_previews(self) -> None:
        self._stop_keeper.set()
        if self._keeper and self._keeper.is_alive() and self._keeper is not threading.current_thread():
            self._keeper.join(timeout=60)

    def _keep(self) -> None:
        while not (self._stop_keeper.is_set() or self._closed or self.running):
            nxt = self._next_stale()
            if nxt is None:
                return
            try:
                self.make_preview(nxt[0])
            except Exception:  # noqa: BLE001
                log.exception("preview of %s failed", nxt[0])
                self._bad.add(nxt)

    def make_preview(self, key: str) -> None:
        """Render the preview of one scan at its current outline and record it in the manifest."""
        m = load_manifest(self.output)
        e = m.get(key) if m else None
        quad, curves, manual = self._quad_curves(e) if e else (None, None, False)
        if quad is None:
            raise api.UsageError(f"{key} has no outline yet")
        margin, flatten = self._settings(m)
        px = (e.get("scale") or {}).get("px_per_mm") or scale_from_size(quad_size(quad)).px_per_mm
        path = self.out_dir / PREVIEW_DIR / f"{Path(key).stem}.jpg"
        info = write_preview(load_rgb(e["input"]), quad, px, curves, manual,
                             {"margin_mm": margin, "flatten": flatten, "long_edge": LONG_EDGE}, path)
        m = load_manifest(self.output)          # fresh: keep what happened in the meantime
        m.entry(key)["preview"] = {**info, "file": m.rel(info["file"])}
        m.save()

    def preview_file(self, key: str) -> Path:
        m = load_manifest(self.output)
        e = m.get(key) if m else None
        pv = (e or {}).get("preview")
        if not pv:
            raise LookupError(key)
        path = (self.out_dir / pv["file"]).resolve()
        if not path.is_file() or (self.out_dir / PREVIEW_DIR).resolve() not in path.parents:
            raise LookupError(key)
        return path

    def close(self, wait: float = 3.0) -> bool:
        """Stop the run (after the scans in flight) and the preview keeper; True when the run has ended."""
        with self._lock:
            self._closed = True
            if self.running:
                self.run.cancel.set()
        self._stop_keeper.set()
        for t in (self._thread, self._keeper):
            if t and t.is_alive() and t is not threading.current_thread():
                t.join(timeout=wait)
        return not self.running
