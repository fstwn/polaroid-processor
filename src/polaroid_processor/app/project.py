"""One project of the app: an input folder and its output folder (with the manifest).

Reads the manifest for the grid and the counts, runs detect and export passes in a background
thread, keeps the crop previews up to date, and re-evaluates a scan after a fix. No HTTP in here.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import cv2
import numpy as np

from .. import api
from ..detection import quad_size
from ..geometry import homography
from ..imageio import load_rgb
from ..item import agent_crop, held_by_hand, is_other, kind_of
from ..lock import holder
from ..manifest import MANIFEST_NAME, Manifest
from ..pipeline import (collect_inputs, file_meta, manual_outline, reevaluate, resolve_scales, retag_ok,
                        settings_from_dict)
from ..preview import LONG_EDGE, PREVIEW_DIR, preview_key, write_preview
from ..scale import explicit_scale, profiles_path, scale_from_size
from ..scan_window import effective_window, fix_in_force, window_state
from ..metadata import format_label as meta_label
from .settings import check_metadata, defaults_of, from_panel, profile_options, to_panel

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


def metadata_key_of(e: dict, key: str, settings) -> str:
    """The metadata key a run would give the exports of scan ``key`` now (``""`` without any filled field): the same
    ``file_meta(...).key()`` the pipeline compares with the entry's ``metadata_key`` (``pipeline.sync_metadata``)."""
    quad = Manifest.effective_quad(e)
    return file_meta(settings.metadata, key, e.get("input") or key, e.get("format"),
                     None if quad is None else quad_size(quad), item=e.get("item_fix")).key()


def export_plan(m: Manifest, settings) -> Dict[str, tuple]:
    """Per scan what a run would give its export now: ``(key, base, tags_ok, metadata_key)`` (see
    ``Settings.export_key``, ``export_base``, ``retag_ok``, ``metadata_key_of``), the same test the export pass uses to
    decide what to export again."""
    keys = [k for k, e in m.items() if Manifest.effective_quad(e) is not None]
    scales = resolve_scales(m, keys, settings)
    out = {}
    for k in keys:
        e = m.get(k)
        if k in scales and e.get("status") != "failed":
            quad, curves, manual = Project._quad_curves(e)
            win = effective_window(e, quad) if settings.photo_crop != "off" else None
            # nothing set and nothing stored: skip the fill (the common case of a project without metadata)
            mkey = metadata_key_of(e, k, settings) if settings.metadata or e.get("metadata_key") else ""
            out[k] = (settings.export_key(quad, scales[k], curves, manual, win, e.get("item_fix")),
                      settings.export_base(quad, scales[k], curves, manual, win, e.get("item_fix")),
                      retag_ok(e, scales[k], settings, curves, manual), mkey)
    return out


def export_state(m: Manifest, e: Optional[dict], plan: Optional[tuple], settings) -> Optional[str]:
    """"done" (exported as a run would now), "headers" (exported as a run would now, but the text headers differ from
    the project's metadata: a run rewrites them in place), "tags" (only the scale differs: a run patches the resolution
    tags, nothing is encoded again), "outdated" (exported, but outline, curves or settings changed since),
    "none" (nothing exported, or files missing), None (nothing to export)."""
    if not e or e.get("status") == "failed" or Manifest.effective_quad(e) is None:
        return None
    outs = e.get("outputs") or {}
    if not outs or any(not m.abs(p).exists() for p in outs.values()):
        return "none"
    if plan is None:
        return "outdated"
    xkey, xbase, tags_ok, mkey = plan
    if e.get("export_key") == xkey:
        return "done" if (e.get("metadata_key") or "") == mkey else "headers"
    if e.get("export_base") == xbase and tags_ok:
        return "tags"
    return "outdated"


def count_scans(m: Optional[Manifest], names: List[str], plans: Optional[Dict[str, tuple]] = None,
                settings=None) -> Dict[str, int]:
    """Scans per status over the manifest and the input folder, plus ``exported``, ``total`` and, when
    ``plans`` (see :func:`export_plan`) is given, ``to_export`` (detected scans without an up to date
    export), ``to_retag`` (exports that need only new resolution tags) and ``to_headers`` (exports whose text headers
    differ from the project's metadata: a run rewrites them in place)."""
    keys = set(names)
    if m:
        keys |= set(m.scans)
    counts = {s: 0 for s in STATUSES}
    exported = to_export = to_retag = to_headers = 0
    for k in keys:
        e = m.get(k) if m else None
        counts[entry_status(e)] += 1
        exported += bool(e and e.get("outputs"))
        if plans is not None and m:
            state = export_state(m, e, plans.get(k), settings)
            to_export += state in ("none", "outdated")
            to_retag += state == "tags"
            to_headers += state == "headers"
    out = {**counts, "exported": exported, "total": len(keys)}
    if plans is not None:
        out["to_export"], out["to_retag"], out["to_headers"] = to_export, to_retag, to_headers
    return out


def scale_problem(settings) -> Optional[str]:
    """Why the explicit scale of ``settings`` (a ppi or a saved profile) cannot be worked out, else None: the profile is
    not in the profiles file, or the file cannot be read (not YAML, not a mapping, a value that is no number). The message
    is the one a run fails with for a missing profile; the cause of any other problem is logged."""
    try:
        explicit_scale(settings.ppi, settings.profile, settings.profiles_file)
    except Exception as err:      # noqa: BLE001  (whatever the profiles file holds)
        if isinstance(err, KeyError) and err.args:
            log.info("scale: %s", err.args[0])
            return str(err.args[0])
        first = (str(err).strip().splitlines() or [""])[0]
        msg = f"the profiles file {profiles_path(settings.profiles_file)} cannot be read ({type(err).__name__}: {first})"
        log.warning("scanner profile %r: %s", settings.profile, msg)      # no traceback: this runs on every recompute
        return msg
    return None


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
    """State of one run (written by the run thread, read by the poller)."""

    def __init__(self, mode: str = "detect"):
        self.mode = mode         # detect, export or all (detect and export)
        self.state = "running"   # running, cancelling, finished, cancelled, error
        self.total: Optional[int] = None
        self.done = 0
        self.errors = 0
        self.started = time.time()
        self.ended: Optional[float] = None
        self.error: Optional[str] = None
        self.current: Optional[str] = None
        self.phase: Optional[str] = None
        self.metadata_updated = 0        # from the run's summary: exports whose text headers were rewritten in place
        self.metadata_skipped = 0        # ... and those that could not be (a file in use): tried again next run
        self.cancel = threading.Event()

    def record(self) -> dict:
        end = self.ended or time.time()
        elapsed = end - self.started
        eta = None
        if self.state in ("running", "cancelling") and self.done and self.total:
            eta = elapsed / self.done * max(0, self.total - self.done)
        return {"mode": self.mode, "state": self.state, "total": self.total, "done": self.done, "errors": self.errors,
                "elapsed": round(elapsed, 1), "eta": None if eta is None else round(eta, 1), "error": self.error,
                "current": self.current, "phase": self.phase if self.state in ("running", "cancelling") else None,
                "metadata_updated": self.metadata_updated, "metadata_skipped": self.metadata_skipped}


class Project:
    def __init__(self, input_dir: Optional[str], output: str, store=None):
        self.store = store
        self.input = str(input_dir) if input_dir else None
        self.output = str(output)
        self.name = Path(self.output).name
        self.run: Optional[Run] = None
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._keeper: Optional[threading.Thread] = None
        self._stop_keeper = threading.Event()
        self._closed = False
        self.fixer = None                # FixApp of this project, made by AppState.fixer
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

    @staticmethod
    def _window_box(e: dict, quad, margin_mm: float) -> Optional[dict]:
        """The photo window as fractions of the crop preview (the straightened print without the margin), for the
        outline on a grid card; None without a window. A flattened print is drawn as if straight: the window lies
        where the unwarp equals the plain homography."""
        w = effective_window(e, quad)
        if window_state(w) != "found":
            return None
        pw, ph = quad_size(quad)
        px = (e.get("scale") or {}).get("px_per_mm") or scale_from_size((pw, ph)).px_per_mm
        m = margin_mm * px
        H, _ = homography(quad, 0.0)
        pts = cv2.perspectiveTransform((np.asarray(w["quad"], float) - 0.5).reshape(-1, 1, 2), H).reshape(-1, 2) + 0.5
        box = np.stack([(pts[:, 0] - m) / (pw - 2 * m), (pts[:, 1] - m) / (ph - 2 * m)], axis=1)
        return {"kind": w.get("kind"), "box": np.round(box, 4).tolist(), "source": w.get("source", "auto")}

    def _record(self, m: Optional[Manifest], key: str, e: Optional[dict], margin: float, flatten: str,
                plans: Optional[Dict[str, tuple]], settings) -> dict:
        status = entry_status(e)
        quad = Manifest.effective_quad(e) if e else None
        rec = {"key": key, "status": status, "warnings": (e or {}).get("warnings", []),
               "error": (e or {}).get("error"), "exported": bool(e and e.get("outputs")),
               "quad_source": "override" if e and e.get("override") else ("auto" if quad is not None else None),
               "format": ((e or {}).get("format") or {}).get("name"), "preview": None, "stale": False,
               "export": None, "rev": len((e or {}).get("history", []))}
        if kind_of(e):      # only with a choice: the card of a scan "as detected" stays as it was
            rec["item"] = kind_of(e)
        if agent_crop(e):
            rec["agent_crop"] = True
        if held_by_hand(e) and kind_of(e):      # a crop with an item on a scan nothing was detected on: a hand crop, not a "fixed" outline (a plain fixed print keeps its card)
            rec["hand_crop"] = True
        if m and e and e.get("outputs") and plans is not None:      # no plans: the scale could not be resolved
            rec["export"] = export_state(m, e, plans.get(key), settings)
        if quad is not None:
            w, h = quad_size(quad)
            rec["size_px"] = [round(w), round(h)]
            q, curves, manual = self._quad_curves(e)
            if is_other(e):      # exported exactly as drawn: no margin, no unwarp
                want = preview_key(q, None, False, 0.0, "off")
            else:
                want = preview_key(q, curves, manual, margin, flatten)
            pv = e.get("preview")
            if pv and (self.out_dir / pv["file"]).exists():
                rec["preview"] = {"url": f"/api/project/preview/{quote(key)}.jpg?rev={pv['key']}",
                                  "size": pv.get("size")}
                rec["stale"] = pv.get("key") != want
            else:
                rec["stale"] = True
            rec["_want"] = want
            if settings is not None and settings.photo_crop != "off":
                box = self._window_box(e, quad, margin)
                if box:
                    rec["window"] = box
                chosen = fix_in_force(e, quad)       # the user's window choice, for the card's tag
                if chosen:
                    rec["window_fix"] = "none" if window_state(chosen) == "none" else "hand"
        return rec

    @property
    def out_dir(self) -> Path:
        return Path(self.output)

    def snapshot(self) -> dict:
        """Scans (input folder and manifest merged), counts and stale-preview count; cached until the
        manifest, the input folder or the profiles file the project's scale comes from changes."""
        sig = (_file_sig(self.out_dir / MANIFEST_NAME), _file_sig(Path(self.input)) if self.input else None)
        with self._lock:
            if self._snap is not None and sig == self._sig:
                pf = self._snap.get("_profiles")      # (path, stamp) of the profiles file a saved profile is looked up in
                if not pf or _file_sig(pf[0]) == pf[1]:
                    return self._snap
            m = load_manifest(self.output)
            names = list_inputs(self.input)
            margin, flatten = self._settings(m)
            plans: Dict[str, tuple] = {}
            settings = settings_from_dict(m.data.get("settings") or {}) if m else None
            scale_error: Optional[str] = None
            profiles = None
            if m:
                # the scale lookup alone is guarded: a saved profile the profiles file lacks, or a file that cannot be read,
                # leaves the grid without export states and counts (a run fails with the same message); a failure of
                # anything below is a bug and is not hidden as a scale error
                scale_error = scale_problem(settings)
                if settings.profile:
                    p = profiles_path(settings.profiles_file)
                    profiles = (p, _file_sig(p))
            if m and not scale_error:
                # a scan fixed by hand (here, on the command line, by an agent) is judged again, in memory only
                fixed = [k for k, e in m.items()
                         if (e.get("override") or e.get("window_fix") or e.get("item_fix") or e.get("agent_crop"))
                         and e.get("status") == "needs_review"]
                if fixed:
                    reevaluate(m, fixed, settings)
                plans = export_plan(m, settings)
            keys = sorted(set(names) | set(m.scans if m else {}))
            scans = [self._record(m, k, m.get(k) if m else None, margin, flatten, None if scale_error else plans, settings)
                     for k in keys]
            counts = count_scans(m, names, plans if m and not scale_error else None, settings)
            if scale_error:      # no export states without a scale: nothing to export, retag or rewrite (a run would fail)
                counts.update(to_export=0, to_retag=0, to_headers=0)
            self._sig = sig
            pstamp = profiles[1][0] if profiles and profiles[1] else 0      # the cards change with the profiles file too
            self._snap = {"rev": f"{sig[0][0] if sig[0] else 0}-{sig[1][0] if sig[1] else 0}-{pstamp}", "scans": scans,
                          "counts": counts, "scale_error": scale_error, "_profiles": profiles,
                          "stale": sum(1 for r in scans if r["stale"] and "_want" in r and (r["key"], r["_want"]) not in self._bad)}
            return self._snap

    def scans(self) -> dict:
        snap = self.snapshot()
        return {"rev": snap["rev"], "counts": snap["counts"], "scale_error": snap["scale_error"],
                "scans": [{k: v for k, v in r.items() if k != "_want"} for r in snap["scans"]]}

    def progress(self) -> dict:
        with self._lock:        # the run first: a finished run must come with a manifest that is at least as new
            run = self.run.record() if self.run else None
        snap = self.snapshot()
        if run is None or run["state"] not in ("running", "cancelling"):
            self.kick_previews()
        other = holder(self.output)
        if other and other["pid"] == os.getpid() and run and run["state"] in ("running", "cancelling"):
            other = None     # our own run
        return {"rev": snap["rev"], "counts": snap["counts"], "scale_error": snap["scale_error"], "run": run,
                "previews_pending": snap["stale"],
                "can_run": bool(self.input and Path(self.input).is_dir()), "lock": other}

    def record(self) -> dict:
        return {"input": self.input, "output": self.output, "name": self.name}

    # ---- settings -----------------------------------------------------------
    def settings_dict(self, m: Optional[Manifest] = None) -> dict:
        """The settings the next run uses: the manifest's; else the project's own (set before its first run);
        else the user's last choices; else the tool's defaults."""
        m = m or load_manifest(self.output)
        if m and m.data.get("settings"):
            return m.data["settings"]
        if self.store:
            own = self.store.project_settings(self.output)
            if own:
                return own
            return from_panel(None, self.store.settings_defaults())
        return from_panel(None, {})

    def get_settings(self) -> dict:
        d = self.settings_dict()
        return {"settings": to_panel(d), "profiles": profile_options(settings_from_dict(d).profiles_file),
                "running": self.running, "locked_by": (holder(self.output) or {}).get("who") if not self.running else None}

    def set_settings(self, update: dict) -> dict:
        """Apply panel fields. Refused while a run goes (here or in another process): a run reads its settings
        once, so a change could not apply to it; the next run uses the new ones."""
        if self.running:
            raise api.UsageError("wait until the run ends before changing settings; they apply to the next run")
        other = holder(self.output)
        if other and other["pid"] != os.getpid():
            raise api.UsageError(f"another run is working on this output folder (pid {other['pid']}, {other['who']}); "
                                 "wait until it ends")
        m = load_manifest(self.output)
        new = from_panel(self.settings_dict(m), update)
        if m:
            m.data["settings"] = new
            m.save()
        elif self.store:
            self.store.set_project_settings(self.output, new)
        if self.store:
            self.store.set_settings_defaults(defaults_of(new))
        with self._lock:
            self._sig = None
        return self.get_settings()

    def metadata_preview(self, given: dict) -> dict:
        """The title and description of the project's first scan as ``given`` (the panel's metadata fields as typed,
        saved or not) would be written, filled by the same code as an export, plus the messages of the fields that
        fail the checks (``errors``, by field). ``scan`` is None when the project has no scans. A scan that is not
        detected yet has no format: ``{format}`` then shows ``?`` (``format_known`` false)."""
        errors = check_metadata(given)
        scans = self.snapshot()["scans"]
        out = {"scan": None, "title": "", "description": "", "format_known": True, "errors": errors}
        if not scans or not isinstance(given, dict):
            return out
        key = scans[0]["key"]
        m = load_manifest(self.output)
        e = (m.get(key) if m else None) or {}
        path = e.get("input") or str(Path(self.input or ".") / key)
        quad = Manifest.effective_quad(e) if e else None
        fm = file_meta(given, key, path, e.get("format"), None if quad is None else quad_size(quad), unknown_format="?",
                       item=e.get("item_fix"))
        known = bool(kind_of(e)) or bool(meta_label(e.get("format"), None if quad is None else quad_size(quad)))
        out.update(scan=key, title=fm.title, description=fm.description, format_known=known)
        return out

    # ---- runs -----------------------------------------------------------
    @property
    def running(self) -> bool:
        return bool(self.run and self.run.state in ("running", "cancelling"))

    def _export_inputs(self) -> List[str]:
        m = load_manifest(self.output)
        paths = [e["input"] for _, e in sorted(m.items()) if m and (e.get("detection") or e.get("override"))
                 and e.get("status") != "failed" and e.get("input") and Path(e["input"]).is_file()] if m else []
        if not paths:
            raise api.UsageError("no detected scans to export; detect first")
        return paths

    def start(self, mode: str = "detect") -> dict:
        """Start a run in the background: ``detect`` (outlines only), ``export`` (the detected scans without an
        up to date export) or ``all`` (detect new scans and export everything)."""
        if mode not in ("detect", "export", "all"):
            raise api.UsageError(f"unknown run mode {mode!r}")
        if mode != "export" and (not self.input or not Path(self.input).is_dir()):
            raise api.UsageError("the input folder of this project is not known or not found")
        other = holder(self.output)
        if other and other["pid"] != os.getpid():
            raise api.UsageError(f"another run is working on this output folder (pid {other['pid']}, {other['who']}); "
                                 "wait until it ends")
        inputs = self._export_inputs() if mode == "export" else [self.input]
        with self._lock:
            if self._closed:
                raise api.UsageError("this project is closed")
            if self.running:
                raise api.UsageError("a run is already going")
            self.run = run = Run(mode)       # first, so that the preview keeper does not start again
        self._pause_previews()
        with self._lock:
            self._thread = threading.Thread(target=self._work, args=(run, inputs), daemon=True,
                                            name=f"polaroid-{mode}")
            self._thread.start()
            return run.record()

    def start_detect(self) -> dict:
        return self.start("detect")

    def cancel(self) -> dict:
        with self._lock:
            if not self.running:
                raise api.UsageError("no run is going")
            if self.run.state == "running":
                self.run.state = "cancelling"
            self.run.cancel.set()
            return self.run.record()

    def _work(self, run: Run, inputs: List[str]) -> None:
        def progress(key: str, what: str) -> None:
            run.done += 1
            run.phase = None
            run.current = key
            if what not in ("detected", "exported"):
                run.errors += 1

        def on_jobs(n: int) -> None:
            run.total = (run.total or 0) + n
            run.phase = None

        def on_phase(text: str) -> None:
            run.phase = text

        state, error = "finished", None
        try:
            settings = settings_from_dict(self.settings_dict())
            settings.export, settings.previews = run.mode != "detect", run.mode != "export"
            summary = api.process(inputs, self.output, settings, progress=progress, cancel=run.cancel.is_set,
                                  on_jobs=on_jobs, on_phase=on_phase)
            run.metadata_updated = summary.get("metadata_updated", 0)
            run.metadata_skipped = summary.get("metadata_skipped", 0)
            if summary.get("cancelled") and run.done < (run.total or 0):
                state = "cancelled"
        except Exception as e:  # noqa: BLE001  (shown in the page)
            log.exception("%s run failed", run.mode)
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

    def after_fix(self, key: str, rgb=None) -> None:
        """A scan was fixed or accepted: judge it again and renew its preview (from ``rgb``, the decoded scan,
        when the caller has it)."""
        m = load_manifest(self.output)
        if m and m.get(key):
            reevaluate(m, [key], settings_from_dict(m.data.get("settings") or {}))
            m.save()
        if m and m.get(key) and Manifest.effective_quad(m.get(key)) is None:
            self._sig = None      # a crop was taken away from a scan nothing was detected on: no outline, no preview
            return
        self.make_preview(key, rgb)
        self._sig = None

    def make_preview(self, key: str, rgb=None) -> None:
        """Render the preview of one scan at its current outline and record it in the manifest."""
        m = load_manifest(self.output)
        e = m.get(key) if m else None
        quad, curves, manual = self._quad_curves(e) if e else (None, None, False)
        if quad is None:
            raise api.UsageError(f"{key} has no outline yet")
        margin, flatten = self._settings(m)
        px = (e.get("scale") or {}).get("px_per_mm") or scale_from_size(quad_size(quad)).px_per_mm
        if is_other(e):      # exported exactly as drawn: no margin, no unwarp
            margin, flatten, curves, manual = 0.0, "off", None, False
        path = self.out_dir / PREVIEW_DIR / f"{Path(key).stem}.jpg"
        info = write_preview(rgb if rgb is not None else load_rgb(e["input"]), quad, px, curves, manual,
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
