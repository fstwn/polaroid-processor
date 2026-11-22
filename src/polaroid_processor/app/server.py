"""Local web server of the app.

A stdlib ``http.server`` bound to 127.0.0.1 (host and origin checks, JSON helpers and error
mapping come from the review server). Routes::

    GET  /                                      the page (static files under /static/)
    GET  /api/home                              recent projects with counts, recent input folders
    POST /api/browse                            body {"initial": path}: native folder dialog, {"path": ...}
    GET  /api/ping                              who answers: another launch finds a running app with it
    GET  /api/project/settings                  the settings panel's fields of the open project
    POST /api/project/settings                  body: fields to change (refused while a run goes); a metadata
                                                field that fails its check gives 400 with {"error", "fields": {name: msg}}
    POST /api/project/metadata/preview          body {"metadata": {...as typed}}: the first scan's filled title and
                                                description, and the field errors, without saving
    POST /api/project/add                       body {"input", "output"?}: onto the recent list, not opened
    POST /api/project/open                      body {"input", "output"?} new project, or {"output"} an existing one
    POST /api/project/close
    POST /api/project/forget                    body {"output"}: drop it from the recent list
    GET  /api/project                           the open project, or null
    GET  /api/project/progress                  run state, counts and the manifest stamp (poll this)
    GET  /api/project/scans                     every scan with status, warnings and preview
    POST /api/project/detect                    start (or resume) the detect run
    POST /api/project/export                    export the detected scans that have no up to date export
    POST /api/project/all                       detect new scans and export everything
    POST /api/project/cancel                    stop after the scans in flight
    GET  /api/project/preview/<key>.jpg         the crop preview of a scan
    GET  /fix                                   the drag-to-fix page (the review page, bound to the project)
    ...  /api/fix/...                           that page's API: the review server's routes (stamp, scans,
                                                scans/<key>/tiles, tile/<name>.png, confirm, save, window)
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Callable, List, Optional

from .. import __version__, api
from ..review.server import Handler, LocalServer, NotFound
from .fix import FixApp
from .project import Project, load_manifest, input_folder_of, list_inputs, project_summary
from .settings import FieldErrors
from .store import Store, same_path

log = logging.getLogger("polaroid_processor.app")

STATIC_DIR = Path(__file__).parent / "static"
REVIEW_STATIC = Path(__file__).parent.parent / "review" / "static"
FIX_CONFIG = '<script>window.POLAROID_API = "/api/fix"; window.POLAROID_EMBED = true;</script>'

CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}

_DIALOG = ("import sys, tkinter as tk\n"
           "from tkinter import filedialog\n"
           "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
           "print(filedialog.askdirectory(title=sys.argv[1], initialdir=sys.argv[2] or None, parent=r) or '')\n")


def pick_folder(title: str = "Select a folder", initial: str = "") -> Optional[str]:
    """Ask for a folder in the native dialog (tkinter, in its own process so the server keeps running)."""
    proc = subprocess.run([sys.executable, "-c", _DIALOG, title, initial], capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise api.UsageError("the folder dialog is not available: " + (proc.stderr.strip().splitlines() or ["?"])[-1])
    path = proc.stdout.strip()
    return str(Path(path)) if path else None


def default_output(input_dir: str) -> str:
    p = Path(input_dir)
    if not p.name:
        raise api.UsageError(f"{input_dir} is a drive root, not a scan folder; pick the folder that holds the scans")
    return str(p.with_name(p.name + "_processed"))


class AppState:
    """The user's recent projects and the one open project (no HTTP in here)."""

    def __init__(self, config: Optional[Path] = None, picker: Callable[..., Optional[str]] = pick_folder,
                 author: str = "human"):
        self.store = Store(config)
        self.author = author
        self.picker = picker
        self.project: Optional[Project] = None
        self._closing: List[Project] = []      # closed projects whose run is still stopping
        self._lock = threading.RLock()

    # ---- home ---------------------------------------------------------------
    def home(self) -> dict:
        projects = []
        for p in self.store.projects():
            info = project_summary(p.get("input"), p["output"])
            projects.append({**p, "name": Path(p["output"]).name, **info,
                             "open": bool(self.project and same_path(self.project.output, p["output"]))})
        return {"projects": projects, "recent_inputs": self.store.recent_inputs(),
                "current": self.project.record() if self.project else None}

    def browse(self, initial: str = "") -> dict:
        start = initial if initial and Path(initial).is_dir() else ""
        return {"path": self.picker("Select a folder", start)}

    # ---- open / close ---------------------------------------------------------
    def _resolve(self, input_dir: Optional[str], output: Optional[str]) -> tuple:
        """Check a project's folders: ``(input or None, output)``. A new project gives ``input_dir`` (``output``
        defaults to ``<input>_processed``); an existing output folder with a manifest gives only ``output``."""
        input_dir = (input_dir or "").strip().strip('"') or None
        output = (output or "").strip().strip('"') or None
        input_dir = os.path.abspath(input_dir) if input_dir else None    # the app may run from anywhere
        output = os.path.abspath(output) if output else None
        if input_dir:
            if not Path(input_dir).is_dir():
                raise api.UsageError(f"{input_dir} is not a folder")
            if not list_inputs(input_dir):
                raise api.UsageError(f"no .tif/.tiff scans in {input_dir}")
            output = output or default_output(input_dir)
            if same_path(input_dir, output):
                raise api.UsageError("the output folder must differ from the input folder")
            if Path(input_dir).resolve() in Path(output).resolve().parents:
                raise api.UsageError("the output folder must not lie inside the input folder")
            if Path(output).exists() and not Path(output).is_dir():
                raise api.UsageError(f"{output} is not a folder")
            m = load_manifest(output)
            known = input_folder_of(m) if m else None
            if known and not same_path(known, input_dir):
                raise api.UsageError(f"{output} already holds the results for {known}")
        else:
            if not output:
                raise api.UsageError("give an input folder, or an existing output folder")
            m = load_manifest(output) if Path(output).is_dir() else None
            if m is None:
                raise api.UsageError(f"no {Path(output) / 'polaroid-manifest.json'}: not an output folder of this tool")
            input_dir = input_folder_of(m)
            if input_dir and not Path(input_dir).is_dir():
                input_dir = None
        return input_dir, str(Path(output))

    def open_project(self, input_dir: Optional[str] = None, output: Optional[str] = None) -> dict:
        """Open a new project (``input_dir``, ``output`` defaulting to ``<input>_processed``) or an existing
        output folder with a manifest (``input_dir`` omitted). Refused while a run goes in another project:
        leaving a project cancels its run, and a caller that only wants to look must not do that."""
        input_dir, output = self._resolve(input_dir, output)
        with self._lock:
            cur = self.project
            if cur and cur.running:
                if same_path(cur.output, output):
                    return cur.record()
                raise api.UsageError(f"a run is going in {cur.name}; stop it (Cancel) or wait until it ends")
            self._close_current()
            self._closing = [p for p in self._closing if p.running]
            if any(same_path(p.output, output) for p in self._closing):
                raise api.UsageError("the run on this output folder is still stopping; try again in a moment")
            self.project = Project(input_dir, output, self.store)
            self.store.touch_project(input_dir, output)
            self.project.kick_previews()
            return self.project.record()

    def add_project(self, input_dir: Optional[str] = None, output: Optional[str] = None) -> dict:
        """Put a project on the recent list without opening it (folders dropped on the launcher)."""
        input_dir, output = self._resolve(input_dir, output)
        self.store.touch_project(input_dir, output)
        return {"input": input_dir, "output": output, "name": Path(output).name}

    def open_path(self, path: str) -> dict:
        """A folder given on the command line: an output folder with a manifest opens as it is, any other folder is
        a new project."""
        if (Path(path) / "polaroid-manifest.json").exists():
            return self.open_project(output=path)
        return self.open_project(input_dir=path)

    def add_path(self, path: str) -> dict:
        if (Path(path) / "polaroid-manifest.json").exists():
            return self.add_project(output=path)
        return self.add_project(input_dir=path)

    def _close_current(self) -> None:
        if self.project:
            if not self.project.close():
                self._closing.append(self.project)
            self.project = None

    def close_project(self) -> dict:
        with self._lock:
            self._close_current()
            self._closing = [p for p in self._closing if p.running]
            return {"ok": True, "stopping": bool(self._closing)}

    def start_run(self, mode: str) -> dict:
        """Start a run of the open project; the state lock is not held while the preview keeper winds down."""
        return self.need_project().start(mode)

    def forget(self, output: str) -> dict:
        self.store.forget_project(output)
        return {"ok": True}

    def need_project(self) -> Project:
        if self.project is None:
            raise api.UsageError("no project is open")
        return self.project

    # ---- drag-to-fix ---------------------------------------------------------------
    def fixer(self) -> FixApp:
        """The review backend of the open project (it keeps the decoded scan the tiles are cut from)."""
        p = self.need_project()
        with self._lock:
            if p.fixer is None:
                p.fixer = FixApp(p, self.author)
            return p.fixer

    def shutdown(self) -> None:
        with self._lock:
            self._close_current()


# --------------------------------------------------------------------------- HTTP
class AppHandler(Handler):
    server_version = "polaroid-app"
    state: AppState   # set on the subclass by make_server

    def _dispatch(self, method: str, segs: List[str]) -> None:
        st = self.state
        if method == "GET":
            if segs == [""] or segs == ["index.html"]:
                return self._static("index.html")
            if len(segs) == 2 and segs[0] == "static":
                return self._static(segs[1])
            if segs == ["fix"]:
                page = (REVIEW_STATIC / "index.html").read_text(encoding="utf-8")
                return self._send(200, page.replace("<!--polaroid-config-->", FIX_CONFIG).encode("utf-8"),
                                  "text/html; charset=utf-8")
            if segs[:2] == ["api", "fix"]:
                return self._fix(method, segs[2:])
            if segs == ["api", "ping"]:
                return self._json({"app": "polaroid-app", "version": __version__, "pid": os.getpid()})
            if segs == ["api", "home"]:
                return self._json(st.home())
            if segs == ["api", "project"]:
                return self._json({"project": st.project.record() if st.project else None})
            if len(segs) == 3 and segs[:2] == ["api", "project"]:
                if segs[2] == "progress":
                    return self._json(st.need_project().progress())
                if segs[2] == "scans":
                    return self._json(st.need_project().scans())
                if segs[2] == "settings":
                    return self._json(st.need_project().get_settings())
            if len(segs) == 4 and segs[:3] == ["api", "project", "preview"] and segs[3].endswith(".jpg"):
                try:
                    path = st.need_project().preview_file(segs[3][:-4])
                except LookupError as e:
                    raise NotFound(f"preview of {e}") from None
                return self._send(200, path.read_bytes(), "image/jpeg")
        elif method == "POST":
            if segs[:2] == ["api", "fix"]:
                return self._fix(method, segs[2:])
            body = self._body()
            if segs == ["api", "browse"]:
                return self._json(st.browse(str(body.get("initial") or "")))
            if segs == ["api", "project", "open"]:
                return self._json(st.open_project(body.get("input"), body.get("output")))
            if segs == ["api", "project", "add"]:
                return self._json(st.add_project(body.get("input"), body.get("output")))
            if segs == ["api", "project", "settings"]:
                try:
                    return self._json(st.need_project().set_settings(body))
                except FieldErrors as e:      # the panel shows each message next to its field
                    return self._json({"error": str(e), "fields": e.fields}, HTTPStatus.BAD_REQUEST)
            if segs == ["api", "project", "metadata", "preview"]:
                return self._json(st.need_project().metadata_preview(body.get("metadata")))
            if segs == ["api", "project", "forget"]:
                return self._json(st.forget(str(body.get("output") or "")))
            if segs == ["api", "project", "close"]:
                return self._json(st.close_project())
            if len(segs) == 3 and segs[:2] == ["api", "project"] and segs[2] in ("detect", "export", "all"):
                return self._json(st.start_run(segs[2]))
            if segs == ["api", "project", "cancel"]:
                return self._json(st.need_project().cancel())
        raise NotFound("/".join(segs))

    def _fix(self, method: str, rest: List[str]) -> None:
        """The review server's routes, served for the open project."""
        fx = self.state.fixer()
        if method == "GET":
            if rest == ["stamp"]:
                return self._json({"stamp": fx.stamp()})
            if rest == ["scans"]:
                return self._json(fx.list_scans())
            if len(rest) == 3 and rest[0] == "scans" and rest[2] == "tiles":
                return self._json(fx.tile_specs(rest[1], *self._tile_args()))
            if len(rest) == 4 and rest[0] == "scans" and rest[2] == "tile" and rest[3].endswith(".png"):
                return self._send(200, fx.tile_png(rest[1], rest[3][:-4], *self._tile_args()), "image/png")
        elif len(rest) == 3 and rest[0] == "scans" and rest[2] in ("confirm", "save", "window"):
            body = self._body()
            if rest[2] == "confirm":
                return self._json(fx.confirm(rest[1]))
            if rest[2] == "window":
                return self._json(fx.window(rest[1], body))
            return self._json(fx.save(rest[1], body.get("corners"), body.get("window")))
        raise NotFound("/".join(rest))

    def _static(self, name: str) -> None:
        path = STATIC_DIR / name
        if name != Path(name).name or not path.is_file():
            raise NotFound(name)
        self._send(HTTPStatus.OK, path.read_bytes(), CONTENT_TYPES.get(path.suffix, "application/octet-stream"))


def make_server(state: AppState, port: int = 0) -> ThreadingHTTPServer:
    """HTTP server for ``state`` on 127.0.0.1 (port 0 picks a free one). Not started."""
    handler = type("BoundAppHandler", (AppHandler,), {"state": state})
    srv = LocalServer(("127.0.0.1", port), handler)
    srv.daemon_threads = True
    return srv


def serve(state: Optional[AppState] = None, port: int = 8780, *, open_browser: bool = True, announce=print,
          path: str = "/", fallback: bool = False) -> None:
    """Run until interrupted. With ``fallback`` a busy ``port`` is replaced by any free one."""
    state = state or AppState()
    try:
        srv = make_server(state, port)
    except OSError:
        if not fallback:
            raise
        srv = make_server(state, 0)
    url = f"http://127.0.0.1:{srv.server_address[1]}{path}"
    announce(f"polaroid app: {url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        state.shutdown()
