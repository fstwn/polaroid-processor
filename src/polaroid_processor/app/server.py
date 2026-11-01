"""Local web server of the app.

A stdlib ``http.server`` bound to 127.0.0.1 (host and origin checks, JSON helpers and error
mapping come from the review server). Routes::

    GET  /                                      the page (static files under /static/)
    GET  /api/home                              recent projects with counts, recent input folders
    POST /api/browse                            body {"initial": path}: native folder dialog, {"path": ...}
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
                                                scans/<key>/tiles, tile/<name>.png, confirm, save)
"""
from __future__ import annotations

import logging
import subprocess
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Callable, List, Optional

from .. import api
from ..review.server import Handler, NotFound
from .fix import FixApp
from .project import Project, load_manifest, input_folder_of, list_inputs, project_summary
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
    return str(p.with_name(p.name + "_processed"))


class AppState:
    """The user's recent projects and the one open project (no HTTP in here)."""

    def __init__(self, config: Optional[Path] = None, picker: Callable[..., Optional[str]] = pick_folder):
        self.store = Store(config)
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
    def open_project(self, input_dir: Optional[str] = None, output: Optional[str] = None) -> dict:
        """Open a new project (``input_dir``, ``output`` defaulting to ``<input>_processed``) or an existing
        output folder with a manifest (``input_dir`` omitted)."""
        input_dir = (input_dir or "").strip().strip('"') or None
        output = (output or "").strip().strip('"') or None
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
        with self._lock:
            self._close_current()
            self._closing = [p for p in self._closing if p.running]
            if any(same_path(p.output, str(Path(output))) for p in self._closing):
                raise api.UsageError("the run on this output folder is still stopping; try again in a moment")
            self.project = Project(input_dir, str(Path(output)))
            self.store.touch_project(input_dir, str(Path(output)))
            self.project.kick_previews()
            return self.project.record()

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
                p.fixer = FixApp(p)
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
            if segs == ["api", "home"]:
                return self._json(st.home())
            if segs == ["api", "project"]:
                return self._json({"project": st.project.record() if st.project else None})
            if len(segs) == 3 and segs[:2] == ["api", "project"]:
                if segs[2] == "progress":
                    return self._json(st.need_project().progress())
                if segs[2] == "scans":
                    return self._json(st.need_project().scans())
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
                return self._json(fx.tile_specs(rest[1]))
            if len(rest) == 4 and rest[0] == "scans" and rest[2] == "tile" and rest[3].endswith(".png"):
                return self._send(200, fx.tile_png(rest[1], rest[3][:-4]), "image/png")
        elif len(rest) == 3 and rest[0] == "scans" and rest[2] in ("confirm", "save"):
            body = self._body()
            if rest[2] == "confirm":
                return self._json(fx.confirm(rest[1]))
            return self._json(fx.save(rest[1], body.get("corners")))
        raise NotFound("/".join(rest))

    def _static(self, name: str) -> None:
        path = STATIC_DIR / name
        if name != Path(name).name or not path.is_file():
            raise NotFound(name)
        self._send(HTTPStatus.OK, path.read_bytes(), CONTENT_TYPES.get(path.suffix, "application/octet-stream"))


def make_server(state: AppState, port: int = 0) -> ThreadingHTTPServer:
    """HTTP server for ``state`` on 127.0.0.1 (port 0 picks a free one). Not started."""
    handler = type("BoundAppHandler", (AppHandler,), {"state": state})
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    srv.daemon_threads = True
    return srv


def serve(state: Optional[AppState] = None, port: int = 8780, *, open_browser: bool = True, announce=print) -> None:
    """Run until interrupted."""
    state = state or AppState()
    srv = make_server(state, port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
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
