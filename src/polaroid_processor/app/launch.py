"""Start the app, or reuse the one that is already running (``polaroid app``, ``polaroid review``)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
import webbrowser
from typing import Optional, Sequence

from .. import api
from .server import AppState, serve

DEFAULT_PORT = 8780


def _call(port: int, path: str, body: Optional[dict] = None, timeout: float = 5.0) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": "application/json"} if data is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error", str(e))
        except ValueError:
            msg = str(e)
        raise api.UsageError(msg) from None


def running(port: int) -> bool:
    """True when a polaroid app answers on 127.0.0.1:``port``. The answer is only a name, a version and a pid:
    any local process could fake it, and the worst it gains is that this launch asks it to open a folder the
    user named (every request stays on 127.0.0.1)."""
    try:
        r = _call(port, "/api/ping", timeout=1.5)
        return r.get("app") == "polaroid-app" and isinstance(r.get("version"), str) and isinstance(r.get("pid"), int)
    except (OSError, ValueError, AttributeError, api.UsageError):
        return False


def launch(paths: Sequence[str] = (), *, output: Optional[str] = None, port: Optional[int] = None,
           open_browser: bool = True, author: str = "human", config: Optional[str] = None,
           filter_name: Optional[str] = None, announce=print) -> int:
    """Open the app on ``output`` (an existing output folder) or on ``paths`` (scan folders become projects, the
    first one is opened; an output folder with a manifest opens as it is). An app that already runs on the port
    (default 8780) is reused: the browser opens on it and on that project; no second server starts. Without an
    explicit ``port`` a busy default port is replaced by a free one."""
    wanted = DEFAULT_PORT if port is None else port
    path = "/" + (f"?filter={filter_name}" if filter_name else "")
    if wanted and running(wanted):
        base = f"http://127.0.0.1:{wanted}"
        _open_on_running(wanted, paths, output, announce)
        announce(f"polaroid app is already running: {base}{path}")
        if author != "human":
            announce("note: the author of fixes is set by the running app; --author is ignored")
        if open_browser:
            webbrowser.open(base + path)
        return 0
    state = AppState(config, author=author)
    if output:
        state.open_project(output=output)
    for i, folder in enumerate(paths):
        try:
            state.open_path(folder) if i == 0 else state.add_path(folder)
        except (api.UsageError, ValueError) as e:
            if len(paths) == 1:
                raise
            announce(f"skipped {folder}: {e}")
    try:
        serve(state, wanted, open_browser=open_browser, announce=announce, path=path, fallback=port is None)
    except OSError as e:
        raise api.UsageError(f"port {wanted} is not available: {e}") from None
    return 0


def _open_on_running(port: int, paths: Sequence[str], output: Optional[str], announce) -> None:
    from pathlib import Path

    def is_output(p):
        return (Path(p) / "polaroid-manifest.json").exists()

    try:
        if output:
            _call(port, "/api/project/open", {"output": output})
        for i, folder in enumerate(paths):
            body = {"output": folder} if is_output(folder) else {"input": folder}
            try:
                _call(port, "/api/project/open" if i == 0 else "/api/project/add", body)
            except api.UsageError as e:
                if len(paths) == 1:
                    raise
                announce(f"skipped {folder}: {e}")
    except api.UsageError as e:
        if not str(e).startswith("a run is going"):
            raise
        announce(f"warning: {e}")      # the running app stays on its project; the page opens on it
