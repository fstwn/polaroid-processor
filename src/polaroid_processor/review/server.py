"""Local web server behind ``polaroid review``.

A stdlib ``http.server`` bound to 127.0.0.1. It serves the static page, lists the scans of an
output folder's manifest, cuts tiles on demand from the (read-only) input scans at the current
outline, and stores fixes through :func:`polaroid_processor.api.fix_scan`.

Routes::

    GET  /                                        the page
    GET  /api/scans                               scans with outline, status, warnings, review state
    GET  /api/scans/<key>/tiles                   tile descriptors of one scan at its current outline
    GET  /api/scans/<key>/tile/<name>.png?rev=N   one tile (raw pixels; the overview is downscaled)
    POST /api/scans/<key>/confirm                 accept the current outline
    POST /api/scans/<key>/save                    body {"corners": {"tl": [x, y], ...}}: store all four corners
"""
from __future__ import annotations

import json
import logging
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import quote, unquote, urlsplit

import cv2
import numpy as np

from .. import api
from ..imageio import load_rgb, read_meta, to_uint8
from ..manifest import Manifest
from ..render import CORNERS, _stretch, inspect_points, is_dark, quad_summary, raw_tile

log = logging.getLogger("polaroid_processor.review")

TILE_PX = 480                  # tile size in original scan pixels
OVERVIEW_PX = 960              # long edge of the overview tile
STATIC_DIR = Path(__file__).parent / "static"
TILE_NAMES = ("tl", "top", "tr", "left", "overview", "right", "bl", "bottom", "br")
MAX_BODY = 64 * 1024
NOTE = "review page"


class NotFound(LookupError):
    pass


class ReviewApp:
    """State and operations of one review session (no HTTP in here)."""

    def __init__(self, output: str, *, show_all: bool = False, author: str = "human", export: bool = False):
        self.output = str(output)
        self.show_all = show_all
        self.author = author
        self.export = export
        self._lock = threading.RLock()   # manifest writes and the one cached scan
        self._cache: Tuple[Optional[tuple], Optional[np.ndarray]] = (None, None)
        if not Manifest(self.output).path.exists():
            raise api.UsageError(f"no manifest in {self.output}")

    # ---- records -------------------------------------------------------
    def _record(self, m: Manifest, key: str) -> dict:
        e = m.get(key)
        quad = Manifest.effective_quad(e)
        hist = [h for h in e.get("history", []) if h.get("action") in ("accept", "fix")]
        meta = e.get("meta") or {}
        review = None
        if e.get("override") and hist:
            review = {k: hist[-1].get(k) for k in ("action", "author", "at", "note")}
        return {
            "key": key,
            "status": e.get("status", "pending"),
            "warnings": e.get("warnings", []),
            "quad": np.round(quad, 2).tolist(),
            "quad_source": "override" if e.get("override") else "auto",
            "size_px": quad_summary(quad)["size_px"],
            "scan_px": [meta["width"], meta["height"]] if meta.get("width") else None,
            "rev": len(e.get("history", [])),
            "review": review,
        }

    def list_scans(self) -> dict:
        with self._lock:
            m = Manifest(self.output)
            keys = [k for k, e in sorted(m.items())
                    if Manifest.effective_quad(e) is not None and (self.show_all or e.get("status") == "needs_review")]
            return {"output": str(m.out_dir), "all": self.show_all, "author": self.author, "export": self.export,
                    "tile_px": TILE_PX, "scans": [self._record(m, k) for k in keys]}

    def _entry(self, key: str) -> Tuple[Manifest, dict]:
        m = Manifest(self.output)
        e = m.get(key)
        if e is None or Manifest.effective_quad(e) is None:
            raise NotFound(key)
        return m, e

    # ---- tiles ---------------------------------------------------------
    def _rgb(self, e: dict) -> np.ndarray:
        """Decode the input scan (read only); the last one is kept because its tiles come in a burst."""
        path = Path(e["input"])
        st = path.stat()
        ident = (str(path), st.st_mtime_ns, st.st_size)
        if self._cache[0] != ident:
            self._cache = (None, None)   # free the old scan before decoding the next
            self._cache = (ident, load_rgb(str(path)))
        return self._cache[1]

    @staticmethod
    def _overview_size(w: int, h: int) -> Tuple[int, int]:
        f = min(1.0, OVERVIEW_PX / max(h, w))
        return max(1, round(w * f)), max(1, round(h * f))

    def tile_specs(self, key: str) -> dict:
        """Where each tile sits for the current outline: scan window (x0, y0, size) and tile image size.

        A scan point (x, y) lies at ((x - x0) * size_w / window_w, (y - y0) * size_h / window_h)
        in the tile image.
        """
        with self._lock:
            m, e = self._entry(key)
            quad = Manifest.effective_quad(e)
            meta = e.get("meta") or {}
            if meta.get("width"):
                w, h = meta["width"], meta["height"]
            else:
                disk = read_meta(e["input"])
                w, h = disk.width, disk.height
            rev = len(e.get("history", []))
            by_name = {}
            for name, p in inspect_points(quad):
                by_name[name] = {"x0": int(round(p[0])) - TILE_PX // 2, "y0": int(round(p[1])) - TILE_PX // 2,
                                 "window": [TILE_PX, TILE_PX], "size": [TILE_PX, TILE_PX]}
            by_name["overview"] = {"x0": 0, "y0": 0, "window": [w, h], "size": list(self._overview_size(w, h))}
            tiles = [{"name": n, **by_name[n], "url": f"/api/scans/{quote(key)}/tile/{n}.png?rev={rev}"}
                     for n in TILE_NAMES]
            return {"key": key, "rev": rev, "scan_px": [w, h], "tiles": tiles}

    def tile_png(self, key: str, name: str) -> bytes:
        if name not in TILE_NAMES:
            raise NotFound(name)
        with self._lock:
            m, e = self._entry(key)
            quad = Manifest.effective_quad(e)
            rgb = self._rgb(e)
            if name == "overview":
                h, w = rgb.shape[:2]
                tile = cv2.resize(to_uint8(rgb), self._overview_size(w, h), interpolation=cv2.INTER_AREA)
                if is_dark(tile):
                    tile = _stretch(tile)
            else:
                tile, _, _, _ = raw_tile(rgb, dict(inspect_points(quad))[name], TILE_PX, "auto")
        ok, buf = cv2.imencode(".png", cv2.cvtColor(tile, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            raise RuntimeError("PNG encoding failed")
        return buf.tobytes()

    # ---- saving --------------------------------------------------------
    def confirm(self, key: str) -> dict:
        with self._lock:
            self._entry(key)
            rec = api.fix_scan(self.output, key, accept=True, author=self.author, note=f"{NOTE}: confirmed",
                               export=self.export)
            return self._saved(key, rec)

    def save(self, key: str, corners: Any) -> dict:
        if not isinstance(corners, dict) or set(corners) != set(CORNERS):
            raise api.UsageError(f"corners must hold exactly {', '.join(CORNERS)}")
        sets = api.check_fix_values(corners, CORNERS, 2, "corners")
        if not all(np.isfinite(v).all() for v in sets.values()):
            raise api.UsageError("corners must be finite numbers")
        with self._lock:
            self._entry(key)
            rec = api.fix_scan(self.output, key, sets=sets, author=self.author, note=f"{NOTE}: corrected",
                               export=self.export)
            return self._saved(key, rec)

    def _saved(self, key: str, rec: dict) -> dict:
        out = self._record(Manifest(self.output), key)
        out["exported"] = bool(rec.get("exported"))
        return out


# --------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "polaroid-review"
    app: ReviewApp   # set on the subclass by make_server

    def log_message(self, fmt, *args):  # quiet by default; -v shows requests
        log.debug("%s %s", self.address_string(), fmt % args)

    # -- helpers
    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, status: int = 200) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, status: int, msg: str) -> None:
        self._json({"error": msg}, status)

    def _host_ok(self) -> bool:
        """Refuse requests whose Host or Origin is not this server (DNS rebinding, other web pages)."""
        port = self.server.server_address[1]
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host", "") not in allowed:
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin in {f"http://{h}" for h in allowed}

    def _route(self, method: str) -> None:
        if not self._host_ok():
            return self._error(HTTPStatus.FORBIDDEN, "bad Host or Origin")
        segs = [unquote(s) for s in urlsplit(self.path).path.strip("/").split("/")]
        try:
            self._dispatch(method, segs)
        except NotFound as e:
            self._error(HTTPStatus.NOT_FOUND, f"not found: {e}")
        except (api.UsageError, ValueError, KeyError) as e:
            self._error(HTTPStatus.BAD_REQUEST, str(e))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # keep serving; the page shows the message
            log.exception("request failed")
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(e).__name__}: {e}")

    def _dispatch(self, method: str, segs: List[str]) -> None:
        app = self.app
        if method == "GET":
            if segs == [""]:
                return self._send(200, (STATIC_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            if segs == ["api", "scans"]:
                return self._json(app.list_scans())
            if len(segs) == 4 and segs[:2] == ["api", "scans"] and segs[3] == "tiles":
                return self._json(app.tile_specs(segs[2]))
            if len(segs) == 5 and segs[:2] == ["api", "scans"] and segs[3] == "tile" and segs[4].endswith(".png"):
                return self._send(200, app.tile_png(segs[2], segs[4][:-4]), "image/png")
        elif method == "POST" and len(segs) == 4 and segs[:2] == ["api", "scans"] and segs[3] in ("confirm", "save"):
            body = self._body()
            if segs[3] == "confirm":
                return self._json(app.confirm(segs[2]))
            return self._json(app.save(segs[2], body.get("corners")))
        raise NotFound("/".join(segs))

    def _body(self) -> dict:
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            raise ValueError("Content-Type must be application/json")
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ValueError("request body too large")
        data = json.loads(self.rfile.read(n) or b"{}")
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        return data

    def do_GET(self):  # noqa: N802
        self._route("GET")

    def do_POST(self):  # noqa: N802
        self._route("POST")


def make_server(app: ReviewApp, port: int = 0) -> ThreadingHTTPServer:
    """HTTP server for ``app`` on 127.0.0.1 (port 0 picks a free one). Not started."""
    handler = type("BoundHandler", (Handler,), {"app": app})
    srv = ThreadingHTTPServer(("127.0.0.1", port), handler)
    srv.daemon_threads = True
    return srv


def serve(app: ReviewApp, port: int = 0, *, open_browser: bool = True, announce=print) -> None:
    """Run until interrupted."""
    srv = make_server(app, port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    announce(f"review page: {url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
