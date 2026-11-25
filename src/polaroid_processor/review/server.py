"""Local web server behind ``polaroid review``.

A stdlib ``http.server`` bound to 127.0.0.1. It serves the static page, lists the scans of an
output folder's manifest, cuts tiles on demand from the (read-only) input scans at the current
outline, and stores fixes through :func:`polaroid_processor.api.fix_scan`.

Routes::

    GET  /                                        the page
    GET  /api/scans                               scans with outline, status, warnings, review state
    GET  /api/stamp                               changes when the manifest on disk does
    GET  /api/scans/<key>/tiles                   tile descriptors of one scan at its current outline; ``?q=x,y,x,y,x,y,x,y``
                                                  (tl, tr, br, bl) cuts the corner tiles at those corners instead (the place
                                                  step's rectangle, not stored yet); a scan without an outline has the overview only
    GET  /api/scans/<key>/start?item=600|go|other the rectangle the place step starts from (``api.start_rectangle``)
    GET  /api/scans/<key>/tile/<name>.png?rev=N   one tile (raw pixels; the overview is downscaled)
    POST /api/scans/<key>/confirm                 accept the current outline (and a missing photo window)
    POST /api/scans/<key>/save                    body {"corners": {"tl": [x, y], ...}}: store all four corners;
                                                  with "window": {"corners": {...}, "kind": ...} the hand-set photo
                                                  window in the same fix (both or neither); with "item": "600" | "go" |
                                                  "other" | "auto" and "scan_ppi" the item choice in the same fix (corners
                                                  may then be left out on a scan with an outline)
    POST /api/scans/<key>/reset                   take the stored crop, item kind and own ppi away (the core's reset, by a human)
    POST /api/scans/<key>/window                  body {"corners": {...}, "kind": "square"|"round"} store the photo
                                                  window; {"none": true} no photo crop; {"reset": true} automatic again

The tile routes take ``?of=window[&kind=square|round]``: the tiles then sit on the corners of the photo window (the
stored one, else the nominal window of that kind).
"""
from __future__ import annotations

import json
import logging
import socket
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import parse_qs, quote, unquote, urlsplit

import cv2
import numpy as np

from .. import api
from ..imageio import load_rgb, read_meta, to_uint8
from ..formats import FORMAT_OF_ITEM
from ..item import MAX_ROTATION_DEG, MAX_SCAN_PPI, MIN_SIDE_PX, agent_crop, is_other, item_of
from ..pipeline import item_record
from ..manifest import MANIFEST_NAME, Manifest
from ..render import CORNERS, _stretch, inspect_points, is_dark, quad_summary, raw_tile
from ..scan_window import (accept_valid, effective_window, fix_outside, fix_window, nominal_window, window_format,
                           window_state)

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

    api_prefix = "/api/scans"   # where the page finds the scans; a host that mounts the routes elsewhere changes it

    def __init__(self, output: str, *, show_all: bool = False, author: str = "human", export: bool = False):
        self.output = str(output)
        self.show_all = show_all
        self.author = author
        self.export = export
        self._stamp = "0"
        self._lock = threading.RLock()   # manifest writes and the one cached scan
        self._cache: Tuple[Optional[tuple], Optional[np.ndarray]] = (None, None)
        if not Manifest(self.output).path.exists():
            raise api.UsageError(f"no manifest in {self.output}")

    # ---- records -------------------------------------------------------
    def _window_info(self, m: Manifest, e: dict, quad: np.ndarray, suggest: Optional[str]) -> Optional[dict]:
        """The photo window as the page needs it (None when the project has no photo crop): its state (found,
        none = no photo crop by choice, missing, unknown = not looked for yet), kind, quad, who set it, whether a
        hand-set window lies outside the outline, and the nominal windows a missing one can be placed from.
        ``suggest``: the kind a new window starts as (the batch's majority when None)."""
        if (m.data.get("settings") or {}).get("photo_crop", "off") == "off" or is_other(e):      # Other: no photo crop
            return None
        suggest = suggest or api.batch_window_kind(m)
        outside = fix_outside(e, quad)
        w = fix_window(e) if outside else effective_window(e, quad)      # outside: the page shows the hand-set window
        state = window_state(w) if w else "unknown"
        info = {"state": state, "source": (w or {}).get("source", "auto"), "reason": (w or {}).get("reason", ""),
                "kind": w.get("kind") if state == "found" else None, "quad": w["quad"] if state == "found" else None,
                "accepted": accept_valid(e, quad), "outside": outside, "suggest": suggest}
        if outside:      # what the export cuts meanwhile: the automatic window, else no photo crop
            info["cut"] = "auto" if window_state(e.get("window")) == "found" else "none"
        if (w or {}).get("source") == "manual":
            info.update({k: w.get(k) for k in ("author", "at", "note")})
        if state != "found":
            info["nominal"] = {}
            for kind in api.KINDS:
                try:
                    info["nominal"][kind] = nominal_window(quad, kind, window_format(e))["quad"]
                except ValueError:
                    pass
        return info

    def _window_quad(self, m: Manifest, e: dict, kind: Optional[str]) -> np.ndarray:
        """Where the window tiles sit: the stored window, else the nominal window of ``kind``."""
        if (m.data.get("settings") or {}).get("photo_crop", "off") == "off":
            raise api.UsageError("this project has no photo crop")
        if is_other(e):
            raise api.UsageError("an item of kind other has no photo window")
        quad = Manifest.effective_quad(e)
        w = fix_window(e) if fix_outside(e, quad) else effective_window(e, quad)
        if window_state(w) == "found":
            return np.array(w["quad"], float)
        kind = kind or api.batch_window_kind(m)
        try:
            return np.array(nominal_window(quad, kind, window_format(e))["quad"], float)
        except ValueError as ex:
            raise api.UsageError(str(ex)) from None

    def _record(self, m: Manifest, key: str, suggest: Optional[str] = None) -> dict:
        e = m.get(key)
        quad = Manifest.effective_quad(e)      # None: a scan whose detection failed and nobody cropped (the place step starts there)
        hist = [h for h in e.get("history", []) if h.get("action") in ("accept", "fix")]
        meta = e.get("meta") or {}
        review = None
        if e.get("override") and hist:
            review = {k: hist[-1].get(k) for k in ("action", "author", "at", "note")}
        rec = {
            "key": key,
            "status": e.get("status", "pending"),
            "warnings": e.get("warnings", []),
            "quad": None if quad is None else np.round(quad, 2).tolist(),
            "quad_source": ("override" if e.get("override") else "auto") if quad is not None else None,
            "size_px": None if quad is None else quad_summary(quad)["size_px"],
            "scan_px": [meta["width"], meta["height"]] if meta.get("width") else None,
            "rev": len(e.get("history", [])),
            "review": review,
            "detected": bool(e.get("detection")),      # a detected outline exists (the item choice may go back to "as detected")
        }
        if quad is None:
            rec["error"] = e.get("error")
            return rec
        win = self._window_info(m, e, quad, suggest)
        if win is not None:
            rec["window"] = win
        if item_of(e):      # the item choice, only when there is one: records of scans "as detected" stay as they were
            rec["item"] = item_record(e)
        if agent_crop(e):
            rec["agent_crop"] = True
        return rec

    def stamp(self) -> str:
        """Changes whenever the manifest file does; a page polls it to reload. Only the file's stamp is
        read; while a writer swaps the file, the last stamp stands (no double reload)."""
        path = Path(self.output) / MANIFEST_NAME
        for _ in range(4):
            try:
                st = path.stat()
                self._stamp = f"{st.st_mtime_ns}-{st.st_size}"
                break
            except OSError:
                time.sleep(0.02)
        return self._stamp

    def list_scans(self) -> dict:
        with self._lock:
            m = Manifest(self.output)
            # a scan whose detection failed and nobody cropped is listed too (the place step starts there); a scan
            # that was never detected stays closed
            keys = [k for k, e in sorted(m.items())
                    if (Manifest.effective_quad(e) is not None and (self.show_all or e.get("status") == "needs_review"))
                    or (Manifest.effective_quad(e) is None and e.get("status") == "failed")]
            suggest = api.batch_window_kind(m) if (m.data.get("settings") or {}).get("photo_crop", "off") != "off" else None
            return {"output": str(m.out_dir), "all": self.show_all, "author": self.author, "export": self.export,
                    "tile_px": TILE_PX, "aspects": {k: f.aspect for k, f in FORMAT_OF_ITEM.items()},
                    "limits": {"min_side_px": MIN_SIDE_PX, "max_rotation_deg": MAX_ROTATION_DEG, "max_scan_ppi": MAX_SCAN_PPI},
                    "scans": [self._record(m, k, suggest) for k in keys]}

    def _entry(self, key: str, need_quad: bool = True) -> Tuple[Manifest, dict]:
        """The manifest and entry of ``key``. ``need_quad=False`` also takes a scan whose detection failed and that has no
        crop yet (the place step works on it)."""
        m = Manifest(self.output)
        e = m.get(key)
        if e is None or (Manifest.effective_quad(e) is None and (need_quad or e.get("status") != "failed")):
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

    def tile_specs(self, key: str, of: str = "print", kind: Optional[str] = None, quad: Optional[Any] = None) -> dict:
        """Where each tile sits for the current outline: scan window (x0, y0, size) and tile image size.

        A scan point (x, y) lies at ((x - x0) * size_w / window_w, (y - y0) * size_h / window_h)
        in the tile image. ``of="window"``: the tiles sit on the photo window's corners (see
        :meth:`_window_quad`) instead of the print's. ``quad`` (4 x 2, tl tr br bl): the corner tiles sit on these
        corners instead of the stored outline (the place step's rectangle, which is not stored yet). A scan without
        an outline and without ``quad`` has the overview only.
        """
        with self._lock:
            m, e = self._entry(key, need_quad=False)
            given = quad
            quad = self._tile_quad(m, e, of, kind, given)
            meta = e.get("meta") or {}
            if meta.get("width"):
                w, h = meta["width"], meta["height"]
            else:
                disk = read_meta(e["input"])
                w, h = disk.width, disk.height
            rev = len(e.get("history", []))
            by_name = {}
            for name, p in (inspect_points(quad) if quad is not None else []):
                by_name[name] = {"x0": int(round(p[0])) - TILE_PX // 2, "y0": int(round(p[1])) - TILE_PX // 2,
                                 "window": [TILE_PX, TILE_PX], "size": [TILE_PX, TILE_PX]}
            by_name["overview"] = {"x0": 0, "y0": 0, "window": [w, h], "size": list(self._overview_size(w, h))}
            extra = f"&of=window&kind={kind}" if of == "window" and kind else ("&of=window" if of == "window" else "")
            if of == "print" and given is not None:
                extra += "&q=" + ",".join(f"{v:.2f}" for v in np.asarray(given, float).ravel())
            tiles = [{"name": n, **by_name[n], "url": f"{self.api_prefix}/{quote(key)}/tile/{n}.png?rev={rev}{extra}"}
                     for n in TILE_NAMES if n in by_name]
            return {"key": key, "rev": rev, "scan_px": [w, h], "tiles": tiles}

    def _tile_quad(self, m: Manifest, e: dict, of: str, kind: Optional[str], given: Optional[Any] = None
                   ) -> Optional[np.ndarray]:
        """The corners the tiles sit on: the window's (``of="window"``), the given ones, else the stored outline
        (None for a scan without one: only its overview can be cut)."""
        if of not in ("print", "window"):
            raise api.UsageError(f"of must be print or window, not {of!r}")
        if kind not in (None, *api.KINDS):
            raise api.UsageError(f"kind must be square or round, not {kind!r}")
        if of == "window":
            if Manifest.effective_quad(e) is None:
                raise api.UsageError("this scan has no outline yet")
            return self._window_quad(m, e, kind)
        if given is not None:
            q = np.asarray(given, float)
            if q.shape != (4, 2) or not np.isfinite(q).all():
                raise api.UsageError("q must be four finite points: x,y,x,y,x,y,x,y (tl, tr, br, bl)")
            return q
        return Manifest.effective_quad(e)

    def tile_png(self, key: str, name: str, of: str = "print", kind: Optional[str] = None, quad: Optional[Any] = None) -> bytes:
        if name not in TILE_NAMES:
            raise NotFound(name)
        with self._lock:
            m, e = self._entry(key, need_quad=False)
            tq = self._tile_quad(m, e, of, kind, quad)
            if name != "overview" and tq is None:
                raise NotFound(f"{name}: this scan has no outline yet")
            quad = tq
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

    @staticmethod
    def _corners(corners: Any) -> dict:
        if not isinstance(corners, dict) or set(corners) != set(CORNERS):
            raise api.UsageError(f"corners must hold exactly {', '.join(CORNERS)}")
        sets = api.check_fix_values(corners, CORNERS, 2, "corners")
        if not all(np.isfinite(v).all() for v in sets.values()):
            raise api.UsageError("corners must be finite numbers")
        return sets

    def save(self, key: str, corners: Any = None, window: Any = None, item: Any = None, scan_ppi: Any = None) -> dict:
        """Store the corrected outline and, in the same fix (so all or nothing), a hand-set photo window and the item
        choice: ``window`` is ``{"corners": {...}, "kind": "square"|"round"}`` as for :meth:`window`; ``item`` is
        ``"600"``, ``"go"``, ``"other"`` or ``"auto"`` (as detected), ``scan_ppi`` the scan's own resolution (0 takes it
        away; kind other only). ``corners`` may be left out on a scan that has an outline when an item or a ppi comes
        with it; a scan without an outline (its detection failed) needs all four. The corner checks of a hand crop, the
        refusals of the item rules and the stale-file check are those of ``api.fix_scan``: a refused save stores nothing."""
        sets = self._corners(corners) if corners is not None else None
        if item is not None and not isinstance(item, str):
            raise api.UsageError("item must be 600, go, other or auto")
        if scan_ppi is not None and (isinstance(scan_ppi, bool) or not isinstance(scan_ppi, (int, float))):
            raise api.UsageError("scan_ppi must be a number")
        wsets, wkind = None, None
        if window is not None:
            if not isinstance(window, dict):
                raise api.UsageError("window must be an object with corners and kind")
            wsets, wkind = self._corners(window.get("corners")), window.get("kind")
        if sets is None and wsets is None and item is None and scan_ppi is None:
            raise api.UsageError("save needs corners, an item or a scan_ppi")
        with self._lock:
            self._entry(key, need_quad=False)      # a scan nothing was found on is open: the core says what it needs (four corners)
            note = f"{NOTE}: corrected" if sets is not None or wsets is not None else f"{NOTE}: item"
            rec = api.fix_scan(self.output, key, sets=sets, window_sets=wsets, window_kind=wkind, item=item,
                               scan_ppi=scan_ppi, author=self.author, note=note, export=self.export)
            return self._saved(key, rec)

    def reset(self, key: str) -> dict:
        """Take the stored crop, the item kind, the own ppi and the agent mark away (the core's ``reset``, by this page's
        author): back to the automatic outline, or to ``failed`` on a scan nothing was detected on. A refusal of the core is
        a 400 and nothing is stored."""
        with self._lock:
            self._entry(key, need_quad=False)
            rec = api.fix_scan(self.output, key, reset=True, author=self.author, note=f"{NOTE}: crop and item taken away",
                               export=self.export)
            return self._saved(key, rec)

    def start(self, key: str, item: str) -> dict:
        """The rectangle the place step starts from for item kind ``item`` (600, go or other): ``api.start_rectangle``,
        plus the format's aspect (width / height; None for other) the page locks the rectangle to."""
        with self._lock:
            self._entry(key, need_quad=False)
            rec = api.start_rectangle(self.output, key, item)
            fmt = FORMAT_OF_ITEM.get(rec["item"])
            rec["aspect"] = fmt.aspect if fmt else None
            return rec

    def window(self, key: str, body: dict) -> dict:
        """Store the photo window: ``corners`` (all four) and/or ``kind``; ``none`` = no photo crop by choice;
        ``reset`` = back to the automatic window."""
        corners, kind = body.get("corners"), body.get("kind")
        sets = self._corners(corners) if corners is not None else None
        none, reset = bool(body.get("none")), bool(body.get("reset"))
        if not (sets or kind or none or reset):
            raise api.UsageError("window needs corners, kind, none or reset")
        what = "no photo crop" if none else "automatic again" if reset else "corrected"
        with self._lock:
            self._entry(key)
            rec = api.fix_scan(self.output, key, window_sets=sets, window_kind=kind, no_photo_crop=none,
                               reset_window=reset, author=self.author, note=f"{NOTE}: photo window {what}",
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
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.send_header("Content-Security-Policy", "frame-ancestors 'self'")
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
            if segs == ["api", "stamp"]:
                return self._json({"stamp": app.stamp()})
            if len(segs) == 4 and segs[:2] == ["api", "scans"] and segs[3] == "tiles":
                return self._json(app.tile_specs(segs[2], *self._tile_args()))
            if len(segs) == 4 and segs[:2] == ["api", "scans"] and segs[3] == "start":
                return self._json(app.start(segs[2], parse_qs(urlsplit(self.path).query).get("item", ["other"])[0]))
            if len(segs) == 5 and segs[:2] == ["api", "scans"] and segs[3] == "tile" and segs[4].endswith(".png"):
                return self._send(200, app.tile_png(segs[2], segs[4][:-4], *self._tile_args()), "image/png")
        elif method == "POST" and len(segs) == 4 and segs[:2] == ["api", "scans"] and segs[3] in ("confirm", "save", "window", "reset"):
            body = self._body()
            if segs[3] == "confirm":
                return self._json(app.confirm(segs[2]))
            if segs[3] == "reset":
                return self._json(app.reset(segs[2]))
            if segs[3] == "window":
                return self._json(app.window(segs[2], body))
            return self._json(app.save(segs[2], body.get("corners"), body.get("window"), body.get("item"),
                                       body.get("scan_ppi")))
        raise NotFound("/".join(segs))

    def _tile_args(self) -> Tuple[str, Optional[str], Optional[list]]:
        """``of``, ``kind`` and ``q`` (corners tl, tr, br, bl as x,y,x,y,x,y,x,y: tiles at these corners instead of the
        stored outline) of a tile request's query string."""
        q = parse_qs(urlsplit(self.path).query)
        corners = None
        if "q" in q:
            try:
                v = [float(x) for x in q["q"][0].split(",")]
            except ValueError:
                raise ValueError("q must be eight numbers: x,y,x,y,x,y,x,y (tl, tr, br, bl)") from None
            if len(v) != 8:
                raise ValueError("q must be eight numbers: x,y,x,y,x,y,x,y (tl, tr, br, bl)")
            corners = [v[0:2], v[2:4], v[4:6], v[6:8]]
        return q.get("of", ["print"])[0], q.get("kind", [None])[0], corners

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


class LocalServer(ThreadingHTTPServer):
    """A threading server that never shares its port: on Windows ``SO_REUSEADDR`` lets a second socket bind a
    port that is listening (two apps would answer on one port), so the port is bound exclusively there."""

    allow_reuse_address = sys.platform != "win32"   # elsewhere it only allows a restart during TIME_WAIT

    def server_bind(self) -> None:
        if sys.platform == "win32":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def make_server(app: ReviewApp, port: int = 0) -> ThreadingHTTPServer:
    """HTTP server for ``app`` on 127.0.0.1 (port 0 picks a free one). Not started."""
    handler = type("BoundHandler", (Handler,), {"app": app})
    srv = LocalServer(("127.0.0.1", port), handler)
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
