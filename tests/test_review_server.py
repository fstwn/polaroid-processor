"""The review page's HTTP API, driven like the browser does (urllib against a real server thread)."""
import io
import json
import shutil
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
import tifffile
from PIL import Image

from polaroid_processor.cli import main
from polaroid_processor.manifest import Manifest
from polaroid_processor.review import ReviewApp, make_server

from synth import PX_PER_MM, make_scan

PPI = PX_PER_MM * 25.4
CORNERS = ("tl", "tr", "br", "bl")


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    base = tmp_path_factory.mktemp("review")
    src = base / "in"
    src.mkdir()
    for i, frame in enumerate(["white", "gold"]):
        rgb, _ = make_scan(frame, 0.3 * i, seed=40 + i)
        tifffile.imwrite(str(src / f"r_{frame}.tif"), rgb, photometric="rgb")
    out = base / "out"
    assert main(["process", str(src), "-o", str(out), "--ppi", str(PPI), "--no-derivatives", "-j", "1", "-q"]) == 0
    return src, out


class Page:
    """A running review server plus a tiny client."""

    def __init__(self, out, **kw):
        self.work = out
        self.app = ReviewApp(str(out), **kw)
        self.srv = make_server(self.app, 0)
        self.port = self.srv.server_address[1]
        self.t = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.t.start()

    def request(self, path, body=None, headers=None):
        h = {"Host": f"127.0.0.1:{self.port}", **(headers or {})}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            h.setdefault("Content-Type", "application/json")
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data, headers=h)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()

    def json(self, path, body=None, **kw):
        status, _, raw = self.request(path, body, **kw)
        return status, json.loads(raw)

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.t.join(timeout=5)


def copy_out(out, dest):
    shutil.copytree(out, dest)
    return dest


@pytest.fixture
def page(processed, tmp_path):
    _, out = processed
    p = Page(copy_out(out, tmp_path / "out"), show_all=True)   # each test edits its own copy
    yield p
    p.close()


def test_page_and_list(page):
    status, headers, raw = page.request("/")
    assert status == 200 and headers["Content-Type"].startswith("text/html") and b"Polaroid review" in raw
    status, data = page.json("/api/scans")
    assert status == 200 and data["author"] == "human" and data["tile_px"] == 480
    assert [s["key"] for s in data["scans"]] == ["r_gold.tif", "r_white.tif"]
    for s in data["scans"]:
        assert len(s["quad"]) == 4 and s["status"] in ("ok", "needs_review")
        assert isinstance(s["warnings"], list) and s["quad_source"] == "auto" and s["review"] is None


def test_list_defaults_to_needs_review(processed):
    _, out = processed
    flagged = sorted(k for k, e in Manifest(str(out)).items() if e.get("status") == "needs_review")
    assert [s["key"] for s in ReviewApp(str(out)).list_scans()["scans"]] == flagged
    assert len(ReviewApp(str(out), show_all=True).list_scans()["scans"]) == 2


def test_tiles(page):
    status, specs = page.json("/api/scans/r_white.tif/tiles")
    assert status == 200 and [t["name"] for t in specs["tiles"]] == [
        "tl", "top", "tr", "left", "overview", "right", "bl", "bottom", "br"]
    quad = np.array([s for s in page.json("/api/scans")[1]["scans"] if s["key"] == "r_white.tif"][0]["quad"])
    for t in specs["tiles"]:
        status, headers, raw = page.request(t["url"])
        assert status == 200 and headers["Content-Type"] == "image/png"
        with Image.open(io.BytesIO(raw)) as im:
            assert list(im.size) == t["size"]
        if t["name"] != "overview":
            assert t["window"] == [480, 480] and t["size"] == [480, 480]
    tl = specs["tiles"][0]    # the tl corner sits in the middle of its tile
    assert abs(tl["x0"] + 240 - quad[0][0]) <= 1 and abs(tl["y0"] + 240 - quad[0][1]) <= 1
    assert page.request("/api/scans/nope.tif/tiles")[0] == 404
    assert page.request("/api/scans/r_white.tif/tile/xx.png")[0] == 404


def test_save_correction(page, processed):
    src, _ = processed
    quad = np.array([s for s in page.json("/api/scans")[1]["scans"] if s["key"] == "r_white.tif"][0]["quad"])
    _, specs0 = page.json("/api/scans/r_white.tif/tiles")
    moved = quad + [[-30.0, 20.0], [0, 0], [0, 0], [0, 0]]
    corners = {n: [float(x), float(y)] for n, (x, y) in zip(CORNERS, moved)}
    status, rec = page.json("/api/scans/r_white.tif/save", {"corners": corners})
    assert status == 200 and rec["quad_source"] == "override" and not rec["exported"]
    assert np.allclose(rec["quad"], moved, atol=0.01)
    assert rec["review"]["action"] == "fix" and rec["review"]["author"] == "human"
    # history and override are in the manifest, the other scan is untouched
    m = Manifest(str(page.work))
    e = m.get("r_white.tif")
    assert e["override"]["author"] == "human" and e["history"][-1]["action"] == "fix"
    assert np.allclose(e["history"][-1]["quad_before"], quad, atol=0.01)
    assert m.get("r_gold.tif").get("override") is None
    # the tiles are re-cut at the new outline
    _, specs1 = page.json("/api/scans/r_white.tif/tiles")
    assert specs1["rev"] == specs0["rev"] + 1
    assert specs1["tiles"][0]["x0"] == specs0["tiles"][0]["x0"] - 30
    assert specs1["tiles"][0]["y0"] == specs0["tiles"][0]["y0"] + 20
    assert specs1["tiles"][0]["url"] != specs0["tiles"][0]["url"]
    assert page.request(specs1["tiles"][0]["url"])[0] == 200
    # the input folder was only read
    assert sorted(p.name for p in src.iterdir()) == ["r_gold.tif", "r_white.tif"]


def test_save_rejects_bad_corners(page):
    ok = {n: [1, 2] for n in CORNERS}
    for bad in ({"tl": [1, 2]}, {**ok, "bl": ["a", 2]}, {**ok, "x": [1, 2]}, {**ok, "bl": [1]}, [1, 2]):
        status, data = page.json("/api/scans/r_white.tif/save", {"corners": bad})
        assert status == 400 and "error" in data, bad
    status, _ = page.json("/api/scans/r_white.tif/save", {"corners": {**ok, "tl": [float("nan"), 0]}})
    assert status == 400
    assert Manifest(str(page.work)).get("r_white.tif").get("override") is None


def test_confirm(page):
    status, rec = page.json("/api/scans/r_gold.tif/confirm", {})
    assert status == 200 and rec["quad_source"] == "override"
    assert rec["review"]["action"] == "accept" and rec["review"]["author"] == "human"
    e = Manifest(str(page.work)).get("r_gold.tif")
    assert e["history"][-1]["action"] == "accept" and e["override"]["author"] == "human"


def test_author_and_export(processed, tmp_path):
    _, out = processed
    p = Page(copy_out(out, tmp_path / "out"), show_all=True, author="alex", export=True)
    try:
        quad = np.array(p.json("/api/scans")[1]["scans"][1]["quad"])
        corners = {n: [float(x), float(y)] for n, (x, y) in zip(CORNERS, quad + [0, 6.0])}
        status, rec = p.json("/api/scans/r_white.tif/save", {"corners": corners})
        assert status == 200 and rec["exported"] and rec["review"]["author"] == "alex"
        assert Manifest(str(p.work)).get("r_white.tif")["history"][-1]["author"] == "alex"
    finally:
        p.close()


def test_rejects_foreign_host_and_origin(page):
    assert page.request("/api/scans", headers={"Host": "evil.example"})[0] == 403
    assert page.request("/api/scans", headers={"Origin": "http://evil.example"})[0] == 403
    status, _, _ = page.request("/api/scans/r_white.tif/confirm", {}, headers={"Content-Type": "text/plain"})
    assert status == 400


def test_cli_without_flagged_scans_does_not_serve(processed, capsys, tmp_path):
    _, out = processed
    work = copy_out(out, tmp_path / "out")
    m = Manifest(str(work))
    for e in m.scans.values():
        e["status"] = "ok"
    m.save()
    assert main(["review", str(work), "--no-browser"]) == 0
    assert "no scans need review" in capsys.readouterr().out
    assert main(["review", str(tmp_path / "missing")]) == 2
