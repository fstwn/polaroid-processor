"""Hand crop, slice 2: the fix page's script run for real in Node (``tests/js/review_harness.js``): a scan nothing was found
on opens at the place step, what a save sends, a refused save loses nothing, the item choice, reset, live updates and the
unsaved-changes question. The geometry is tested on its own in ``test_crop_page.py``. Skipped without ``node`` on the PATH."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not on the PATH")

HARNESS = Path(__file__).parent / "js" / "review_harness.js"
PAGE = Path(__file__).parent.parent / "src" / "polaroid_processor" / "review" / "static" / "index.html"

A600, AGO = 88 / 106.85, 53.5 / 65.8
NAMES = ["tl", "top", "tr", "left", "overview", "right", "bl", "bottom", "br"]
START = {"tl": [160.0, 200.0], "tr": [1440.0, 200.0], "br": [1440.0, 1800.0], "bl": [160.0, 1800.0]}
START600 = {"tl": [300.0, 250.0], "tr": [1300.0, 250.0], "br": [1300.0, 1450.0], "bl": [300.0, 1450.0]}
QUAD = [[100, 100], [900, 100], [900, 1100], [100, 1100]]


def run(spec):
    p = subprocess.run([NODE, str(HARNESS), str(PAGE), json.dumps(spec)], capture_output=True, text=True, timeout=60, encoding="utf-8")
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


def failed(key="ticket.tif", **kw):
    r = {"key": key, "status": "failed", "warnings": [], "quad": None, "quad_source": None, "size_px": None, "scan_px": [1600, 2000],
         "rev": 0, "review": None, "detected": False, "error": "detection failed: no plausible print outline at any threshold"}
    r.update(kw)
    return r


def detected(key="p0.tif", **kw):
    r = {"key": key, "status": "ok", "warnings": [], "quad": QUAD, "quad_source": "auto", "size_px": [800, 1000], "scan_px": [1600, 2000],
         "rev": 0, "review": None, "detected": True}
    r.update(kw)
    return r


def listing(*scans):
    return {"output": "/o", "all": True, "author": "human", "export": False, "tile_px": 480, "aspects": {"600": A600, "go": AGO},
            "limits": {"min_side_px": 50, "max_rotation_deg": 45.0, "max_scan_ppi": 50000.0}, "scans": list(scans)}


def specs(key, full=True):
    names = NAMES if full else ["overview"]
    return {"key": key, "rev": 0, "scan_px": [1600, 2000],
            "tiles": [{"name": n, "x0": 0, "y0": 0, "window": [480, 480] if n != "overview" else [1600, 2000], "size": [480, 480],
                       "url": f"/api/scans/{key}/tile/{n}.png?rev=0"} for n in names]}


def start(kind="other", corners=START):
    return {"item": kind, "scan_px": [1600, 2000], "px_per_mm": None, "corners": corners, "size_px": [1280, 1600],
            "aspect": None if kind == "other" else (A600 if kind == "600" else AGO)}


def routes(*scans, extra=None):
    r = {"GET /api/scans": {"body": listing(*scans)}, "GET /api/stamp": {"body": {"stamp": "1"}}}
    for s in scans:
        r[f"GET /api/scans/{s['key']}/tiles"] = {"body": specs(s["key"], full=s["quad"] is not None)}
        if s["quad"] is None:
            for kind, corners in (("other", START), ("600", START600), ("go", START600)):
                r[f"GET /api/scans/{s['key']}/start?item={kind}"] = {"body": start(kind, corners)}
    r.update(extra or {})
    return r


def posts(out):
    return [c for c in out["calls"] if c["method"] == "POST"]


STATE = """return {step: S.cur.step, quad: S.cur.quad, item: S.cur.item, dirty: isDirty(), touched: S.cur.touched,
  save: $("save").disabled, confirm: $("confirm").disabled, fine: $("st-fine").disabled, place: $("st-place").disabled,
  options: [...$("item").children].map(o => o.value), itemValue: $("item").value, ppiHidden: $("ppirow").hidden,
  rotHidden: $("rotrow").hidden, placeHidden: $("place").hidden, gridOff: $("grid").style.display, state: $("cropstate").textContent,
  hint: $("hint").hidden ? "" : $("hint").textContent, info: $("rectinfo").textContent, msg: $("msg").textContent, name: $("name").textContent};"""


# --------------------------------------------------------------------------- opening
def test_a_scan_nothing_was_found_on_opens_at_the_place_step_with_the_start_rectangle():
    out = run({"routes": routes(failed()), "hash": "ticket.tif", "steps": [STATE]})
    s = out["results"][0]
    assert s["name"] == "ticket.tif" and s["step"] == "place" and s["placeHidden"] is False and s["gridOff"] == "none"
    assert s["quad"] == [START[n] for n in ("tl", "tr", "br", "bl")]
    assert s["item"] == {"kind": "other", "ppi": None} and s["itemValue"] == "other" and s["options"] == ["600", "go", "other"]
    assert s["dirty"] is False and s["touched"] is False                      # browsing past it asks nothing
    assert s["save"] is False and s["confirm"] is True and s["fine"] is False   # the start rectangle can be saved as it is; nothing to confirm
    assert s["ppiHidden"] is False and s["rotHidden"] is False and "No outline was found" in s["state"] and "detection failed" in s["state"]
    assert s["hint"] == "" and "1280 × 1600" in s["info"]               # no proportions hint for an untouched start rectangle
    paths = [c["path"] for c in out["calls"]]
    assert "/api/scans/ticket.tif/tiles" in paths and "/api/scans/ticket.tif/start?item=other" in paths


def test_a_detected_scan_opens_at_the_fine_tune_step_and_offers_as_detected():
    out = run({"routes": routes(detected()), "hash": "p0.tif", "steps": [STATE]})
    s = out["results"][0]
    assert s["step"] == "fine" and s["placeHidden"] is True and s["gridOff"] == "" and s["rotHidden"] is True
    assert s["options"] == ["auto", "600", "go", "other"] and s["itemValue"] == "auto" and s["ppiHidden"] is True
    assert s["dirty"] is False and s["save"] is True and s["confirm"] is False
    assert not any("/start" in c["path"] for c in out["calls"])


def test_a_scan_with_a_crop_and_an_item_does_not_offer_as_detected_and_shows_its_ppi():
    r = detected("t.tif", quad_source="override", detected=False, item={"kind": "other", "scan_ppi": 800.0, "agent_crop": True}, agent_crop=True,
                 status="needs_review", warnings=["cropped by agent"])
    out = run({"routes": routes(r), "hash": "t.tif", "steps": [STATE, 'return $("ppi").value;']})
    s = out["results"][0]
    assert s["options"] == ["600", "go", "other"] and s["itemValue"] == "other" and s["ppiHidden"] is False and out["results"][1] == 800
    assert "Cropped by an agent" in s["state"] and s["dirty"] is False and s["confirm"] is False


# --------------------------------------------------------------------------- saving
SAVE_OK = {"POST /api/scans/ticket.tif/save": {"body": failed(status="ok", quad=QUAD, quad_source="override", detected=False, error=None,
                                                          item={"kind": "other", "scan_ppi": 800.0, "agent_crop": False})}}


def test_an_untouched_start_rectangle_is_saved_as_a_first_crop_named_other():
    out = run({"routes": routes(failed(), extra=SAVE_OK), "hash": "ticket.tif", "embed": True,
               "steps": ['$("save").click(); await new Promise(r => setTimeout(r, 30)); return $("msg").textContent;']})
    (post,) = posts(out)
    assert post["path"] == "/api/scans/ticket.tif/save"
    assert post["body"] == {"corners": {"tl": [160, 200], "tr": [1440, 200], "br": [1440, 1800], "bl": [160, 1800]}, "item": "other"}
    assert out["results"][0].startswith("Saved ticket.tif") and out["posted"] == [{"type": "fix-saved", "key": "ticket.tif"}]


def test_a_placed_rotated_rectangle_goes_with_the_item_and_the_own_ppi():
    steps = ["""
      const g = geom(); setRect(rotateRect(currentRect(), 6.4, {max: g.max, W: g.W, H: g.H}));
      $("ppi").value = "800"; $("ppi").fire("input");
      $("save").click(); await new Promise(r => setTimeout(r, 30));
      return {dirtyAfter: isDirty(), msg: $("msg").textContent};"""]
    out = run({"routes": routes(failed(), extra=SAVE_OK), "hash": "ticket.tif", "embed": True, "steps": steps})
    (post,) = posts(out)
    c = post["body"]["corners"]
    assert post["body"]["item"] == "other" and post["body"]["scan_ppi"] == 800
    assert c["tr"][1] > c["tl"][1] and c["bl"][0] < c["tl"][0]                # turned clockwise by 6.4 deg about the centre
    assert out["results"][0]["dirtyAfter"] is False                            # saved: nothing unsaved is left


def test_a_refused_save_shows_the_message_and_loses_no_edit():
    refused = {"POST /api/scans/ticket.tif/save": [{"status": 400, "body": {"error": "crop must be convex: the corner br bends the wrong way"}},
                                                  SAVE_OK["POST /api/scans/ticket.tif/save"]]}
    steps = ["""
      S.cur.quad = [[100,100],[1000,100],[500,300],[100,1000]]; S.cur.touched = true; S.cur.item.kind = "600"; renderSide(); updateButtons();
      $("save").click(); await new Promise(r => setTimeout(r, 30));
      return {msg: $("msg").textContent, info: $("msg").className, quad: S.cur.quad, item: S.cur.item, dirty: isDirty(), busy: S.busy, save: $("save").disabled};""",
             """$("save").click(); await new Promise(r => setTimeout(r, 30)); return $("msg").textContent;"""]
    out = run({"routes": routes(failed(), extra=refused), "hash": "ticket.tif", "embed": True, "steps": steps})
    first = out["results"][0]
    assert first["msg"] == "Not saved: crop must be convex: the corner br bends the wrong way" and first["info"] == ""
    assert first["quad"] == [[100, 100], [1000, 100], [500, 300], [100, 1000]] and first["item"]["kind"] == "600"
    assert first["dirty"] is True and first["busy"] is False and first["save"] is False
    assert len(posts(out)) == 2 and posts(out)[0]["body"] == posts(out)[1]["body"] and posts(out)[0]["body"]["item"] == "600"
    assert out["results"][1].startswith("Saved ticket.tif")


def test_an_item_alone_is_saved_on_a_detected_scan_and_only_the_changed_part_is_sent():
    ok = {"POST /api/scans/p0.tif/save": {"body": detected(item={"kind": "go", "agent_crop": False})}}
    steps = ["""const sel = $("item"); sel.value = "go"; sel.fire("change"); await new Promise(r => setTimeout(r, 30));
      const r = {dirty: isDirty(), save: $("save").disabled, confirm: $("confirm").disabled, ppiHidden: $("ppirow").hidden};
      $("save").click(); await new Promise(r => setTimeout(r, 30)); return r;"""]
    out = run({"routes": routes(detected(), extra=ok), "hash": "p0.tif", "embed": True, "steps": steps})
    assert out["results"][0] == {"dirty": True, "save": False, "confirm": True, "ppiHidden": True}
    (post,) = posts(out)
    assert post["body"] == {"item": "go"}


def test_an_item_other_sends_no_window_and_takes_the_photo_window_box_away():
    win = {"state": "found", "source": "auto", "reason": "", "kind": "square", "quad": [[120, 120], [880, 120], [880, 900], [120, 900]],
           "accepted": False, "outside": False, "suggest": "square"}
    r = detected(window=win)
    steps = ["""const was = {box: $("winbox").hidden, has: hasWin()};
      S.cur.win.quad[0] = [130, 130];                          // an edit of the window
      const dirtyWin = winDirty();
      $("item").value = "other"; $("item").fire("change"); await new Promise(r => setTimeout(r, 30));
      return {was, dirtyWin, hasWin: hasWin(), winDirty: winDirty(), box: $("winbox").hidden};"""]
    out = run({"routes": routes(r), "hash": "p0.tif", "steps": steps})
    res = out["results"][0]
    assert res["was"] == {"box": False, "has": True} and res["dirtyWin"] is True
    assert res["hasWin"] is False and res["winDirty"] is False and res["box"] is True


# --------------------------------------------------------------------------- the item choice on a scan nothing was found on
def test_the_item_choice_restarts_an_untouched_rectangle_but_never_a_touched_one():
    steps = ["""$("item").value = "600"; $("item").fire("change"); await new Promise(r => setTimeout(r, 30));
      const a = {quad: S.cur.quad, item: S.cur.item.kind, dirty: isDirty(), aspect: aspectOf(S.cur.item.kind)};
      const g = geom(); setRect(moveRect(currentRect(), 20, 10, g.W, g.H));
      $("item").value = "go"; $("item").fire("change"); await new Promise(r => setTimeout(r, 30));
      return {a, b: {quad: S.cur.quad, item: S.cur.item.kind, touched: S.cur.touched}};"""]
    out = run({"routes": routes(failed()), "hash": "ticket.tif", "steps": steps})
    a, b = out["results"][0]["a"], out["results"][0]["b"]
    assert a["quad"] == [START600[n] for n in ("tl", "tr", "br", "bl")] and a["item"] == "600" and a["dirty"] is True and a["aspect"] == pytest.approx(A600)
    # a touched rectangle takes the format's proportions at once (decision 4 after the fifth review): same centre, same area
    q = b["quad"]
    w, h = q[1][0] - q[0][0], q[3][1] - q[0][1]
    assert b["item"] == "go" and b["touched"] is True and w / h == pytest.approx(AGO, rel=1e-4)
    assert ((q[0][0] + q[2][0]) / 2, (q[0][1] + q[2][1]) / 2) == pytest.approx((820.0, 860.0), abs=1e-6)      # the 600 start rectangle's centre (800, 850), moved by (20, 10)
    assert w * h == pytest.approx(1000 * 1200, rel=1e-3)                                                     # the same area
    assert [c["path"] for c in out["calls"] if "/start" in c["path"]] == ["/api/scans/ticket.tif/start?item=other", "/api/scans/ticket.tif/start?item=600"]


def test_the_proportions_hint_shows_for_a_placed_other_crop_and_changes_nothing():
    steps = ["""const g = geom(); const r = rectFromQuad(S.cur.quad);
      setRect({cx: 800, cy: 1000, w: 880, h: 1068.5, angle: 0});            // the proportions of a 600
      const hint = {text: $("hint").hidden ? "" : $("hint").textContent, kind: S.cur.item.kind, quad: S.cur.quad};
      setRect({cx: 800, cy: 1000, w: 900, h: 1500, angle: 0});
      const none = $("hint").hidden;
      $("item").value = "600"; $("item").fire("change"); await new Promise(r => setTimeout(r, 30));
      return {hint, none, hiddenWith600: $("hint").hidden};"""]
    out = run({"routes": routes(failed()), "hash": "ticket.tif", "steps": steps})
    res = out["results"][0]
    assert "proportions of a 600 or a Go print" in res["hint"]["text"] and "choose 600 or Go" in res["hint"]["text"]
    assert res["hint"]["kind"] == "other" and res["none"] is True
    assert res["hiddenWith600"] is True


def test_reset_throws_away_the_rectangle_the_item_and_the_ppi():
    steps = ["""const g = geom(); setRect(moveRect(currentRect(), 100, 100, g.W, g.H));
      $("item").value = "600"; $("item").fire("change"); await new Promise(r => setTimeout(r, 30));
      $("ppi").value = "500"; $("ppi").fire("input");
      const dirty = isDirty(); $("reset").click(); await new Promise(r => setTimeout(r, 30));
      return {dirty, after: {dirty: isDirty(), item: S.cur.item, quad: S.cur.quad, touched: S.cur.touched, value: $("item").value}};"""]
    out = run({"routes": routes(failed()), "hash": "ticket.tif", "steps": steps})
    res = out["results"][0]
    assert res["dirty"] is True and res["after"]["dirty"] is False and res["after"]["item"] == {"kind": "other", "ppi": None}
    assert res["after"]["quad"] == [START[n] for n in ("tl", "tr", "br", "bl")] and res["after"]["touched"] is False and res["after"]["value"] == "other"


# --------------------------------------------------------------------------- the two steps
def test_fine_tune_cuts_its_tiles_at_the_rectangle_the_place_step_made():
    steps = ["""const g = geom(); setRect(moveRect(currentRect(), 100, 50, g.W, g.H));
      await setStep("fine"); return {step: S.cur.step, tiles: S.cur.specs.tiles.length, fine: $("place").hidden, grid: $("grid").style.display, quad: S.cur.quad};""",
             """await setStep("place"); const back = S.cur.step; await setStep("fine"); return back;"""]
    q = "260.00,250.00,1540.00,250.00,1540.00,1850.00,260.00,1850.00"
    extra = {f"GET /api/scans/ticket.tif/tiles?q={q}": {"body": specs("ticket.tif")}}
    out = run({"routes": routes(failed(), extra=extra), "hash": "ticket.tif", "steps": steps})
    res = out["results"][0]
    assert res == {"step": "fine", "tiles": 9, "fine": True, "grid": "", "quad": [[260, 250], [1540, 250], [1540, 1850], [260, 1850]]}
    tile_calls = [c["path"] for c in out["calls"] if "/tiles" in c["path"]]
    assert tile_calls == ["/api/scans/ticket.tif/tiles", f"/api/scans/ticket.tif/tiles?q={q}"]      # no new request when the corners did not move


def test_a_saved_scan_goes_to_the_fine_step_and_the_place_step_is_offered_on_a_detected_scan():
    steps = ["""await setStep("place"); const a = {step: S.cur.step, rot: $("rot").value, info: $("rectinfo").textContent, hidden: $("place").hidden};
      await setStep("fine"); return {a, step: S.cur.step};"""]
    out = run({"routes": routes(detected()), "hash": "p0.tif", "steps": steps})
    res = out["results"][0]
    assert res["a"]["step"] == "place" and res["a"]["rot"] == "0.0" and res["a"]["info"] == "800 × 1000 px" and res["a"]["hidden"] is False
    assert res["step"] == "fine" and not any("tiles?q=" in c["path"] for c in out["calls"])


# --------------------------------------------------------------------------- unsaved changes, next/previous, live updates
def test_leaving_a_touched_scan_asks_and_an_untouched_one_does_not():
    steps = ["""$("next").click(); await new Promise(r => setTimeout(r, 30)); const untouched = {dlg: $("discard").style.display || "", idx: S.idx};
      return untouched;""",
             """const g = geom(); setRect(moveRect(currentRect(), 40, 40, g.W, g.H)); const p = go(0);
      await new Promise(r => setTimeout(r, 30)); const asked = $("discard").style.display; $("d-stay").click(); await p;
      return {asked, idx: S.idx, dirty: isDirty()};"""]
    out = run({"routes": routes(failed(), failed("sheet.tif")), "hash": "ticket.tif", "steps": steps})
    assert out["results"][0] == {"dlg": "", "idx": 1}                              # no question: nothing was touched
    assert out["results"][1] == {"asked": "flex", "idx": 1, "dirty": True}         # the question, and "keep editing" keeps them


def test_a_scan_cropped_elsewhere_is_taken_over_unless_it_has_unsaved_edits():
    after = failed(status="needs_review", quad=QUAD, quad_source="override", detected=False, error=None, rev=1, agent_crop=True,
                   item={"kind": "other", "agent_crop": True}, warnings=["cropped by agent"])
    steps = ["""await refreshScans(); await new Promise(r => setTimeout(r, 30));
      return {step: S.cur.step, status: $("status").textContent, note: $("msg").textContent, quad: S.cur.quad, state: $("cropstate").textContent};"""]
    r = routes(failed())
    r["GET /api/scans"] = [{"body": listing(failed())}, {"body": listing(after)}]
    r["GET /api/scans/ticket.tif/tiles"] = [{"body": specs("ticket.tif", full=False)}, {"body": specs("ticket.tif")}]
    out = run({"routes": r, "hash": "ticket.tif", "steps": steps})
    res = out["results"][0]
    assert res["step"] == "fine" and res["status"] == "needs_review" and res["quad"] == QUAD and "Cropped by an agent" in res["state"]
    assert res["note"] == "This scan was changed elsewhere; showing the new outline."
    # with edits in the page: kept, Reset takes the new one
    steps = ["""const g = geom(); setRect(moveRect(currentRect(), 40, 40, g.W, g.H)); const q = S.cur.quad;
      await refreshScans(); const kept = {quad: S.cur.quad, msg: $("msg").textContent, same: JSON.stringify(q) === JSON.stringify(S.cur.quad)};
      $("reset").click(); await new Promise(r => setTimeout(r, 40)); return {kept, step: S.cur.step, quad: S.cur.quad};"""]
    out = run({"routes": r, "hash": "ticket.tif", "steps": steps})
    res = out["results"][0]
    assert res["kept"]["same"] is True and res["kept"]["msg"].startswith("This scan was changed elsewhere. Reset takes over")
    assert res["step"] == "fine" and res["quad"] == QUAD


def test_the_side_list_marks_failed_scans_items_and_agent_crops():
    a = detected("a.tif", item={"kind": "go", "agent_crop": False})
    b = detected("b.tif", quad_source="override", detected=False, item={"kind": "other", "agent_crop": True}, agent_crop=True)
    out = run({"routes": routes(a, b, failed("c.tif")), "hash": "a.tif", "steps": [
        """return [...$("list").children].map(d => d.children.map(c => c.textContent));"""]})
    rows = out["results"][0]
    assert rows[0][1:] == ["a.tif", "Go"] and rows[1][1:] == ["b.tif", "Other", "by agent"] and rows[2][1:] == ["c.tif", "no outline"]


def test_a_failed_scan_with_a_narrow_window_fits_the_canvas_and_keeps_the_pad():
    out = run({"routes": routes(failed()), "hash": "ticket.tif", "narrow": True, "steps": [
        """const cv = $("pcanvas"); return {w: cv.style.width, h: cv.style.height, h0: $("place").style.height};"""]})
    r = out["results"][0]
    assert r["h0"] == "auto" and int(r["w"][:-2]) > 2 * 38 and int(r["h"][:-2]) > 2 * 38


# =========================================================================== after the fifth review (R29 to R33, decisions 1, 4, 5)
def hand_crop(key="t.tif", item=None, **kw):
    """A scan nothing was detected on that has a stored hand crop (an item Other by default)."""
    r = detected(key, quad_source="override", detected=False, item=item or {"kind": "other", "agent_crop": False}, **kw)
    r["quad"] = [[300, 400], [1100, 400], [1100, 1400], [300, 1400]]
    return r


# R29: the own ppi lives in the editor, not in the kind
def test_the_own_ppi_survives_a_trip_through_600_and_back_and_a_save_with_600_sends_none():
    r = detected("p0.tif", item={"kind": "other", "scan_ppi": 800.0, "agent_crop": False})
    steps = ["""$("item").value = "600"; $("item").fire("change"); await new Promise(r => setTimeout(r, 20));
      const as600 = {ppi: S.cur.item.ppi, dirty: isDirty(), hidden: $("ppirow").hidden};
      $("item").value = "other"; $("item").fire("change"); await new Promise(r => setTimeout(r, 20));
      return {as600, back: {ppi: S.cur.item.ppi, dirty: isDirty(), field: $("ppi").value, hidden: $("ppirow").hidden}};""",
             """$("ppi").value = "950"; $("ppi").fire("input"); $("item").value = "go"; $("item").fire("change"); await new Promise(r => setTimeout(r, 20));
      const typed = S.cur.item.ppi; $("item").value = "other"; $("item").fire("change"); await new Promise(r => setTimeout(r, 20));
      return {typed, kept: S.cur.item.ppi, field: $("ppi").value};"""]
    out = run({"routes": routes(r), "hash": "p0.tif", "steps": steps})
    a, b = out["results"]
    assert a["as600"] == {"ppi": 800, "dirty": True, "hidden": True}
    assert a["back"] == {"ppi": 800, "dirty": False, "field": 800, "hidden": False}      # back at the stored choice: nothing unsaved, no scan_ppi 0
    assert b == {"typed": 950, "kept": 950, "field": 950}                                  # a typed ppi is not lost either
    ok = {"POST /api/scans/p0.tif/save": {"body": detected("p0.tif", item={"kind": "go", "agent_crop": False})}}
    out = run({"routes": routes(r, extra=ok), "hash": "p0.tif", "embed": True, "steps": [
        """$("ppi").value = "950"; $("ppi").fire("input"); $("item").value = "go"; $("item").fire("change"); await new Promise(r => setTimeout(r, 20));
      $("save").click(); await new Promise(r => setTimeout(r, 30));"""]})
    (post,) = posts(out)
    assert post["body"] == {"item": "go"}                                                      # 600 / Go: no ppi (the core drops a stored one)


# R30: only an empty field means none
def test_a_ppi_that_is_no_positive_number_blocks_save_and_only_an_empty_field_removes_it():
    r = detected("p0.tif", item={"kind": "other", "scan_ppi": 800.0, "agent_crop": False})
    ok = {"POST /api/scans/p0.tif/save": {"body": detected("p0.tif", item={"kind": "other", "agent_crop": False})}}
    steps = ["""const res = {};
      for (const t of ["-3", "0", "1e9"]) { $("ppi").value = t; $("ppi").fire("input");
        res[t] = {save: $("save").disabled, err: $("ppierr").textContent, bad: $("ppi").classList.contains("bad"), title: $("save").title}; }
      $("ppi").value = ""; $("ppi").validity.badInput = true; $("ppi").fire("input");               // "abc": a number input reads it as ""
      res.abc = {save: $("save").disabled, err: $("ppierr").textContent};
      $("ppi").validity.badInput = false; $("ppi").value = "700"; $("ppi").fire("input");
      res.good = {save: $("save").disabled, err: $("ppierr").textContent, bad: $("ppi").classList.contains("bad")};
      $("ppi").value = ""; $("ppi").fire("input");
      res.empty = {save: $("save").disabled, err: $("ppierr").textContent, ppi: S.cur.item.ppi, dirty: isDirty()};
      $("save").click(); await new Promise(r => setTimeout(r, 30));
      return res;"""]
    out = run({"routes": routes(r, extra=ok), "hash": "p0.tif", "embed": True, "steps": steps})
    res = out["results"][0]
    for t in ("-3", "0", "1e9"):
        assert res[t]["save"] is True and "above 0" in res[t]["err"] and res[t]["bad"] is True and res[t]["title"] == res[t]["err"]
    assert res["abc"]["save"] is True and "above 0" in res["abc"]["err"]
    assert res["good"] == {"save": False, "err": "", "bad": False}
    assert res["empty"] == {"save": False, "err": "", "ppi": None, "dirty": True}               # empty = none: the stored ppi goes
    (post,) = posts(out)
    assert post["body"] == {"scan_ppi": 0}


def test_a_ppi_error_does_not_hold_up_an_item_that_is_not_other():
    r = detected("p0.tif", item={"kind": "other", "agent_crop": False})
    out = run({"routes": routes(r), "hash": "p0.tif", "steps": [
        """$("ppi").value = "-5"; $("ppi").fire("input"); $("item").value = "go"; $("item").fire("change"); await new Promise(r => setTimeout(r, 20));
      return {save: $("save").disabled, err: $("ppierr").textContent};"""]})
    assert out["results"][0] == {"save": False, "err": ""}


# R31: after a first crop the editor holds the stored record
def test_after_the_first_crop_is_saved_the_page_shows_a_stored_crop_and_a_second_save_stores_nothing():
    steps = ["""$("save").click(); await new Promise(r => setTimeout(r, 40));
      const a = {rec: S.cur.rec.quad !== null, confirm: $("confirm").disabled, save: $("save").disabled, state: $("cropstate").textContent,
                 dirty: isDirty(), item0: S.cur.item0, source: $("source").textContent};
      $("save").click(); await new Promise(r => setTimeout(r, 40));
      return a;"""]
    noppi = {"POST /api/scans/ticket.tif/save": {"body": failed(status="ok", quad=QUAD, quad_source="override", detected=False, error=None,
                                                             item={"kind": "other", "agent_crop": False})}}
    out = run({"routes": routes(failed(), extra=noppi), "hash": "ticket.tif", "embed": True, "steps": steps})
    a = out["results"][0]
    assert a["rec"] is True and a["confirm"] is False and a["save"] is True and "No outline was found" not in a["state"]
    assert a["dirty"] is False and a["item0"] == {"kind": "other", "ppi": None} and a["source"] == "outline: manual"
    assert len(posts(out)) == 1                                                                # no second fix


# R32: a drag while the start rectangle loads wins
def test_a_drag_while_the_start_rectangle_loads_is_not_overwritten_and_gets_the_formats_proportions():
    steps = ["""const g = geom(); const p = setItem("600");                       // the start rectangle is on its way
      setRect(moveRect(currentRect(), 50, 30, g.W, g.H)); await p;
      const q = S.cur.quad, w = q[1][0] - q[0][0], h = q[3][1] - q[0][1];
      return {touched: S.cur.touched, kind: S.cur.item.kind, ratio: w / h, cx: (q[0][0] + q[2][0]) / 2, cy: (q[0][1] + q[2][1]) / 2};"""]
    out = run({"routes": routes(failed()), "hash": "ticket.tif", "steps": steps})
    r = out["results"][0]
    assert r["touched"] is True and r["kind"] == "600" and r["ratio"] == pytest.approx(A600, rel=1e-4)
    assert (r["cx"], r["cy"]) == pytest.approx((850.0, 1030.0), abs=1e-6)                     # where it was dragged, not the start rectangle's centre
    # another choice made meanwhile also wins
    steps = ["""const p = setItem("600"); await setItem("other"); await p; return {kind: S.cur.item.kind, quad: S.cur.quad};"""]
    out = run({"routes": routes(failed()), "hash": "ticket.tif", "steps": steps})
    assert out["results"][0]["kind"] == "other" and out["results"][0]["quad"] == [START[n] for n in ("tl", "tr", "br", "bl")]


# R33: no row click during a save, and the answer goes to its own scan
def test_a_save_in_flight_blocks_a_move_to_another_scan_and_its_answer_lands_on_its_own_scan():
    ok = {"POST /api/scans/p0.tif/save": {"body": detected("p0.tif", item={"kind": "go", "agent_crop": False}, rev=1)}}
    steps = ["""$("item").value = "go"; $("item").fire("change"); await new Promise(r => setTimeout(r, 20));
      const p = commit("save"); await go(1); const during = {idx: S.idx, key: S.cur.rec.key}; await p; await new Promise(r => setTimeout(r, 40));
      return {during, idx: S.idx, marks: S.marks, rows: S.scans.map(s => [s.key, s.item ? s.item.kind : null]), cur: S.cur.rec.key, busy: S.busy};"""]
    out = run({"routes": routes(detected("p0.tif"), detected("p1.tif"), extra=ok), "hash": "p0.tif", "steps": steps})
    r = out["results"][0]
    assert r["during"] == {"idx": 0, "key": "p0.tif"} and r["idx"] == 0 and r["cur"] == "p0.tif" and r["busy"] is False
    assert r["marks"] == {"p0.tif": "fixed"} and r["rows"] == [["p0.tif", "go"], ["p1.tif", None]]


def test_a_save_that_comes_back_after_the_page_moved_on_marks_only_its_own_scan():
    ok = {"POST /api/scans/p0.tif/save": {"body": detected("p0.tif", item={"kind": "go", "agent_crop": False}, rev=1)}}
    steps = ["""$("item").value = "go"; $("item").fire("change"); await new Promise(r => setTimeout(r, 20));
      const cur = S.cur, rec = JSON.parse(JSON.stringify(cur.rec)); rec.item = {kind: "go", agent_crop: false}; rec.rev = 1;
      S.busy = true; S.idx = 1; S.cur = null;                               // as if another scan had been opened meanwhile
      saved(cur, rec, "fixed");
      return {marks: S.marks, rows: S.scans.map(s => [s.key, s.item ? s.item.kind : null]), busy: S.busy, idx: S.idx};"""]
    out = run({"routes": routes(detected("p0.tif"), detected("p1.tif"), extra=ok), "hash": "p0.tif", "steps": steps})
    assert out["results"][0] == {"marks": {"p0.tif": "fixed"}, "rows": [["p0.tif", "go"], ["p1.tif", None]], "busy": False, "idx": 1}


# decision 5: a touched rectangle takes the format's proportions at once
SHAPE = """const shapeOf = () => { const r = rectFromQuad(S.cur.quad); return {ratio: r.w / r.h, area: r.w * r.h, cx: r.cx, cy: r.cy, angle: r.angle,
  touched: S.cur.touched, dirty: isDirty()}; };"""
CHOOSE = """const choose = async (k) => { $("item").value = k; $("item").fire("change"); await new Promise(r => setTimeout(r, 20)); };"""


def test_choosing_600_or_go_reshapes_a_touched_rectangle_a_stored_hand_crop_but_not_a_detected_outline_or_a_non_rectangle():
    # a failed scan, touched, turned: same centre and angle, the format's proportions, about the same area
    steps = [SHAPE + CHOOSE + """setRect({cx: 700, cy: 900, w: 900, h: 700, angle: 12}); await choose("go"); const a = shapeOf();
      await choose("other"); const back = shapeOf(); return {a, backRatio: back.ratio};""",
             """$("reset").click(); await new Promise(r => setTimeout(r, 40)); return {quad: S.cur.quad, dirty: isDirty(), kind: S.cur.item.kind};"""]
    out = run({"routes": routes(failed()), "hash": "ticket.tif", "steps": steps})
    a = out["results"][0]["a"]
    assert a["ratio"] == pytest.approx(1 / AGO, rel=1e-4) and a["area"] == pytest.approx(900 * 700, rel=1e-3)
    assert (a["cx"], a["cy"], a["angle"]) == pytest.approx((700, 900, 12), abs=1e-6) and a["touched"] is True
    assert out["results"][0]["backRatio"] == pytest.approx(1 / AGO, rel=1e-4)                  # back to Other does not undo it
    assert out["results"][1]["quad"] == [START[n] for n in ("tl", "tr", "br", "bl")] and out["results"][1]["dirty"] is False   # Reset does
    # a stored hand crop on a failed scan (not touched, not detected): reshaped, and then unsaved
    out = run({"routes": routes(hand_crop()), "hash": "t.tif", "steps": [SHAPE + CHOOSE + 'await choose("600"); return shapeOf();']})
    a = out["results"][0]
    assert a["ratio"] == pytest.approx(A600, rel=1e-4) and a["dirty"] is True and a["area"] == pytest.approx(800 * 1000, rel=1e-3)
    assert (a["cx"], a["cy"]) == pytest.approx((700, 900), abs=1e-6)
    # a detected outline is not changed by an item choice, neither is a quad that is no rectangle
    out = run({"routes": routes(detected("p0.tif")), "hash": "p0.tif", "steps": [CHOOSE + """const before = JSON.stringify(S.cur.quad); await choose("600");
      return {same: JSON.stringify(S.cur.quad) === before, touched: S.cur.touched};"""]})
    assert out["results"][0] == {"same": True, "touched": False}
    skew = hand_crop()
    skew["quad"] = [[300, 400], [1100, 440], [1080, 1400], [290, 1380]]
    out = run({"routes": routes(skew), "hash": "t.tif", "steps": [CHOOSE + """const before = JSON.stringify(S.cur.quad); await choose("go");
      return {same: JSON.stringify(S.cur.quad) === before, dirty: isDirty()};"""]})
    assert out["results"][0] == {"same": True, "dirty": True}                                   # the item changed, the corners did not


def test_a_reshaped_rectangle_is_sent_with_the_item_in_one_save():
    ok = {"POST /api/scans/t.tif/save": {"body": hand_crop(item={"kind": "600", "agent_crop": False})}}
    out = run({"routes": routes(hand_crop(), extra=ok), "hash": "t.tif", "embed": True, "steps": [
        """$("item").value = "600"; $("item").fire("change"); await new Promise(r => setTimeout(r, 20)); $("save").click(); await new Promise(r => setTimeout(r, 40));"""]})
    (post,) = posts(out)
    c = post["body"]["corners"]
    assert post["body"]["item"] == "600" and (c["tr"][0] - c["tl"][0]) / (c["bl"][1] - c["tl"][1]) == pytest.approx(A600, rel=1e-3)


# decision 1: the button that takes the stored crop, item and own ppi away
def test_the_take_away_button_is_offered_only_when_a_crop_without_detection_an_item_or_a_ppi_is_stored():
    cases = {
        "plain": detected("a.tif"),                                                                  # a print without an item choice: the page as before
        "fixed": detected("b.tif", quad_source="override"),                                         # a plain outline fix of a detected print: no button
        "item": detected("c.tif", item={"kind": "go", "agent_crop": False}),
        "ppi": detected("d.tif", item={"kind": "other", "scan_ppi": 500.0, "agent_crop": False}),
        "crop": hand_crop("e.tif"),
        "failed": failed("f.tif"),
    }
    out = run({"routes": routes(*cases.values()), "hash": "a.tif", "steps": [
        "const res = {}; for (let i = 0; i < S.scans.length; i++) { await go(i); res[S.scans[i].key] = [$(\"takeaway\").hidden, $(\"takeaway\").textContent]; } return res;"]})
    res = out["results"][0]
    assert res["a.tif"][0] is True and res["b.tif"][0] is True and res["f.tif"][0] is True
    assert res["c.tif"] == [False, "Back to automatic…"] and res["d.tif"][0] is False
    assert res["e.tif"] == [False, "Remove hand crop…"]


def test_the_take_away_dialog_says_what_goes_asks_first_and_covers_unsaved_edits():
    r = detected("p0.tif", quad_source="override", item={"kind": "other", "scan_ppi": 800.0, "agent_crop": False})
    after = detected("p0.tif", rev=2)
    ok = {"POST /api/scans/p0.tif/reset": {"body": after}}
    steps = ["""const dlg = () => ({shown: $("takedlg").style.display || "", text: $("take-text").textContent});
      const p = $("takeaway").click(); await new Promise(r => setTimeout(r, 10)); const first = dlg(); $("t-stay").click(); await p;
      const kept = {item: S.cur.item.kind, dlg: dlg().shown};
      $("ppi").value = "900"; $("ppi").fire("input");
      const q = $("takeaway").click(); await new Promise(r => setTimeout(r, 10)); const second = dlg(); $("t-go").click(); await q; await new Promise(r => setTimeout(r, 50));
      return {first, kept, second, after: {item: S.cur.item, rec: S.cur.rec.item || null, step: S.cur.step, hidden: $("takeaway").hidden, dirty: isDirty(), msg: $("msg").textContent}};"""]
    out = run({"routes": routes(r, extra=ok), "hash": "p0.tif", "steps": steps})
    res = out["results"][0]
    assert res["first"]["shown"] == "flex" and "the outline you or an agent set" in res["first"]["text"]
    assert "Other, own ppi 800" in res["first"]["text"] and "back to its automatic outline" in res["first"]["text"]
    assert "the outline you or an agent set and the item choice (Other, own ppi 800)." in res["first"]["text"]      # joined with "and"
    assert res["first"]["text"].endswith("Files already exported for this scan stay on disk.")
    assert "unsaved edits" not in res["first"]["text"] and res["kept"] == {"item": "other", "dlg": "none"}
    assert "Your unsaved edits on this scan are dropped too." in res["second"]["text"]
    assert [c["path"] for c in posts(out)] == ["/api/scans/p0.tif/reset"] and posts(out)[0]["body"] == {}
    a = res["after"]
    assert a["item"] == {"kind": "auto", "ppi": None} and a["rec"] is None and a["step"] == "fine" and a["hidden"] is True and a["dirty"] is False
    assert a["msg"].startswith("Took the crop and item of p0.tif away")


def test_taking_a_hand_crop_away_brings_a_failed_scan_back_to_the_place_step_with_a_fresh_start_rectangle():
    r = hand_crop("ticket.tif", item={"kind": "other", "scan_ppi": 400.0, "agent_crop": True}, agent_crop=True,
                  status="needs_review", warnings=["cropped by agent"], window={"state": "found", "source": "manual", "kind": "square",
                                                                              "quad": QUAD, "accepted": False, "outside": False, "suggest": "square"})
    after = failed("ticket.tif", rev=3)
    ok = {"POST /api/scans/ticket.tif/reset": {"body": after},
          "GET /api/scans/ticket.tif/tiles": [{"body": specs("ticket.tif")}, {"body": specs("ticket.tif", full=False)}],
          "GET /api/scans/ticket.tif/start?item=other": {"body": start()}}
    steps = ["""const p = $("takeaway").click(); await new Promise(r => setTimeout(r, 10)); const text = $("take-text").textContent; $("t-go").click(); await p;
      await new Promise(r => setTimeout(r, 60));
      return {text, step: S.cur.step, quad: S.cur.quad, state: $("cropstate").textContent, hidden: $("takeaway").hidden, status: $("status").textContent,
              row: S.scans[0].quad, options: [...$("item").children].map(o => o.value), item: S.cur.item};"""]
    out = run({"routes": routes(r, extra=ok), "hash": "ticket.tif", "steps": steps})
    res = out["results"][0]
    assert "the hand crop, the item choice (Other, own ppi 400) and the \"cropped by agent\" mark." in res["text"] and "goes back to failed" in res["text"]
    assert "A photo window set by hand stays" in res["text"]
    assert res["step"] == "place" and res["quad"] == [START[n] for n in ("tl", "tr", "br", "bl")] and res["row"] is None
    assert "No outline was found" in res["state"] and res["hidden"] is True and res["status"] == "failed"
    assert res["options"] == ["600", "go", "other"] and res["item"] == {"kind": "other", "ppi": None}


def test_a_refused_take_away_shows_the_message_and_keeps_the_page():
    r = detected("p0.tif", item={"kind": "go", "agent_crop": False})
    refused = {"POST /api/scans/p0.tif/reset": {"status": 400, "body": {"error": "this scan has not been detected yet; run detect first"}}}
    out = run({"routes": routes(r, extra=refused), "hash": "p0.tif", "steps": [
        """$("item").value = "600"; $("item").fire("change"); await new Promise(r => setTimeout(r, 20));
      const p = $("takeaway").click(); await new Promise(r => setTimeout(r, 10)); $("t-go").click(); await p; await new Promise(r => setTimeout(r, 40));
      return {msg: $("msg").textContent, busy: S.busy, item: S.cur.item.kind, dirty: isDirty(), step: S.cur.step};"""]})
    assert out["results"][0] == {"msg": "Not taken away: this scan has not been detected yet; run detect first", "busy": False, "item": "600",
                                 "dirty": True, "step": "fine"}


def test_the_take_away_button_is_not_called_reset_and_reset_stays_the_unsaved_edits_button():
    page = PAGE.read_text(encoding="utf-8")
    assert 'id="takeaway"' in page and 'id="reset" title="Throw away edits that are not saved yet"' in page


# R37: the question is modal
def test_the_take_away_question_cannot_be_opened_twice_and_keeps_every_other_action_out_while_open():
    r = detected("p0.tif", item={"kind": "go", "agent_crop": False})
    steps = ["""$("item").value = "600"; $("item").fire("change"); await new Promise(r => setTimeout(r, 20));
      const first = $("takeaway").click(); await new Promise(r => setTimeout(r, 10));
      const handler = $("t-go").onclick;
      const second = $("takeaway").click(); await new Promise(r => setTimeout(r, 10));                    // a second click: nothing new
      const same = $("t-go").onclick === handler;
      await commit("save"); await go(1); await setItem("go"); await setStep("place"); const back = await closeToApp();
      const during = {modal: S.modal, busy: S.busy, idx: S.idx, step: S.cur.step, item: S.cur.item.kind, save: $("save").disabled,
                      rot: $("rot").disabled, ppi: $("ppi").disabled, item_disabled: $("item").disabled, dlg: $("takedlg").style.display, back};
      document.fire("keydown", {key: "Escape"});                                                          // Escape keeps the crop
      await first; await second;
      return {same, during, after: {modal: S.modal, busy: S.busy, dlg: $("takedlg").style.display, t: $("t-go").onclick || null, item: S.cur.item.kind, save: $("save").disabled}};"""]
    out = run({"routes": routes(r, detected("p1.tif")), "hash": "p0.tif", "steps": steps})
    res = out["results"][0]
    assert res["same"] is True
    assert res["during"] == {"modal": True, "busy": True, "idx": 0, "step": "fine", "item": "600", "save": True, "rot": True, "ppi": True,
                             "item_disabled": True, "dlg": "flex", "back": False}
    assert res["after"] == {"modal": False, "busy": False, "dlg": "none", "t": None, "item": "600", "save": False}
    assert posts(out) == []                                                                              # nothing was sent


def test_the_take_away_question_goes_on_after_it_was_answered_once():
    r = detected("p0.tif", item={"kind": "go", "agent_crop": False})
    ok = {"POST /api/scans/p0.tif/reset": {"body": detected("p0.tif", rev=2)}}
    steps = ["""const p = $("takeaway").click(); await new Promise(r => setTimeout(r, 10)); $("t-stay").click(); await p;
      const again = $("takeaway").click(); await new Promise(r => setTimeout(r, 10)); const open = S.modal; $("t-go").click(); await again;
      await new Promise(r => setTimeout(r, 60)); return {open, modal: S.modal, busy: S.busy};"""]
    out = run({"routes": routes(r, extra=ok), "hash": "p0.tif", "steps": steps})
    assert out["results"][0] == {"open": True, "modal": False, "busy": False} and len(posts(out)) == 1


# R36: a fine-tuned corner is not a rectangle any more, so an item choice and the first Place edit leave it alone
def test_a_corner_fine_tuned_by_a_fraction_of_a_pixel_survives_an_item_choice():
    for dx in (1.0, 0.5, 0.3):
        skew = hand_crop()
        skew["quad"] = [[300 + dx, 400], [1100, 400], [1100, 1400], [300, 1400]]
        out = run({"routes": routes(skew), "hash": "t.tif", "steps": [CHOOSE + """const before = JSON.stringify(S.cur.quad); await choose("go");
          return {same: JSON.stringify(S.cur.quad) === before, dirty: isDirty(), rect: isRectQuad(S.cur.quad)};"""]})
        assert out["results"][0] == {"same": True, "dirty": True, "rect": False}, dx
    # a rectangle that went through the rounding of a save is still reshaped
    rounded = hand_crop()
    rounded["quad"] = [[300.01, 399.99], [1100.01, 400.0], [1100.0, 1400.01], [300.0, 1400.0]]
    out = run({"routes": routes(rounded), "hash": "t.tif", "steps": [SHAPE + CHOOSE + 'await choose("600"); return shapeOf();']})
    assert out["results"][0]["ratio"] == pytest.approx(A600, rel=1e-3)


def test_an_arrow_key_at_fine_tune_and_a_drag_count_as_a_fine_tuned_corner():
    out = run({"routes": routes(hand_crop()), "hash": "t.tif", "steps": [CHOOSE + """const q0 = JSON.stringify(S.cur.quad); const was = isRectQuad(S.cur.quad);
      openZoom("tl"); nudge(1, 0); const nudged = {rect: isRectQuad(S.cur.quad), moved: S.cur.quad[0][0] - 300};
      closeZoom(); S.cur.quad = JSON.parse(q0);
      S.cur.quad[2] = [S.cur.quad[2][0] + 1.4, S.cur.quad[2][1] - 0.9];                                      // a drag of a few css px on a tile
      const dragged = isRectQuad(S.cur.quad);
      const before = JSON.stringify(S.cur.quad); await choose("go");
      return {was, nudged, dragged, unchanged: JSON.stringify(S.cur.quad) === before};"""]})
    assert out["results"][0] == {"was": True, "nudged": {"rect": False, "moved": 1}, "dragged": False, "unchanged": True}
