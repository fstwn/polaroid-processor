"""Hand crop, slice 2: what the fix page (review server and app) serves and stores for a hand crop.

The page's own code is tested in ``test_crop_page.py`` (geometry, save body, in Node). Here: a scan whose detection
failed is listed and opened (the place step starts at its overview), the corner tiles can sit on a rectangle that is not
stored yet, the start rectangle comes from the core, a save stores outline and item choice in one fix, the corner checks
and the item rules answer 400 and store nothing, and the app's grid cards carry the hand crop, its item and the agent mark.
"""
import json

import numpy as np
import pytest

from polaroid_processor import api
from polaroid_processor.manifest import Manifest

from synth import make_scan  # noqa: F401
from test_app import AppClient
from test_hand_crop import TICKET, base, box, edit, entry, out, src, stored  # noqa: F401
from test_review_server import Page


@pytest.fixture
def page(out):
    p = Page(out, show_all=True)
    yield p
    p.close()


def by_key(listing):
    return {r["key"]: r for r in listing["scans"]}


# --------------------------------------------------------------------------- listing and opening
def test_a_failed_scan_is_listed_with_no_outline_and_the_limits_the_page_draws_with(page):
    status, data = page.json("/api/scans")
    recs = by_key(data)
    assert status == 200 and {"ticket.tif", "sheet.tif", "lift.tif"} <= set(recs)
    t = recs["ticket.tif"]
    assert t["status"] == "failed" and t["quad"] is None and t["size_px"] is None and t["quad_source"] is None
    assert t["detected"] is False and t["error"].startswith("detection failed") and t["scan_px"] == [1600, 2000]
    assert "item" not in t and "window" not in t and "agent_crop" not in t
    assert recs["p0.tif"]["detected"] is True and recs["p0.tif"]["quad"] is not None and "error" not in recs["p0.tif"]
    assert data["aspects"]["600"] == pytest.approx(88 / 106.85) and data["aspects"]["go"] == pytest.approx(53.5 / 65.8)
    assert data["limits"] == {"min_side_px": 50, "max_rotation_deg": 45.0, "max_scan_ppi": 50000.0}


def test_the_default_list_shows_failed_scans_next_to_the_ones_that_need_review(out):
    p = Page(out)      # not show_all: only scans that need review, and now the failed ones too
    try:
        keys = set(by_key(p.json("/api/scans")[1]))
        assert {"ticket.tif", "sheet.tif", "lift.tif"} <= keys
        assert not any(k.startswith("p") and entry(out, k)["status"] == "ok" for k in keys)
    finally:
        p.close()


def test_a_scan_that_was_never_detected_stays_closed(page, out):
    edit(out, "lift.tif", status="pending", error=None)
    assert "lift.tif" not in by_key(page.json("/api/scans")[1])
    assert page.request("/api/scans/lift.tif/tiles")[0] == 404
    assert page.request("/api/scans/lift.tif/start?item=other")[0] == 404
    assert page.json("/api/scans/lift.tif/save", {"corners": TICKET, "item": "other"})[0] == 404      # the page does not open it (the core would refuse it too)
    assert entry(out, "lift.tif").get("override") is None


# --------------------------------------------------------------------------- tiles at a rectangle that is not stored
def test_a_scan_without_an_outline_has_its_overview_only(page):
    status, specs = page.json("/api/scans/ticket.tif/tiles")
    assert status == 200 and [t["name"] for t in specs["tiles"]] == ["overview"] and specs["scan_px"] == [1600, 2000]
    assert page.request(specs["tiles"][0]["url"])[2][:4] == b"\x89PNG"
    assert page.request("/api/scans/ticket.tif/tile/tl.png")[0] == 404
    assert page.request("/api/scans/ticket.tif/tiles?of=window")[0] == 400


def test_corner_tiles_can_sit_on_corners_given_in_the_query(page):
    q = "400,300,1100,300,1100,1500,400,1500"
    status, specs = page.json(f"/api/scans/ticket.tif/tiles?q={q}")
    t = {x["name"]: x for x in specs["tiles"]}
    assert status == 200 and len(specs["tiles"]) == 9
    assert t["tl"]["x0"] + 240 == 400 and t["tl"]["y0"] + 240 == 300 and t["br"]["x0"] + 240 == 1100 and t["br"]["y0"] + 240 == 1500
    assert t["tl"]["url"].endswith("&q=400.00,300.00,1100.00,300.00,1100.00,1500.00,400.00,1500.00")
    status, _, png = page.request(t["bl"]["url"])
    assert status == 200 and png[:4] == b"\x89PNG"
    # a detected scan: the same, the stored outline is the default
    _, stored_specs = page.json("/api/scans/p0.tif/tiles")
    _, moved = page.json("/api/scans/p0.tif/tiles?q=100,100,500,100,500,600,100,600")
    assert stored_specs["tiles"][0]["x0"] != moved["tiles"][0]["x0"] and moved["tiles"][0]["x0"] + 240 == 100
    for bad in ("1,2,3", "1,2,3,4,5,6,7,x", "1,2,3,4,5,6,7,nan"):
        assert page.request(f"/api/scans/ticket.tif/tiles?q={bad}")[0] == 400


# --------------------------------------------------------------------------- the start rectangle
def test_the_start_route_gives_the_core_rectangle_and_the_formats_aspect(page, out):
    status, rec = page.json("/api/scans/ticket.tif/start?item=other")
    assert status == 200 and rec["aspect"] is None and rec["scan_px"] == [1600, 2000]
    assert rec["corners"] == {"tl": [160.0, 200.0], "tr": [1440.0, 200.0], "br": [1440.0, 1800.0], "bl": [160.0, 1800.0]}
    assert page.json("/api/scans/ticket.tif/start")[1]["item"] == "other"
    for kind, aspect in (("600", 88 / 106.85), ("go", 53.5 / 65.8)):
        _, rec = page.json(f"/api/scans/ticket.tif/start?item={kind}")
        c = rec["corners"]
        assert rec["aspect"] == pytest.approx(aspect)
        assert (c["tr"][0] - c["tl"][0]) / (c["bl"][1] - c["tl"][1]) == pytest.approx(aspect, rel=0.01)
    assert page.json("/api/scans/ticket.tif/start?item=auto")[0] == 400
    assert page.json("/api/scans/ticket.tif/start?item=card")[0] == 400
    assert page.request("/api/scans/nope.tif/start?item=other")[0] == 404


# --------------------------------------------------------------------------- saving
def test_a_first_crop_is_saved_with_its_item_in_one_fix(page, out):
    status, rec = page.json("/api/scans/ticket.tif/save", {"corners": TICKET, "item": "other", "scan_ppi": 800})
    e = entry(out, "ticket.tif")
    assert status == 200 and rec["status"] == "ok" and rec["quad"][0] == [400.0, 300.0] and rec["detected"] is False
    assert rec["item"]["kind"] == "other" and rec["item"]["scan_ppi"] == 800.0 and rec["item"]["agent_crop"] is False
    assert e["override"] and e["item_fix"]["author"] == "human" and e["item_fix"]["by_human"] is True
    assert e["item_fix"]["note"] == "review page: corrected" and [h["action"] for h in e["history"]][:2] == ["fix", "item"]
    # the first crop of a scan nothing was found on needs all four corners
    status, body = page.json("/api/scans/sheet.tif/save", {"item": "go"})
    assert status == 400 and "four corners" in body["error"] and entry(out, "sheet.tif")["status"] == "failed"
    # a format named: the page sends it
    status, rec = page.json("/api/scans/sheet.tif/save", {"corners": box(500, 600, 1900, 2200), "item": "600"})
    assert status == 200 and rec["item"]["kind"] == "600" and rec["status"] in ("ok", "needs_review")


def test_a_refused_save_stores_nothing_and_says_why(page, out):
    before = stored(out)
    cases = [
        ({"corners": box(400, 300, 440, 1500), "item": "other"}, "50 px"),                                   # a side below 50 px
        ({"corners": {"tl": [400, 300], "tr": [1100, 300], "br": [400, 1500], "bl": [1100, 1500]}, "item": "other"}, "cross"),
        ({"corners": box(-5, 300, 1100, 1500), "item": "other"}, "inside the scan"),
        ({"corners": box(400, 300, 1100, 1500), "item": "card"}, "item must be"),
        ({"corners": box(400, 300, 1100, 1500), "item": 600}, "item must be 600"),
        ({"corners": box(400, 300, 1100, 1500), "item": "go", "scan_ppi": 500}, "own resolution|does not apply"),
        ({"corners": box(400, 300, 1100, 1500), "item": "other", "scan_ppi": True}, "scan_ppi must be a number"),
        ({"corners": box(400, 300, 1100, 1500), "item": "auto"}, "first crop needs an item kind"),
        ({"corners": {"tl": [1, 2]}, "item": "other"}, "corners must hold exactly"),
        ({}, "save needs"),
    ]
    for body, why in cases:
        status, res = page.json("/api/scans/ticket.tif/save", body)
        assert status == 400 and __import__("re").search(why, res["error"]), (body, res)
        assert stored(out) == before
    # the corner checks of a rotated crop
    status, res = page.json("/api/scans/ticket.tif/save", {"corners": {"tl": [800, 300], "tr": [1100, 820], "br": [580, 1120], "bl": [280, 600]},
                                                           "item": "other"})
    assert status == 400 and "rotated" in res["error"] and stored(out) == before


def test_an_item_alone_is_saved_on_a_detected_scan_and_the_rules_of_the_core_hold(page, out):
    status, rec = page.json("/api/scans/p1.tif/save", {"item": "go"})
    assert status == 200 and rec["item"]["kind"] == "go" and rec["quad_source"] == "auto" and entry(out, "p1.tif").get("override") is None
    assert entry(out, "p1.tif")["item_fix"]["note"] == "review page: item"
    status, rec = page.json("/api/scans/p1.tif/save", {"item": "auto"})
    assert status == 200 and "item" not in rec and entry(out, "p1.tif").get("item_fix") is None
    status, rec = page.json("/api/scans/p1.tif/save", {"item": "other", "scan_ppi": 600})
    assert status == 200 and rec["item"]["scan_ppi"] == 600.0
    assert page.json("/api/scans/p1.tif/save", {"scan_ppi": 0})[1]["item"].get("scan_ppi") is None      # the own ppi taken away
    # item and corners in one fix on a detected scan
    det = np.array(entry(out, "p2.tif")["detection"]["quad"])
    moved = {n: [float(v[0] + 4), float(v[1])] for n, v in zip(("tl", "tr", "br", "bl"), det)}
    status, rec = page.json("/api/scans/p2.tif/save", {"corners": moved, "item": "600"})
    e = entry(out, "p2.tif")
    assert status == 200 and e["override"] and e["item_fix"]["kind"] == "600" and rec["quad"][0][0] == pytest.approx(det[0][0] + 4, abs=0.01)
    # with an item Other a hand window is refused (photo crop is off here: the message says so), nothing is stored
    before = stored(out)
    status, res = page.json("/api/scans/p0.tif/save", {"item": "other", "window": {"corners": moved, "kind": "square"}})
    assert status == 400 and stored(out) == before


def test_a_humans_item_alone_keeps_the_agent_mark_and_a_corner_save_clears_it(page, out):
    api.fix_scan(str(out), "ticket.tif", sets=TICKET, author="agent")
    status, rec = page.json("/api/scans/ticket.tif/save", {"item": "go"})
    assert status == 200 and rec["item"]["agent_crop"] is True and rec["status"] == "needs_review"
    status, rec = page.json("/api/scans/ticket.tif/save", {"corners": TICKET, "item": "other"})
    assert status == 200 and rec["status"] == "ok" and rec["item"]["agent_crop"] is False and entry(out, "ticket.tif")["agent_crop"] is None
    # Confirm of a crop changes no corner
    api.fix_scan(str(out), "sheet.tif", sets=box(500, 600, 1900, 2200), author="agent")
    status, rec = page.json("/api/scans/sheet.tif/confirm", {})
    assert status == 200 and rec["status"] == "ok" and "agent_crop" not in rec and entry(out, "sheet.tif")["agent_crop"] is None
    assert page.json("/api/scans/lift.tif/confirm", {})[0] == 404      # nothing to confirm without an outline


def test_a_save_stores_the_edit_after_the_scan_was_changed_elsewhere_or_refuses_a_changed_file(page, out, tmp_path):
    # the page's save goes through the same check of the scan file as every other entry point
    e = entry(out, "ticket.tif")
    m = Manifest(str(out))
    m.get("ticket.tif")["size_bytes"] = e["size_bytes"] + 1
    m.save()
    before = stored(out)
    status, res = page.json("/api/scans/ticket.tif/save", {"corners": TICKET, "item": "other"})
    assert status == 400 and "changed since the last run" in res["error"] and stored(out) == before


# --------------------------------------------------------------------------- the app
@pytest.fixture
def app(tmp_path, out):
    c = AppClient(tmp_path)
    assert c.json("/api/project/open", {"output": str(out)})[0] == 200
    yield c
    c.close()


def test_the_app_serves_the_new_routes_and_the_grid_cards_carry_a_hand_crop(app, out):
    cards = by_key({"scans": app.json("/api/project/scans")[1]["scans"]})
    t = cards["ticket.tif"]
    assert t["status"] == "failed" and t["error"].startswith("detection failed") and "hand_crop" not in t and "item" not in t
    status, rec = app.json("/api/fix/scans/ticket.tif/save", {"corners": TICKET, "item": "other"})
    assert status == 200 and rec["status"] == "ok"
    cards = by_key({"scans": app.json("/api/project/scans")[1]["scans"]})
    t = cards["ticket.tif"]
    assert t["status"] == "ok" and t["hand_crop"] is True and t["item"] == "other" and not t["error"] and "agent_crop" not in t
    assert t["preview"] is not None or t["stale"]      # the card shows the crop as drawn (the preview is made in the background)
    assert "hand_crop" not in cards["p0.tif"] and "item" not in cards["p0.tif"]      # a scan without a choice: the card as before
    status, rec = app.json("/api/fix/scans/sheet.tif/start?item=go")
    assert status == 200 and rec["aspect"] == pytest.approx(53.5 / 65.8)
    status, specs = app.json("/api/fix/scans/lift.tif/tiles")
    assert status == 200 and [x["name"] for x in specs["tiles"]] == ["overview"] and specs["tiles"][0]["url"].startswith("/api/fix/scans/lift.tif/tile/")
    status, specs = app.json("/api/fix/scans/lift.tif/tiles?q=400,300,1100,300,1100,1500,400,1500")
    assert status == 200 and "&q=400.00" in specs["tiles"][0]["url"] and app.request(specs["tiles"][1]["url"])[0] == 200
    # an agent's crop shows its tag on the card, and counts as needs review, not as failed
    api.fix_scan(str(out), "lift.tif", sets=TICKET, author="agent")
    cards = by_key({"scans": app.json("/api/project/scans")[1]["scans"]})
    assert cards["lift.tif"]["agent_crop"] is True and cards["lift.tif"]["hand_crop"] is True and cards["lift.tif"]["status"] == "needs_review"
    counts = app.json("/api/project/progress")[1]["counts"]
    assert counts["failed"] == 1 and counts["needs_review"] >= 1      # sheet.tif is the one failed scan left


def test_the_app_page_opens_a_failed_scan_and_has_the_card_tags():
    from pathlib import Path
    js = (Path(api.__file__).parent / "app" / "static" / "app.js").read_text(encoding="utf-8")
    assert 'scan.status !== "failed"' in js and "hand crop" in js and "cropped by agent" in js and "ITEM_LABEL" in js
    assert 'No outline was found for this scan.' not in js      # the old refusal is gone


# --------------------------------------------------------------------------- after the fifth review: the route behind "take the crop away", R34
def test_the_reset_route_takes_crop_item_and_ppi_away_and_the_scan_goes_back_to_failed(page, out):
    page.json("/api/scans/ticket.tif/save", {"corners": TICKET, "item": "other", "scan_ppi": 800})
    assert entry(out, "ticket.tif")["status"] == "ok"
    status, rec = page.json("/api/scans/ticket.tif/reset", {})
    e = entry(out, "ticket.tif")
    assert status == 200 and rec["status"] == "failed" and rec["quad"] is None and rec["detected"] is False and "item" not in rec
    assert rec["error"].startswith("detection failed") and e.get("override") is None and e.get("item_fix") is None
    assert [h["action"] for h in e["history"]][-2:] == ["reset", "item-reset"] and e["history"][-1]["author"] == "human"
    assert e["history"][-1]["note"] == "review page: crop and item taken away"
    # the scan is a failed one again: its overview only, the place step starts again
    assert [t["name"] for t in page.json("/api/scans/ticket.tif/tiles")[1]["tiles"]] == ["overview"]
    # a scan that was never cropped answers as before; one that was never detected stays closed
    assert page.json("/api/scans/sheet.tif/reset", {})[0] == 200
    edit(out, "lift.tif", status="pending", error=None)
    assert page.json("/api/scans/lift.tif/reset", {})[0] == 404
    assert page.json("/api/scans/nope.tif/reset", {})[0] == 404


def test_the_reset_route_on_a_detected_scan_removes_the_item_and_the_hand_outline_and_keeps_the_detected_one(page, out):
    det = np.array(entry(out, "p1.tif")["detection"]["quad"])
    moved = {n: [float(v[0] + 4), float(v[1])] for n, v in zip(("tl", "tr", "br", "bl"), det)}
    page.json("/api/scans/p1.tif/save", {"corners": moved, "item": "go"})
    assert entry(out, "p1.tif")["override"] and entry(out, "p1.tif")["item_fix"]
    status, rec = page.json("/api/scans/p1.tif/reset", {})
    e = entry(out, "p1.tif")
    assert status == 200 and rec["quad_source"] == "auto" and rec["detected"] is True and "item" not in rec
    assert e.get("override") is None and e.get("item_fix") is None and rec["quad"][0] == pytest.approx(det[0].tolist(), abs=0.01)


def test_a_reset_by_an_agent_page_is_refused_when_a_human_set_the_item(out):
    api.fix_scan(str(out), "p2.tif", item="go", author="human")
    p = Page(out, show_all=True, author="agent")
    try:
        before = stored(out)
        status, body = p.json("/api/scans/p2.tif/reset", {})
        assert status == 400 and "a human set" in body["error"] and stored(out) == before
    finally:
        p.close()


def test_the_app_route_resets_renews_the_card_and_leaves_no_preview_error(app, out):
    assert app.json("/api/fix/scans/ticket.tif/save", {"corners": TICKET, "item": "other"})[0] == 200
    card = by_key({"scans": app.json("/api/project/scans")[1]["scans"]})["ticket.tif"]
    assert card["hand_crop"] is True and card["item"] == "other"
    status, rec = app.json("/api/fix/scans/ticket.tif/reset", {})
    assert status == 200 and rec["status"] == "failed" and rec["quad"] is None
    card = by_key({"scans": app.json("/api/project/scans")[1]["scans"]})["ticket.tif"]
    assert card["status"] == "failed" and "hand_crop" not in card and "item" not in card and card["error"].startswith("detection failed")
    assert card["preview"] is None
    assert app.json("/api/project/progress")[1]["counts"]["failed"] == 3      # sheet, ticket and lift are failed scans again
    # a detected print with an item: the card loses the item
    app.json("/api/fix/scans/p0.tif/save", {"item": "go"})
    assert by_key({"scans": app.json("/api/project/scans")[1]["scans"]})["p0.tif"]["item"] == "go"
    assert app.json("/api/fix/scans/p0.tif/reset", {})[0] == 200
    assert "item" not in by_key({"scans": app.json("/api/project/scans")[1]["scans"]})["p0.tif"]


def test_a_plain_fixed_print_whose_detection_failed_keeps_its_fixed_card_and_only_a_crop_with_an_item_is_a_hand_crop(app, out):
    api.fix_scan(str(out), "p0.tif", shifts={"left": [1.0]}, author="human")
    edit(out, "p0.tif", detection=None)      # a hand-fixed print whose later detection failed: held by hand, but no item choice
    card = by_key({"scans": app.json("/api/project/scans")[1]["scans"]})["p0.tif"]
    assert card["quad_source"] == "override" and "hand_crop" not in card and "item" not in card      # the card it had: "fixed"
    api.fix_scan(str(out), "p0.tif", item="600", author="human")
    card = by_key({"scans": app.json("/api/project/scans")[1]["scans"]})["p0.tif"]
    assert card["hand_crop"] is True and card["item"] == "600"
