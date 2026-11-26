"""The app's page code (``app/static/app.js``) run for real in Node, against a small DOM stub (``tests/js/page_harness.js``):
what the settings panel sends on a save, and the scanner profile dropdown. Skipped without ``node`` on the PATH."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not on the PATH")

HARNESS = Path(__file__).parent / "js" / "page_harness.js"
APP_JS = Path(__file__).parent.parent / "src" / "polaroid_processor" / "app" / "static" / "app.js"


def run_page(scenario, data=None, js=APP_JS):
    cmd = [NODE, str(HARNESS), str(js), scenario] + ([json.dumps(data)] if data is not None else [])
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60, encoding="utf-8")
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


# --------------------------------------------------------------------------- metadata fields of a save
def test_the_page_sends_only_the_metadata_fields_that_were_edited():
    """Keywords with a comma (set by CLI/MCP) and a CRLF description stay as they are on a save that edits other
    things: a field goes to the server only when its text in the page changed since it was shown."""
    out = run_page("metadata_save")
    assert out["shown_keywords"] == "Hamburg, Germany, Garten"
    assert out["margin_only"] == {"margin_mm": 0.2}
    assert out["title_edit"] == {"metadata": {"title": "New {n}"}}
    assert out["keyword_edit"] == {"metadata": {"keywords": ["Hamburg", "Germany", "Garten", "Neu"]}}      # documented panel limit


def test_the_harness_tells_the_old_comparison_from_the_new(tmp_path):
    """The same harness on the pre-fix page (the typed value compared with the saved one after splitting) sends the
    comma keyword and the description on a margin-only save."""
    js = APP_JS.read_text(encoding="utf-8")
    new = 'if ($("m-" + f).value !== S.shownMeta[f]) meta[f] = typed[f];'
    old = ('if (JSON.stringify(typed[f]) !== JSON.stringify(((S.loadedSettings.metadata || {})[f]) || (f === "keywords" ? [] : ""))) '
           'meta[f] = typed[f];')
    assert new in js
    old_js = tmp_path / "app_old.js"
    old_js.write_text(js.replace(new, old), encoding="utf-8")
    out = run_page("metadata_save", js=old_js)
    assert out["margin_only"]["metadata"]["keywords"] == ["Hamburg", "Germany", "Garten"]      # the bug: re-split, sent, changed
    assert "description" in out["margin_only"]["metadata"]                                    # CRLF against the textarea's LF


# --------------------------------------------------------------------------- the scanner profile dropdown
ITEMS = [{"name": "epson-v850", "ppi": 3200.0}, {"name": "plustek", "ppi": 3600.0}]


def profiles(items=ITEMS, path="C:/cfg/profiles.yaml", error=None):
    return {"path": path, "items": items, "error": error}


def test_the_dropdown_lists_none_then_every_profile_with_its_ppi_and_selects_the_saved_one():
    out = run_page("profile_select", {"profile": "plustek", "profiles": profiles(),
                                      "picks": {"to_other": "epson-v850", "to_none": ""}})
    assert out["options"] == [{"value": "", "text": "none"}, {"value": "epson-v850", "text": "epson-v850 (3200 ppi)"},
                              {"value": "plustek", "text": "plustek (3600 ppi)"}]
    assert out["value"] == "plustek" and not out["hint_warns"] and "C:/cfg/profiles.yaml" in out["hint"]
    assert out["margin_only"] == {"margin_mm": 0.2}                                   # an unrelated save leaves the profile alone
    assert out["to_other"] == {"profile": "epson-v850"} and out["to_none"] == {"profile": ""}


def test_no_profile_selected_shows_none_and_sends_nothing_for_it():
    out = run_page("profile_select", {"profile": None, "profiles": profiles(), "picks": {"pick": "epson-v850"}})
    assert out["value"] == "" and out["margin_only"] == {"margin_mm": 0.2}
    assert out["pick"] == {"profile": "epson-v850"}


def test_a_saved_profile_that_is_not_in_the_file_stays_as_an_extra_selected_option():
    out = run_page("profile_select", {"profile": "old-scanner", "profiles": profiles(), "picks": {"to_none": ""}})
    assert out["options"][-1] == {"value": "old-scanner", "text": "old-scanner (not in profiles file)"}
    assert [o["value"] for o in out["options"]] == ["", "epson-v850", "plustek", "old-scanner"]
    assert out["value"] == "old-scanner" and out["hint_warns"] and "old-scanner" in out["hint"]
    assert "choose \"none\" and type the profile's ppi in the same save" in out["hint"]      # the way that keeps the scale
    assert out["margin_only"] == {"margin_mm": 0.2}                                   # kept: the save does not name the profile
    assert out["to_none"] == {"profile": ""}                                          # the way out is "none"


def test_without_profiles_the_dropdown_has_only_none_and_a_hint_where_they_live():
    out = run_page("profile_select", {"profile": None, "profiles": profiles([], path="C:/cfg/profiles.yaml")})
    assert out["options"] == [{"value": "", "text": "none"}]
    assert "C:/cfg/profiles.yaml" in out["hint"] and "profiles: {name: ppi}" in out["hint"] and not out["hint_warns"]


def test_an_unreadable_profiles_file_is_said_in_the_hint():
    out = run_page("profile_select", {"profile": None, "profiles": profiles([], error="could not be read (ValueError: bad)")})
    assert out["options"] == [{"value": "", "text": "none"}] and out["hint_warns"]
    assert "Problem with the profiles file" in out["hint"] and "ValueError: bad" in out["hint"]


def test_the_progress_bar_shows_the_scale_error():
    out = run_page("scale_error")
    assert out["plain"] == {"text": "", "hidden": True}
    assert out["error"]["hidden"] is False and out["error"]["cls"] == "panel runbar locked"
    assert "scanner profile 'x' not found in /p.yaml" in out["error"]["text"] and "Settings > Advanced" in out["error"]["text"]


# --------------------------------------------------------------------------- grid cards of hand crops (hand crop, slice 2)
def card(key, status, **kw):
    return {"key": key, "status": status, "warnings": [], "error": None, "exported": False, "quad_source": None, "format": None,
            "preview": None, "stale": False, "export": None, "rev": 0, **kw}


def test_grid_cards_show_a_hand_crop_its_item_and_the_agent_mark_and_failed_scans_open():
    scans = [
        card("plain.tif", "ok", quad_source="auto", size_px=[800, 1000], preview={"url": "/p.jpg", "size": [10, 10]}),
        card("fixed.tif", "ok", quad_source="override", size_px=[800, 1000], item="go", preview={"url": "/p.jpg", "size": [10, 10]}),
        card("ticket.tif", "ok", quad_source="override", size_px=[700, 1200], item="other", hand_crop=True, error=None,
             preview={"url": "/t.jpg", "size": [10, 10]}),
        card("sheet.tif", "needs_review", quad_source="override", size_px=[700, 1200], item="600", hand_crop=True, agent_crop=True,
             warnings=["cropped by agent", "scale_unverified"], preview={"url": "/s.jpg", "size": [10, 10]}),
        card("fail.tif", "failed", error="detection failed: no plausible print outline at any threshold"),
        card("wait.tif", "pending"),
    ]
    out = run_page("cards", {"scans": scans})
    c = out["cards"]
    assert c["plain.tif"]["tags"] == ["ok"]                                                                # a scan without a choice: the card as before
    assert c["fixed.tif"]["tags"] == ["ok", "fixed", "item: Go"]
    assert c["ticket.tif"]["tags"] == ["ok", "hand crop", "item: Other"]                                  # no "fixed": it is a hand crop
    assert c["ticket.tif"]["warns"] == []                                                                  # the stored detection error is not shown
    assert c["sheet.tif"]["tags"] == ["needs review", "hand crop", "item: 600", "cropped by agent"]
    assert c["sheet.tif"]["warns"] == ["scale_unverified"]                                                 # the mark is a tag, not a second line
    assert c["fail.tif"]["tags"] == ["failed"] and c["fail.tif"]["warns"] == [scans[4]["error"]]
    assert c["fail.tif"]["img"] == "no outline: click to crop by hand" and c["wait.tif"]["img"] == "waiting for detection"
    o = out["opened"]
    assert o["fail.tif"] == {"url": "/fix#fail.tif", "shown": True, "msg": ""}                           # a failed scan opens on the fix page
    assert o["ticket.tif"]["shown"] is True and o["sheet.tif"]["url"] == "/fix#sheet.tif"
    assert o["wait.tif"] == {"url": None, "shown": False, "msg": "This scan has not been detected yet."}     # not detected yet: closed


def test_a_card_is_redrawn_when_the_item_or_the_agent_mark_changes():
    out = run_page("sig", {"a": card("a.tif", "ok", quad_source="override", size_px=[1, 1]),
                           "b": card("a.tif", "ok", quad_source="override", size_px=[1, 1], item="go"),
                           "c": card("a.tif", "ok", quad_source="override", size_px=[1, 1], agent_crop=True),
                           "d": card("a.tif", "ok", quad_source="override", size_px=[1, 1], hand_crop=True)})
    assert len(set(out.values())) == 4


# --------------------------------------------------------------------------- the grid's "To review" filter
def test_to_review_shows_needs_review_and_failed_and_needs_review_still_hides_failed():
    scans = [card("a.tif", "ok"), card("b.tif", "needs_review"), card("c.tif", "failed", error="detection failed: x"),
             card("d.tif", "needs_review"), card("e.tif", "pending")]
    # chip order: All, To review, OK, Needs review, Failed, Not checked yet
    out = run_page("filters", {"scans": scans, "clicks": {"needs_review": 3, "failed": 4, "all": 0, "again": 1}})
    assert out["start"]["chips"] == ["All 5", "To review 3", "OK 1", "Needs review 2", "Failed 1", "Not checked yet 1"]
    assert out["start"]["on"] == ["To review 3"]                              # the address `polaroid review` opens
    assert out["start"]["shown"] == ["b.tif", "c.tif", "d.tif"]
    assert out["needs_review"]["shown"] == ["b.tif", "d.tif"] and out["needs_review"]["on"] == ["Needs review 2"]
    assert out["failed"]["shown"] == ["c.tif"]
    assert out["all"]["shown"] == [s["key"] for s in scans]
    assert out["again"] == out["start"] and out["start"]["empty"] is None
    clean = run_page("filters", {"scans": [card("a.tif", "ok")], "clicks": {"failed": 4}})
    assert clean["start"]["shown"] == [] and clean["start"]["empty"] == "Nothing to review."
    assert clean["failed"]["empty"] == "No scans with this status."
