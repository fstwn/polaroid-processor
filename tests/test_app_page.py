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
