"""Export metadata, slice 2: the app's panel API (save, validation, preview, defaults), the Export count with a
metadata-only change, and the warning and count of a file whose headers could not be rewritten."""
import json
import shutil
from pathlib import Path

import pytest
import tifffile

from polaroid_processor import api, cli, pipeline
from polaroid_processor.app.project import export_plan, export_state
from polaroid_processor.manifest import MANIFEST_NAME, Manifest
from polaroid_processor.mcp_server import TOOLS, tool_process
from polaroid_processor.pipeline import settings_from_dict

from test_app import app, finished, scans  # noqa: F401  (fixtures and helpers)
from test_metadata_pipeline import PPI, copy, files_of, project, run, src, tags, xp  # noqa: F401

EMPTY = {"title": "", "author": "", "copyright": "", "description": "", "keywords": [], "date_taken": ""}


@pytest.fixture
def roll(scans, tmp_path):
    """A project folder of two scans named like a roll: ``260328_Roll/Roll_3.tif`` and ``Roll_4.tif``."""
    d = tmp_path / "260328_Roll"
    d.mkdir()
    shutil.copy(scans / "a_white.tif", d / "Roll_3.tif")
    shutil.copy(scans / "a_gold.tif", d / "Roll_4.tif")
    return d


def panel(app):
    return app.json("/api/project/settings")[1]["settings"]


def counts(app):
    return app.json("/api/project/progress")[1]["counts"]


def export_states(app):
    return {s["key"]: s["export"] for s in app.json("/api/project/scans")[1]["scans"]}


# --------------------------------------------------------------------------- the panel API
def test_the_panel_saves_metadata_and_keeps_it_before_the_first_run(app, roll):
    app.json("/api/project/open", {"input": str(roll)})
    assert panel(app)["metadata"] == EMPTY
    meta = {"title": "Müller {stem}", "author": "Jörg", "keywords": ["Garten", "Übung"], "date_taken": "1987-07",
            "description": "Zeile 1\nZeile 2", "copyright": "© 2026 {project}"}
    status, data = app.json("/api/project/settings", {"metadata": meta})
    assert status == 200 and data["settings"]["metadata"] == meta
    out = app.state.project.output
    assert not (Path(out) / MANIFEST_NAME).exists()                      # no run yet: the settings wait in app.json
    assert app.state.store.project_settings(out)["metadata"] == meta
    # another panel change leaves the metadata alone, a field set to "" clears that field only
    assert panel(app)["metadata"] == meta
    app.json("/api/project/settings", {"margin_mm": 0.2})
    assert panel(app)["margin_mm"] == 0.2 and panel(app)["metadata"] == meta
    app.json("/api/project/settings", {"metadata": {"title": "", "keywords": []}})
    assert panel(app)["metadata"] == {**meta, "title": "", "keywords": []}
    app.json("/api/project/settings", {"metadata": {"title": "T {n}"}})
    # the first run takes them into the manifest
    app.json("/api/project/detect", {})
    app.wait(finished)
    saved = Manifest(out).data["settings"]["metadata"]
    assert saved == {"title": "T {n}", "author": "Jörg", "copyright": "© 2026 {project}",
                     "description": "Zeile 1\nZeile 2", "date_taken": "1987-07"}      # empty fields are left out
    assert panel(app)["metadata"]["title"] == "T {n}"
    # and without any metadata the manifest's settings carry no key (unchanged manifests)
    app.json("/api/project/settings", {"metadata": {f: "" if f != "keywords" else [] for f in EMPTY}})
    assert "metadata" not in Manifest(out).data["settings"] and panel(app)["metadata"] == EMPTY


@pytest.mark.parametrize("bad, fields", [
    ({"title": "{bogus}", "date_taken": "1987-13", "author": "fine"}, {"title", "date_taken"}),
    ({"copyright": "a {"}, {"copyright"}),
    ({"keywords": ["ok", "{nope}"]}, {"keywords"}),
    ({"description": "x" * 4001}, {"description"}),
    ({"colour": "red"}, {"colour"}),
])
def test_a_metadata_field_that_fails_its_check_is_named_and_nothing_is_saved(app, roll, bad, fields):
    app.json("/api/project/open", {"input": str(roll)})
    status, data = app.json("/api/project/settings", {"margin_mm": 0.3, "metadata": bad})
    assert status == 400 and set(data["fields"]) == fields and data["error"] in data["fields"].values()
    assert panel(app)["margin_mm"] == 0.1 and panel(app)["metadata"] == EMPTY        # not even the margin


def test_the_error_texts_are_those_of_the_api(app, roll):
    app.json("/api/project/open", {"input": str(roll)})
    _, data = app.json("/api/project/settings", {"metadata": {"title": "{bogus}", "date_taken": "1987-13"}})
    with pytest.raises(ValueError) as t:
        api.make_settings(metadata={"title": "{bogus}"})
    with pytest.raises(ValueError) as d:
        api.make_settings(metadata={"date_taken": "1987-13"})
    assert data["fields"] == {"title": str(t.value), "date_taken": str(d.value)} and "bogus" in data["fields"]["title"]
    assert app.json("/api/project/settings", {"metadata": "x"})[1]["fields"].keys() == {"metadata"}
    total = {f: "{stem}" * 10 for f in ("title", "author", "copyright", "description")}      # 2600 each at worst
    status, data = app.json("/api/project/settings", {"metadata": total})      # fine one by one, too long together
    assert status == 400 and set(data["fields"]) == {"metadata"} and "add up" in data["fields"]["metadata"]


# --------------------------------------------------------------------------- the preview
def test_the_preview_fills_the_first_scan_with_the_real_fill_logic(app, roll):
    app.json("/api/project/open", {"input": str(roll)})
    typed = {"title": "Müller {stem} #{n} ({format}, {project})", "description": "Straße {{x}} {n}", "author": "Jörg"}
    status, pv = app.json("/api/project/metadata/preview", {"metadata": typed})
    # not detected yet: the format is not known, the rest is filled
    assert status == 200 and pv["scan"] == "Roll_3.tif" and pv["errors"] == {} and pv["format_known"] is False
    assert pv["title"] == "Müller Roll_3 #3 (?, 260328_Roll)" and pv["description"] == "Straße {x} 3"
    assert panel(app)["metadata"] == EMPTY                                  # a preview saves nothing
    # bad values: the messages by field, 200 (the panel shows them while typing)
    status, pv = app.json("/api/project/metadata/preview", {"metadata": {"title": "{bogus}", "date_taken": "1987-13"}})
    assert status == 200 and set(pv["errors"]) == {"title", "date_taken"}
    # an empty panel previews nothing
    pv = app.json("/api/project/metadata/preview", {"metadata": EMPTY})[1]
    assert pv["title"] == "" and pv["description"] == "" and pv["errors"] == {}
    # after Detect the format is known, and an export writes what the preview showed
    app.json("/api/project/settings", {"metadata": typed})
    app.json("/api/project/all", {})
    app.wait(finished)
    pv = app.json("/api/project/metadata/preview", {"metadata": typed})[1]
    assert pv["format_known"] is True and pv["title"] == "Müller Roll_3 #3 (600, 260328_Roll)"
    out = app.state.project.output
    t = tags(Path(out) / Manifest(out).get("Roll_3.tif")["outputs"]["tiff"])
    assert xp(t[40091]) == pv["title"] and t[270] == "Straße {x} 3"           # XPTitle; ImageDescription = description


def test_the_preview_of_a_project_without_a_manifest_or_input_scans(app, roll):
    app.json("/api/project/open", {"input": str(roll)})
    assert app.json("/api/project/metadata/preview", {"metadata": "nonsense"})[1]["errors"].keys() == {"metadata"}
    assert app.json("/api/project/metadata/preview", {})[0] == 200          # a missing body is an empty panel


# --------------------------------------------------------------------------- defaults for new projects
def test_author_and_copyright_carry_over_to_new_projects_the_rest_does_not(app, roll, scans, tmp_path):
    app.json("/api/project/open", {"input": str(roll)})
    full = {"title": "Roll 3", "author": "Jörg", "copyright": "© {project}", "description": "d", "keywords": ["k"],
            "date_taken": "1987"}
    app.json("/api/project/settings", {"metadata": full})
    assert app.state.store.settings_defaults()["metadata"] == {"author": "Jörg", "copyright": "© {project}"}
    other = tmp_path / "second"
    other.mkdir()
    shutil.copy(scans / "a_white.tif", other / "s.tif")
    app.json("/api/project/close", {})
    app.json("/api/project/open", {"input": str(other)})
    assert panel(app)["metadata"] == {**EMPTY, "author": "Jörg", "copyright": "© {project}"}
    # the first project keeps all its own
    app.json("/api/project/close", {})
    app.json("/api/project/open", {"input": str(roll)})
    assert panel(app)["metadata"] == full
    # the last panel choice wins, an emptied author too
    app.json("/api/project/close", {})
    app.json("/api/project/open", {"input": str(other)})
    app.json("/api/project/settings", {"metadata": {"author": "", "copyright": "(c) new"}})
    assert app.state.store.settings_defaults()["metadata"] == {"author": "", "copyright": "(c) new"}


def test_old_defaults_without_metadata_still_open_new_projects(app, roll):
    app.state.store.set_settings_defaults({"margin_mm": 0.3})                 # a config from before slice 2
    app.json("/api/project/open", {"input": str(roll)})
    assert panel(app)["margin_mm"] == 0.3 and panel(app)["metadata"] == EMPTY


# --------------------------------------------------------------------------- the Export count
def test_a_metadata_only_change_shows_in_the_export_count_and_a_run_clears_it(app, roll):
    app.json("/api/project/open", {"input": str(roll)})
    app.json("/api/project/all", {})
    prog = app.wait(finished)
    assert prog["counts"]["to_export"] == 0 and prog["counts"]["to_headers"] == 0 and prog["counts"]["exported"] == 2
    out = app.state.project.output
    keys_before = {k: (e["export_key"], e["exported_at"]) for k, e in Manifest(out).items()}
    # no metadata anywhere before: setting one shows in the count (the export has no key, the wanted one is not empty)
    app.json("/api/project/settings", {"metadata": {"title": "Müller {stem}"}})
    c = counts(app)
    assert c["to_headers"] == 2 and c["to_export"] == 0 and c["to_retag"] == 0
    assert export_states(app) == {"Roll_3.tif": "headers", "Roll_4.tif": "headers"}
    assert app.json("/api/project/export", {})[0] == 200
    prog = app.wait(finished)
    assert prog["run"]["metadata_updated"] == 2 and prog["run"]["metadata_skipped"] == 0 and prog["run"]["total"] == 0
    assert prog["counts"]["to_headers"] == 0 and prog["counts"]["to_export"] == 0
    assert export_states(app) == {"Roll_3.tif": "done", "Roll_4.tif": "done"}
    m = Manifest(out)
    assert {k: (e["export_key"], e["exported_at"]) for k, e in m.items()} == keys_before      # no pixel work
    assert xp(tags(files_of(out, "Roll_3.tif")["tiff"])[40091]) == "Müller Roll_3"
    # a changed text, then a cleared project: each shows and each run brings it back to 0
    app.json("/api/project/settings", {"metadata": {"title": "Neu {n}"}})
    assert counts(app)["to_headers"] == 2
    app.json("/api/project/export", {})
    assert app.wait(finished)["counts"]["to_headers"] == 0
    assert xp(tags(files_of(out, "Roll_4.tif")["tiff"])[40091]) == "Neu 4"
    app.json("/api/project/settings", {"metadata": {"title": ""}})
    assert counts(app)["to_headers"] == 2                                      # the old headers still say "Neu 4"
    app.json("/api/project/export", {})
    prog = app.wait(finished)
    assert prog["counts"]["to_headers"] == 0 and prog["run"]["metadata_updated"] == 2
    assert 40091 not in tags(files_of(out, "Roll_4.tif")["tiff"])
    assert "metadata_key" not in Manifest(out).get("Roll_4.tif")


def test_headers_and_an_export_that_is_needed_count_once_each(app, roll):
    app.json("/api/project/open", {"input": str(roll)})
    app.json("/api/project/all", {})
    app.wait(finished)
    out = app.state.project.output
    app.json("/api/project/settings", {"metadata": {"title": "X"}})
    api.fix_scan(out, "Roll_3.tif", shifts={"right": [6.0]}, author="agent")        # a new outline: a real export
    c = counts(app)
    assert c["to_export"] == 1 and c["to_headers"] == 1
    assert export_states(app) == {"Roll_3.tif": "outdated", "Roll_4.tif": "headers"}


def test_export_state_uses_the_same_metadata_key_as_the_pipeline(app, roll):
    app.json("/api/project/open", {"input": str(roll)})
    app.json("/api/project/settings", {"metadata": {"title": "{project}/{stem}", "keywords": ["a"]}})
    app.json("/api/project/all", {})
    app.wait(finished)
    m = Manifest(app.state.project.output)
    s = settings_from_dict(m.data["settings"])
    plan = export_plan(m, s)
    for k in ("Roll_3.tif", "Roll_4.tif"):
        assert plan[k][3] == m.get(k)["metadata_key"] != "" and export_state(m, m.get(k), plan[k], s) == "done"


def test_a_project_without_metadata_has_empty_keys_and_no_headers_count(app, roll):
    app.json("/api/project/open", {"input": str(roll)})
    app.json("/api/project/all", {})
    app.wait(finished)
    m = Manifest(app.state.project.output)
    s = settings_from_dict(m.data["settings"])
    assert all(p[3] == "" for p in export_plan(m, s).values()) and all("metadata_key" not in e for _, e in m.items())
    assert counts(app)["to_headers"] == 0


# --------------------------------------------------------------------------- a file that cannot be rewritten
def lock_all(monkeypatch):
    def locked(path, fm):
        raise PermissionError(13, "in use by a viewer", path)

    real = pipeline.set_metadata
    monkeypatch.setattr(pipeline, "set_metadata", locked)
    return real


def test_a_locked_file_is_a_scan_warning_and_a_skipped_count_in_every_interface(project, src, copy, monkeypatch, capsys):
    out = copy(project)
    old = {n: Manifest(str(out)).get(n)["metadata_key"] for n in ("Roll_3.tif", "Roll_4.tif")}
    real = lock_all(monkeypatch)
    emitted = []
    summary = api.process([str(src)], str(out), api.make_settings(workers=1, ppi=PPI, derivatives=["web=800:jpg:85", "png=x0.25:png"],
                                                                   photo_crop="tiff", metadata={"title": "Locked {stem}"}),
                          emit=emitted.append)
    assert summary["metadata_skipped"] == 2 and summary["metadata_updated"] == 0 and summary["exported"] == 0
    assert not summary["failed"] and summary["needs_review"] == []
    for n in old:
        e = Manifest(str(out)).get(n)
        warns = [w for w in e["warnings"] if w.startswith("metadata_not_updated: ")]
        assert len(warns) == 1 and "in use by a viewer" in warns[0] and e["status"] == "ok"
        assert e["metadata_key"] == old[n]                                       # still the old headers, and it says so
    rec = next(r for r in emitted if r.get("file") == "Roll_3.tif")
    assert any(w.startswith("metadata_not_updated: ") for w in rec["warnings"])      # in the streamed result too
    cli._human({**summary})
    printed = capsys.readouterr().out
    assert "metadata NOT updated 2" in printed
    cli._human(rec)
    assert "metadata_not_updated: " in capsys.readouterr().out                 # the warning is not hidden like a hint
    # the next run tries again, finds the file free, and the warning goes
    monkeypatch.setattr(pipeline, "set_metadata", real)
    summary = run(src, out)
    assert summary["metadata_updated"] == 2 and summary["metadata_skipped"] == 0
    assert not any(w.startswith("metadata_not_updated") for _, e in Manifest(str(out)).items() for w in e["warnings"])
    assert xp(tags(files_of(out, "Roll_3.tif")["tiff"])[40091]) == "Locked Roll_3"


def test_the_mcp_result_carries_the_skipped_count_and_the_warning(project, src, copy, monkeypatch):
    out = copy(project)
    lock_all(monkeypatch)
    res = tool_process({"inputs": [str(src)], "output": str(out), "ppi": PPI, "workers": 1,
                        "derivatives": ["web=800:jpg:85", "png=x0.25:png"], "photo_crop": "tiff",
                        "metadata": {"title": "MCP {stem}"}}, lambda *a: None)
    body = res["structuredContent"]
    assert body["summary"]["metadata_skipped"] == 2
    assert all(any(w.startswith("metadata_not_updated: ") for w in r["warnings"]) for r in body["results"])
    desc = next(t for t in TOOLS if t["name"] == "polaroid_process")["description"]
    assert "metadata_skipped" in desc and "metadata_not_updated" in desc


def test_a_normal_run_reports_no_skipped_files(project, src, copy):
    out = copy(project)
    summary = run(src, out, metadata={"title": "Again {stem}"})
    assert summary["metadata_updated"] == 2 and summary["metadata_skipped"] == 0
    assert not any(w.startswith("metadata_not_updated") for _, e in Manifest(str(out)).items() for w in e["warnings"])


def test_the_app_shows_a_locked_file_and_counts_it_until_a_run_gets_through(app, roll, monkeypatch):
    app.json("/api/project/open", {"input": str(roll)})
    app.json("/api/project/all", {})
    app.wait(finished)
    app.json("/api/project/settings", {"metadata": {"title": "Müller {stem}"}})
    real = lock_all(monkeypatch)
    app.json("/api/project/export", {})
    prog = app.wait(finished)
    assert prog["run"]["metadata_skipped"] == 2 and prog["run"]["metadata_updated"] == 0 and prog["run"]["state"] == "finished"
    assert prog["counts"]["to_headers"] == 2                                      # still to do
    scans_ = {s["key"]: s for s in app.json("/api/project/scans")[1]["scans"]}
    assert all(any(w.startswith("metadata_not_updated: ") for w in s["warnings"]) and s["export"] == "headers"
               for k, s in scans_.items() if k.startswith("Roll"))
    monkeypatch.setattr(pipeline, "set_metadata", real)
    app.json("/api/project/export", {})
    prog = app.wait(finished)
    assert prog["run"]["metadata_updated"] == 2 and prog["run"]["metadata_skipped"] == 0
    assert prog["counts"]["to_headers"] == 0
    scans_ = app.json("/api/project/scans")[1]["scans"]
    assert not any(w.startswith("metadata_not_updated") for s in scans_ for w in s["warnings"])


# --------------------------------------------------------------------------- fix round 1 of the review
def test_saving_an_unrelated_setting_keeps_keywords_with_a_comma_and_the_metadata_key(app, roll):
    """The page sends a metadata field only when its text changed since it was shown; the server side of that contract:
    a margin-only save leaves a keyword with a comma (set by CLI/MCP) and a CRLF description exactly as they are."""
    app.json("/api/project/open", {"input": str(roll)})
    app.json("/api/project/all", {})
    app.wait(finished)
    out = app.state.project.output
    api.process([str(roll)], out, api.make_settings(ppi=PPI, workers=1, metadata={
        "keywords": ["Hamburg, Germany", "Garten"], "description": "Zeile 1\r\nZeile 2"}))
    keys = {k: e["metadata_key"] for k, e in Manifest(out).items()}
    assert panel(app)["metadata"]["keywords"] == ["Hamburg, Germany", "Garten"]
    status, data = app.json("/api/project/settings", {"margin_mm": 0.1})          # what the page sends for "nothing edited"
    assert status == 200 and data["settings"]["metadata"]["keywords"] == ["Hamburg, Germany", "Garten"]
    assert {k: e["metadata_key"] for k, e in Manifest(out).items()} == keys and counts(app)["to_headers"] == 0


def test_the_total_limit_holds_for_the_merged_result_across_saves(app, roll):
    app.json("/api/project/open", {"input": str(roll)})
    big = "{stem}" * 12                                                   # 3120 characters filled at the worst
    assert app.json("/api/project/settings", {"metadata": {"title": big}})[0] == 200
    assert app.json("/api/project/settings", {"metadata": {"description": big}})[0] == 200      # 6240 together
    status, data = app.json("/api/project/settings", {"metadata": {"author": "{stem}" * 10}})  # 8840: too much
    assert status == 400 and set(data["fields"]) == {"metadata"} and "add up" in data["fields"]["metadata"]
    assert panel(app)["metadata"]["author"] == "" and panel(app)["metadata"]["title"] == big      # nothing of the third save
    # clearing a field makes room again
    assert app.json("/api/project/settings", {"metadata": {"description": "", "author": "{stem}" * 10}})[0] == 200


def test_the_total_limit_across_saves_in_merge_and_for_api_cli_and_mcp(project, src, copy, capsys):
    from polaroid_processor import metadata as M
    assert M.merge({"title": "{stem}" * 12}, {"description": "{stem}" * 12}) == {"title": "{stem}" * 12, "description": "{stem}" * 12}
    with pytest.raises(ValueError, match="add up"):
        M.merge({"title": "{stem}" * 12, "description": "{stem}" * 12}, {"author": "{stem}" * 10})
    # a saved value that fails the checks on its own (hand-edited) is not counted, and is kept
    kept = M.merge({"title": "x" * 600, "description": "{stem}" * 12}, {"author": "{stem}" * 10})
    assert kept["title"] == "x" * 600 and kept["author"] == "{stem}" * 10
    out = copy(project)
    p = out / MANIFEST_NAME
    m = json.loads(p.read_text(encoding="utf-8"))
    m["settings"]["metadata"] = {"title": "{stem}" * 12, "description": "{stem}" * 12}
    p.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    before = p.read_bytes()
    more = {"author": "{stem}" * 10}
    with pytest.raises(ValueError, match="add up"):                            # API: before anything is written
        api.process([str(src)], str(out), api.make_settings(workers=1, ppi=PPI, metadata=more))
    assert cli.main(["process", str(src), "-o", str(out), "--author", "{stem}" * 10]) == 2
    assert "add up" in capsys.readouterr().err
    with pytest.raises(ValueError, match="add up"):
        tool_process({"inputs": [str(src)], "output": str(out), "metadata": more}, lambda *a: None)
    assert p.read_bytes() == before                                              # the manifest was not touched
    ok = api.process([str(src)], str(out), api.make_settings(workers=1, ppi=PPI, metadata={"author": "ok"}))
    assert not ok["failed"]


def test_defaults_leave_out_a_saved_author_or_copyright_that_fails_the_checks(app, roll):
    from polaroid_processor.app.settings import defaults_of
    assert defaults_of({"metadata": {"author": "a {", "copyright": "ok"}})["metadata"] == {"author": "", "copyright": "ok"}
    # a hand-edited manifest with such an author must not break new projects after a settings save
    app.json("/api/project/open", {"input": str(roll)})
    app.json("/api/project/detect", {})
    app.wait(finished)
    out = app.state.project.output
    m = Manifest(out)
    m.data["settings"]["metadata"] = {"author": "bad {", "title": "t"}
    m.save()
    assert app.json("/api/project/settings", {"margin_mm": 0.2})[0] == 200
    assert app.state.store.settings_defaults()["metadata"] == {"author": "", "copyright": ""}
    other = roll.parent / "second"
    other.mkdir()
    shutil.copy(roll / "Roll_3.tif", other / "s.tif")
    app.json("/api/project/close", {})
    assert app.json("/api/project/open", {"input": str(other)})[0] == 200
    assert panel(app)["metadata"] == EMPTY


# --------------------------------------------------------------------------- nits B and C of the slice 2 review
BIG = "{stem}" * 12                                                       # 3120 characters filled at the worst case


def over_limit_manifest(out):
    """Hand-edit a saved metadata that is already over the 8000 characters (as through several runs of 9a33b84)."""
    p = out / MANIFEST_NAME
    m = json.loads(p.read_text(encoding="utf-8"))
    m["settings"]["metadata"] = {"title": BIG, "author": BIG, "copyright": BIG}
    p.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    return p


def test_a_call_that_sets_nothing_is_not_refused_on_a_project_over_the_limit(project, src, copy):
    from polaroid_processor import metadata as M
    stored = {"title": BIG, "author": BIG, "copyright": BIG}
    for nothing in (None, {}, {"title": None}, {"title": None, "keywords": None, "date_taken": None}):
        assert M.merge(stored, nothing) == stored, nothing
    assert M.merge(stored, {"author": ""}) == {"title": BIG, "copyright": BIG}      # clearing never adds: not refused either
    assert M.merge(stored, {"keywords": [], "description": ""}) == stored
    with pytest.raises(ValueError, match="add up"):                                 # setting something is still checked
        M.merge(stored, {"description": "x"})
    with pytest.raises(ValueError, match="add up"):
        M.merge(stored, {"title": BIG})                                              # also a field replaced by an equal one
    out = copy(project)
    p = over_limit_manifest(out)
    before = json.loads(p.read_text(encoding="utf-8"))["settings"]["metadata"]
    for given in ({}, {"title": None}):                                              # API and MCP: runs, keeps what is saved
        summary = api.process([str(src)], str(out), api.make_settings(workers=1, ppi=PPI, metadata=given))
        assert not summary["failed"]
        assert Manifest(out).data["settings"]["metadata"] == before
    res = tool_process({"inputs": [str(src)], "output": str(out), "ppi": PPI, "workers": 1, "metadata": {}}, lambda *a: None)
    assert not res["structuredContent"]["summary"]["failed"]
    assert Manifest(out).data["settings"]["metadata"] == before


def test_the_total_limit_message_names_the_saved_fields_that_count(app, roll, project, src, copy, capsys):
    from polaroid_processor import metadata as M
    saved = {"title": BIG, "description": BIG}
    with pytest.raises(ValueError) as e:
        M.merge(saved, {"author": "{stem}" * 10})
    assert str(e.value) == ("metadata texts add up to more than 8000 characters, with the output folder's title, description: "
                            "shorten or clear one")
    saved_kw = {"title": BIG, "keywords": [f"{{stem}}-{i}" for i in range(11)]}      # keywords count too (11 x 266 = 2926)
    with pytest.raises(ValueError) as e:
        M.merge({**saved_kw, "copyright": BIG}, {"author": "{stem}" * 10})
    assert "with the output folder's title, copyright, keywords: shorten or clear one" in str(e.value)
    with pytest.raises(ValueError) as e:                                              # a field the call replaces is not "saved"
        M.merge({"title": BIG, "description": BIG}, {"title": BIG, "author": BIG})
    assert str(e.value).endswith("with the output folder's description: shorten or clear one")
    # nothing saved counts: the call alone is too long, so the message is the plain one
    with pytest.raises(ValueError) as e:
        M.normalize({f: "{stem}" * 10 for f in ("title", "author", "copyright", "description")})
    assert str(e.value) == "metadata texts add up to more than 8000 characters"
    # API, CLI (exit 2) and the panel (400) all carry it
    out = copy(project)
    m = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    m["settings"]["metadata"] = saved
    (out / MANIFEST_NAME).write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    tail = "with the output folder's title, description: shorten or clear one"
    with pytest.raises(ValueError, match=tail):
        api.process([str(src)], str(out), api.make_settings(workers=1, ppi=PPI, metadata={"author": "{stem}" * 10}))
    capsys.readouterr()
    assert cli.main(["process", str(src), "-o", str(out), "--author", "{stem}" * 10]) == 2
    assert tail in capsys.readouterr().err
    app.json("/api/project/open", {"input": str(roll)})
    assert app.json("/api/project/settings", {"metadata": saved})[0] == 200
    status, data = app.json("/api/project/settings", {"metadata": {"author": "{stem}" * 10}})
    assert status == 400 and data["fields"] == {"metadata": f"metadata texts add up to more than 8000 characters, with the output folder's title, description: shorten or clear one"}
    assert tail in data["error"]
