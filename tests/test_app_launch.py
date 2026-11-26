"""The v2 app, slice 3: settings panel, `polaroid app` / `polaroid review`, server reuse, scale retag rule."""
import http.server
import json
import shutil
import threading
import time
import urllib.request

import numpy as np
import pytest
import tifffile

from polaroid_processor import api
import importlib

launch_mod = importlib.import_module("polaroid_processor.app.launch")   # (the package re-exports the function launch)
from polaroid_processor.app.launch import launch, running
from polaroid_processor.app.settings import from_panel, to_panel
from polaroid_processor.cli import main
from polaroid_processor.manifest import Manifest
from polaroid_processor.pipeline import settings_from_dict

from test_app import AppClient, PPI, app, fake_running as fake_running_state, finished, scans, settings, src  # noqa: F401


@pytest.fixture
def home(tmp_path, monkeypatch):
    """The user's app data in a temp folder, so a test never touches the real one."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    return tmp_path / "appdata" / "polaroid-processor" / "app.json"


def empty_output(tmp_path, name="out"):
    out = tmp_path / name
    Manifest(str(out)).save()           # a manifest without scans is enough to open it
    return out


# --------------------------------------------------------------------------- settings
def test_panel_fields_round_trip_and_keep_other_derivatives():
    base = {"margin_mm": 0.2, "flatten": "off", "derivatives": [
        {"name": "high", "long_edge": 5000, "scale": None, "fmt": "jpg", "quality": 90},
        {"name": "png", "long_edge": None, "scale": 0.25, "fmt": "png", "quality": 92}]}
    panel = to_panel(base)
    assert panel["margin_mm"] == 0.2 and panel["flatten"] == "off" and panel["shadows"] == "auto"
    assert panel["high_on"] and panel["high_edge"] == 5000 and panel["high_quality"] == 90
    assert not panel["web_on"] and panel["other_derivatives"] == 1 and panel["ppi"] is None
    new = from_panel(base, {"web_on": True, "web_edge": 1500, "high_on": False, "ppi": 1270, "overlay": True,
                            "tiff_compression": "none", "interpolation": "lanczos"})
    s = settings_from_dict(new)
    assert [(d.name, d.long_edge) for d in s.derivatives] == [("png", None), ("web", 1500)]
    assert s.ppi == 1270 and s.overlay and s.tiff_compression == "none" and s.interpolation == "lanczos"
    assert to_panel(new)["web_on"] and not to_panel(new)["high_on"]
    assert to_panel(None)["margin_mm"] == 0.1 and to_panel(None)["high_edge"] == 6000 and to_panel(None)["web_edge"] == 2048


def test_photo_crop_panel_fields():
    assert to_panel(None)["photo_crop"] == "off" and to_panel(None)["photo_bleed_mm"] == 0.3
    new = from_panel(None, {"photo_crop": "tiff", "photo_bleed_mm": 0.4})
    s = settings_from_dict(new)
    assert (s.photo_crop, s.photo_bleed_mm) == ("tiff", 0.4) and to_panel(new)["photo_crop"] == "tiff"
    back = from_panel(new, {"photo_crop": "off", "photo_bleed_mm": 0.3})
    assert "photo_crop" not in back and "photo_bleed_mm" not in back          # defaults stay out of the manifest


@pytest.mark.parametrize("bad", [{"photo_crop": "all"}, {"photo_crop": True}, {"photo_bleed_mm": -0.1},
                                 {"photo_bleed_mm": 9}, {"photo_bleed_mm": "x"}])
def test_panel_refuses_bad_photo_values(bad):
    with pytest.raises(api.UsageError):
        from_panel(None, bad)


@pytest.mark.parametrize("bad", [{"margin_mm": -1}, {"margin_mm": "x"}, {"flatten": "maybe"}, {"shadows": 1},
                                 {"tiff_compression": "lzw"}, {"interpolation": "nearest"}, {"ppi": 5},
                                 {"high_quality": 500}, {"web_edge": 5}, {"nope": 1}, {"ppi": 1200, "profile": "epson"}])
def test_panel_refuses_bad_values(bad):
    with pytest.raises(api.UsageError):
        from_panel(None, bad)


def test_settings_through_the_api_and_the_defaults_for_new_projects(app, src, tmp_path):
    status, data = app.json("/api/project/settings")
    assert status == 400                                       # no project open
    app.json("/api/project/open", {"input": str(src)})
    status, data = app.json("/api/project/settings")
    assert status == 200 and data["settings"]["margin_mm"] == 0.1 and data["settings"]["web_on"] and not data["running"]
    status, data = app.json("/api/project/settings", {"margin_mm": 0.3, "flatten": "off", "web_on": False,
                                                      "high_edge": 4000, "ppi": 1200})
    assert status == 200 and data["settings"]["margin_mm"] == 0.3 and not data["settings"]["web_on"]
    assert not (tmp_path / "scans_processed").exists()          # nothing is created before the first run
    # the first run takes them over into the manifest
    app.json("/api/project/detect", {})
    app.wait(finished)
    saved = Manifest(app.state.project.output).data["settings"]
    assert saved["margin_mm"] == 0.3 and saved["flatten"] == "off" and saved["ppi"] == 1200
    assert [(d["name"], d["long_edge"]) for d in saved["derivatives"]] == [("high", 4000)]
    # a new project starts from the last choices, except the scale
    other = tmp_path / "other"
    other.mkdir()
    shutil.copy(src / "a_white.tif", other / "o.tif")
    app.json("/api/project/close", {})
    app.json("/api/project/open", {"input": str(other)})
    fresh = app.json("/api/project/settings")[1]["settings"]
    assert fresh["margin_mm"] == 0.3 and fresh["flatten"] == "off" and fresh["high_edge"] == 4000 and not fresh["web_on"]
    assert fresh["ppi"] is None
    # the first project keeps its own
    app.json("/api/project/close", {})
    app.json("/api/project/open", {"output": app.state.store.projects()[1]["output"]})
    assert app.json("/api/project/settings")[1]["settings"]["ppi"] == 1200


def test_bad_settings_and_a_run_going_are_refused(app, src):
    app.json("/api/project/open", {"input": str(src)})
    for bad in ({"margin_mm": -2}, {"flatten": "x"}, {"mystery": 1}, {"ppi": 1200, "profile": "p"}):
        assert app.json("/api/project/settings", bad)[0] == 400, bad
    assert app.json("/api/project/settings")[1]["settings"]["margin_mm"] == 0.1
    run = fake_running_state(app.state.project)
    status, data = app.json("/api/project/settings", {"margin_mm": 0.2})
    assert status == 400 and "wait until the run ends" in data["error"]
    assert app.json("/api/project/settings")[1]["running"]
    run.state = "finished"
    assert app.json("/api/project/settings", {"margin_mm": 0.2})[0] == 200


def test_a_settings_change_shows_in_the_export_count(app, src):
    app.json("/api/project/open", {"input": str(src)})
    app.json("/api/project/all", {})
    prog = app.wait(finished)
    assert prog["counts"]["to_export"] == 0
    app.json("/api/project/settings", {"margin_mm": 0.4})
    prog = app.json("/api/project/progress")[1]
    assert prog["counts"]["to_export"] == 2
    app.json("/api/project/settings", {"margin_mm": 0.1})
    assert app.json("/api/project/progress")[1]["counts"]["to_export"] == 0
    app.json("/api/project/settings", {"web_on": False})            # fewer derivatives: a new export too
    assert app.json("/api/project/progress")[1]["counts"]["to_export"] == 2
    app.json("/api/project/settings", {"web_on": True, "ppi": 1200})  # a deliberate scale: a new export
    assert app.json("/api/project/progress")[1]["counts"]["to_export"] == 2
    app.json("/api/project/settings", {"web_on": True, "ppi": None})      # back to the exports made at the start
    assert app.json("/api/project/progress")[1]["counts"]["to_export"] == 0
    for update in ({"photo_crop": "derivatives"}, {"photo_crop": "tiff"}, {"photo_bleed_mm": 0.5},
                   {"photo_crop": "off", "photo_bleed_mm": 0.3}):         # each is a new export; back to off is the old one
        app.json("/api/project/settings", update)
        want = 0 if update.get("photo_crop") == "off" else 2
        assert app.json("/api/project/progress")[1]["counts"]["to_export"] == want, update


# --------------------------------------------------------------------------- polaroid app / review
@pytest.fixture
def served(monkeypatch):
    """`launch` without a real server: the arguments of `serve` are recorded."""
    calls = []

    def fake_serve(state, port, **kw):
        calls.append({"state": state, "port": port, **kw})

    monkeypatch.setattr(launch_mod, "serve", fake_serve)
    monkeypatch.setattr(launch_mod, "running", lambda port: False)      # whatever runs on this machine
    return calls


def test_app_command_makes_projects_of_folders(served, home, src, tmp_path, capsys):
    other = tmp_path / "second"
    other.mkdir()
    shutil.copy(src / "a_white.tif", other / "s.tif")
    cfg = tmp_path / "cfg.json"
    assert main(["app", str(src), str(other), "--no-browser", "--port", "1", "--config", str(cfg)]) == 0
    (call,) = served
    assert call["port"] == 1 and call["open_browser"] is False and call["path"] == "/" and call["fallback"] is False
    state = call["state"]
    assert state.project.input == str(src) and state.project.output == str(src) + "_processed"
    assert {p["output"] for p in state.store.projects()} == {str(src) + "_processed", str(other) + "_processed"}
    assert not (tmp_path / "scans_processed").exists()
    # no folder: the home page; a bad single folder is a usage error
    served.clear()
    assert main(["app", "--no-browser", "--config", str(cfg)]) == 0
    assert served[0]["state"].project is None and served[0]["port"] == 8780 and served[0]["fallback"] is True
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["app", str(empty), "--no-browser", "--config", str(cfg)]) == 2
    assert "no .tif/.tiff scans" in capsys.readouterr().err


def test_a_scan_folder_dropped_with_an_output_folder_opens_the_output(served, home, src, tmp_path):
    out = tmp_path / "res"
    api.process([str(src)], str(out), settings(export=False))
    assert main(["app", str(out), "--no-browser", "--config", str(tmp_path / "c.json")]) == 0
    assert served[0]["state"].project.output == str(out) and served[0]["state"].project.input == str(src)


def test_review_command_opens_the_app_on_the_output(served, home, tmp_path, capsys):
    out = empty_output(tmp_path)
    assert main(["review", str(out), "--all", "--no-browser", "--port", "5"]) == 0
    call = served[-1]
    assert call["port"] == 5 and call["open_browser"] is False and call["path"] == "/" and call["fallback"] is False
    assert call["state"].project.output == str(out)
    assert main(["review", str(out), "--no-browser", "--author", "alex", "--export"]) == 0
    call = served[-1]
    assert call["path"] == "/?filter=review" and call["state"].author == "alex" and call["fallback"] is True
    assert "--export is ignored" in capsys.readouterr().err
    assert main(["review", str(tmp_path / "missing"), "--no-browser"]) == 2
    assert "not an output folder" in capsys.readouterr().err


def test_review_keeps_the_dev_workflow_flags(served, home, tmp_path):
    out = empty_output(tmp_path)
    assert main(["review", str(out), "--all", "--no-browser", "--port", "8771"]) == 0
    assert served[-1]["port"] == 8771


def test_ping_and_running_find_only_the_app(app, tmp_path):
    status, data = app.json("/api/ping")
    assert status == 200 and data["app"] == "polaroid-app"
    assert running(app.port)

    class Other(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"hello")

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Other)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert not running(srv.server_address[1])
    finally:
        srv.shutdown()
        srv.server_close()
    assert not running(srv.server_address[1])


def test_a_running_app_is_reused_and_pointed_at_the_project(app, home, tmp_path, monkeypatch, src):
    opened, said = [], []
    monkeypatch.setattr(launch_mod.webbrowser, "open", opened.append)

    def no_second_server(*a, **k):
        raise AssertionError("a second server was started")

    monkeypatch.setattr(launch_mod, "serve", no_second_server)
    out = empty_output(tmp_path, "res")
    assert launch(output=str(out), port=app.port, announce=said.append, filter_name="needs_review") == 0
    assert app.state.project.output == str(out)
    assert opened == [f"http://127.0.0.1:{app.port}/?filter=needs_review"] and "already running" in said[-1]
    # a scan folder dropped on a running app becomes a project there
    assert launch([str(src)], port=app.port, open_browser=False, announce=said.append) == 0
    assert app.state.project.input == str(src) and len(opened) == 1
    # a bad folder is refused with the app's message
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(api.UsageError, match="no .tif/.tiff scans"):
        launch([str(empty)], port=app.port, open_browser=False, announce=said.append)
    # the app keeps its project while a run goes there; the browser still opens on it
    fake_running_state(app.state.project)
    keep = app.state.project.output
    assert launch(output=str(out), port=app.port, open_browser=True, announce=said.append) == 0
    assert app.state.project.output == keep and any(m.startswith("warning: a run is going") for m in said)
    assert len(opened) == 2
    app.state.project.run.state = "finished"


def test_open_refuses_to_leave_a_running_project(app, src, tmp_path):
    app.json("/api/project/open", {"input": str(src)})
    fake_running_state(app.state.project)
    out = empty_output(tmp_path, "res")
    status, data = app.json("/api/project/open", {"output": str(out)})
    assert status == 400 and "a run is going" in data["error"]
    assert app.json("/api/project/open", {"input": str(src)})[0] == 200          # the same project: nothing changes
    app.state.project.run.state = "finished"
    assert app.json("/api/project/open", {"output": str(out)})[0] == 200


def test_a_busy_explicit_port_is_an_error(home, tmp_path, monkeypatch):
    def busy(*a, **k):
        raise OSError("address in use")

    monkeypatch.setattr(launch_mod, "running", lambda port: False)
    monkeypatch.setattr(launch_mod, "serve", busy)
    with pytest.raises(api.UsageError, match="not available"):
        launch(port=8123, open_browser=False, announce=lambda m: None, config=str(tmp_path / "c.json"))


# --------------------------------------------------------------------------- pass A exports at a provisional scale
def test_a_fresh_export_cut_at_a_scale_more_than_half_a_percent_off_is_exported_again(tmp_path):
    from synth import make_scan
    src = tmp_path / "in"
    src.mkdir()
    for i in range(6):
        rgb, _ = make_scan("white", 0.2 * i, seed=90 + i)
        tifffile.imwrite(str(src / f"q_{i}.tif"), rgb, photometric="rgb")
    out = tmp_path / "out"
    detect_only = api.make_settings(workers=1, export=False)
    api.process([str(src / f"q_{i}.tif") for i in range(3)], str(out), detect_only)
    for i in range(3):                                          # three big prints: the history says a larger scale
        api.fix_scan(str(out), f"q_{i}.tif", shifts={"right": [60.0]}, author="agent")
    jobs = []
    summary = api.process([str(src)], str(out), api.make_settings(workers=1, no_derivatives=True), on_jobs=jobs.append)
    m = Manifest(str(out))
    new = [f"q_{i}.tif" for i in range(3, 6)]
    # pass A cut the new scans at the history's scale (three big prints); the batch median came out about 1 % lower:
    # they are exported again, so pass B has all six scans (the three fixed ones are new to it too)
    assert jobs == [3, 6] and summary["retagged"] == 0
    for k in new:
        assert abs(m.get(k)["export_geom_ppm"] - m.get(k)["scale"]["px_per_mm"]) < 1e-3


# --------------------------------------------------------------------------- review of slice 3
def test_a_second_server_cannot_bind_a_busy_port_and_the_default_port_falls_back(tmp_path):
    from polaroid_processor.app import AppState, make_server
    from polaroid_processor.app.server import serve
    first = make_server(AppState(tmp_path / "a.json"), 0)
    port = first.server_address[1]
    try:
        with pytest.raises(OSError):                             # Windows used to let a second socket share the port
            make_server(AppState(tmp_path / "b.json"), port)
        with pytest.raises(OSError):
            serve(AppState(tmp_path / "c.json"), port, open_browser=False, announce=lambda m: None)
        said = []
        t = threading.Thread(target=serve, args=(AppState(tmp_path / "d.json"), port),
                             kwargs={"open_browser": False, "announce": said.append, "fallback": True}, daemon=True)
        t.start()
        end = time.time() + 10
        while not said and time.time() < end:
            time.sleep(0.05)
        url = said[0].split()[2]
        assert f":{port}/" not in url and running(int(url.split(":")[2].rstrip("/")))    # another port, a live app
        assert running(port) is False                            # the first one is not an app (nothing serves it)
    finally:
        first.server_close()


@pytest.mark.parametrize("bad", [{"overlay": "false"}, {"web_on": "false"}, {"high_on": 0}, {"overlay": 1}])
def test_panel_flags_must_be_real_bools(bad):
    with pytest.raises(api.UsageError, match="true or false"):
        from_panel(None, bad)


def test_an_unknown_profile_is_refused_a_known_one_accepted(tmp_path, monkeypatch):
    profiles = tmp_path / "profiles.yaml"
    profiles.write_text("profiles:\n  epson-test: 3200\n", encoding="utf-8")
    monkeypatch.setenv("POLAROID_PROFILES", str(profiles))
    with pytest.raises(api.UsageError, match="nope"):
        from_panel(None, {"profile": "nope"})
    assert from_panel(None, {"profile": "epson-test"})["profile"] == "epson-test"


def test_a_margin_only_save_keeps_a_derivative_made_on_the_command_line():
    base = {"derivatives": [{"name": "high", "long_edge": 6000, "scale": None, "fmt": "jpg", "quality": 95},
                            {"name": "web", "long_edge": None, "scale": 0.5, "fmt": "png", "quality": 92}]}
    kept = settings_from_dict(from_panel(base, {"margin_mm": 0.2}))
    assert kept.margin_mm == 0.2
    web = [d for d in kept.derivatives if d.name == "web"][0]
    assert (web.scale, web.fmt, web.long_edge) == (0.5, "png", None)            # untouched
    assert [d.name for d in kept.derivatives] == ["high", "web"]
    changed = settings_from_dict(from_panel(base, {"web_quality": 80}))        # edited on purpose: a jpg by edge
    web = [d for d in changed.derivatives if d.name == "web"][0]
    assert (web.scale, web.fmt, web.long_edge, web.quality) == (None, "jpg", 2048, 80)


def test_reuse_says_that_author_is_not_applied(app, home, tmp_path, monkeypatch):
    said = []
    out = empty_output(tmp_path, "res")
    assert launch(output=str(out), port=app.port, open_browser=False, author="alex", announce=said.append) == 0
    assert any("author" in m and "ignored" in m for m in said)
    said.clear()
    launch(output=str(out), port=app.port, open_browser=False, announce=said.append)
    assert not any("author" in m for m in said)


def test_running_wants_a_version_and_a_pid_not_just_a_name():
    class Fake(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = json.dumps({"app": "polaroid-app"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert not running(srv.server_address[1])
    finally:
        srv.shutdown()
        srv.server_close()


def test_a_drive_root_is_not_a_project(tmp_path):
    from polaroid_processor.app.server import default_output
    with pytest.raises(api.UsageError, match="drive root"):
        default_output(tmp_path.anchor)


# --------------------------------------------------------------------------- the scanner profile dropdown (server side)
@pytest.fixture
def profiles_file(tmp_path, monkeypatch):
    """A temp profiles file named by POLAROID_PROFILES (the user's real file is never read or written by these tests)."""
    p = tmp_path / "profiles.yaml"
    p.write_text("profiles:\n  plustek: 3600\n  Epson-test: 3200\n  alpha: {ppi: 2400.5}\n", encoding="utf-8")
    monkeypatch.setenv("POLAROID_PROFILES", str(p))
    return p


def test_the_settings_payload_lists_the_profiles_sorted_by_name(app, src, profiles_file):
    app.json("/api/project/open", {"input": str(src)})
    status, data = app.json("/api/project/settings")
    assert status == 200 and data["profiles"]["error"] is None and data["profiles"]["path"] == str(profiles_file)
    assert data["profiles"]["items"] == [{"name": "alpha", "ppi": 2400.5}, {"name": "Epson-test", "ppi": 3200.0},
                                         {"name": "plustek", "ppi": 3600.0}]      # by name, ignoring case
    assert data["settings"]["profile"] is None
    # choosing one saves it, and the list stays; the payload of the save carries it too
    status, data = app.json("/api/project/settings", {"profile": "plustek"})
    assert status == 200 and data["settings"]["profile"] == "plustek" and len(data["profiles"]["items"]) == 3


def test_the_projects_own_profiles_file_wins_over_the_environment(app, src, profiles_file, tmp_path):
    from polaroid_processor.app.settings import profile_options
    own = tmp_path / "own.yaml"
    own.write_text("profiles:\n  mine: 1200\n", encoding="utf-8")
    assert [i["name"] for i in profile_options(str(own))["items"]] == ["mine"]
    assert profile_options(str(own))["path"] == str(own)
    assert profile_options()["path"] == str(profiles_file)


def test_no_empty_or_unreadable_profiles_file_gives_an_empty_list_and_the_path(app, src, tmp_path, monkeypatch):
    from polaroid_processor.app.settings import profile_options
    missing = tmp_path / "nowhere" / "profiles.yaml"
    monkeypatch.setenv("POLAROID_PROFILES", str(missing))
    assert profile_options() == {"path": str(missing), "items": [], "error": None}
    missing.parent.mkdir()
    missing.write_text("", encoding="utf-8")
    assert profile_options() == {"path": str(missing), "items": [], "error": None}
    missing.write_text("profiles: {a: [1, 2]}\n", encoding="utf-8")                 # a value that is no number
    bad = profile_options()
    assert bad["items"] == [] and bad["error"] and bad["path"] == str(missing)
    missing.write_text("profiles: [unclosed\n", encoding="utf-8")                  # not YAML
    assert profile_options()["error"] and profile_options()["items"] == []
    app.json("/api/project/open", {"input": str(src)})                             # the panel still loads
    status, data = app.json("/api/project/settings")
    assert status == 200 and data["profiles"]["error"] and data["settings"]["margin_mm"] == 0.1


def test_the_dropdown_rule_either_ppi_or_a_profile_and_unknown_profiles(app, src, profiles_file):
    app.json("/api/project/open", {"input": str(src)})
    status, data = app.json("/api/project/settings", {"ppi": 1200, "profile": "plustek"})
    assert status == 400 and data["error"] == "set either ppi or a scanner profile, not both"
    assert app.json("/api/project/settings", {"ppi": 1200})[0] == 200
    status, data = app.json("/api/project/settings", {"profile": "plustek"})        # ppi is still set
    assert status == 400 and data["error"] == "set either ppi or a scanner profile, not both"
    assert app.json("/api/project/settings", {"ppi": None, "profile": "plustek"})[0] == 200
    status, data = app.json("/api/project/settings", {"profile": "nope"})
    assert status == 400 and "nope" in data["error"] and str(profiles_file) in data["error"]
    assert app.json("/api/project/settings")[1]["settings"]["profile"] == "plustek"


@pytest.fixture
def missing_profile(app, src, profiles_file):
    """A project that saved the profile ``plustek``, which the profiles file has lost since."""
    app.json("/api/project/open", {"input": str(src)})
    assert app.json("/api/project/settings", {"profile": "plustek"})[0] == 200
    profiles_file.write_text("profiles:\n  other: 1000\n", encoding="utf-8")        # plustek is gone from the file
    return app


def test_a_saved_profile_missing_from_the_file_is_kept_and_shown(missing_profile):
    status, data = missing_profile.json("/api/project/settings")
    assert status == 200 and data["settings"]["profile"] == "plustek"             # still there (the page keeps it selected)
    assert [i["name"] for i in data["profiles"]["items"]] == ["other"]


def test_an_unrelated_save_passes_with_a_saved_profile_missing_from_the_file(missing_profile):
    """Only a newly set or changed profile is checked against the file; other settings save, and the profile stays."""
    app = missing_profile
    status, data = app.json("/api/project/settings", {"margin_mm": 0.3, "metadata": {"title": "T {stem}"}})
    assert status == 200 and data["settings"]["margin_mm"] == 0.3 and data["settings"]["metadata"]["title"] == "T {stem}"
    assert data["settings"]["profile"] == "plustek"
    # naming the saved profile again is no change either
    assert app.json("/api/project/settings", {"profile": "plustek", "flatten": "off"})[0] == 200
    assert app.json("/api/project/settings")[1]["settings"]["profile"] == "plustek"


def test_a_run_with_a_saved_profile_missing_from_the_file_still_fails_with_its_message(missing_profile):
    app = missing_profile
    assert app.json("/api/project/settings", {"margin_mm": 0.3})[0] == 200
    app.json("/api/project/detect", {})
    prog = app.wait(lambda p: p["run"] and p["run"]["state"] in ("error", "finished"))
    assert prog["run"]["state"] == "error"
    assert "scanner profile 'plustek' not found in" in prog["run"]["error"]


def test_choosing_another_unknown_profile_is_refused_a_known_one_accepted(missing_profile):
    app = missing_profile
    status, data = app.json("/api/project/settings", {"profile": "nope", "margin_mm": 0.3})
    assert status == 400 and "nope" in data["error"] and "not found" in data["error"]
    got = app.json("/api/project/settings")[1]["settings"]
    assert got["profile"] == "plustek" and got["margin_mm"] == 0.1                  # nothing of that save
    status, data = app.json("/api/project/settings", {"profile": "other"})
    assert status == 200 and data["settings"]["profile"] == "other"


def test_none_with_the_ppi_in_one_save_replaces_a_missing_profile(missing_profile):
    """What the page's hint offers: choose "none" and type the ppi in the same save; the scale stays."""
    app = missing_profile
    status, data = app.json("/api/project/settings", {"profile": "", "ppi": 3600})
    assert status == 200 and data["settings"]["profile"] is None and data["settings"]["ppi"] == 3600
    assert app.json("/api/project/settings", {"margin_mm": 0.3})[0] == 200          # and later saves are plain


def test_a_ppi_without_clearing_a_missing_profile_is_refused_as_both(missing_profile):
    status, data = missing_profile.json("/api/project/settings", {"ppi": 3600})
    assert status == 400 and data["error"] == "set either ppi or a scanner profile, not both"


def test_profiles_that_are_not_a_positive_finite_number_are_left_out_and_reported(tmp_path, monkeypatch):
    """``.inf`` / ``.nan`` would put Infinity / NaN into the settings JSON, which the page cannot parse."""
    from polaroid_processor.app.settings import profile_options
    p = tmp_path / "profiles.yaml"
    p.write_text("profiles:\n  good: 3200\n  inf: .inf\n  nan: .nan\n  zero: 0\n  neg: -600\n", encoding="utf-8")
    monkeypatch.setenv("POLAROID_PROFILES", str(p))
    got = profile_options()
    assert got["items"] == [{"name": "good", "ppi": 3200.0}]
    assert got["error"] == "ignored: inf, nan, neg, zero (the ppi must be a number above 0)"
    json.dumps(got, allow_nan=False)                                         # what the page gets is plain JSON
    p.write_text("profiles:\n  good: 3200\n", encoding="utf-8")
    assert profile_options()["error"] is None


def test_the_settings_payload_with_a_bad_ppi_in_the_profiles_file_is_plain_json(app, src, tmp_path, monkeypatch):
    p = tmp_path / "profiles.yaml"
    p.write_text("profiles:\n  good: 3200\n  inf: .inf\n", encoding="utf-8")
    monkeypatch.setenv("POLAROID_PROFILES", str(p))
    app.json("/api/project/open", {"input": str(src)})
    status, _, raw = app.request("/api/project/settings")
    data = json.loads(raw, parse_constant=lambda c: pytest.fail(f"{c} in the settings payload"))
    assert status == 200 and [i["name"] for i in data["profiles"]["items"]] == ["good"] and "inf" in data["profiles"]["error"]


def test_the_grid_and_progress_still_answer_when_the_saved_profile_is_gone_from_the_file(app, src, profiles_file):
    """A project that has a manifest (a run took the profile) and then loses the profile from the file: the scale cannot
    be worked out, so scans and progress answer 200 with the scans, no export states and the message as ``scale_error``,
    instead of a 500 that leaves the page without a grid."""
    profiles_file.write_text(f"profiles:\n  plustek: {PPI}\n", encoding="utf-8")
    app.json("/api/project/open", {"input": str(src)})
    assert app.json("/api/project/settings", {"profile": "plustek"})[0] == 200
    app.json("/api/project/all", {})
    app.wait(finished)
    out = app.state.project.output
    assert Manifest(out).data["settings"]["profile"] == "plustek"
    status, data = app.json("/api/project/scans")
    assert status == 200 and data["scale_error"] is None and data["counts"]["to_export"] == 0
    assert {s["export"] for s in data["scans"] if s["exported"]} == {"done"}
    revs = [data["rev"]]
    # the profile leaves the file
    profiles_file.write_text("profiles:\n  other: 1000\n", encoding="utf-8")        # (the cached snapshot notices the file)
    status, data = app.json("/api/project/scans")
    assert status == 200 and len(data["scans"]) == 3 and "plustek" in data["scale_error"] and "not found" in data["scale_error"]
    revs.append(data["rev"])
    assert all(s["export"] is None for s in data["scans"]) and all(s["status"] for s in data["scans"])
    assert data["counts"]["to_export"] == data["counts"]["to_retag"] == data["counts"]["to_headers"] == 0
    assert data["counts"]["exported"] == 2 and data["counts"]["total"] == 3
    status, prog = app.json("/api/project/progress")
    assert status == 200 and prog["scale_error"] == data["scale_error"] and prog["counts"]["to_export"] == 0
    assert app.json("/api/project/settings")[0] == 200                              # the panel works as well
    # the profile is added back: everything is as before
    profiles_file.write_text(f"profiles:\n  plustek: {PPI}\n", encoding="utf-8")
    status, data = app.json("/api/project/scans")
    assert status == 200 and data["scale_error"] is None
    assert {s["export"] for s in data["scans"] if s["exported"]} == {"done"}
    revs.append(data["rev"])
    # the page re-reads the scans only when ``rev`` changes: it must change with the profiles file (cards show export states)
    assert len(set(revs)) == 3 and app.json("/api/project/progress")[1]["rev"] == revs[-1]
    # a file that cannot be read gives the same answer, with the reason (not a 500 from the YAML parser)
    for text in ("profiles: [epson: 3200\n", "- epson\n- plustek\n", "profiles: {plustek: [1, 2]}\n"):
        profiles_file.write_text(text, encoding="utf-8")
        for path in ("/api/project/scans", "/api/project/progress"):
            status, data = app.json(path)
            assert status == 200 and data["scale_error"], (text, path)
            assert str(profiles_file) in data["scale_error"] and "cannot be read" in data["scale_error"]
            assert data["counts"]["to_export"] == 0
    # and the profile file is read again when it changes back
    profiles_file.write_text(f"profiles:\n  plustek: {PPI}\n", encoding="utf-8")
    assert app.json("/api/project/progress")[1]["scale_error"] is None


def test_a_failure_in_the_export_plan_is_not_shown_as_a_scale_error(app, src, profiles_file, monkeypatch):
    """Only the scale lookup is guarded: a bug further down must stay a loud error, not a quiet 'scale cannot be worked out'."""
    from polaroid_processor.app import project as project_mod
    profiles_file.write_text(f"profiles:\n  plustek: {PPI}\n", encoding="utf-8")
    app.json("/api/project/open", {"input": str(src)})
    app.json("/api/project/settings", {"profile": "plustek"})
    app.json("/api/project/detect", {})
    app.wait(finished)
    def boom(m, settings):
        raise RuntimeError("a bug in the plan")
    monkeypatch.setattr(project_mod, "export_plan", boom)
    app.state.project._sig = None
    assert app.json("/api/project/scans")[0] == 500


def test_the_fix_page_confirm_works_while_the_scale_cannot_be_worked_out(app, src, profiles_file):
    profiles_file.write_text(f"profiles:\n  plustek: {PPI}\n", encoding="utf-8")
    app.json("/api/project/open", {"input": str(src)})
    app.json("/api/project/settings", {"profile": "plustek"})
    app.json("/api/project/detect", {})
    app.wait(finished)
    profiles_file.write_text("profiles:\n  other: 1000\n", encoding="utf-8")        # plustek is gone
    status, data = app.json("/api/project/scans")
    assert status == 200 and "plustek" in data["scale_error"]
    key = next(s["key"] for s in data["scans"] if s["status"] in ("ok", "needs_review"))
    status, body = app.json(f"/api/fix/scans/{key}/confirm", {})
    assert status == 200 and body["quad_source"] == "override"
    status, data = app.json("/api/project/scans")                                    # the grid still answers after the fix
    assert status == 200 and "plustek" in data["scale_error"]
