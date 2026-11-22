"""Export metadata, slice 1: settings, the export, changes after export (headers rewritten in place), and the
interfaces (CLI, API, MCP, the app's settings round trip)."""
import hashlib
import json
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from polaroid_processor import api, cli, metadata as M, pipeline
from polaroid_processor.app import settings as app_settings
from polaroid_processor.manifest import MANIFEST_NAME, Manifest
from polaroid_processor.meta_patch import PatchError
from polaroid_processor.mcp_server import TOOLS, tool_process
from polaroid_processor.pipeline import Settings, settings_from_dict, settings_to_dict
from polaroid_processor.scale import MM_PER_INCH, Scale

from synth import PX_PER_MM, make_scan

PPI = PX_PER_MM * MM_PER_INCH
SQUARE = (5.5, 6.3, 76.9, 78.9)
META = {"title": "Müller {stem} #{n} ({format}, {project})", "author": "Jörg", "copyright": "© 2026 {project}",
        "description": "Straße", "keywords": ["Garten", "Übung"], "date_taken": "1987-07"}
DERIV = ["web=800:jpg:85", "png=x0.25:png"]


def xp(b):
    return b.decode("utf-16-le").rstrip("\x00")


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    d = tmp_path_factory.mktemp("meta_in") / "260328_Roll"
    d.mkdir()
    for i, n in enumerate(("Roll_3", "Roll_4")):
        rgb, _ = make_scan("white", 0.2 * i - 0.1, seed=70 + i, window_mm=SQUARE)
        tifffile.imwrite(str(d / f"{n}.tif"), rgb, photometric="rgb",
                         extratags=[(271, "s", 0, "EPSON", True), (272, "s", 0, "Perfection V850", True),
                                    (305, "s", 0, "SilverFast 8", True)])
    return d


def run(src, out, **kw):
    kw.setdefault("workers", 1)
    kw.setdefault("ppi", PPI)
    kw.setdefault("derivatives", DERIV)
    kw.setdefault("photo_crop", "tiff")
    return api.process([str(src)], str(out), api.make_settings(**kw))


@pytest.fixture(scope="module")
def project(src, tmp_path_factory):
    """Two scans exported with metadata (and photo crops); the output folder is copied by the tests that change it."""
    out = tmp_path_factory.mktemp("meta_out")
    summary = run(src, out, metadata=META)
    assert summary["exported"] == 2 and summary["metadata_updated"] == 0 and not summary["failed"]
    return out


@pytest.fixture(scope="module")
def plain(src, tmp_path_factory):
    """The same scans exported without any metadata."""
    out = tmp_path_factory.mktemp("meta_plain")
    run(src, out)
    return out


@pytest.fixture
def copy(tmp_path):
    def make(folder):
        dst = tmp_path / "copy"
        shutil.copytree(folder, dst)
        return dst
    return make


def files_of(out, name):
    e = Manifest(str(out)).get(name)
    return {k: Path(out) / v for k, v in e["outputs"].items() if k != "overlay"}


def tags(path):
    with tifffile.TiffFile(path) as tf:
        return {t.code: t.value for t in tf.pages[0].tags.values()}


def pixels(path):
    """What must not change by a header rewrite: decoded pixels of every format, plus the encoded bytes."""
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix == ".tif":
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            enc = b"".join(raw[o:o + n] for o, n in zip(page.dataoffsets, page.databytecounts))
        return hashlib.sha1(tifffile.imread(path).tobytes() + enc).hexdigest()
    with Image.open(path) as im:
        im.load()
        dec = np.asarray(im).tobytes()
    if path.suffix == ".jpg":
        enc = raw[raw.index(b"\xff\xda"):]
    else:      # PNG: the IDAT chunks
        enc, i = b"", 8
        while i < len(raw):
            ln = struct.unpack(">I", raw[i:i + 4])[0]
            if raw[i + 4:i + 8] == b"IDAT":
                enc += raw[i + 8:i + 8 + ln]
            i += 12 + ln
    return hashlib.sha1(dec + enc).hexdigest()


def all_pixels(out):
    return {(n, k): pixels(p) for n in ("Roll_3.tif", "Roll_4.tif") for k, p in files_of(out, n).items()}


# --------------------------------------------------------------------------- the export
def test_every_exported_file_carries_the_metadata(project):
    files = files_of(project, "Roll_3.tif")
    assert set(files) == {"tiff", "web", "png", "photo_tiff", "photo_web", "photo_png"}
    for name, p in files.items():
        if p.suffix == ".tif":
            t = tags(p)
            assert xp(t[40091]) == "Müller Roll_3 #3 (600, 260328_Roll)" and xp(t[40093]) == "Jörg", name
            assert t[33432] == "© 2026 260328_Roll" and t[270] == "Straße", name
            assert t[271] == "EPSON" and t[272] == "Perfection V850" and t[305].startswith("polaroid-processor "), name
            assert 34665 not in t                       # a date to the month: no EXIF date
            pkt = t[700]
        else:
            raw = p.read_bytes()
            pkt = raw[raw.index(b"<?xpacket begin"):raw.index(b'<?xpacket end="w"?>') + 19]
            with Image.open(p) as im:
                exif = dict(im.getexif())
            assert xp(exif[40091]) == "Müller Roll_3 #3 (600, 260328_Roll)" and exif[271] == "EPSON", name
        got = M.read_xmp_fields(pkt)
        assert got["title"] == "Müller Roll_3 #3 (600, 260328_Roll)" and got["date_taken"] == "1987-07", name
        assert got["keywords"] == ["Garten", "Übung"], name


def test_placeholders_differ_per_scan(project):
    titles = {n: M.read_xmp_fields(tags(files_of(project, n)["tiff"])[700])["title"] for n in ("Roll_3.tif", "Roll_4.tif")}
    assert titles == {"Roll_3.tif": "Müller Roll_3 #3 (600, 260328_Roll)", "Roll_4.tif": "Müller Roll_4 #4 (600, 260328_Roll)"}


def test_not_in_previews_or_overlays(src, tmp_path):
    run(src, tmp_path, metadata=META, overlay=True, no_derivatives=True, derivatives=None)
    for f in list((tmp_path / "_overlays").glob("*.jpg")):
        assert b"xpacket" not in f.read_bytes()


def test_no_metadata_still_drops_tifffile_defaults_and_keeps_scanner_tags(plain):
    t = tags(files_of(plain, "Roll_3.tif")["tiff"])
    assert 270 not in t and 700 not in t and 33432 not in t and 315 not in t
    assert t[271] == "EPSON" and t[272] == "Perfection V850" and t[305].startswith("polaroid-processor ")
    e = Manifest(str(plain)).get("Roll_3.tif")
    assert "metadata_key" not in e
    assert "metadata" not in json.loads((plain / MANIFEST_NAME).read_text(encoding="utf-8"))["settings"]


def test_metadata_is_not_part_of_the_export_key(project, plain):
    for n in ("Roll_3.tif", "Roll_4.tif"):
        a, b = Manifest(str(project)).get(n), Manifest(str(plain)).get(n)
        assert a["export_key"] == b["export_key"] and a["export_base"] == b["export_base"]
        assert a["metadata_key"] and "metadata_key" not in b


def test_export_key_pinned():
    q = np.array([[10.0, 20.0], [1000.5, 21.0], [1001.0, 1300.25], [9.5, 1299.0]])
    for s in (Settings(), Settings(metadata=META)):
        assert s.export_key(q, Scale(25.6, "ppi")) == "2fecc94a02844b31"
        assert s.export_base(q, Scale(25.6, "ppi")) == "5e4aa68ee0da87b8"


def test_dry_run_records_do_not_depend_on_metadata(src):
    def records(**kw):
        out = []
        api.process([str(src / "Roll_3.tif")], None, api.make_settings(dry_run=True, ppi=PPI, workers=1, **kw),
                    emit=out.append)
        return [{k: v for k, v in r.items() if k != "time_s"} for r in out if r["type"] == "result"]

    assert records() == records(metadata=META)


# --------------------------------------------------------------------------- changes after export
def test_changing_the_title_rewrites_the_headers_in_place(project, src, copy):
    out = copy(project)
    before_px = all_pixels(out)
    keys = {n: Manifest(str(out)).get(n)["export_key"] for n in ("Roll_3.tif", "Roll_4.tif")}
    mkeys = {n: Manifest(str(out)).get(n)["metadata_key"] for n in keys}
    summary = run(src, out, metadata={"title": "New {stem}"})
    assert summary["exported"] == 0 and summary["retagged"] == 0 and summary["metadata_updated"] == 2
    assert all_pixels(out) == before_px
    for n in keys:
        e = Manifest(str(out)).get(n)
        assert e["export_key"] == keys[n] and e["metadata_key"] != mkeys[n]
        for k, p in files_of(out, n).items():
            if p.suffix == ".tif":
                t = tags(p)
                assert xp(t[40091]) == f"New {Path(n).stem}"
                assert xp(t[40093]) == "Jörg" and M.read_xmp_fields(t[700])["author"] == "Jörg"   # kept
                assert 33432 in t and t[271] == "EPSON"
            else:
                with Image.open(p) as im:
                    assert xp(dict(im.getexif())[40091]) == f"New {Path(n).stem}"
    assert json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))["settings"]["metadata"]["title"] == "New {stem}"
    again = run(src, out)      # a run that leaves metadata out keeps the project's values and has nothing to do
    assert again["exported"] == 0 and again["metadata_updated"] == 0
    same = run(src, out, metadata={"title": "New {stem}"})
    assert same["metadata_updated"] == 0


def test_an_empty_string_clears_one_field(project, src, copy):
    out = copy(project)
    summary = run(src, out, metadata={"author": "", "keywords": []})
    assert summary["metadata_updated"] == 2 and summary["exported"] == 0
    t = tags(files_of(out, "Roll_3.tif")["tiff"])
    assert 315 not in t and 40093 not in t and 40094 not in t and xp(t[40091]).startswith("Müller")
    got = M.read_xmp_fields(t[700])
    assert "author" not in got and "keywords" not in got and got["title"].startswith("Müller")


def test_clearing_everything_removes_the_headers_and_the_key(project, src, copy):
    out = copy(project)
    summary = run(src, out, metadata={f: "" for f in ("title", "author", "copyright", "description", "date_taken")}
                  | {"keywords": []})
    assert summary["metadata_updated"] == 2
    t = tags(files_of(out, "Roll_3.tif")["tiff"])
    assert {270, 315, 33432, 700, 34665, 40091}.isdisjoint(t) and t[271] == "EPSON"
    assert "metadata_key" not in Manifest(str(out)).get("Roll_3.tif")
    assert "metadata" not in json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))["settings"]
    assert run(src, out)["metadata_updated"] == 0


def test_old_exports_without_a_key_are_rewritten_once_metadata_is_set(plain, src, copy):
    out = copy(plain)
    before_px = all_pixels(out)
    summary = run(src, out, metadata=META)
    assert summary["exported"] == 0 and summary["metadata_updated"] == 2
    assert all_pixels(out) == before_px
    assert xp(tags(files_of(out, "Roll_4.tif")["tiff"])[40091]) == "Müller Roll_4 #4 (600, 260328_Roll)"


def test_no_metadata_leaves_exports_without_a_key_alone(plain, src, copy):
    out = copy(plain)
    files = [p for n in ("Roll_3.tif", "Roll_4.tif") for p in files_of(out, n).values()]
    before = {p: p.read_bytes() for p in files}
    summary = run(src, out)
    assert summary["exported"] == 0 and summary["metadata_updated"] == 0
    assert {p: p.read_bytes() for p in files} == before


def test_a_file_that_cannot_be_patched_is_exported_again(project, src, copy, monkeypatch):
    out = copy(project)

    def refuse(path, fm):
        raise PatchError("not what the writers make")

    monkeypatch.setattr(pipeline, "set_metadata", refuse)
    summary = run(src, out, metadata={"title": "Again {stem}"})
    assert summary["metadata_updated"] == 0 and summary["exported"] == 2
    for n in ("Roll_3.tif", "Roll_4.tif"):
        assert xp(tags(files_of(out, n)["tiff"])[40091]) == f"Again {Path(n).stem}"
        assert Manifest(str(out)).get(n)["metadata_key"]


def test_a_file_that_is_locked_keeps_its_old_key_and_is_tried_again(project, src, copy, monkeypatch):
    out = copy(project)
    old = {n: Manifest(str(out)).get(n)["metadata_key"] for n in ("Roll_3.tif", "Roll_4.tif")}
    real = pipeline.set_metadata

    def locked(path, fm):
        raise PermissionError("in use by a viewer")

    monkeypatch.setattr(pipeline, "set_metadata", locked)
    summary = run(src, out, metadata={"title": "Locked {stem}"})
    assert summary["exported"] == 0 and summary["metadata_updated"] == 0 and not summary["failed"]
    assert {n: Manifest(str(out)).get(n)["metadata_key"] for n in old} == old      # still the old headers, and it says so
    assert all(Manifest(str(out)).get(n)["status"] == "ok" for n in old)
    monkeypatch.setattr(pipeline, "set_metadata", real)
    summary = run(src, out)
    assert summary["metadata_updated"] == 2 and summary["exported"] == 0
    assert xp(tags(files_of(out, "Roll_3.tif")["tiff"])[40091]) == "Locked Roll_3"


def test_a_writer_version_bump_rewrites_existing_exports_once(project, src, copy, monkeypatch):
    out = copy(project)
    before = all_pixels(out)
    monkeypatch.setattr(M, "WRITER_VERSION", M.WRITER_VERSION + 1)
    summary = run(src, out)
    assert summary["metadata_updated"] == 2 and summary["exported"] == 0 and all_pixels(out) == before
    assert run(src, out)["metadata_updated"] == 0


def test_the_phase_is_announced_before_headers_are_rewritten(project, src, copy):
    out = copy(project)
    phases = []
    api.process([str(src)], str(out), api.make_settings(workers=1, ppi=PPI, derivatives=DERIV, photo_crop="tiff",
                                                        metadata={"title": "P {stem}"}), on_phase=phases.append)
    assert phases.count("updating metadata") == 1


def test_a_stale_patch_file_is_removed_at_the_start_of_a_run(project, src, copy):
    out = copy(project)
    stale = [out / "web" / "Roll_3.jpg.patch", out / "photo" / "png" / "Roll_4.png.part.patch",
             out / "Roll_4.tif.patch"]
    for f in stale:
        f.write_bytes(b"half")
    run(src, out)
    assert not any(f.exists() for f in stale)


def test_a_missing_output_is_exported_again_with_the_metadata(project, src, copy):
    out = copy(project)
    files_of(out, "Roll_3.tif")["web"].unlink()
    summary = run(src, out)
    assert summary["exported"] == 1 and summary["metadata_updated"] == 0
    assert xp(tags(files_of(out, "Roll_3.tif")["tiff"])[40091]).startswith("Müller")


def test_a_forced_run_writes_the_metadata_again(project, src, copy):
    out = copy(project)
    summary = run(src, out, force=True)
    assert summary["exported"] == 2 and summary["metadata_updated"] == 0
    assert Manifest(str(out)).get("Roll_3.tif")["metadata_key"]


# --------------------------------------------------------------------------- settings, validation, interfaces
def test_settings_dict_leaves_metadata_out_when_empty():
    assert "metadata" not in settings_to_dict(Settings())
    assert "metadata" not in settings_to_dict(Settings(metadata={}))
    d = settings_to_dict(Settings(metadata={"title": "t"}))
    assert d["metadata"] == {"title": "t"} and settings_from_dict(d).metadata == {"title": "t"}
    assert settings_to_dict(settings_from_dict(settings_to_dict(Settings()))) == settings_to_dict(Settings())
    assert settings_from_dict({}).metadata is None


def test_a_hand_edited_bad_stored_value_does_not_break_loading_or_export():
    s = settings_from_dict({"metadata": {"title": "{nope}", "date_taken": "soon"}})
    assert s.metadata["title"] == "{nope}"
    assert M.fill(s.metadata, stem="a", project="p", format_name="600").title == "{nope}"


def test_the_apps_settings_round_trip_keeps_metadata():
    current = settings_to_dict(Settings(metadata={"title": "T {stem}", "keywords": ["a"], "date_taken": "1987"}))
    after = app_settings.from_panel(current, {"margin_mm": 0.2, "photo_crop": "derivatives"})
    assert after["metadata"] == current["metadata"] and after["margin_mm"] == 0.2
    assert app_settings.from_panel(current, {})["metadata"] == current["metadata"]
    assert app_settings.to_panel(after)["margin_mm"] == 0.2
    assert "metadata" not in app_settings.from_panel(settings_to_dict(Settings()), {"margin_mm": 0.3})
    assert app_settings.defaults_of(after)["margin_mm"] == 0.2


@pytest.mark.parametrize("bad", [{"title": "{bogus}"}, {"title": "a {"}, {"date_taken": "20.7.1987"},
                                 {"date_taken": "1987-02-30"}, {"color": "red"}, {"keywords": "a,b"}, {"title": 1}])
def test_bad_metadata_is_refused_when_given(bad):
    with pytest.raises(ValueError):
        api.make_settings(metadata=bad)
    with pytest.raises(api.UsageError):
        api.make_settings(metadata=bad)


def test_make_settings_keeps_what_was_given_for_the_merge():
    assert api.make_settings().metadata is None
    assert api.make_settings(metadata={}).metadata == {}
    assert api.make_settings(metadata={"title": " t ", "author": ""}).metadata == {"title": "t", "author": ""}


def test_cli_refuses_a_bad_placeholder_with_exit_2_before_writing(src, tmp_path, capsys):
    out = tmp_path / "out"
    assert cli.main(["process", str(src), "-o", str(out), "--title", "{bogus}"]) == 2
    assert "bogus" in capsys.readouterr().err
    assert not out.exists()
    assert cli.main(["process", str(src), "-o", str(out), "--date-taken", "1987-13"]) == 2
    assert not out.exists()


def test_cli_flags_set_the_metadata(src, tmp_path, capsys):
    out = tmp_path / "out"
    rc = cli.main(["process", str(src), "-o", str(out), "--json", "--ppi", str(PPI), "--no-derivatives", "-j", "1",
                   "--title", "Tür {n}", "--author", "Me", "--copyright", "(c)", "--description", "d",
                   "--keyword", "a", "--keyword", "b ü", "--date-taken", "1987-07-20"])
    assert rc == 0
    summary = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.strip()][-1]
    assert summary["metadata_updated"] == 0
    m = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert m["settings"]["metadata"] == {"title": "Tür {n}", "author": "Me", "copyright": "(c)", "description": "d",
                                         "keywords": ["a", "b ü"], "date_taken": "1987-07-20"}
    t = tags(out / "Roll_3.tif")
    assert xp(t[40091]) == "Tür 3" and t[34665]["DateTimeOriginal"] == "1987:07:20 00:00:00"
    rc = cli.main(["process", str(src), "-o", str(out), "--json", "--ppi", str(PPI), "--no-derivatives", "-j", "1",
                   "--keyword", ""])      # clears the keywords only
    assert rc == 0
    capsys.readouterr()
    m = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert "keywords" not in m["settings"]["metadata"] and m["settings"]["metadata"]["title"] == "Tür {n}"
    assert 40094 not in tags(out / "Roll_3.tif")


def test_mcp_tool_takes_a_metadata_object(src, tmp_path):
    tool = next(t for t in TOOLS if t["name"] == "polaroid_process")
    prop = tool["inputSchema"]["properties"]["metadata"]
    assert prop["type"] == "object" and "{stem}" in prop["description"] and "metadata_updated" in prop["description"]
    assert set(prop["properties"]) == {"title", "author", "copyright", "description", "keywords", "date_taken"}
    out = tmp_path / "out"
    res = tool_process({"inputs": [str(src)], "output": str(out), "ppi": PPI, "no_derivatives": True, "workers": 1,
                        "metadata": {"title": "MCP {stem}"}}, lambda *a: None)
    assert xp(tags(out / "Roll_3.tif")[40091]) == "MCP Roll_3"
    assert json.dumps(res)
    with pytest.raises(ValueError):
        tool_process({"inputs": [str(src)], "output": str(out), "metadata": {"title": "{x}"}}, lambda *a: None)


# --------------------------------------------------------------------------- saved values that fail the checks
def set_saved(out, value):
    p = Path(out) / MANIFEST_NAME
    m = json.loads(p.read_text(encoding="utf-8"))
    m["settings"]["metadata"] = value
    p.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")


def test_saved_values_that_fail_the_checks_are_kept_not_wiped(project, src, copy):
    out = copy(project)
    set_saved(out, {"title": "X {foo}", "author": "A", "keywords": ["k"], "date_taken": "1987-7"})
    summary = run(src, out)          # a run without metadata: the saved values stay, literally
    assert summary["exported"] == 0 and summary["metadata_updated"] == 2 and not summary["failed"]
    saved = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))["settings"]["metadata"]
    assert saved == {"title": "X {foo}", "author": "A", "keywords": ["k"], "date_taken": "1987-7"}
    t = tags(files_of(out, "Roll_3.tif")["tiff"])
    assert xp(t[40091]) == "X {foo}" and xp(t[40093]) == "A" and 34665 not in t and 33432 not in t
    assert run(src, out)["metadata_updated"] == 0
    assert run(src, out, force=True)["exported"] == 2             # and an export is fine with them too
    assert xp(tags(files_of(out, "Roll_3.tif")["tiff"])[40091]) == "X {foo}"


def test_saved_values_that_are_not_text_do_not_fail_an_export(project, src, copy):
    out = copy(project)
    set_saved(out, {"title": 5, "author": "A", "keywords": [1, "k"], "copyright": ["c"]})
    summary = run(src, out, force=True)
    assert summary["exported"] == 2 and not summary["failed"]
    t = tags(files_of(out, "Roll_3.tif")["tiff"])
    assert xp(t[40093]) == "A" and xp(t[40094]) == "k" and 40091 not in t and 33432 not in t
    assert all("export failed" not in str(Manifest(str(out)).get(n).get("error")) for n in ("Roll_3.tif", "Roll_4.tif"))


@pytest.mark.parametrize("value", ["x", ["title"], 5, True])
def test_a_saved_value_that_is_not_an_object_does_not_crash_anything(project, src, copy, value):
    out = copy(project)
    set_saved(out, value)
    d = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))["settings"]
    assert settings_from_dict(d).metadata in (None, {})          # the app's load, the panel, fix
    assert app_settings.to_panel(d)["margin_mm"] == 0.1
    assert "metadata" not in app_settings.from_panel(d, {"margin_mm": 0.2})
    summary = run(src, out)
    assert not summary["failed"] and summary["exported"] == 0


def test_the_apps_settings_load_keeps_saved_texts_that_fail_the_checks():
    d = {"metadata": {"title": "X {foo}", "date_taken": "1987-7", "author": 3}}
    assert settings_from_dict(d).metadata == {"title": "X {foo}", "date_taken": "1987-7"}
    assert app_settings.from_panel(d, {"margin_mm": 0.2})["metadata"] == {"title": "X {foo}", "date_taken": "1987-7"}


# --------------------------------------------------------------------------- with other things that move files
def test_a_fix_with_export_writes_the_stored_metadata(project, copy):
    out = copy(project)
    rec = api.fix_scan(str(out), "Roll_3.tif", shifts={"right": [4.0]}, author="agent", export=True)
    assert "skipped" not in rec.get("note", "")
    e = Manifest(str(out)).get("Roll_3.tif")
    t = tags(Path(out) / e["outputs"]["tiff"])
    assert xp(t[40091]) == "Müller Roll_3 #3 (600, 260328_Roll)" and t[271] == "EPSON"
    assert e["metadata_key"]
    with Image.open(Path(out) / e["outputs"]["web"]) as im:
        assert xp(dict(im.getexif())[40091]) == "Müller Roll_3 #3 (600, 260328_Roll)"


def test_a_retag_and_a_metadata_change_in_one_run(tmp_path):
    import time
    from synth import make_scan
    src = tmp_path / "in"
    src.mkdir()
    for i in range(3):                                           # no resolution tag: the scale is inferred
        rgb, _ = make_scan("white", 0.3 * i, seed=70 + i)
        tifffile.imwrite(str(src / f"s_{i}.tif"), rgb, photometric="rgb")
    out = tmp_path / "out"
    kw = dict(workers=1, no_derivatives=True)
    api.process([str(src)], str(out), api.make_settings(workers=1, export=False))
    api.fix_scan(str(out), "s_0.tif", shifts={"right": [6.0]}, author="agent")
    api.fix_scan(str(out), "s_1.tif", shifts={"right": [-6.0]}, author="agent")
    api.process([str(src)], str(out), api.make_settings(**kw))
    before = {k: dict(e) for k, e in Manifest(str(out)).items()}
    assert not any("metadata_key" in e for e in before.values())
    time.sleep(1.1)
    api.fix_scan(str(out), "s_2.tif", shifts={"right": [12.0]}, author="agent")      # the median moves
    summary = api.process([str(src)], str(out), api.make_settings(metadata={"title": "R {stem}"}, **kw))
    assert summary["retagged"] == 2 and summary["metadata_updated"] == 2 and summary["exported"] == 1
    m = Manifest(str(out))
    for k in ("s_0.tif", "s_1.tif"):
        e = m.get(k)
        assert e["exported_at"] == before[k]["exported_at"] and e["metadata_key"]      # not encoded again
        with tifffile.TiffFile(str(m.abs(e["outputs"]["tiff"]))) as tf:
            num, den = tf.pages[0].tags["XResolution"].value
            t = {x.code: x.value for x in tf.pages[0].tags.values()}
        assert abs(num / den - e["scale"]["px_per_mm"] * 10) < 0.5            # the new scale and the new title together
        assert xp(t[40091]) == f"R {Path(k).stem}"
    assert m.get("s_2.tif")["metadata_key"]
    again = api.process([str(src)], str(out), api.make_settings(**kw))
    assert again["retagged"] == 0 and again["metadata_updated"] == 0 and again["exported"] == 0
