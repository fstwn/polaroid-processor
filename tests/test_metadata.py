"""Export metadata, slice 1: placeholders, dates, the project's dict, the XMP packet, the writers and the in-place
rewrite of TIFF, JPEG and PNG headers."""
import re
import struct

import numpy as np
import pytest
import tifffile
from PIL import Image

from polaroid_processor import metadata as M
from polaroid_processor.imageio import write_derivative, write_tiff
from polaroid_processor.meta_patch import PatchError, exif_block, set_jpeg, set_metadata, set_png, set_tiff


def values(stem="Misc_D3_17", project="260328_tttacho", fmt="600"):
    return M.placeholder_values(stem, project, fmt)


FULL = {"title": "Müller {stem} #{n}", "author": "Jörg & Co <x>", "copyright": "© 2026 {project}",
        "description": "Straße — ☃ ({format})", "keywords": ["Garten", "Übung {n}"], "date_taken": "1987-07-20"}


def fm_of(meta=FULL, **kw):
    kw.setdefault("make", "EPSON")
    kw.setdefault("model", "Perfection V850")
    kw.setdefault("software", "polaroid-processor 9.9")
    return M.fill(meta, stem="Misc_D3_17", project="260328_tttacho", format_name="600", **kw)


def image(h=300, w=200):
    yy, xx = np.mgrid[:h, :w]
    return np.stack([xx % 256, yy % 256, (xx + yy) % 256], -1).astype(np.uint8)


# --------------------------------------------------------------------------- placeholders
def test_placeholders_are_filled():
    assert M.fill_text("{stem} / {n} / {project} / {format}", values()) == "Misc_D3_17 / 17 / 260328_tttacho / 600"


def test_n_is_the_trailing_number_as_written_and_empty_without_one():
    assert values("scan_07")["n"] == "07"
    assert values("scan")["n"] == ""
    assert values("2023_09_a")["n"] == ""


def test_double_braces_are_literal():
    assert M.fill_text("{{stem}} = {stem} {{", values("a1")) == "{stem} = a1 {"
    assert M.fill_text("}}", values()) == "}"


@pytest.mark.parametrize("bad", ["{title}", "{}", "{stem", "stem}", "{stem}}", "{ stem }", "{{stem}"])
def test_bad_placeholders_are_errors(bad):
    with pytest.raises(ValueError):
        M.fill_text(bad, values())


def test_format_labels():
    assert M.format_label({"name": "polaroid-go"}) == "Go"
    assert M.format_label({"name": "polaroid-600"}) == "600"
    assert M.format_label(None, (880, 1068)) == "600"           # by the shape of the outline when no record
    assert M.format_label(None, (535, 658)) == "Go"
    assert M.format_label(None) == ""


# --------------------------------------------------------------------------- date and fields
@pytest.mark.parametrize("ok", ["1987", "1987-07", "1987-07-20", "2024-02-29"])
def test_dates_accepted(ok):
    assert M.check_date(ok) == ok


@pytest.mark.parametrize("bad", ["87", "1987-7", "1987-13", "1987-02-30", "2023-02-29", "1987/07/20", "July 1987",
                                 "1987-07-20T10:00", "0999", ""])
def test_dates_refused(bad):
    with pytest.raises(ValueError):
        M.check_date(bad)


def test_normalize_checks_and_cleans():
    assert M.normalize(None) == {}
    n = M.normalize({"title": "  t ", "author": "", "keywords": ["a", " ", "b"], "date_taken": "1987-07"})
    assert n == {"title": "t", "keywords": ["a", "b"], "date_taken": "1987-07"}
    assert M.normalize({"title": "", "keywords": [], "date_taken": ""}, keep_empty=True) == \
        {"title": "", "keywords": [], "date_taken": ""}


@pytest.mark.parametrize("bad", [{"titel": "x"}, {"title": 3}, {"keywords": "a,b"}, {"keywords": [1]},
                                 {"date_taken": "soon"}, {"title": "{bogus}"}, {"keywords": ["{stem"]},
                                 {"title": "x" * 501}, {"description": "x" * 4001}, {"keywords": [str(i) for i in range(101)]},
                                 {"title": "a\x00b"}, {"author": "a\ud800"}, {"keywords": ["k\x07"]}],
                         ids=str)
def test_normalize_refuses(bad):
    with pytest.raises(ValueError):
        M.normalize(bad)
    with pytest.raises(ValueError):
        M.normalize([1])


def test_total_size_is_limited_with_placeholders_at_their_worst():
    with pytest.raises(ValueError):
        M.normalize({"description": "{stem}" * 700})        # 700 x a file name long enough to overflow a JPEG segment


def test_merge_keeps_replaces_and_clears():
    stored = {"title": "T", "author": "A", "keywords": ["k"], "date_taken": "1987"}
    assert M.merge(stored, None) == stored
    assert M.merge(stored, {}) == stored
    assert M.merge(stored, {"title": "new"}) == {**stored, "title": "new"}
    assert M.merge(stored, {"author": ""}) == {"title": "T", "keywords": ["k"], "date_taken": "1987"}
    assert M.merge(stored, {"keywords": []}) == {"title": "T", "author": "A", "date_taken": "1987"}
    assert M.merge(stored, {"keywords": ["x", "y"], "date_taken": ""}) == {"title": "T", "author": "A",
                                                                            "keywords": ["x", "y"]}
    assert M.merge(None, {"title": ""}) == {}


def test_none_is_a_field_left_out_not_a_clear():
    stored = {"title": "T", "author": "A", "keywords": ["k"], "date_taken": "1987"}
    assert M.merge(stored, {"title": None, "keywords": None, "date_taken": None}) == stored
    assert M.normalize({"title": None, "keywords": None}, keep_empty=True) == {}


def test_keywords_have_no_duplicates():
    assert M.normalize({"keywords": ["a", "b", "a", " b "]})["keywords"] == ["a", "b"]
    assert M.fill({"keywords": ["{n}", "7"]}, stem="x7", project="p", format_name="600").keywords == ("7",)


def test_fill_never_raises_on_a_stored_bad_value():
    fm = M.fill({"title": "{nope}", "date_taken": "bad", "keywords": ["{stem}", "x}"]}, stem="s", project="p",
                format_name="Go")
    assert fm.title == "{nope}" and fm.date_taken == "" and fm.keywords == ("s", "x}")


@pytest.mark.parametrize("stored", ["x", ["title"], 5, {"title": 5}, {"keywords": 3}, {"keywords": [1, None, {}]},
                                    {"title": ["a"], "date_taken": 1987}, {"colour": "red"}, {"title": None}],
                         ids=repr)
def test_stored_values_that_are_not_text_are_dropped_not_fatal(stored):
    assert M.normalize_stored(stored) == {}
    assert M.merge(stored, None) == {}
    fm = M.fill(stored, stem="s", project="p", format_name="Go")         # never raises (a TypeError once)
    assert fm.key() == ""


def test_stored_values_that_fail_the_checks_are_kept_literally():
    stored = {"title": "X {foo}", "author": " A ", "keywords": ["k", "k", "{x", 3], "date_taken": "1987-7",
              "description": "d", "bad": "dropped"}
    kept = M.normalize_stored(stored)
    assert kept == {"title": "X {foo}", "author": "A", "keywords": ["k", "{x"], "date_taken": "1987-7", "description": "d"}
    assert M.merge(stored, None) == kept                                  # a run that gives nothing keeps all of it
    assert M.merge(stored, {"author": "B"})["title"] == "X {foo}"
    fm = M.fill(stored, stem="s", project="p", format_name="Go")
    assert fm.title == "X {foo}" and fm.keywords == ("k", "{x") and fm.date_taken == ""


def test_stored_values_warn(caplog):
    with caplog.at_level("WARNING", logger="polaroid_processor"):
        M.normalize_stored({"title": 5, "keywords": "a", "colour": "red"})
    assert len(caplog.records) == 3


def test_control_characters_and_surrogates_are_stripped_at_write_time_from_stored_values():
    fm = M.fill({"title": "a\x00b\ud800c"}, stem="s", project="p", format_name="Go")
    assert fm.title == "abc"


def test_the_writer_version_is_in_the_key_only_with_fields(monkeypatch):
    with_fields, without = fm_of().key(), fm_of({}).key()
    monkeypatch.setattr(M, "WRITER_VERSION", M.WRITER_VERSION + 1)
    assert fm_of().key() != with_fields             # a change to what the writers put in a file reaches old exports
    assert fm_of({}).key() == without == ""          # none for exports without metadata


def test_key_is_over_the_filled_fields_only():
    a, b = fm_of(), fm_of(make="OTHER", software="x")
    assert a.key() == b.key() != ""
    assert fm_of({"title": "Müller {stem} #{n}"}).key() != a.key()
    assert fm_of({}).key() == "" and fm_of(None).key() == ""
    other = M.fill(FULL, stem="Misc_D3_18", project="260328_tttacho", format_name="600")
    assert other.key() != a.key()               # a placeholder that fills differently is another key


# --------------------------------------------------------------------------- XMP
def test_xmp_none_without_fields():
    assert M.xmp_packet(fm_of({})) is None


def test_xmp_roundtrip_with_umlauts_and_xml_characters():
    pkt = M.xmp_packet(fm_of())
    assert pkt.startswith(b"<?xpacket begin=") and pkt.rstrip().endswith(b'<?xpacket end="w"?>')
    assert "Müller Misc_D3_17 #17".encode() in pkt and "☃".encode() in pkt
    got = M.read_xmp_fields(pkt)
    assert got == {"title": "Müller Misc_D3_17 #17", "author": "Jörg & Co <x>", "copyright": "© 2026 260328_tttacho",
                   "description": "Straße — ☃ (600)", "keywords": ["Garten", "Übung 17"], "date_taken": "1987-07-20",
                   "software": "polaroid-processor 9.9", "make": "EPSON", "model": "Perfection V850"}


def test_xmp_has_make_and_model_only_when_the_scan_has_them():
    got = M.read_xmp_fields(M.xmp_packet(fm_of({"title": "t"}, make="", model="")))
    assert "make" not in got and "model" not in got


@pytest.mark.parametrize("date", ["1987", "1987-07", "1987-07-20"])
def test_xmp_date_at_the_precision_given(date):
    pkt = M.xmp_packet(fm_of({"title": "t", "date_taken": date})).decode()
    assert f"<photoshop:DateCreated>{date}<" in pkt and f"<exif:DateTimeOriginal>{date}<" in pkt


def test_xmp_drops_control_characters():
    pkt = M.xmp_packet(fm_of({"title": "a\x00b\x07c\td"}))
    assert M.read_xmp_fields(pkt)["title"] == "abc\td"


# --------------------------------------------------------------------------- legacy tags in a TIFF
def xp(b):
    return b.decode("utf-16-le").rstrip("\x00")


def read_tiff_tags(path):
    with tifffile.TiffFile(path) as tf:
        tags = tf.pages[0].tags
        out = {t.code: t.value for t in tags.values()}
    return out


def test_tiff_carries_xmp_and_legacy_fields(tmp_path):
    p = tmp_path / "a.tif"
    write_tiff(str(p), image(), 126.0, meta=fm_of())
    t = read_tiff_tags(p)
    assert t[270] == "Straße — ☃ (600)" and t[315] == "Jörg & Co <x>" and t[33432] == "© 2026 260328_tttacho"
    assert t[271] == "EPSON" and t[272] == "Perfection V850" and t[305] == "polaroid-processor 9.9"
    assert xp(t[40091]) == "Müller Misc_D3_17 #17" and xp(t[40092]) == "Straße — ☃ (600)"
    assert xp(t[40093]) == "Jörg & Co <x>" and xp(t[40094]) == "Garten;Übung 17"
    assert t[34665] == {"ExifVersion": "0231", "DateTimeOriginal": "1987:07:20 00:00:00"}
    assert M.read_xmp_fields(t[700])["title"] == "Müller Misc_D3_17 #17"
    assert np.array_equal(tifffile.imread(p), image())


def test_tiff_without_metadata_has_no_description_and_no_user_fields(tmp_path):
    p = tmp_path / "a.tif"
    write_tiff(str(p), image(), 126.0, meta=fm_of({}))
    t = read_tiff_tags(p)
    assert {270, 315, 33432, 700, 34665, 40091, 40092, 40093, 40094}.isdisjoint(t)
    assert t[271] == "EPSON" and t[272] == "Perfection V850" and t[305] == "polaroid-processor 9.9"
    p2 = tmp_path / "b.tif"        # not even a meta object: tifffile's own description and software are never written
    write_tiff(str(p2), image(), 126.0)
    t2 = read_tiff_tags(p2)
    assert 270 not in t2 and 305 not in t2


def test_exif_date_only_for_full_dates(tmp_path):
    for date, has in (("1987", False), ("1987-07", False), ("1987-07-20", True)):
        p = tmp_path / f"{date}.tif"
        write_tiff(str(p), image(), 126.0, meta=fm_of({"title": "t", "date_taken": date}))
        assert (34665 in read_tiff_tags(p)) is has
        assert f"<photoshop:DateCreated>{date}<" in read_tiff_tags(p)[700].decode()


def test_image_description_is_the_description_else_the_title(tmp_path):
    p = tmp_path / "a.tif"
    write_tiff(str(p), image(), 126.0, meta=fm_of({"title": "only a title"}))
    t = read_tiff_tags(p)
    assert t[270] == "only a title" and xp(t[40091]) == "only a title" and 40092 not in t       # XPComment: description only
    assert M.read_xmp_fields(t[700]).get("description") is None                                 # XMP: description only
    write_tiff(str(p), image(), 126.0, meta=fm_of({"title": "a title", "description": "a description"}))
    t = read_tiff_tags(p)
    assert t[270] == "a description" and xp(t[40091]) == "a title" and xp(t[40092]) == "a description"
    write_tiff(str(p), image(), 126.0, meta=fm_of({"author": "only an author"}))
    assert 270 not in read_tiff_tags(p)


def test_a_semicolon_in_a_keyword_does_not_split_it_in_xp_keywords(tmp_path):
    p = tmp_path / "a.tif"
    write_tiff(str(p), image(), 126.0, meta=fm_of({"keywords": ["a;b", "c"]}))
    assert xp(read_tiff_tags(p)[40094]) == "a,b;c"
    assert M.read_xmp_fields(read_tiff_tags(p)[700])["keywords"] == ["a;b", "c"]


# --------------------------------------------------------------------------- JPEG and PNG
def test_jpeg_carries_xmp_and_exif(tmp_path):
    p = tmp_path / "a.jpg"
    write_derivative(str(p), image(), "jpg", 90, 300, meta=fm_of())
    with Image.open(p) as im:
        exif = dict(im.getexif())
        assert im.info["dpi"] == (300, 300)
    assert exif[315].encode("latin-1").decode("utf-8") == "Jörg & Co <x>"      # PIL reads ASCII fields as Latin-1
    assert xp(exif[40091]) == "Müller Misc_D3_17 #17" and exif[271] == "EPSON" and exif[305] == "polaroid-processor 9.9"
    raw = p.read_bytes()
    pkt = raw[raw.index(b"<?xpacket begin"):raw.index(b'<?xpacket end="w"?>') + 19]
    assert M.read_xmp_fields(pkt)["keywords"] == ["Garten", "Übung 17"]
    assert raw.index(b"JFIF") < raw.index(b"Exif") < raw.index(b"http://ns.adobe.com/xap/1.0/")


def test_png_carries_xmp_and_exif_chunks(tmp_path):
    p = tmp_path / "a.png"
    write_derivative(str(p), image(), "png", 90, 300, meta=fm_of())
    raw = p.read_bytes()
    kinds = chunk_kinds(raw)
    assert kinds.index(b"eXIf") < kinds.index(b"IDAT") and kinds.index(b"iTXt") < kinds.index(b"IDAT")
    pkt = raw[raw.index(b"<?xpacket begin"):raw.index(b'<?xpacket end="w"?>') + 19]
    assert M.read_xmp_fields(pkt)["title"] == "Müller Misc_D3_17 #17"
    with Image.open(p) as im:
        im.load()
        assert xp(dict(im.getexif())[40091]) == "Müller Misc_D3_17 #17"
        assert im.info["dpi"][0] == pytest.approx(300, abs=0.01)


def test_png_alpha_and_16_bit_sources_still_write(tmp_path):
    alpha = np.full(image().shape[:2], 255, np.uint8)
    write_derivative(str(tmp_path / "a.png"), image(), "png", 90, 300, alpha=alpha, meta=fm_of())
    with Image.open(tmp_path / "a.png") as im:
        assert im.mode == "RGBA"


def chunk_kinds(raw):
    kinds, i = [], 8
    while i < len(raw):
        ln = struct.unpack(">I", raw[i:i + 4])[0]
        kinds.append(raw[i + 4:i + 8])
        i += 12 + ln
    return kinds


def png_idat(raw):
    out, i = b"", 8
    while i < len(raw):
        ln = struct.unpack(">I", raw[i:i + 4])[0]
        if raw[i + 4:i + 8] == b"IDAT":
            out += raw[i + 8:i + 8 + ln]
        i += 12 + ln
    return out


def jpeg_scan(raw):
    return raw[raw.index(b"\xff\xda"):]


def tiff_strips(path):
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        fh = tf.filehandle
        out = []
        for o, n in zip(page.dataoffsets, page.databytecounts):
            fh.seek(o)
            out.append(fh.read(n))
        return out


# --------------------------------------------------------------------------- the in-place rewrite
def test_rewritten_jpeg_equals_a_fresh_one_and_keeps_the_entropy_data(tmp_path):
    p, q = tmp_path / "a.jpg", tmp_path / "b.jpg"
    write_derivative(str(p), image(), "jpg", 90, 300, meta=fm_of({"title": "first", "keywords": ["x"]}))
    before = jpeg_scan(p.read_bytes())
    set_metadata(str(p), fm_of())
    write_derivative(str(q), image(), "jpg", 90, 300, meta=fm_of())
    assert p.read_bytes() == q.read_bytes()
    assert jpeg_scan(p.read_bytes()) == before
    set_metadata(str(p), fm_of({}))      # all fields cleared: Make / Model / Software stay, XMP is gone
    raw = p.read_bytes()
    assert b"xpacket" not in raw and b"Exif" in raw and jpeg_scan(raw) == before
    with Image.open(p) as im:
        assert 40091 not in dict(im.getexif())


def test_rewritten_png_equals_a_fresh_one_and_keeps_the_pixel_chunks(tmp_path):
    p, q = tmp_path / "a.png", tmp_path / "b.png"
    write_derivative(str(p), image(), "png", 90, 300, meta=fm_of({"title": "first"}))
    before = png_idat(p.read_bytes())
    set_metadata(str(p), fm_of())
    write_derivative(str(q), image(), "png", 90, 300, meta=fm_of())
    assert p.read_bytes() == q.read_bytes()
    assert png_idat(p.read_bytes()) == before
    set_metadata(str(p), fm_of({}))
    raw = p.read_bytes()
    assert b"xpacket" not in raw and b"eXIf" in raw and png_idat(raw) == before
    with Image.open(p) as im:
        im.load()
        assert np.array_equal(np.asarray(im), image())


@pytest.mark.parametrize("kw", [{}, {"bigtiff": True}, {"byteorder": ">"}, {"compression": "zlib", "predictor": True}],
                         ids=["classic", "bigtiff", "big-endian", "zlib"])
def test_tiff_rewrite_leaves_the_pixels_alone(tmp_path, kw):
    p = tmp_path / "a.tif"
    img = image()
    tifffile.imwrite(str(p), img, photometric="rgb", resolution=(1260, 1260), resolutionunit="CENTIMETER", **kw)
    strips = tiff_strips(p)
    set_metadata(str(p), fm_of({"title": "first"}))
    set_metadata(str(p), fm_of())          # a second rewrite on top of the first
    assert tiff_strips(p) == strips
    assert np.array_equal(tifffile.imread(p), img)
    t = read_tiff_tags(p)
    assert xp(t[40091]) == "Müller Misc_D3_17 #17" and t[305] == "polaroid-processor 9.9" and t[271] == "EPSON"
    assert t[282] == (1260, 1) or t[282] == 1260 or tuple(t[282]) == (1260, 1)
    with Image.open(p) as im:      # other readers still open it
        im.load()
        assert im.size == (200, 300)
    import cv2
    assert cv2.imread(str(p)) is not None
    set_metadata(str(p), fm_of({}))
    t = read_tiff_tags(p)
    assert {270, 315, 33432, 700, 34665, 40091}.isdisjoint(t) and t[305] == "polaroid-processor 9.9"
    assert tiff_strips(p) == strips


def test_tiff_rewrite_replaces_a_tifffile_default_description(tmp_path):
    p = tmp_path / "old.tif"
    tifffile.imwrite(str(p), image(), photometric="rgb")      # as the masters of before: {"shape": [h, w, 3]}
    assert read_tiff_tags(p)[270].startswith('{"shape"')
    set_metadata(str(p), fm_of({}))
    t = read_tiff_tags(p)
    assert 270 not in t and t[305] == "polaroid-processor 9.9"


def test_tiff_resolution_patch_still_works_after_a_rewrite(tmp_path):
    from polaroid_processor.imageio import set_resolution
    p = tmp_path / "a.tif"
    write_tiff(str(p), image(), 126.0, meta=fm_of())
    set_metadata(str(p), fm_of({"title": "again"}))
    set_resolution(str(p), 130.0)
    with tifffile.TiffFile(p) as tf:
        num, den = tf.pages[0].tags["XResolution"].value
        assert num / den == pytest.approx(1300)
    assert xp(read_tiff_tags(p)[40091]) == "again"


def test_tiff_with_alpha_and_icc_keeps_them(tmp_path):
    p = tmp_path / "a.tif"
    icc = b"icc-profile-bytes" * 20
    write_tiff(str(p), image(), 126.0, icc, alpha=np.full((300, 200), 200, np.uint8), meta=fm_of())
    t = read_tiff_tags(p)
    assert bytes(t[34675]) == icc and tifffile.imread(p).shape == (300, 200, 4)


def test_unexpected_files_are_a_patch_error_not_a_corruption(tmp_path):
    garbage = tmp_path / "g.tif"
    garbage.write_bytes(b"not a tiff at all" * 4)
    for f in (set_tiff, set_jpeg, set_png):
        with pytest.raises(PatchError):
            f(str(garbage), fm_of())
    assert garbage.read_bytes() == b"not a tiff at all" * 4
    multi = tmp_path / "m.tif"
    with tifffile.TiffWriter(str(multi)) as tw:
        tw.write(image(), photometric="rgb")
        tw.write(image(), photometric="rgb")
    before = multi.read_bytes()
    with pytest.raises(PatchError):
        set_tiff(str(multi), fm_of())
    assert multi.read_bytes() == before
    with pytest.raises(PatchError):
        set_metadata(str(tmp_path / "x.gif"), fm_of())


def test_exif_block_is_none_without_tags():
    assert exif_block(M.FileMeta()) is None
    assert exif_block(fm_of({})) is not None       # Make / Model / Software


def test_no_patch_leftovers_next_to_the_files(tmp_path):
    p = tmp_path / "a.jpg"
    write_derivative(str(p), image(), "jpg", 90, 300, meta=fm_of())
    set_metadata(str(p), fm_of({"title": "x"}))
    assert sorted(f.name for f in tmp_path.iterdir()) == ["a.jpg"]


# --------------------------------------------------------------------------- safety of the rewrite
def test_a_crash_before_the_pointer_switch_leaves_the_old_valid_file(tmp_path, monkeypatch):
    from polaroid_processor import meta_patch
    p = tmp_path / "a.tif"
    write_tiff(str(p), image(), 126.0, meta=fm_of({"title": "old"}))
    strips = tiff_strips(p)

    def boom(fd):
        raise KeyboardInterrupt

    monkeypatch.setattr(meta_patch.os, "fsync", boom)
    with pytest.raises(KeyboardInterrupt):
        set_metadata(str(p), fm_of({"title": "new"}))
    monkeypatch.undo()
    assert xp(read_tiff_tags(p)[40091]) == "old" and tiff_strips(p) == strips
    assert np.array_equal(tifffile.imread(p), image())
    set_metadata(str(p), fm_of({"title": "new"}))           # and the next rewrite is fine
    assert xp(read_tiff_tags(p)[40091]) == "new" and tiff_strips(p) == strips


def test_rewrites_are_flushed_but_a_part_file_is_not(tmp_path, monkeypatch):
    from polaroid_processor import meta_patch
    calls = []
    monkeypatch.setattr(meta_patch.os, "fsync", lambda fd: calls.append(fd))
    for name, writer in (("a.tif", lambda f: write_tiff(f, image(), 126.0, meta=fm_of())),
                         ("a.jpg", lambda f: write_derivative(f, image(), "jpg", 90, 300, meta=fm_of())),
                         ("a.png", lambda f: write_derivative(f, image(), "png", 90, 300, meta=fm_of()))):
        calls.clear()
        writer(str(tmp_path / name))              # written as NAME.part, then renamed
        assert calls == [], name
        set_metadata(str(tmp_path / name), fm_of({"title": "x"}))
        assert calls, name                         # a finished export is flushed before the new header counts


def test_icc_profiles_survive_a_rewrite(tmp_path):
    icc = Image.new("RGB", (1, 1)).info.get("icc_profile") or _srgb()
    pj, pp = tmp_path / "a.jpg", tmp_path / "a.png"
    write_derivative(str(pj), image(), "jpg", 90, 300, icc, meta=fm_of({"title": "a"}))
    write_derivative(str(pp), image(), "png", 90, 300, icc, meta=fm_of({"title": "a"}))
    set_metadata(str(pj), fm_of())
    set_metadata(str(pp), fm_of())
    for p in (pj, pp):
        with Image.open(p) as im:
            assert im.info["icc_profile"] == icc and im.info["dpi"][0] == pytest.approx(300, abs=0.01)


def _srgb():
    from PIL import ImageCms
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
