"""An export is stamped with the outline it was cut from: a refit or a fix merged during the run exports again;
scans whose file names share a stem are refused."""
import numpy as np
import pytest
import tifffile

from polaroid_processor import api, pipeline
from polaroid_processor.manifest import Manifest
from polaroid_processor.scale import MM_PER_INCH

from synth import PX_PER_MM, make_scan

PPI = PX_PER_MM * MM_PER_INCH
SHIFT = 4.0      # px the refit / the fix moves the left side


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    d = tmp_path_factory.mktemp("outline_scans")
    rgb, _ = make_scan("white", 0.3, seed=7)
    tifffile.imwrite(str(d / "one.tif"), rgb, photometric="rgb")
    return d


def run(src, out, **kw):
    return api.process([str(src)], str(out), api.make_settings(workers=1, ppi=PPI, flatten="off", **kw))


def width(out, name="one.tif"):
    with tifffile.TiffFile(str(out / name)) as tf:
        return tf.pages[0].shape[1]


@pytest.fixture(scope="module")
def plain_width(src, tmp_path_factory):
    out = tmp_path_factory.mktemp("outline_plain")
    run(src, out)
    return width(out)


def test_refit_after_the_export_cuts_the_scan_again(src, plain_width, tmp_path, monkeypatch):
    real_evaluate = pipeline.evaluate

    def evaluate(entry, *a, **kw):                       # needs_review until the (patched) refit has been applied
        status, warnings, fmt = real_evaluate(entry, *a, **kw)
        if not (entry.get("detection") or {}).get("refit"):
            status = "needs_review"
        return status, warnings, fmt

    def size_refit(entry, *a, **kw):
        quad = np.array(entry["detection"]["quad"], float)
        quad[[0, 3], 0] -= SHIFT                           # the left side moves outward
        return quad, {"side": "left", "shift_px": SHIFT}

    monkeypatch.setattr(pipeline, "evaluate", evaluate)
    monkeypatch.setattr(pipeline, "size_refit", size_refit)
    out = tmp_path / "out"
    summary = run(src, out)
    monkeypatch.undo()
    e = Manifest(str(out)).get("one.tif")
    assert e["detection"]["refit"] and summary["exported"] == 1
    assert width(out) == pytest.approx(plain_width + SHIFT, abs=1.5)         # cut from the refit outline, not retagged
    assert e["output_size_px"][0] == width(out)
    assert run(src, out)["exported"] == 0                                    # and the stamped key is the right one


def test_fix_merged_during_the_run_cuts_the_scan_again(src, plain_width, tmp_path, monkeypatch):
    real_sizes = pipeline.reference_sizes
    out = tmp_path / "out"

    fixes = []

    def reference_sizes(manifest):                       # a fix saved by someone else while the sizes are checked
        sizes = real_sizes(manifest)
        if not fixes:
            fixes.append(api.fix_scan(str(out), "one.tif", shifts={"left": [SHIFT]}, author="agent", export=False))
        return sizes

    monkeypatch.setattr(pipeline, "reference_sizes", reference_sizes)
    summary = run(src, out)
    monkeypatch.undo()
    e = Manifest(str(out)).get("one.tif")
    assert e["override"] and summary["exported"] == 1
    assert width(out) == pytest.approx(plain_width + SHIFT, abs=1.5)
    assert run(src, out)["exported"] == 0


@pytest.mark.parametrize("names, listed", [(("a.tif", "a.tiff", "b.tif"), "a.tif.*a.tiff"),
                                           (("A.tif", "a.tiff", "b.tif"), "A.tif.*a.tiff")])      # also across case
def test_names_with_the_same_stem_are_refused(tmp_path, names, listed):
    d = tmp_path / "in"
    d.mkdir()
    for name in names:
        tifffile.imwrite(str(d / name), np.zeros((10, 10, 3), np.uint8), photometric="rgb")
    with pytest.raises(ValueError, match=f"same stem.*{listed}"):
        api.process([str(d)], str(tmp_path / "out"), api.make_settings(workers=1))
    assert not (tmp_path / "out" / "polaroid-manifest.json").exists()
