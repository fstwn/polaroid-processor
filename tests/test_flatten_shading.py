"""Curved-edge unwarping and edge-shadow removal on synthetic scans with known geometry."""
import numpy as np
import pytest
import cv2

from polaroid_processor import detect
from polaroid_processor import flatten as flat
from polaroid_processor import shading
from polaroid_processor.geometry import rectify

from synth import PRINT_MM, PX_PER_MM, make_scan

BOW_MM = 0.9


def bowed_scan(bow_mm: float = BOW_MM, seed: int = 3):
    """A straight print whose left side is pushed inward by up to ``bow_mm`` in its middle,
    fading out towards the right (a curled print seen from above)."""
    rgb, quad = make_scan("white", 0.0, seed=seed, rim=False)
    h, w = rgb.shape[:2]
    x_left, x_right = quad[0, 0], quad[1, 0]
    y_top, y_bot = quad[0, 1], quad[3, 1]
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32) + 0.5, np.arange(h, dtype=np.float32) + 0.5)
    u = np.clip((X - x_left) / (x_right - x_left), 0, 1)
    v = np.clip((Y - y_top) / (y_bot - y_top), 0, 1)
    d = bow_mm * PX_PER_MM * 4 * v * (1 - v) * (1 - u) ** 3
    mapx, mapy = (X - d - 0.5).astype(np.float32), (Y - 0.5).astype(np.float32)
    warped = cv2.remap(rgb, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped, quad


def side_spread_mm(img: np.ndarray, ext_px: float, thr: int) -> dict:
    """p95 deviation from straight of each traced side of a rectified image extended by ext_px."""
    h, w = img.shape[:2]
    rect = np.array([[ext_px, ext_px], [w - ext_px, ext_px], [w - ext_px, h - ext_px], [ext_px, h - ext_px]])
    curves = flat.trace(img, rect, PX_PER_MM, thr, flat.FlattenParams(max_dev_mm=9))
    out = {}
    for s, c in curves.items():
        o = c.offset(np.linspace(0, c.length, 101))
        out[s] = float(np.percentile(np.abs(o - np.median(o)), 95)) / PX_PER_MM
    return out


@pytest.fixture(scope="module")
def bowed():
    rgb, truth = bowed_scan()
    det = detect(rgb)
    curves = flat.trace(rgb, det.quad, PX_PER_MM, det.speck_threshold)
    return rgb, truth, det, curves


def test_trace_finds_the_bow_and_plans_flattening(bowed):
    _, _, _, curves = bowed
    assert all(c.reliable for c in curves.values()), {s: c.reason for s, c in curves.items()}
    assert curves["left"].bend_px() / PX_PER_MM == pytest.approx(BOW_MM, abs=0.15)
    for s in ("top", "right", "bottom"):
        assert curves[s].bend_px() / PX_PER_MM < 0.1
    p = flat.plan(curves, PX_PER_MM)
    assert p.apply and "left" in p.sides


def test_unwarp_straightens_the_bowed_side(bowed):
    rgb, _, det, curves = bowed
    ext = 1.5 * PX_PER_MM
    straight = side_spread_mm(rectify(rgb, det.quad, -ext), ext, det.speck_threshold)
    flattened = side_spread_mm(flat.unwarp(rgb, curves, det.quad, -ext), ext, det.speck_threshold)
    assert straight["left"] > 0.5 * BOW_MM
    assert max(flattened.values()) < 0.1, flattened


def test_fade_straightens_the_side_and_leaves_the_inside_plain(bowed):
    rgb, _, det, curves = bowed
    ext, fade = 1.5 * PX_PER_MM, 5.0 * PX_PER_MM
    flattened = side_spread_mm(flat.unwarp(rgb, curves, det.quad, -ext, fade_px=fade), ext, det.speck_threshold)
    assert max(flattened.values()) < 0.1, flattened
    # beyond the fade the map is the bilinear map between the corners: no curvature along rows
    # or columns (Coons would carry the bow of the left side across the print)
    for f, plain in ((fade, True), (None, False)):
        mapx, mapy, _ = flat.coons_map(curves, det.quad, fade_px=f)
        k = int(fade) + 20
        inner_x, inner_y = mapx[k:-k:50, k:-k:50], mapy[k:-k:50, k:-k:50]
        curv = max(np.abs(np.diff(inner_x, 2, axis=0)).max(), np.abs(np.diff(inner_y, 2, axis=1)).max())
        assert (curv < 0.01) == plain, (f, curv)


def test_unwarp_keeps_the_print_size(bowed):
    _, _, det, curves = bowed
    W, H = flat.output_size(curves, det.quad)
    assert W / PX_PER_MM == pytest.approx(PRINT_MM[0], abs=0.15)
    assert H / PX_PER_MM == pytest.approx(PRINT_MM[1], abs=0.15)


def test_straight_print_is_not_flattened():
    rgb, _ = make_scan("white", 0.4, seed=5)
    det = detect(rgb)
    curves = flat.trace(rgb, det.quad, PX_PER_MM, det.speck_threshold)
    assert not flat.plan(curves, PX_PER_MM).apply


def test_curves_round_trip_through_dict(bowed):
    _, _, det, curves = bowed
    again = flat.curves_from_dict({s: c.to_dict() for s, c in curves.items()})
    assert np.allclose(flat.corners(again, det.quad), flat.corners(curves, det.quad), atol=1e-6)


# --------------------------------------------------------------------------- shading
def frame_image(kind: str = "white", h: int = 900, w: int = 700) -> np.ndarray:
    """A rectified print border region: frame colour, photo far inside."""
    img = np.full((h, w, 3), (232, 234, 228), np.uint8)
    if kind == "stripes":  # printed frame: graphic bands parallel to the bottom edge
        for y in range(h - 160, h, 20):
            img[y: y + 10] = (40, 40, 40)
    return img


def shadow(img: np.ndarray, depth_mm: float = 0.4, strength: float = 0.15) -> np.ndarray:
    d = np.arange(img.shape[0])[::-1].astype(np.float32) / PX_PER_MM  # distance from the bottom edge
    g = 1 - strength * np.exp(-d / depth_mm)
    return np.clip(img.astype(np.float32) * g[:, None, None], 0, 255).astype(np.uint8)


def bottom_profile(img: np.ndarray) -> np.ndarray:
    return img[::-1].astype(np.float32).mean(axis=(1, 2))


def test_shadow_on_plain_frame_is_removed():
    clean = frame_image()
    shaded = shadow(clean)
    fixed, info = shading.correct(shaded, PX_PER_MM)
    assert "bottom" in info["sides"]
    near = int(0.1 * PX_PER_MM)
    before = bottom_profile(shaded)[near] / bottom_profile(shaded)[-1]
    after = bottom_profile(fixed)[near] / bottom_profile(fixed)[-1]
    assert before < 0.9 and abs(after - 1) < 0.02
    # sides without a shadow are left alone
    assert "top" not in info["sides"] and np.array_equal(fixed[:50, 100:-100], shaded[:50, 100:-100])


def test_printed_frame_is_left_alone():
    shaded = shadow(frame_image("stripes"))
    fixed, info = shading.correct(shaded, PX_PER_MM)
    assert "bottom" not in info["sides"]
    assert np.array_equal(fixed[-100:], shaded[-100:])


def test_process_flattens_by_default_and_can_be_switched_off(tmp_path, capsys):
    import json
    import tifffile
    from polaroid_processor.cli import main
    src = tmp_path / "in"
    src.mkdir()
    rgb, _ = bowed_scan()
    tifffile.imwrite(str(src / "bowed.tif"), rgb, photometric="rgb")
    ppi = str(PX_PER_MM * 25.4)
    for mode, expect in (("auto", True), ("off", False)):
        out = tmp_path / f"out_{mode}"
        rc = main(["process", str(src), "-o", str(out), "--ppi", ppi, "--no-derivatives", "-j", "1",
                   "--flatten", mode, "--json"])
        recs = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.strip()]
        r = [x for x in recs if x["type"] == "result"][0]
        assert rc == 0
        assert any(w.startswith("flattened:") for w in r["warnings"]) is expect, r["warnings"]
        assert r["flatten"]["apply"] is expect
        # a flattened export is the print's arc-length size, minus the margin on each side
        if expect:
            w_px = tifffile.TiffFile(r["outputs"]["tiff"]).pages[0].shape[1]
            assert w_px / PX_PER_MM == pytest.approx(PRINT_MM[0] - 0.4, abs=0.2)


def test_accept_keeps_flattening_and_a_manual_fix_turns_it_off(tmp_path, capsys):
    import json
    import tifffile
    from polaroid_processor.cli import main
    src = tmp_path / "in"
    src.mkdir()
    tifffile.imwrite(str(src / "bowed.tif"), bowed_scan()[0], photometric="rgb")
    out = tmp_path / "out"
    main(["process", str(src), "-o", str(out), "--ppi", str(PX_PER_MM * 25.4), "--no-derivatives", "-j", "1"])
    capsys.readouterr()

    def fix(*args):
        main(["fix", str(out), "bowed", *args, "--export", "--json"])
        recs = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.strip()]
        return [r for r in recs if r["type"] == "result"][0]

    assert fix("--accept")["flatten"]["apply"] is True
    assert fix("--nudge", "tl=3,0")["flatten"]["apply"] is False
