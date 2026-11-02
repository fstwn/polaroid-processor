"""Photo-window finder on synthetic straightened prints with a known window."""
import dataclasses
import json
import math

import cv2
import numpy as np
import pytest

from polaroid_processor.formats import POLAROID_600, POLAROID_GO
from polaroid_processor.window import NOMINAL, WindowResult, find_window

PX = 12.0  # px per mm
PX_FINE = 20.0  # for the tests that pin a fraction of a pixel
SPEC = NOMINAL[("polaroid-600", "square")]


def _mask(shape, poly, ss=4):
    """Coverage mask (0..1) of a polygon in continuous px. Four corners (a parallelogram): analytic edge
    coverage, clip(distance to each edge + 0.5), exact for the near-axis-aligned edges used here.
    Other shapes: fillPoly on an ss-times finer grid."""
    h, w = shape
    if len(poly) != 4:
        m = np.zeros((h * ss, w * ss), np.uint8)
        cv2.fillPoly(m, [np.round(poly * ss - 0.5).astype(np.int32).reshape(-1, 1, 2)], 255)
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    x0, y0 = max(0, int(poly[:, 0].min()) - 2), max(0, int(poly[:, 1].min()) - 2)
    x1, y1 = min(w, int(poly[:, 0].max()) + 3), min(h, int(poly[:, 1].max()) + 3)
    gx, gy = np.meshgrid(np.arange(x0, x1, dtype=np.float32) + 0.5, np.arange(y0, y1, dtype=np.float32) + 0.5)
    centre = poly.mean(axis=0)
    cover = np.ones(gx.shape, np.float32)
    for i in range(4):
        a, b = poly[i], poly[(i + 1) % 4]
        n = np.array([-(b - a)[1], (b - a)[0]]) / np.linalg.norm(b - a)
        if np.dot(centre - a, n) < 0:
            n = -n  # inward
        dist = (gx - a[0]) * n[0] + (gy - a[1]) * n[1]
        cover *= np.clip(dist + 0.5, 0, 1)
    out = np.zeros((h, w), np.float32)
    out[y0:y1, x0:x1] = cover
    return out


def window_quad(rot_deg=0.0, shear_deg=0.0, spec=SPEC, px=PX):
    """Window parallelogram in continuous px: horizontal borders at angle rot, vertical ones at -rot + shear."""
    a, s = math.radians(rot_deg), math.radians(shear_deg)
    ex = np.array([math.cos(a), math.sin(a)])
    ey = np.array([math.tan(-a + s), 1.0])
    c = np.array([spec.left + spec.width / 2, spec.top + spec.height / 2]) * px
    hw, hh = spec.width * px / 2, spec.height * px / 2
    return np.array([c - hw * ex - hh * ey, c + hw * ex - hh * ey, c + hw * ex + hh * ey, c - hw * ex + hh * ey])


def make_print(fmt=POLAROID_600, px=PX, frame=235.0, photo_level=None, texture=True, quad=None, circle=None,
               stripes=False, seed=0, no_window=False, frame_line=None):
    """RGB straightened print. The photo is smooth noise (flat at ``photo_level`` without ``texture``).

    ``frame_line``: (quad of the line's outer edge, quad of its inner edge, level) draws a printed line in the frame."""
    rng = np.random.default_rng(seed)
    w, h = int(round(fmt.width_mm * px)), int(round(fmt.height_mm * px))
    img = np.full((h, w), frame, np.float32) + rng.normal(0, 0.8, (h, w)).astype(np.float32)
    if stripes:  # artwork in the frame: dark diagonal bars that run into the window border
        yy, xx = np.mgrid[:h, :w]
        img = np.where(((xx + yy * 0.5) // (3.0 * px)) % 2 == 0, frame * 0.35, img)
    if frame_line is not None:
        outer, inner, level = frame_line
        band = np.clip(_mask((h, w), outer) - _mask((h, w), inner), 0, 1)
        img = img * (1 - band) + level * band
    if not no_window:
        base = rng.uniform(30, 190, (h // 40 + 2, w // 40 + 2)).astype(np.float32) if texture else np.zeros((h // 40 + 2, w // 40 + 2), np.float32)
        photo = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
        if photo_level is not None:
            photo = photo - photo.mean() + photo_level
        if circle is not None:
            cx, cy, r = circle
            t = np.linspace(0, 2 * np.pi, 720, endpoint=False)
            m = _mask((h, w), np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], -1))
        else:
            m = _mask((h, w), quad if quad is not None else window_quad(px=px))
        img = img * (1 - m) + photo * m
    return np.clip(np.repeat(img[..., None], 3, axis=2), 0, 255).astype(np.uint8)


def _err_mm(res, truth, px=PX):
    return np.linalg.norm(res.quad - truth, axis=1).max() / px


def test_axis_aligned_window_to_a_fraction_of_a_pixel():
    # flat photo: the half-height crossing is then exactly the drawn edge. A half-pixel error in the
    # continuous-coordinate convention would be 0.025 mm here, a quarter pixel 0.0125.
    quad = window_quad(px=PX_FINE)
    r = find_window(make_print(px=PX_FINE, photo_level=60.0, texture=False, quad=quad), PX_FINE, POLAROID_600)
    assert r.found, r.reason
    assert r.kind == "square"
    assert _err_mm(r, quad, PX_FINE) < 0.01
    assert abs(r.rotation_deg) < 0.02 and abs(r.shear_deg) < 0.02
    assert abs(r.size_mm[0] - SPEC.width) < 0.02 and abs(r.size_mm[1] - SPEC.height) < 0.02


def test_textured_photo_stays_within_a_tenth_of_a_millimetre():
    r = find_window(make_print(), PX, POLAROID_600)
    assert r.found, r.reason
    assert _err_mm(r, window_quad()) < 0.08


@pytest.mark.parametrize("rot,shear", [(0.4, 0.0), (-0.3, 0.45), (0.0, -0.5)])
def test_rotated_and_sheared_window_keeps_its_own_quad(rot, shear):
    quad = window_quad(rot, shear)
    r = find_window(make_print(quad=quad), PX, POLAROID_600)
    assert r.found, r.reason
    assert _err_mm(r, quad) < 0.1
    assert abs(r.rotation_deg - rot) < 0.05
    assert abs(r.shear_deg - shear) < 0.05


def test_thin_dark_line_at_the_border_gives_the_outer_flank():
    # a 3 px (0.25 mm) dark line just outside a bright photo: the answer must be the line's outer
    # edge; the photo side would be 0.25 mm further in
    quad = window_quad(px=PX_FINE)
    grow = 0.25 * PX_FINE
    outer = quad + np.array([[-grow, -grow], [grow, -grow], [grow, grow], [-grow, grow]])
    img = make_print(px=PX_FINE, photo_level=200.0, texture=False, quad=quad, frame_line=(outer, quad, 40.0))
    r = find_window(img, PX_FINE, POLAROID_600)
    assert r.found, r.reason
    assert _err_mm(r, outer, PX_FINE) < 0.03
    assert _err_mm(r, quad, PX_FINE) > 0.2


def test_artwork_crossing_the_border():
    quad = window_quad(0.3, 0.0)
    r = find_window(make_print(quad=quad, stripes=True), PX, POLAROID_600)
    assert r.found, r.reason
    assert _err_mm(r, quad) < 0.1


def test_printed_line_parallel_to_the_border_is_never_a_wrong_window():
    # a dark printed line 1.5 mm outside the window, as strong as the border and as long
    quad = window_quad()
    far = 1.5 * PX
    outer = quad + np.array([[-far - 6, -far - 6], [far + 6, -far - 6], [far + 6, far + 6], [-far - 6, far + 6]])
    inner = quad + np.array([[-far, -far], [far, -far], [far, far], [-far, far]])
    img = make_print(quad=quad, frame_line=(outer, inner, 20.0))
    r = find_window(img, PX, POLAROID_600, kind="square")
    # the line wins the search (it is as strong as the border); only the size check stands between
    # it and a wrong window, and it must hold
    assert not r.found
    assert r.reason.startswith("size")
    assert _err_mm(r, quad) > 2.0


def test_dark_photo_on_a_black_frame():
    quad = window_quad(-0.2, 0.2)
    img = make_print(frame=8.0, photo_level=45.0, quad=quad)
    r = find_window(img, PX, POLAROID_600)
    assert r.found, r.reason
    assert _err_mm(r, quad) < 0.15


def test_dark_photo_edge_below_the_contrast_limit_is_a_clean_not_found():
    # photo and black frame differ by 4 levels: there is no border to see
    img = make_print(frame=8.0, photo_level=12.0, texture=False)
    r = find_window(img, PX, POLAROID_600)
    assert not r.found
    assert r.kind == "square" and "no border" in r.reason and r.quad is None


def test_round_window():
    cx, cy = (SPEC.left + SPEC.width / 2) * PX + 3.0, (7.7 + SPEC.width / 2) * PX - 4.0
    rad = 38.3 * PX
    r = find_window(make_print(circle=(cx, cy, rad)), PX, POLAROID_600)
    assert r.found, r.reason
    assert r.kind == "round"
    assert abs(r.circle[0] - cx) < 0.1 * PX and abs(r.circle[1] - cy) < 0.1 * PX
    assert abs(r.circle[2] - rad) < 0.1 * PX
    assert np.allclose(r.quad[2] - r.quad[0], 2 * r.circle[2])  # bounding square of the circle
    assert r.size_mm[0] >= r.size_mm[1]  # long diameter first, like the ellipse
    assert r.ellipse[2] >= r.ellipse[3]
    assert abs(2 * r.ellipse[2] / PX - r.size_mm[0]) < 1e-6


def test_ellipse_is_not_a_round_window():
    # 1.2 mm wider than tall, same mean diameter as a circle: only the ellipse check can reject it: scanner anisotropy or a wrong window, not a circle
    cx, cy = (SPEC.left + SPEC.width / 2) * PX, (7.7 + SPEC.width / 2) * PX
    h, w = int(POLAROID_600.height_mm * PX), int(POLAROID_600.width_mm * PX)
    img = make_print()
    img[:] = 235
    m = np.zeros((h * 4, w * 4), np.uint8)
    cv2.ellipse(m, (int(cx * 4), int(cy * 4)), (int(38.65 * PX * 4), int(38.05 * PX * 4)), 0, 0, 360, 255, -1)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255
    img = (img * (1 - m) + 60 * m).astype(np.uint8)
    r = find_window(img, PX, POLAROID_600, kind="round")
    assert not r.found and r.reason.startswith("round border is an ellipse")


def test_failed_square_search_is_not_reported_as_round():
    # square window whose bottom border has no step (photo as bright as the frame there)
    quad = window_quad()
    img = make_print(quad=quad, photo_level=60.0, texture=False)
    bottom = int(quad[2][1])
    img[bottom - int(8 * PX):bottom + 2, :, :] = 235
    r = find_window(img, PX, POLAROID_600)
    assert not r.found
    assert r.kind == "square"
    assert r.notes[0].startswith("square:") and r.notes[1].startswith("round:")


def test_kind_hint_limits_the_search():
    r = find_window(make_print(), PX, POLAROID_600, kind="round")
    assert not r.found and r.kind == "round"


def test_no_window_is_a_clean_not_found():
    r = find_window(make_print(no_window=True), PX, POLAROID_600)
    assert not r.found
    assert r.reason
    assert r.quad is None


def test_wrong_size_is_rejected_by_the_size_check():
    # window 2 mm too big on both axes, centred on the nominal one: the borders are inside the
    # search zone, so only the size check can reject it
    big = dataclasses.replace(SPEC, left=SPEC.left - 1.0, top=SPEC.top - 1.0, width=SPEC.width + 2.0, height=SPEC.height + 2.0)
    quad = window_quad(spec=big)
    r = find_window(make_print(quad=quad), PX, POLAROID_600, kind="square")
    assert not r.found
    assert r.reason.startswith("size")
    assert abs(r.size_mm[0] - big.width) < 0.15 and abs(r.size_mm[1] - big.height) < 0.15  # measured, just not plausible


def test_go_window():
    spec = NOMINAL[("polaroid-go", "square")]
    quad = window_quad(0.4, 0.2, spec=spec)
    r = find_window(make_print(POLAROID_GO, quad=quad), PX, POLAROID_GO)
    assert r.found, r.reason
    assert _err_mm(r, quad) < 0.1


def test_works_at_low_resolution_and_16_bit():
    px = 8.0
    quad = window_quad(0.2, 0.1, px=px)
    rgb = make_print(quad=window_quad(0.2, 0.1))
    small = cv2.resize(rgb, None, fx=px / PX, fy=px / PX, interpolation=cv2.INTER_AREA)
    r = find_window(small.astype(np.uint16) * 257, px, POLAROID_600)
    assert r.found, r.reason
    assert _err_mm(r, quad, px) < 0.15


def test_margin_shifts_the_search():
    margin = 0.2
    quad = window_quad(0.2, 0.0)
    m = int(round(margin * PX))
    cut = make_print(quad=quad)[m:-m, m:-m]
    r = find_window(cut, PX, POLAROID_600, margin_mm=margin)
    assert r.found, r.reason
    assert _err_mm(r, quad - m) < 0.1


def _round_result():
    cx, cy = (SPEC.left + SPEC.width / 2) * PX, (7.7 + SPEC.width / 2) * PX
    return find_window(make_print(circle=(cx, cy, 38.3 * PX)), PX, POLAROID_600)


def test_scaled_copies_everything_and_leaves_the_original_alone():
    sq = find_window(make_print(quad=window_quad(0.2, 0.1)), PX, POLAROID_600)
    big = sq.scaled(2.0)
    assert np.allclose(big.quad, sq.quad * 2) and big.px_per_mm == 2 * sq.px_per_mm
    assert big.size_mm == sq.size_mm and big.rotation_deg == sq.rotation_deg
    big.rms_mm["top"] = 99.0
    big.notes.append("x")
    assert sq.rms_mm["top"] < 1.0 and "x" not in sq.notes
    rd = _round_result()
    assert rd.found, rd.reason
    bd = rd.scaled(2.0)
    assert np.allclose(bd.circle, np.array(rd.circle) * 2)
    assert np.allclose(bd.ellipse[:4], np.array(rd.ellipse[:4]) * 2) and bd.ellipse[4] == rd.ellipse[4]
    assert np.allclose(bd.quad, rd.quad * 2)


@pytest.mark.parametrize("make", [lambda: find_window(make_print(quad=window_quad(0.2, 0.1)), PX, POLAROID_600),
                                  _round_result,
                                  lambda: find_window(make_print(no_window=True), PX, POLAROID_600)])
def test_dict_round_trip(make):
    r = make()
    d = json.loads(json.dumps(r.to_dict()))  # what the manifest does
    back = WindowResult.from_dict(d)
    assert back.found == r.found and back.kind == r.kind and back.reason == r.reason
    assert back.notes == r.notes and abs(back.px_per_mm - r.px_per_mm) < 1e-3
    for name in ("quad", "size_mm", "circle", "ellipse", "nominal_mm"):
        a, b = getattr(r, name), getattr(back, name)
        assert (a is None) == (b is None)
        if a is not None:
            assert np.allclose(a, b, atol=2e-3)
    assert back.rms_mm.keys() == r.rms_mm.keys()
    assert back.rotation_deg == r.rotation_deg or abs(back.rotation_deg - r.rotation_deg) < 1e-4


# ---------------------------------------------------------------- real scans

SCAN_CASES = [  # (label file, relative path, expected kind)
    ("HOLDOUT", "05_roller_roids/250728_SICKalps_L2_D2/250728_SICKalps_L2_D2_21.tif", "square"),  # gold
    ("HOLDOUT", "02_polaroid/260226_Polaroids_Samsoe/260315_Polaroids_Samsoe_03.tif", "square"),  # printed, artwork at the border
    ("HOLDOUT", "05_roller_roids/250728_SICKalps_L2_D2/250728_SICKalps_L2_D2_02.tif", "round"),
    ("SAMPLES", "scans/269999_Polaroid_24.tif", "square"),  # Go
    ("SAMPLES", "white_bg/150101_Polaroid_Misc_2023_01.tif", "square"),  # camera shot
]


@pytest.mark.scans
@pytest.mark.parametrize("which,rel,kind", SCAN_CASES)
def test_window_on_labelled_scans(which, rel, kind):
    import test_regression_scans as reg
    from polaroid_processor import load_rgb
    from polaroid_processor.detection import quad_size
    from polaroid_processor.formats import classify, get_format
    from polaroid_processor.geometry import rectify

    path = reg.LIBRARY / rel if which == "HOLDOUT" else reg.SAMPLES_ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} not present")
    label = dict(reg._labels(reg.HOLDOUT if which == "HOLDOUT" else reg.SAMPLES))[rel]
    quad = np.array(label["quad"], float)
    fmt = get_format(label["format"]) if label.get("format") else classify(quad_size(quad))
    rgb = load_rgb(str(path))
    f = 0.25
    small = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    img = rectify(small, quad * f)
    r = find_window(img, img.shape[1] / fmt.width_mm, fmt)  # the basis of NOMINAL: print width / format width
    assert r.found, r.reason
    assert r.kind == kind
    spec = NOMINAL[(fmt.name, kind)]
    assert abs(r.size_mm[0] - spec.width) < 0.5 and abs(r.size_mm[1] - spec.height) < 0.5
