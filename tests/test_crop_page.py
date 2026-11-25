"""Hand crop, slice 2: the place step's geometry and the save body of the fix page, run for real in Node.

The functions are the pure block at the top of the page's script (``review/static/index.html``, between the
"crop geometry" markers; ``tests/js/crop_runner.js`` loads that block on its own). A crop rectangle is
``{cx, cy, w, h, angle}`` in scan px and degrees; its corners run tl, tr, br, bl. Skipped without ``node`` on the PATH.
"""
import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not on the PATH")

RUNNER = Path(__file__).parent / "js" / "crop_runner.js"
PAGE = Path(__file__).parent.parent / "src" / "polaroid_processor" / "review" / "static" / "index.html"

W, H = 4000, 3000
A600 = 88.0 / 106.85
AGO = 53.5 / 65.8
O = {"W": W, "H": H, "min": 50}


def geo(*calls):
    p = subprocess.run([NODE, str(RUNNER), str(PAGE), "-"], input=json.dumps([list(c) for c in calls]), capture_output=True, text=True,
                       timeout=120, encoding="utf-8")
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout)
    return out if len(calls) > 1 else out[0]


def rect(cx=2000, cy=1500, w=1000, h=1200, angle=0.0):
    return {"cx": cx, "cy": cy, "w": w, "h": h, "angle": angle}


def corners(r):
    return geo(("rectToQuad", r))


def close(a, b, tol=1e-6):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def inside(r):
    return geo(("insideScan", r, W, H))


# --------------------------------------------------------------------------- rectangle and corners
def test_a_rectangle_gives_its_corners_clockwise_from_the_top_left_and_back():
    q = corners(rect(w=1000, h=1200))
    assert q == [[1500, 900], [2500, 900], [2500, 2100], [1500, 2100]]
    for angle in (0, 12.5, -30, 45, -45):
        r = rect(cx=1800, cy=1400, w=700, h=900, angle=angle)
        back = geo(("rectFromQuad", corners(r)))
        assert close([back[k] for k in ("cx", "cy", "w", "h", "angle")], [r[k] for k in ("cx", "cy", "w", "h", "angle")], 1e-6)
        q = corners(r)
        assert (q[1][1] - q[0][1]) / (q[1][0] - q[0][0]) == pytest.approx(math.tan(math.radians(angle)), abs=1e-9)
        # clockwise on the scan (y down): positive signed area, as the corner checks of the core want
        area = sum(q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1] for i in range(4))
        assert area > 0


def test_a_quad_that_is_no_rectangle_is_fitted_by_its_means_and_not_called_one():
    skew = [[100, 100], [900, 140], [880, 1000], [90, 960]]
    assert geo(("isRectQuad", corners(rect(angle=7)))) is True and geo(("isRectQuad", skew)) is False
    fit = geo(("rectFromQuad", skew))
    assert fit["cx"] == pytest.approx(492.5) and fit["angle"] == pytest.approx(math.degrees(math.atan2(40, 800)))
    assert fit["w"] == pytest.approx((math.hypot(800, 40) + math.hypot(790, 40)) / 2)


# --------------------------------------------------------------------------- move
def test_move_translates_and_stops_at_the_scan_edges():
    r = rect()
    assert geo(("moveRect", r, 100, -50, W, H))["cx"] == 2100
    far = geo(("moveRect", r, 9000, 9000, W, H))
    assert far["cx"] == W - 500 and far["cy"] == H - 600 and inside(far)
    far = geo(("moveRect", r, -9000, -9000, W, H))
    assert far["cx"] == 500 and far["cy"] == 600
    rot = rect(angle=30)
    moved = geo(("moveRect", rot, 9000, 0, W, H))
    assert inside(moved) and moved["angle"] == 30 and max(p[0] for p in corners(moved)) == pytest.approx(W)


# --------------------------------------------------------------------------- resize
@pytest.mark.parametrize("angle", [0, 25, -40])
def test_a_corner_resize_keeps_the_opposite_corner_in_place(angle):
    r = rect(angle=angle)
    q0 = corners(r)
    for handle, fixed in (("br", 0), ("tl", 2), ("tr", 3), ("bl", 1)):
        target = [q0[{"tl": 0, "tr": 1, "br": 2, "bl": 3}[handle]][0] + 90, q0[{"tl": 0, "tr": 1, "br": 2, "bl": 3}[handle]][1] + 60]
        out = geo(("resizeRect", r, handle, target, {**O, "aspect": 0}))
        q = corners(out)
        assert close(q[fixed], q0[fixed], 1e-6)
        assert out["angle"] == angle


def test_a_corner_resize_follows_the_pointer_when_the_corner_is_free():
    r = rect()
    out = geo(("resizeRect", r, "br", [2700, 2400], {**O, "aspect": 0}))
    assert out == {"cx": 2100.0, "cy": 1650.0, "w": 1200.0, "h": 1500.0, "angle": 0}
    # the pointer on the other side of the fixed corner does not flip the rectangle: it stops at the smallest side
    out = geo(("resizeRect", r, "br", [1000, 500], {**O, "aspect": 0}))
    assert out["w"] == 50 and out["h"] == 50 and corners(out)[0] == [1500, 900]


@pytest.mark.parametrize("angle", [0, -17])
def test_a_side_resize_moves_one_side_and_leaves_the_other_dimension(angle):
    r = rect(angle=angle)
    q0 = corners(r)
    n = [math.cos(math.radians(angle)), math.sin(math.radians(angle))]       # the right side's outward direction
    mid = [(q0[1][0] + q0[2][0]) / 2, (q0[1][1] + q0[2][1]) / 2]
    out = geo(("resizeRect", r, "right", [mid[0] + 80 * n[0], mid[1] + 80 * n[1]], {**O, "aspect": 0}))
    assert out["w"] == pytest.approx(1080) and out["h"] == pytest.approx(1200) and out["angle"] == angle
    q = corners(out)
    assert close(q[0], q0[0], 1e-6) and close(q[3], q0[3], 1e-6)               # the left side stays
    top = [(q0[0][0] + q0[1][0]) / 2, (q0[0][1] + q0[1][1]) / 2]
    out = geo(("resizeRect", r, "top", [top[0] - 70 * math.sin(math.radians(angle)), top[1] + 70 * math.cos(math.radians(angle))],
               {**O, "aspect": 0}))                                          # the top side moved 70 px inward along its normal
    assert out["h"] == pytest.approx(1130) and out["w"] == pytest.approx(1000) and close(corners(out)[2], q0[2], 1e-6)


def test_the_aspect_lock_keeps_the_ratio_and_the_opposite_corner():
    r = rect(w=800, h=800 / A600)
    q0 = corners(r)
    out = geo(("resizeRect", r, "br", [q0[2][0] + 200, q0[2][1] + 40], {**O, "aspect": A600}))
    assert out["w"] / out["h"] == pytest.approx(A600) and close(corners(out)[0], q0[0], 1e-6)
    assert out["w"] == pytest.approx(1000)                                    # the pointer's larger extent decides
    out = geo(("resizeRect", r, "br", [q0[2][0] + 10, q0[2][1] + 300], {**O, "aspect": A600}))
    assert out["w"] / out["h"] == pytest.approx(A600) and out["h"] == pytest.approx(r["h"] + 300)
    # a side: the dragged dimension decides, the other follows about the middle line
    out = geo(("resizeRect", r, "right", [q0[1][0] + 100, 700], {**O, "aspect": A600}))
    assert out["w"] == pytest.approx(900) and out["h"] == pytest.approx(900 / A600) and out["cy"] == r["cy"]
    assert out["cx"] - out["w"] / 2 == pytest.approx(r["cx"] - r["w"] / 2)
    out = geo(("resizeRect", r, "bottom", [2000, q0[2][1] + 100], {**O, "aspect": A600}))
    assert out["h"] == pytest.approx(r["h"] + 100) and out["w"] / out["h"] == pytest.approx(A600) and out["cx"] == r["cx"]
    # the smallest side holds for both sides of a locked rectangle
    out = geo(("resizeRect", r, "br", [q0[0][0] + 5, q0[0][1] + 5], {**O, "aspect": A600}))
    assert out["w"] >= 50 - 1e-9 and out["h"] >= 50 - 1e-9 and out["w"] / out["h"] == pytest.approx(A600)


def test_the_locked_aspect_follows_the_orientation_of_the_rectangle():
    assert geo(("lockedAspect", rect(w=800, h=1000), AGO)) == pytest.approx(AGO)
    assert geo(("lockedAspect", rect(w=1000, h=800), AGO)) == pytest.approx(1 / AGO)
    assert geo(("lockedAspect", rect(), 0)) == 0
    out = geo(("resizeRect", rect(w=1200, h=900), "br", [2900, 2300], {**O, "aspect": 1 / A600}))
    assert out["w"] / out["h"] == pytest.approx(1 / A600)


def test_a_resize_cannot_leave_the_scan_and_comes_back_with_the_pointer():
    r = rect()
    out = geo(("resizeRect", r, "br", [W + 500, H + 500], {**O, "aspect": 0}))
    assert inside(out) and max(p[0] for p in corners(out)) <= W + 1e-6 and out["w"] > r["w"]
    out = geo(("resizeRect", r, "right", [W + 5000, 1500], {**O, "aspect": 0}))
    assert corners(out)[1][0] == pytest.approx(W, abs=0.5)
    # every call starts from the rectangle the drag began with: the same pointer gives the same answer, and a pointer back
    # inside gives the free result again
    again = geo(("resizeRect", r, "br", [2600, 2200], {**O, "aspect": 0}))
    assert again["w"] == 1100 and again["h"] == 1300
    rot = rect(w=1800, h=1800, angle=30)
    assert inside(geo(("resizeRect", rot, "br", [3900, 2900], {**O, "aspect": 0})))


# --------------------------------------------------------------------------- rotate
def test_rotation_snaps_to_tenths_is_limited_to_45_and_keeps_the_centre():
    r = rect()
    o = {"max": 45, "W": W, "H": H}
    out = geo(("rotateRect", r, 12.34, o))
    assert out["angle"] == 12.3 and out["cx"] == 2000 and out["cy"] == 1500 and out["w"] == 1000
    assert geo(("rotateRect", r, 80, o))["angle"] == 45 and geo(("rotateRect", r, -80, o))["angle"] == -45
    assert geo(("rotateRect", r, 0.04, o))["angle"] == 0


def test_rotation_at_the_scan_edge_slides_the_rectangle_in_or_turns_less():
    o = {"max": 45, "W": W, "H": H}
    near = rect(cx=2000, cy=1500, w=1000, h=2900)               # nearly the scan's height: a turn cannot fit
    out = geo(("rotateRect", near, 30, o))
    assert inside(out) and 0 <= out["angle"] < 30
    edge = rect(cx=600, cy=1500, w=1000, h=1200)               # at the left edge: it slides right
    out = geo(("rotateRect", edge, 20, o))
    assert out["angle"] == 20 and inside(out) and out["cx"] > 600
    fits = rect(cx=2000, cy=1500, w=3000, h=2000)
    assert inside(geo(("rotateRect", fits, 45, o)))


def test_the_rotation_handle_gives_the_angle_of_the_pointer_seen_from_the_centre():
    r = rect()
    assert geo(("handleAngle", r, [2000, 100])) == pytest.approx(0)            # straight above
    assert geo(("handleAngle", r, [3000, 1500])) == pytest.approx(90)
    assert geo(("handleAngle", r, [2500, 1000])) == pytest.approx(45)
    assert geo(("handleAngle", r, [1500, 1000])) == pytest.approx(-45)
    assert geo(("handleAngle", r, [2000, 2900])) == pytest.approx(180) or geo(("handleAngle", r, [2000, 2900])) == pytest.approx(-180)


# --------------------------------------------------------------------------- draw a new rectangle
def test_a_new_rectangle_is_drawn_between_two_points_in_any_direction():
    o = {**O, "aspect": 0}
    assert geo(("drawRect", [1000, 800], [1600, 1500], o)) == {"cx": 1300.0, "cy": 1150.0, "w": 600.0, "h": 700.0, "angle": 0}
    out = geo(("drawRect", [1600, 1500], [1000, 800], o))                       # dragged up and to the left
    assert out["cx"] == 1300 and out["cy"] == 1150 and out["w"] == 600 and out["h"] == 700
    assert geo(("drawRect", [1000, 800], [1020, 1500], o)) is None              # a side below the smallest: nothing yet
    out = geo(("drawRect", [3500, 2500], [9000, 9000], o))                      # the scan's edge stops it
    assert out["w"] == 500 and out["h"] == 500 and inside(out)


def test_a_drawn_rectangle_keeps_the_aspect_of_a_locked_format_in_the_drag_orientation():
    o = {**O, "aspect": A600}
    out = geo(("drawRect", [1000, 800], [1400, 1700], o))
    assert out["w"] / out["h"] == pytest.approx(A600) and out["h"] == pytest.approx(900)
    out = geo(("drawRect", [1000, 800], [2400, 1200], o))                       # a wide drag draws a landscape one
    assert out["w"] / out["h"] == pytest.approx(1 / A600) and out["w"] == pytest.approx(1400)
    out = geo(("drawRect", [3500, 2500], [3999, 2999], o))
    assert inside(out) and out["w"] / out["h"] == pytest.approx(A600)


# --------------------------------------------------------------------------- what a point grabs
def test_hit_test_finds_handle_corner_side_inside_or_nothing():
    r = rect()                                                                  # corners (1500, 900) .. (2500, 2100)
    scale = 0.25                                                                # css px per scan px
    hit = lambda p, rr=r: geo(("hitTest", rr, p, scale, {"hit": 14, "rotDist": 30}))
    assert hit([2000, 900 - 120])["type"] == "rotate"                           # 30 css px above the top side = 120 scan px
    assert hit([1510, 910]) == {"type": "corner", "name": "tl"}
    assert hit([2495, 2095]) == {"type": "corner", "name": "br"}
    assert hit([2000, 905]) == {"type": "side", "name": "top"} and hit([2000, 2095]) == {"type": "side", "name": "bottom"}
    assert hit([1505, 1500]) == {"type": "side", "name": "left"} and hit([2495, 1500]) == {"type": "side", "name": "right"}
    assert hit([2000, 1500]) == {"type": "inside"} and hit([300, 300]) == {"type": "none"}
    rot = rect(angle=90)                                                        # turned: the handle is on the right now
    cx = geo(("rectToQuad", rot))
    top_mid = [(cx[0][0] + cx[1][0]) / 2, (cx[0][1] + cx[1][1]) / 2]
    assert hit([top_mid[0] + 120, top_mid[1]], rot)["type"] == "rotate"


# --------------------------------------------------------------------------- the hint
def test_the_proportions_hint_names_a_600_or_go_crop_in_either_orientation_within_one_and_a_half_percent():
    aspects = {"600": A600, "go": AGO}
    assert geo(("proportionsOf", 880, 1068.5, aspects)) == ["600", "go"]       # exactly a 600: 1.3 % off a Go, so both fit
    assert geo(("proportionsOf", 1068.5, 880, aspects)) == ["600", "go"]       # landscape
    assert geo(("proportionsOf", 1000 * 0.835, 1000, aspects)) == ["600"]      # 1.4 % over a 600, 2.7 % over a Go
    assert geo(("proportionsOf", 535, 658, aspects)) == ["go", "600"]          # the closer one first
    assert geo(("proportionsOf", 1000 * 0.805, 1000, aspects)) == ["go"]
    assert geo(("proportionsOf", 1000, 1000, aspects)) == [] and geo(("proportionsOf", 700, 1200, aspects)) == []
    assert geo(("proportionsOf", 800, 1000, aspects)) == []                    # 0.8 is 2.9 % off a 600 and 1.6 % off a Go: outside 1.5 %
    assert geo(("proportionsOf", 800, 1000, aspects, 0.03)) == ["go", "600"]   # the tolerance can be given


# --------------------------------------------------------------------------- the page's limits against the core's checks
def core_check(corners, W, H):
    from polaroid_processor.item import check_crop_quad
    rounded = [[math.floor(v * 100 + 0.5) / 100 for v in p] for p in corners]      # as commit() rounds them
    check_crop_quad(rounded, W, H)


PAGE_MIN, PAGE_MAX = 50.5, 44.9      # the core's 50 px + 0.5 and 45 deg - 0.1: what geom() gives the functions


def locked(r, aspect):
    return 0 if not aspect else (1 / aspect if r["w"] > r["h"] else aspect)


def base_rects():
    """Rectangles at several angles, shapes and places (edges included), made valid with limitRect first."""
    raw = []
    for (W, H) in ((1600, 2000), (4000, 3000)):
        for angle in (0, 5, 30, 44.8, 44.9, -44.9, -30):
            for (w, h) in ((700, 900), (900, 700), (600, 600)):
                for pos in ("middle", "left", "bottom right", "top"):
                    cx = {"middle": W / 2, "left": w / 2 + 3, "bottom right": W - w / 2 - 3, "top": W / 2}[pos]
                    cy = {"middle": H / 2, "left": H / 2, "bottom right": H - h / 2 - 3, "top": h / 2 + 3}[pos]
                    raw.append((W, H, {"cx": cx, "cy": cy, "w": w, "h": h, "angle": angle}))
    lim = lambda W, H: {"W": W, "H": H, "min": PAGE_MIN, "max": PAGE_MAX}
    fixed = geo(*[["limitRect", r, lim(W, H)] for W, H, r in raw])
    return [(W, H, r) for (W, H, _), r in zip(raw, fixed)]


def limit_cases():
    """Calls with the page's limits: every one gives a rectangle that must pass the core's corner checks once rounded."""
    calls = []
    for W, H, r in base_rects():
        lim = {"W": W, "H": H, "min": PAGE_MIN, "max": PAGE_MAX}
        cx, cy = r["cx"], r["cy"]
        for aspect in (0, A600, AGO):
            for handle in ("tl", "tr", "br", "bl", "top", "right", "bottom", "left"):
                for target in ([cx, cy], [-9000, -9000], [W * 3, H * 3], [cx + 1, cy - 1]):
                    o = {"W": W, "H": H, "min": PAGE_MIN, "aspect": locked(r, aspect)}
                    calls.append(("limit", ["resizeRect", r, handle, target, o], W, H))
        for deg in (45, 44.9, -45, -44.9, 90, 12.34):
            calls.append(("limit", ["rotateRect", r, deg, lim], W, H))
        for kind_aspect in (A600, AGO):
            calls.append(("limit", ["reshapeRect", r, kind_aspect, lim], W, H))
    for (W, H) in ((1600, 2000), (4000, 3000)):
        for p0, p1 in (([100, 100], [160, 160]), ([W - 100, H - 100], [W - 150, H - 170]), ([5, 5], [60, 400])):
            for aspect in (0, A600):
                calls.append(("draw", ["drawRect", p0, p1, {"W": W, "H": H, "min": PAGE_MIN, "aspect": aspect}], W, H))
    return calls


def test_rectangles_made_with_the_pages_limits_pass_the_cores_corner_checks_after_rounding():
    cases = limit_cases()
    out = geo(*[c[1] for c in cases])
    quads = iter(geo(*[["rectToQuad", r] for r in out if r is not None]))
    checked = 0
    for (kind, call, W, H), r in zip(cases, out):
        if r is None:
            assert kind == "draw"                                       # a drag still below the smallest side draws nothing
            continue
        try:
            core_check(next(quads), W, H)
        except ValueError as ex:
            raise AssertionError(f"{call[0]} {call[1:]}: {ex}") from None
        checked += 1
    assert checked > 3000


def test_the_cores_own_numbers_as_limits_would_be_refused_after_rounding():
    """The same operations with 50 px and 45 deg give rectangles the core refuses once the corners are rounded."""
    bad = 0
    for angle in (30, 44.9, 45):
        r = {"cx": 800.37, "cy": 900.21, "w": 500, "h": 600, "angle": angle}
        out = geo(["resizeRect", r, "right", [-5000, -5000], {"W": 1600, "H": 2000, "min": 50, "aspect": 0}],
                  ["rotateRect", {**r, "angle": 0}, angle, {"max": 45, "W": 1600, "H": 2000}])
        for rc in out:
            try:
                core_check(geo(("rectToQuad", rc)), 1600, 2000)
            except ValueError:
                bad += 1
    assert bad >= 2


def inside_in(r, W, H):
    return geo(("insideScan", r, W, H))


def test_limit_rect_keeps_a_rectangle_inside_the_scan_the_angle_inside_the_limit_and_the_sides_above_the_smallest():
    o = {"W": 1000, "H": 800, "min": PAGE_MIN, "max": PAGE_MAX}
    out = geo(["limitRect", rect(cx=500, cy=400, w=40, h=900, angle=60), o], ["limitRect", rect(cx=500, cy=400, w=300, h=200, angle=3), o])
    assert out[0]["angle"] == PAGE_MAX and out[0]["w"] >= PAGE_MIN and inside_in(out[0], 1000, 800)
    assert out[1] == rect(cx=500, cy=400, w=300, h=200, angle=3)               # a rectangle that fits is left alone


def test_reshape_gives_the_formats_proportions_with_the_same_centre_angle_orientation_and_about_the_same_area():
    o = {"W": 4000, "H": 3000, "min": PAGE_MIN, "max": PAGE_MAX}
    r = rect(cx=1800, cy=1400, w=900, h=700, angle=12)                          # landscape, turned
    for aspect in (A600, AGO):
        out = geo(("reshapeRect", r, aspect, o))
        assert out["w"] / out["h"] == pytest.approx(1 / aspect) and out["angle"] == 12 and (out["cx"], out["cy"]) == (1800, 1400)
        assert out["w"] * out["h"] == pytest.approx(900 * 700)
    portrait = geo(("reshapeRect", rect(w=800, h=1100), A600, o))
    assert portrait["w"] / portrait["h"] == pytest.approx(A600) and portrait["w"] * portrait["h"] == pytest.approx(800 * 1100)
    # near an edge it slides in, too big for the scan it shrinks with the proportions, never below the smallest side
    edge = geo(("reshapeRect", rect(cx=300, cy=1500, w=700, h=1200), AGO, o))
    assert inside_in(edge, 4000, 3000) and edge["w"] / edge["h"] == pytest.approx(AGO) and edge["cx"] >= edge["w"] / 2 - 1e-6
    small = geo(("reshapeRect", rect(cx=500, cy=400, w=60, h=60), A600, {"W": 1000, "H": 800, "min": PAGE_MIN, "max": PAGE_MAX}))
    assert small["w"] >= PAGE_MIN - 1e-9 and small["h"] >= PAGE_MIN - 1e-9 and small["w"] / small["h"] == pytest.approx(A600)
    tight = geo(("reshapeRect", rect(cx=500, cy=400, w=900, h=700), A600, {"W": 1000, "H": 800, "min": PAGE_MIN, "max": PAGE_MAX}))
    assert inside_in(tight, 1000, 800)


def test_the_own_ppi_field_reads_only_an_empty_field_as_none():
    ok = lambda text, bad=False: geo(("parsePpi", text, bad, 50000))
    assert ok("") == {"value": None, "error": ""} and ok("  ") == {"value": None, "error": ""}
    assert ok("800") == {"value": 800, "error": ""} and ok("3200.5")["value"] == 3200.5
    for text in ("-3", "0", "abc", "1e9", "NaN", "Infinity"):
        r = ok(text)
        assert r["value"] is None and "above 0" in r["error"], text
    r = ok("", True)                                                           # a number input gives "" for text it cannot read
    assert r["value"] is None and "above 0" in r["error"]


# --------------------------------------------------------------------------- what a save sends
Q = [[100, 100], [900, 100], [900, 1100], [100, 1100]]


def body(**kw):
    s = {"storedQuad": Q, "quad": Q, "printDirty": False, "item": {"kind": "auto", "ppi": None},
         "item0": {"kind": "auto", "ppi": None}, "win": None}
    s.update(kw)
    return geo(("saveBody", s))


def test_a_first_crop_sends_its_corners_and_always_names_the_kind():
    out = body(storedQuad=None, item={"kind": "other", "ppi": None}, item0=None)
    assert out == {"corners": Q, "item": "other"}
    out = body(storedQuad=None, item={"kind": "go", "ppi": None}, item0=None)
    assert out == {"corners": Q, "item": "go"}
    out = body(storedQuad=None, item={"kind": "other", "ppi": 800}, item0=None)
    assert out == {"corners": Q, "item": "other", "scan_ppi": 800}


def test_a_save_sends_only_what_changed():
    assert body() == {}
    assert body(printDirty=True) == {"corners": Q}                                    # corners with the unchanged item
    assert body(item={"kind": "go", "ppi": None}) == {"item": "go"}                    # an item alone on a detected scan
    assert body(item={"kind": "auto", "ppi": None}, item0={"kind": "go", "ppi": None}) == {"item": "auto"}
    both = body(printDirty=True, item={"kind": "600", "ppi": None})
    assert both == {"corners": Q, "item": "600"}
    # the own ppi: set, changed, taken away (0), not sent for another kind or when it did not change
    other = {"kind": "other", "ppi": 500}
    assert body(item={"kind": "other", "ppi": None}, item0=other) == {"scan_ppi": 0}
    assert body(item={"kind": "other", "ppi": 700}, item0=other) == {"scan_ppi": 700}
    assert body(item=other, item0=other) == {}
    assert body(item={"kind": "other", "ppi": 500}, item0={"kind": "go", "ppi": None}) == {"item": "other", "scan_ppi": 500}
    assert body(item={"kind": "go", "ppi": None}, item0=other) == {"item": "go"}


def test_the_photo_window_goes_along_except_for_an_item_other():
    win = {"corners": {"tl": [1, 1]}, "kind": "square"}
    assert body(printDirty=True, win=win) == {"corners": Q, "window": win}
    assert body(win=win) == {"window": win}
    assert body(printDirty=True, win=win, item={"kind": "other", "ppi": None}) == {"corners": Q, "item": "other"}
    assert body(win=win, item={"kind": "go", "ppi": None}) == {"item": "go", "window": win}


# --------------------------------------------------------------------------- the page script
def test_the_page_script_has_the_new_controls_and_is_valid_javascript(tmp_path):
    import re
    page = PAGE.read_text(encoding="utf-8")
    for needle in ('id="st-place"', 'id="st-fine"', 'id="item"', 'id="ppi"', 'id="rot"', 'id="pcanvas"', 'id="hint"'):
        assert needle in page
    js = re.search(r"<script>\n(.*?)</script>", page, re.S).group(1)
    f = tmp_path / "review.js"
    f.write_text(js, encoding="utf-8")
    done = subprocess.run([NODE, "--check", str(f)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


# --------------------------------------------------------------------------- what counts as a rectangle (R36)
def page_round(v):
    return math.floor(v * 100 + 0.5) / 100      # the rounding of commit() and of the stored record


def test_a_rectangle_stays_a_rectangle_after_the_rounding_of_a_save_and_two_save_load_rounds():
    calls = []
    for W in (1600, 4000):
        for angle in (0, 0.3, 5, 17.77, 30, 44.9, -44.9, -12.345):
            for (w, h) in ((50.5, 50.5), (300, 900), (900, 300), (1234.567, 2345.678), (3100, 2800)):
                for (cx, cy) in ((W / 2 + 0.123, 1500.456), (w / 2 + 3.3, h / 2 + 2.2), (W - 77.777, 999.999)):
                    calls.append(["rectToQuad", {"cx": cx, "cy": cy, "w": w, "h": h, "angle": angle}])
    quads = geo(*calls)
    saved = [[[page_round(v) for v in p] for p in q] for q in quads]                   # saved: rounded corners
    results = geo(*[["isRectQuad", q] for q in saved])
    assert all(results), [i for i, ok in enumerate(results) if not ok][:5]
    # loaded, edited with no change, saved again, twice
    for _ in range(2):
        back = geo(*[["rectFromQuad", q] for q in saved])
        saved = [[[page_round(v) for v in p] for p in q] for q in geo(*[["rectToQuad", r] for r in back])]
        assert all(geo(*[["isRectQuad", q] for q in saved]))
    assert len(calls) == 240


def test_one_corner_moved_by_a_pixel_or_less_is_no_rectangle_but_a_move_of_the_whole_one_is_still_one():
    base = {"cx": 800.3, "cy": 900.7, "w": 700, "h": 900, "angle": 6.5}
    q = [[page_round(v) for v in p] for p in geo(("rectToQuad", base))]
    calls = []
    for corner in range(4):
        for (dx, dy) in ((1, 0), (0, 1), (-1, 0), (0.5, 0), (0, -0.5), (0.3, 0), (0, 0.3), (0.3, 0.3), (0.2, 0)):      # 0.1 px along a side is inside the tolerance (the fit absorbs most of it)
            moved = [p[:] for p in q]
            moved[corner] = [moved[corner][0] + dx, moved[corner][1] + dy]
            calls.append(["isRectQuad", moved])
    assert not any(geo(*calls))
    shifted = [[p[0] + 12.34, p[1] - 5.6] for p in q]
    assert geo(("isRectQuad", shifted)) is True
