"""The posed renderer draws moving parts as rotated blocks, matching the static look at rest."""

import runpy
import time
from pathlib import Path

import numpy as np
import pytest

from cubist import Model, Part, box, hinge, sphere
from cubist.build.runner import load_palette
from cubist.core.grid import VoxelGrid
from cubist.core.model import AssetDef
from cubist.core.palette import Palette
from cubist.motion.pose import world_matrices
from cubist.render import raster
from cubist.render.contact_sheet import asset_region, choose_scale
from cubist.render.posed import PosedScene

DEMO = Path(__file__).parent.parent / "projects" / "demo"
P = Palette({"wood": ["#3b2414", "#5e3a1f", "#8a5a32"]})


def build(fn, size=None):
    return AssetDef(fn, "t", size, ()).build(P)


def demo_assets():
    palette = load_palette(DEMO / "style.toml")
    out = []
    for script in sorted((DEMO / "assets").glob("*.py")):
        [a] = [v for v in runpy.run_path(str(script)).values() if isinstance(v, AssetDef)]
        out.append(a.build(palette))
    return out, palette.rgba_table()


@pytest.fixture(scope="module")
def demo():
    return demo_assets()


@pytest.mark.parametrize("view", ["iso", "front", "right", "top"])
def test_rest_pose_matches_static_renderer(view, demo):
    assets, table = demo
    for a in assets:
        grid = a.visible_grid()  # what contact sheets show: parts hidden at rest left out
        region = asset_region(grid, a.size)
        scale = choose_scale([h - lo for lo, h in zip(*region, strict=True)])
        if view == "iso":
            want = raster.iso(grid, table, scale, region)
        else:
            want = raster.ortho(grid, table, view, scale, region)
        hidden = frozenset(a.hidden_parts())
        got = PosedScene(a, table).render({}, view, scale, region, hidden)
        assert got.rgb.shape == want.rgb.shape, a.name
        agree = np.mean(got.mask == want.mask)
        assert agree >= 0.99, (a.name, agree)
        both = got.mask & want.mask
        diff = np.abs(got.rgb[both].astype(int) - want.rgb[both].astype(int)).mean()
        assert diff <= 3, (a.name, diff)
        if view != "iso":
            assert got.u_ticks == want.u_ticks and got.v_ticks == want.v_ticks


def test_single_voxel_matches_exactly():
    a = build(lambda p: box((1, 1, 1), color=p.wood.mid))
    table = P.rgba_table()
    region = ((0, 0, 0), (1, 1, 1))
    for scale in (2, 4, 8):
        want = raster.iso(a.merged_grid(), table, scale, region)
        got = PosedScene(a, table).render({}, "iso", scale, region)
        assert np.array_equal(got.mask, want.mask)
        assert np.abs(got.rgb.astype(int) - want.rgb.astype(int)).max() <= 1


def plate(p):
    shape = box((8, 8, 1), color=p.wood.mid)
    return Model([Part("plate", shape, pivot=(0, 0, 0), joint=hinge("z", (0, 90)))])


def test_rotated_plate_renders_as_rotated_blocks_not_staircase():
    a = build(plate)
    m = np.eye(4)
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    m[:2, :2] = [[c, -s], [s, c]]
    region = ((-5, 0, 0), (8, 12, 1))
    scale = 8
    v = PosedScene(a, P.rgba_table()).render({"plate": m}, "front", scale, region)
    # Rows 0.5..6.4 voxels above the base: the left boundary is the edge from (0,0) to (-4, 6.93).
    h = v.mask.shape[0]
    rows, lefts = [], []
    for py in range(h):
        y = (h - py - 0.5) / scale  # world y of this pixel row's center
        if 0.5 < y < 6.4 and v.mask[py].any():
            rows.append(y)
            lefts.append(np.argmax(v.mask[py]))
    fit = np.polyfit(rows, lefts, 1)
    rms = np.sqrt(np.mean((np.polyval(fit, rows) - lefts) ** 2))
    assert len(rows) > 30 and rms < 0.75


def test_back_faces_are_culled_and_nearest_face_wins():
    # Two boxes, one in front of the other: from the front only the nearer color shows.
    def two(p):
        back = Part("back", box((4, 4, 1), color=p.wood.dark))
        front = Part("front", box((4, 4, 1), color=p.wood.light).at(0, 0, 2))
        return Model([back, front])

    a = build(two)
    table = P.rgba_table()
    v = PosedScene(a, table).render({}, "front", 4, ((0, 0, 0), (4, 4, 3)))
    light = P.wood.light.index
    assert v.mask.all()
    # flat front faces: color = palette * side light (0.86) * depth cue 1.0, AO open
    expected = np.rint(table[light, :3] * 0.86)
    center = v.rgb[4:12, 4:12].reshape(-1, 3)
    assert np.abs(center - expected).max() <= 1


def test_canvas_depends_only_on_region_and_scale():
    a = build(plate)
    scene = PosedScene(a, P.rgba_table())
    region = ((-8, -8, -8), (8, 8, 8))
    m = world_matrices(a, None, 0)
    r0 = scene.render(m, "iso", 4, region)
    turned = {"plate": np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])}
    r1 = scene.render(turned, "iso", 4, region)
    assert r0.rgb.shape == r1.rgb.shape and not np.array_equal(r0.mask, r1.mask)


def test_bounds_cover_the_posed_part():
    a = build(plate)
    scene = PosedScene(a, P.rgba_table())
    quarter = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
    assert scene.bounds({"plate": quarter}) == ((-8, 0, 0), (0, 8, 1))
    assert scene.bounds({}) == ((0, 0, 0), (8, 8, 1))


def test_empty_part_renders_nothing():
    a = build(plate)
    a.parts[0].grid = VoxelGrid(np.zeros((1, 1, 1), dtype=np.uint8))
    v = PosedScene(a, P.rgba_table()).render({}, "iso", 4, ((0, 0, 0), (2, 2, 2)))
    assert not v.mask.any()


def test_render_speed_64_cubed():
    a = build(lambda p: sphere(32, color=p.wood.mid).at(32, 32, 32))
    scene = PosedScene(a, P.rgba_table())
    region = ((0, 0, 0), (64, 64, 64))
    start = time.perf_counter()
    scene.render({}, "iso", 4, region)
    assert time.perf_counter() - start < 0.5
