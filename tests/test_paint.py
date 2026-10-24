"""Paint ops and selections."""

import pytest

from cubist.core.palette import Palette
from cubist.dsl import ShapeError, band, box, building, colored, faces, region, sphere

P = Palette(
    {
        "wood": ["#100000", "#200000", "#300000", "#400000"],
        "grass": "#00ff00",
        "white": "#ffffff",
    }
)
WD = P.wood


def colors_by_layer(grid, axis=1):
    out = {}
    for (x, y, z), c in zip(
        map(tuple, grid.voxels()[0].tolist()), grid.voxels()[1].tolist(), strict=True
    ):
        out.setdefault((x, y, z)[axis], set()).add(c)
    return out


def test_paint_everything():
    g = box((2, 2, 2), color=WD.dark).paint(P.white).voxelize()
    assert g.colors_used() == {P.white.index: 8}


def test_paint_a_band():
    g = box((3, 4, 3), color=WD.mid).paint(WD.dark, where=band(axis="y", at=1)).voxelize()
    layers = colors_by_layer(g)
    assert layers[1] == {WD.dark.index}
    assert layers[0] == layers[2] == layers[3] == {WD.mid.index}


def test_paint_where_a_shape_is():
    ball = sphere(1.5, color=P.white).at(0, 0, 0)
    g = box((4, 4, 4), color=WD.mid).paint(P.white, where=ball).voxelize()
    assert g.get(0, 0, 0) == P.white.index
    assert g.get(3, 3, 3) == WD.mid.index


def test_paint_faces_top_with_depth():
    top = box((3, 3, 3), color=WD.dark).paint_faces("top", P.grass).voxelize()
    assert top.colors_used()[P.grass.index] == 9
    deep = box((3, 3, 3), color=WD.dark).paint_faces("top", P.grass, depth=2).voxelize()
    assert deep.colors_used()[P.grass.index] == 18


def test_paint_faces_front_is_max_z():
    g = box((2, 2, 3), color=WD.dark).paint_faces("front", P.grass).voxelize()
    assert g.get(0, 0, 2) == P.grass.index and g.get(0, 0, 1) == WD.dark.index


def test_unknown_face_direction_is_an_error():
    with pytest.raises(ShapeError, match="sideways"):
        box((2, 2, 2), color=WD.dark).paint_faces("sideways", P.grass)


def test_gradient_runs_dark_to_light_along_an_axis():
    g = box((1, 4, 1), color=WD.dark).gradient("y", WD).voxelize()
    assert [g.get(0, y, 0) for y in range(4)] == [c.index for c in WD.ramp]
    r = box((1, 4, 1), color=WD.dark).gradient("y", WD, reverse=True).voxelize()
    assert [r.get(0, y, 0) for y in range(4)] == [c.index for c in reversed(WD.ramp)]


def test_dither_is_seeded_and_stays_on_the_ramp():
    base = box((8, 8, 8), color=WD[1])
    a = base.dither(WD, seed=3).voxelize()
    b = base.dither(WD, seed=3).voxelize()
    c = base.dither(WD, seed=4).voxelize()
    assert a == b and a != c
    assert set(a.colors_used()) <= {WD[0].index, WD[1].index, WD[2].index}
    assert len(a.colors_used()) == 3
    assert base.dither(WD, seed=3, amount=0).voxelize() == base.voxelize()


def test_dither_leaves_colors_outside_the_ramp():
    g = box((4, 4, 4), color=P.white).dither(WD, seed=1).voxelize()
    assert g.colors_used() == {P.white.index: 64}


def test_dither_follows_the_build_seed():
    base = box((8, 8, 8), color=WD[1]).dither(WD, seed=3)
    with building(P, seed=0):
        a = base.voxelize()
    with building(P, seed=9):
        b = box((8, 8, 8), color=WD[1]).dither(WD, seed=3).voxelize()
    assert a != b


def test_darken_crevices_darkens_inside_corners_only():
    floor_and_wall = box((6, 1, 4), color=WD[2]) | box((1, 5, 4), color=WD[2])
    with building(P):
        g = floor_and_wall.darken_crevices().voxelize()
    assert g.get(1, 0, 1) == WD[1].index  # floor next to the wall
    assert g.get(4, 0, 1) == WD[2].index  # open floor


def test_highlight_edges_lightens_convex_edges():
    with building(P):
        g = box((4, 4, 4), color=WD[1]).highlight_edges().voxelize()
    assert g.get(0, 0, 0) == WD[2].index
    assert g.get(1, 1, 0) == WD[1].index  # face, not edge


def test_ramp_ops_need_a_palette():
    with pytest.raises(ShapeError, match="palette"):
        box((4, 4, 4), color=WD[1]).darken_crevices().voxelize()


def test_selections_combine():
    sel = faces("top") & ~band(axis="x", at=0)
    g = box((3, 2, 3), color=WD.dark).paint(P.grass, where=sel).voxelize()
    assert g.colors_used()[P.grass.index] == 6
    only_dark = box((2, 2, 2), color=WD.dark) | box((2, 2, 2), color=P.white).at(2, 0, 0)
    g2 = only_dark.paint(P.grass, where=colored(WD.dark)).voxelize()
    assert g2.colors_used() == {P.grass.index: 8, P.white.index: 8}
    g3 = box((4, 1, 1), color=WD.dark).paint(P.grass, where=region((1, 0, 0), (3, 1, 1))).voxelize()
    assert g3.colors_used() == {P.grass.index: 2, WD.dark.index: 2}


def test_ops_used_are_recorded_while_building():
    with building(P) as ctx:
        box((2, 2, 2), color=WD[1]).dither(WD, seed=1).voxelize()
    assert "dither" in ctx.ops_used


def test_darken_crevices_ignores_bevel_steps():
    with building(P):
        rounded = box((6, 6, 6), color=WD[2]).round(1)
        g = rounded.darken_crevices().voxelize()
        assert g == rounded.voxelize()
