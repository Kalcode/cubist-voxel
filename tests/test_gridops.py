"""Grid-stage shapes: painted shapes keep working with CSG and transforms, plus morphology."""

import numpy as np
import pytest

from cubist.core.palette import Palette
from cubist.dsl import ShapeError, box, sphere

P = Palette({"wood": ["#100000", "#200000", "#300000"], "white": "#ffffff", "red": "#ff0000"})
W, R = P.white, P.red


def painted(size=(4, 4, 4)):
    """A box that has gone through the grid stage."""
    return box(size, color=W).paint(R)


def coords(grid):
    return sorted(map(tuple, grid.voxels()[0].tolist()))


def test_painted_shape_is_not_a_distance_field():
    assert box((2, 2, 2), color=W).is_sdf
    assert not painted().is_sdf


def test_union_with_painted_shape_later_wins():
    g = painted().union(box((2, 2, 2), color=W).at(1, 1, 1)).voxelize()
    assert g.colors_used() == {R.index: 56, W.index: 8}
    g2 = box((2, 2, 2), color=W).at(1, 1, 1).union(painted()).voxelize()
    assert g2.colors_used() == {R.index: 64}


def test_subtract_and_intersect_on_the_grid():
    hole = box((2, 2, 2), color=W).at(1, 1, 1)
    assert (painted() - hole).voxelize().count() == 56
    assert (painted() & hole).voxelize().colors_used() == {R.index: 8}
    assert (box((4, 4, 4), color=W) - painted((2, 2, 2))).voxelize().count() == 56


def test_smooth_union_rejects_painted_shapes():
    with pytest.raises(ShapeError, match="before painting"):
        painted().smooth_union(box((1, 1, 1), color=W), k=2)


def test_taper_and_displace_reject_painted_shapes():
    with pytest.raises(ShapeError, match="before painting"):
        painted().taper(top=0.5)
    with pytest.raises(ShapeError, match="before painting"):
        painted().displace(amp=1)


def test_integer_translation_of_painted_shape():
    g = painted().at(3, -2, 1).voxelize()
    assert g.bounds() == ((3, -2, 1), (7, 2, 5))
    assert g.count() == 64


def test_rotate_painted_shape_90_degrees_is_exact():
    g = box((4, 1, 2), color=W).paint(R).rotate(90, axis="y").voxelize()
    lo, hi = g.bounds()
    assert tuple(b - a for a, b in zip(lo, hi, strict=True)) == (2, 1, 4)
    assert g.count() == 8


def test_scale_painted_shape():
    g = box((1, 2, 1), color=W).paint(R).scale(2, about=(0, 0, 0)).voxelize()
    assert g.count() == 16
    assert g.bounds() == ((0, 0, 0), (2, 4, 2))


def test_mirror_painted_shape_is_symmetric():
    shape = box((2, 3, 2), color=W).at(5, 0, 0).union(box((1, 1, 2), color=R).at(7, 1, 0))
    g = shape.paint(R, where=box((1, 1, 2)).at(5, 2, 0)).mirror("x", about=8).voxelize()
    voxels = dict(zip(map(tuple, g.voxels()[0].tolist()), g.voxels()[1].tolist(), strict=True))
    assert voxels
    for (x, y, z), c in voxels.items():
        assert voxels[(15 - x, y, z)] == c


def test_mirror_painted_shape_needs_a_grid_aligned_plane():
    with pytest.raises(ShapeError, match="plane"):
        painted().mirror("x", about=2.3)


# --- morphology -------------------------------------------------------------


def test_hollow_keeps_a_shell_of_given_thickness():
    assert box((6, 6, 6), color=W).hollow(1).voxelize().count() == 216 - 64
    assert box((6, 6, 6), color=W).hollow(2).voxelize().count() == 216 - 8


def test_erode_and_dilate():
    assert box((6, 6, 6), color=W).erode(1).voxelize().count() == 64
    grown = box((2, 2, 2), color=W).at(3, 3, 3).dilate(1).voxelize()
    assert grown.count() == 8 + 6 * 4
    assert grown.colors_used() == {W.index: 32}


def test_dilate_with_a_new_color():
    grown = box((2, 2, 2), color=W).dilate(1, color=R).voxelize()
    assert grown.colors_used() == {W.index: 8, R.index: 24}


def test_round_on_a_composite_shape_opens_it():
    spiky = box((6, 6, 6), color=W) | box((1, 4, 1), color=W).at(3, 6, 3)
    g = spiky.round(1).voxelize()
    original = set(coords(spiky.voxelize()))
    assert set(coords(g)) <= original
    assert not any(g.get(3, y, 3) for y in (7, 8, 9))  # the thin spike is gone
    assert g.get(0, 0, 0) == 0 and g.get(5, 5, 0) == 0  # box corners and edges rounded
    assert g.get(3, 3, 0) and g.get(3, 5, 3)  # faces stay
    assert not spiky.round(1).is_sdf


def test_chamfer_cuts_convex_edges():
    assert box((6, 6, 6), color=W).chamfer(1).voxelize().count() == 216 - 56
    assert box((6, 6, 6), color=W).chamfer(2).voxelize().count() < 216 - 56


def test_morphology_keeps_colors():
    two_tone = box((6, 6, 6), color=W) | box((6, 3, 6), color=R)
    g = two_tone.hollow(1).voxelize()
    assert set(g.colors_used()) == {W.index, R.index}


def test_grid_stage_result_is_cached_and_deterministic():
    shape = sphere(5, color=W).at(6, 6, 6).hollow(1)
    a, b = shape.voxelize(), shape.voxelize()
    assert a == b and np.array_equal(a.data, b.data)


def test_hollow_shell_of_a_curved_shape_is_face_connected():
    from scipy import ndimage

    g = sphere(7.3, color=W).at(8, 8, 8).hollow(1).voxelize()
    _, pieces = ndimage.label(g.data > 0)  # 6-connectivity
    assert pieces == 1
