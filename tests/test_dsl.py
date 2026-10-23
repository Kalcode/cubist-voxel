import itertools

import numpy as np
import pytest

from cubist.core.palette import Palette
from cubist.dsl import ShapeError, box, capsule, cone, cylinder, ellipsoid, sphere

P = Palette({"red": "#ff0000", "blue": "#0000ff", "green": "#00ff00"})
RED, BLUE, GREEN = P.red, P.blue, P.green


def brute(pred, lo, hi):
    """Voxel coords in [lo, hi) whose centers satisfy pred."""
    out = []
    for x, y, z in itertools.product(*(range(a, b) for a, b in zip(lo, hi, strict=True))):
        if pred(np.array([x + 0.5, y + 0.5, z + 0.5])):
            out.append((x, y, z))
    return sorted(out)


def coords(grid):
    return sorted(map(tuple, grid.voxels()[0].tolist()))


# --- primitives -------------------------------------------------------------


def test_box_fills_exactly_its_cells_from_min_corner():
    g = box((2, 3, 4), color=RED).at(1, 1, 1).voxelize()
    assert g.count() == 24
    assert g.bounds() == ((1, 1, 1), (3, 4, 5))
    assert g.colors_used() == {RED.index: 24}


def test_sphere_matches_center_distance_rule():
    g = sphere(3, color=RED).at(8, 8, 8).voxelize()
    expected = brute(lambda c: np.linalg.norm(c - 8) <= 3, (0, 0, 0), (16, 16, 16))
    assert coords(g) == expected


def test_ellipsoid_matches_implicit_equation():
    radii = np.array([4.0, 2.0, 3.0])
    g = ellipsoid((4, 2, 3), color=RED).at(8, 8, 8).voxelize()
    expected = brute(lambda c: np.sum(((c - 8) / radii) ** 2) <= 1, (0, 0, 0), (16, 16, 16))
    assert coords(g) == expected


def test_cylinder_is_a_stack_of_identical_disks_from_its_base():
    g = cylinder(r=2.5, h=3, color=RED).at(5, 0, 5).voxelize()
    lo, hi = g.bounds()
    assert (lo[1], hi[1]) == (0, 3)
    disk = brute(lambda c: np.hypot(c[0] - 5, c[2] - 5) <= 2.5, (0, 0, 0), (10, 1, 10))
    for y in range(3):
        layer = sorted((x, z) for x, yy, z in coords(g) if yy == y)
        assert layer == [(x, z) for x, _, z in disk]


def test_cylinder_along_x_axis():
    g = cylinder(r=1, h=5, axis="x", color=RED).at(0, 4, 4).voxelize()
    lo, hi = g.bounds()
    assert (lo[0], hi[0]) == (0, 5)


def test_cone_narrows_toward_its_apex():
    g = cone(r=4, h=8, color=RED).at(8.5, 0, 8.5).voxelize()
    per_layer = [sum(1 for _, y, _ in coords(g) if y == yy) for yy in range(8)]
    assert per_layer == sorted(per_layer, reverse=True)
    assert per_layer[0] > 4 * per_layer[-1]
    lo, hi = g.bounds()
    assert (lo[1], hi[1]) == (0, 8)


def test_capsule_matches_segment_distance_rule():
    a, b = np.array([5.0, 2.0, 5.0]), np.array([5.0, 9.0, 5.0])

    def inside(c):
        t = np.clip(np.dot(c - a, b - a) / np.dot(b - a, b - a), 0, 1)
        return np.linalg.norm(c - (a + t * (b - a))) <= 1.5

    g = capsule(tuple(a), tuple(b), r=1.5, color=RED).voxelize()
    assert coords(g) == brute(inside, (0, 0, 0), (12, 12, 12))


def test_shape_without_color_cannot_be_voxelized():
    with pytest.raises(ShapeError, match="color"):
        box((2, 2, 2)).voxelize()


def test_oversized_shape_is_rejected_before_sampling():
    with pytest.raises(ShapeError, match="too large"):
        box((5000, 1, 1), color=RED).voxelize()


# --- CSG --------------------------------------------------------------------


def test_union_later_shape_wins_color_where_inside():
    g = box((4, 4, 4), color=RED).union(box((2, 2, 2), color=BLUE).at(1, 1, 1)).voxelize()
    assert g.count() == 64
    assert g.colors_used() == {RED.index: 56, BLUE.index: 8}
    assert g.get(1, 1, 1) == BLUE.index


def test_union_accepts_many_shapes_and_operator():
    a = box((1, 1, 1), color=RED)
    b = box((1, 1, 1), color=BLUE).at(2, 0, 0)
    c = box((1, 1, 1), color=GREEN).at(4, 0, 0)
    assert a.union(b, c).voxelize() == (a | b | c).voxelize()
    assert (a | b | c).voxelize().count() == 3


def test_subtract_keeps_left_color():
    g = (box((4, 4, 4), color=RED) - box((2, 2, 2), color=BLUE).at(1, 1, 1)).voxelize()
    assert g.colors_used() == {RED.index: 56}
    assert g.get(1, 1, 1) == 0


def test_intersect_keeps_left_color():
    g = (box((4, 4, 4), color=RED) & box((4, 4, 4), color=BLUE).at(2, 2, 2)).voxelize()
    assert g.colors_used() == {RED.index: 8}
    assert g.bounds() == ((2, 2, 2), (4, 4, 4))


def test_smooth_union_adds_a_fillet_between_shapes():
    a = sphere(3, color=RED).at(6, 8, 8)
    b = sphere(3, color=BLUE).at(13, 8, 8)
    hard = a.union(b).voxelize()
    smooth = a.smooth_union(b, k=3).voxelize()
    assert smooth.count() > hard.count()
    assert set(coords(hard)) <= set(coords(smooth))
    assert set(smooth.colors_used()) == {RED.index, BLUE.index}


# --- transforms -------------------------------------------------------------


def test_at_translates():
    a = box((2, 2, 2), color=RED).voxelize()
    b = box((2, 2, 2), color=RED).at(3, -1, 7).voxelize()
    assert b.bounds() == ((3, -1, 7), (5, 1, 9))
    assert b.count() == a.count()


def test_rotate_90_about_y_swaps_x_and_z_extents():
    g = box((4, 1, 2), color=RED).rotate(90, axis="y").voxelize()
    lo, hi = g.bounds()
    size = tuple(b - a for a, b in zip(lo, hi, strict=True))
    assert size == (2, 1, 4)
    assert g.count() == 8


def test_rotate_45_resamples_without_losing_much_volume():
    g = box((8, 2, 8), color=RED).at(4, 0, 4).rotate(45, axis="y").voxelize()
    assert abs(g.count() - 128) <= 16


def test_mirror_keeps_original_and_adds_reflection():
    g = box((1, 1, 1), color=RED).at(6, 18, 8).mirror("x", about=8).voxelize()
    assert coords(g) == [(6, 18, 8), (9, 18, 8)]


def test_reflect_replaces_with_reflection():
    g = box((1, 1, 1), color=RED).at(6, 18, 8).reflect("x", about=8).voxelize()
    assert coords(g) == [(9, 18, 8)]


def test_mirror_of_off_center_shape_is_symmetric():
    shape = sphere(2.2, color=RED).at(4.3, 5, 5).union(box((2, 3, 1), color=BLUE).at(2, 1, 4))
    g = shape.mirror("x", about=8).voxelize()
    voxels = dict(zip(map(tuple, g.voxels()[0].tolist()), g.voxels()[1].tolist(), strict=True))
    for (x, y, z), c in voxels.items():
        assert voxels.get((15 - x, y, z)) == c


def test_mirror_of_shape_straddling_the_plane_has_symmetric_colors():
    shape = box((4, 2, 2), color=RED).at(5, 0, 0).union(box((1, 2, 2), color=BLUE).at(7, 0, 0))
    g = shape.mirror("x", about=8).voxelize()
    voxels = dict(zip(map(tuple, g.voxels()[0].tolist()), g.voxels()[1].tolist(), strict=True))
    assert len(voxels) == 6 * 2 * 2
    for (x, y, z), c in voxels.items():
        assert voxels[(15 - x, y, z)] == c


def test_scale_uniform_about_min_corner():
    g = box((1, 2, 1), color=RED).at(3, 3, 3).scale(2, about=(3, 3, 3)).voxelize()
    assert g.bounds() == ((3, 3, 3), (5, 7, 5))


def test_voxelize_is_deterministic():
    shape = sphere(5, color=RED).at(7.3, 6.1, 8.9).rotate(33, axis="z")
    a, b = shape.voxelize(), shape.voxelize()
    assert a == b and np.array_equal(a.data, b.data)
