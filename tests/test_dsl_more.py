"""Milestone 3 primitives and SDF-level operations."""

import itertools

import numpy as np
import pytest
from scipy import ndimage

from cubist.core.palette import Palette
from cubist.dsl import (
    ShapeError,
    box,
    extrude,
    heightmap,
    lathe,
    line,
    sphere,
    torus,
    wedge,
)

P = Palette({"wood": ["#100000", "#200000", "#300000", "#400000"], "white": "#ffffff"})
W = P.white


def brute(pred, lo, hi):
    out = []
    for x, y, z in itertools.product(*(range(a, b) for a, b in zip(lo, hi, strict=True))):
        if pred(np.array([x + 0.5, y + 0.5, z + 0.5])):
            out.append((x, y, z))
    return sorted(out)


def coords(grid):
    return sorted(map(tuple, grid.voxels()[0].tolist()))


# --- primitives -------------------------------------------------------------


def test_torus_matches_its_implicit_equation_and_has_a_hole():
    def inside(c):
        q = np.hypot(np.hypot(c[0] - 8, c[2] - 8) - 5, c[1] - 2)
        return q <= 1.5

    g = torus(R=5, r=1.5, color=W).at(8, 2, 8).voxelize()
    assert coords(g) == brute(inside, (0, -2, 0), (16, 6, 16))
    assert g.get(8, 2, 8) == 0
    lo, hi = g.bounds()
    assert (lo[1], hi[1]) == (1, 3)


@pytest.mark.parametrize(
    "rise, column_heights",
    [("+x", [1, 2, 3, 4]), ("-x", [4, 3, 2, 1])],
)
def test_wedge_rises_along_its_axis(rise, column_heights):
    g = wedge((4, 4, 2), rise=rise, color=W).voxelize()
    heights = [sum(1 for x, y, z in coords(g) if x == col and z == 0) for col in range(4)]
    assert heights == column_heights
    assert g.count() == 2 * sum(column_heights)


def test_wedge_rising_along_z():
    g = wedge((2, 3, 3), rise="+z", color=W).voxelize()
    heights = [sum(1 for x, y, z in coords(g) if z == row and x == 0) for row in range(3)]
    assert heights == [1, 2, 3]


def test_wedge_bad_rise_is_an_error():
    with pytest.raises(ShapeError, match="rise"):
        wedge((2, 2, 2), rise="up", color=W)


def test_line_is_a_connected_one_voxel_path():
    g = line((0.5, 0.5, 0.5), (9.5, 4.5, 0.5), color=W).voxelize()
    assert g.get(0, 0, 0) and g.get(9, 4, 0)
    _, components = ndimage.label(g.data > 0, structure=np.ones((3, 3, 3)))
    assert components == 1
    assert 10 <= g.count() <= 14
    lo, hi = g.bounds()
    assert (lo[2], hi[2]) == (0, 1)


def test_lathe_revolves_a_piecewise_linear_profile():
    profile = [(0, 3), (4, 3), (6, 1)]

    def radius(h):
        return 3.0 if h <= 4 else 3 - (h - 4)

    def inside(c):
        h = c[1]
        return 0 <= h <= 6 and np.hypot(c[0] - 5, c[2] - 5) <= radius(h)

    g = lathe(profile, color=W).at(5, 0, 5).voxelize()
    assert coords(g) == brute(inside, (0, 0, 0), (10, 8, 10))


def test_lathe_needs_two_points_in_height_order():
    with pytest.raises(ShapeError, match="profile"):
        lathe([(0, 2)], color=W)
    with pytest.raises(ShapeError, match="profile"):
        lathe([(4, 2), (0, 2)], color=W)


def test_extrude_triangle():
    g = extrude([(0, 0), (6, 0), (0, 6)], depth=2, color=W).voxelize()
    assert g.count() == 2 * 21
    assert g.get(0, 5, 1) and g.get(4, 1, 0)
    assert not g.get(5, 1, 0) and not g.get(3, 3, 0)


def test_extrude_along_y_uses_x_and_z():
    g = extrude([(0, 0), (4, 0), (4, 2), (0, 2)], depth=3, axis="y", color=W).voxelize()
    assert g.bounds() == ((0, 0, 0), (4, 3, 2))


def test_heightmap_fills_columns_from_the_ground():
    g = heightmap([[0, 1], [2, 3]], color=W).voxelize()
    assert g.count() == 6
    assert g.get(1, 2, 1) and not g.get(1, 3, 1) and not g.get(0, 0, 0)


# --- SDF operations ---------------------------------------------------------


def test_array_repeats_along_a_step():
    g = box((1, 1, 1), color=W).array(3, step=(2, 0, 0)).voxelize()
    assert coords(g) == [(0, 0, 0), (2, 0, 0), (4, 0, 0)]


def test_radial_copies_around_an_axis():
    g = box((1, 1, 1), color=W).at(5, 0, 0).radial(4, about=(0.5, 0, 0.5)).voxelize()
    assert g.count() == 4
    assert g.bounds() == ((-5, 0, -5), (6, 1, 6))


def test_taper_narrows_toward_the_top():
    g = box((8, 8, 8), color=W).taper(top=0.5).voxelize()
    layer = [sum(1 for _, y, _ in coords(g) if y == yy) for yy in range(8)]
    assert layer[0] == 64
    assert layer[-1] < layer[0] / 2
    assert layer == sorted(layer, reverse=True)


def test_displace_is_seeded_and_deterministic():
    base = sphere(6, color=W).at(8, 8, 8)
    a = base.displace(amp=1.5, scale=3, seed=1).voxelize()
    b = base.displace(amp=1.5, scale=3, seed=1).voxelize()
    c = base.displace(amp=1.5, scale=3, seed=2).voxelize()
    assert a == b
    assert a != c
    assert base.displace(amp=0, seed=1).voxelize() == base.voxelize()


def test_round_takes_voxels_off_edges_without_growing():
    one = box((6, 6, 6), color=W).round(1).voxelize()
    assert one.count() == 216 - 56
    assert one.bounds() == ((0, 0, 0), (6, 6, 6))
    assert one.get(0, 0, 0) == 0 and one.get(3, 3, 0)
    two = box((6, 6, 6), color=W).round(2).voxelize()
    assert two.count() < one.count()
    moved = box((6, 6, 6), color=W).at(1, 1, 1).round(1).voxelize()
    assert moved.count() == one.count()
