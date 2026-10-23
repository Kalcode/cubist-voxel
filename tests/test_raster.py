import numpy as np
import pytest

from cubist.core.grid import VoxelGrid
from cubist.render.raster import iso, ortho

PAL = np.zeros((256, 4), dtype=np.uint8)
PAL[1] = (200, 100, 50, 255)
PAL[2] = (50, 100, 200, 255)


def grid(points):
    coords = np.array(list(points.keys()))
    colors = np.array(list(points.values()), dtype=np.uint8)
    return VoxelGrid.from_voxels(coords, colors)


def cells(view, scale):
    """Set of (col, row) voxel cells that contain object pixels."""
    rows, cols = np.nonzero(view.mask)
    return {(c // scale, r // scale) for r, c in zip(rows, cols, strict=True)}


REGION = ((0, 0, 0), (4, 3, 3))


@pytest.mark.parametrize(
    "view, expected",
    [
        ("front", (2, 1)),  # u = x, v = y from the top
        ("back", (1, 1)),  # x mirrored
        ("right", (2, 1)),  # u = z descending (front on the left)
        ("left", (0, 1)),  # u = z ascending
        ("top", (2, 0)),  # u = x, v = z from the back
        ("bottom", (2, 2)),  # v = z from the front
    ],
)
def test_ortho_view_orientation(view, expected):
    g = grid({(2, 1, 0): 1})
    v = ortho(g, PAL, view, scale=2, region=REGION)
    assert cells(v, 2) == {expected}


def test_ortho_view_size_follows_region_and_scale():
    v = ortho(grid({(0, 0, 0): 1}), PAL, "front", scale=3, region=REGION)
    assert v.rgb.shape == (3 * 3, 4 * 3, 3)
    assert ortho(grid({(0, 0, 0): 1}), PAL, "right", scale=3, region=REGION).rgb.shape == (9, 9, 3)
    assert ortho(grid({(0, 0, 0): 1}), PAL, "top", scale=3, region=REGION).rgb.shape == (9, 12, 3)


def test_ortho_nearest_voxel_wins():
    g = grid({(0, 0, 0): 1, (0, 0, 2): 2})
    front = ortho(g, PAL, "front", scale=4)
    back = ortho(g, PAL, "back", scale=4)
    center = (2, 2)
    assert front.rgb[center][2] > front.rgb[center][0]  # blue voxel at z=2 is in front
    assert back.rgb[center][0] > back.rgb[center][2]  # red voxel at z=0 faces the back


def _cell_mean(view, col, row, scale):
    # Interior pixels only, so outlines don't count.
    block = view.rgb[
        row * scale + 1 : (row + 1) * scale - 1, col * scale + 1 : (col + 1) * scale - 1
    ]
    return block.astype(float).mean()


def test_ortho_ambient_occlusion_darkens_recessed_faces():
    # A sits at z=0 between two walls at z=1; B is an open voxel at the same depth.
    g = grid({(1, 0, 0): 1, (0, 0, 1): 2, (2, 0, 1): 2, (6, 0, 0): 1})
    v = ortho(g, PAL, "front", scale=8, region=((0, 0, 0), (7, 1, 2)))
    assert _cell_mean(v, 1, 0, 8) < _cell_mean(v, 6, 0, 8) - 5


def test_ortho_depth_cue_darkens_farther_faces():
    g = grid({(0, 0, 3): 1, (4, 0, 0): 1})
    v = ortho(g, PAL, "front", scale=8)
    assert _cell_mean(v, 4, 0, 8) < _cell_mean(v, 0, 0, 8) - 5


def test_ortho_background_is_not_masked():
    v = ortho(grid({(0, 0, 0): 1}), PAL, "front", scale=2, region=REGION)
    assert v.mask.sum() == 4


def test_ortho_ticks_are_world_coordinates():
    g = grid({(5, 7, 2): 1})
    region = ((4, 6, 0), (8, 9, 3))
    front = ortho(g, PAL, "front", scale=2, region=region)
    assert front.u_ticks[0] == (0, 4) and front.u_ticks[-1] == (8, 8)
    assert front.v_ticks[0] == (0, 9) and front.v_ticks[-1] == (6, 6)
    back = ortho(g, PAL, "back", scale=2, region=region)
    assert back.u_ticks[0] == (0, 8) and back.u_ticks[-1] == (8, 4)


def iso_area(scale):
    a, b = scale, scale // 2
    c = round(1.2247 * scale)
    return a, b, c


@pytest.mark.parametrize("scale", [2, 4, 6, 12])
def test_iso_single_voxel_covers_its_exact_hexagon(scale):
    a, b, c = iso_area(scale)
    v = iso(grid({(0, 0, 0): 1}), PAL, scale=scale)
    # top rhombus 2ab plus two side faces a*c each
    assert v.mask.sum() == 2 * a * b + 2 * a * c


@pytest.mark.parametrize("scale", [2, 4, 6])
def test_iso_faces_tile_without_gaps_or_overlaps(scale):
    a, b, c = iso_area(scale)
    cube = grid({(x, y, z): 1 for x in range(3) for y in range(3) for z in range(3)})
    v = iso(cube, PAL, scale=scale)
    # A 3x3x3 block projects to the hexagon of a cube with side 3.
    assert v.mask.sum() == 9 * (2 * a * b + 2 * a * c)


def test_iso_top_is_brighter_than_front_brighter_than_right():
    a, b, c = iso_area(12)
    v = iso(grid({(0, 0, 0): 1}), PAL, scale=12)
    # Face centers projected: top (0.5, 1, 0.5), front (0.5, 0.5, 1), right (1, 0.5, 0.5).
    side_row = int(1.5 * b + 0.5 * c)
    top = v.rgb[b, a].astype(int).sum()
    front = v.rgb[side_row, a // 2].astype(int).sum()
    right = v.rgb[side_row, a + a // 2].astype(int).sum()
    assert top > front > right


def test_iso_nearer_voxel_occludes_farther():
    near_only = iso(grid({(1, 1, 1): 2}), PAL, scale=8, region=((0, 0, 0), (2, 2, 2)))
    both = iso(grid({(0, 0, 0): 1, (1, 1, 1): 2}), PAL, scale=8, region=((0, 0, 0), (2, 2, 2)))
    inner = near_only.mask.copy()
    for axis in (0, 1):
        inner &= np.roll(near_only.mask, 1, axis) & np.roll(near_only.mask, -1, axis)
    assert np.array_equal(both.rgb[inner], near_only.rgb[inner])


def test_render_is_deterministic():
    g = grid({(x, y, (x * y) % 3): 1 + (x + y) % 2 for x in range(6) for y in range(5)})
    a, b = iso(g, PAL, scale=4), iso(g, PAL, scale=4)
    assert np.array_equal(a.rgb, b.rgb)
