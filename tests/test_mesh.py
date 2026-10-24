import numpy as np
import pytest

from cubist.core.grid import VoxelGrid
from cubist.io.mesh import greedy_mesh, triangle_count


def grid(points):
    coords = np.array(list(points.keys()))
    colors = np.array(list(points.values()), dtype=np.uint8)
    return VoxelGrid.from_voxels(coords, colors)


def solid(size, color=1, origin=(0, 0, 0)):
    data = np.full(size, color, dtype=np.uint8)
    return VoxelGrid(data, origin)


def exposed_faces(g):
    filled = np.pad(g.data > 0, 1)
    total = 0
    for axis in range(3):
        for shift in (1, -1):
            total += (filled & ~np.roll(filled, shift, axis)).sum()
    return int(total)


def area(quads):
    return sum((q.u1 - q.u0) * (q.v1 - q.v0) for q in quads)


def test_single_voxel_is_six_quads():
    quads = greedy_mesh(grid({(0, 0, 0): 1}))
    assert len(quads) == 6
    assert triangle_count(grid({(0, 0, 0): 1})) == 12


@pytest.mark.parametrize("size", [(4, 4, 4), (2, 1, 1), (5, 3, 7)])
def test_solid_box_of_one_color_is_six_quads(size):
    assert len(greedy_mesh(solid(size))) == 6


def test_colors_split_quads():
    g = solid((4, 4, 4))
    g.data[:, 2:, :] = 2
    quads = greedy_mesh(g)
    assert len(quads) == 10  # 4 side faces split in two, top and bottom whole
    assert {q.color for q in quads} == {1, 2}


def test_quads_cover_every_exposed_face_exactly():
    rng = np.random.default_rng(3)
    data = (rng.random((6, 5, 7)) < 0.5).astype(np.uint8) * rng.integers(1, 4, (6, 5, 7))
    g = VoxelGrid(data.astype(np.uint8), (2, -1, 3))
    assert area(greedy_mesh(g)) == exposed_faces(g)


def test_quad_planes_are_in_world_coordinates():
    quads = greedy_mesh(grid({(5, -2, 7): 1}))
    top = next(q for q in quads if q.axis == 1 and q.sign == 1)
    assert top.layer == -1
    assert (top.u0, top.u1, top.v0, top.v1) == (5, 6, 7, 8)
    bottom = next(q for q in quads if q.axis == 1 and q.sign == -1)
    assert bottom.layer == -2


def test_empty_grid_has_no_quads():
    assert greedy_mesh(VoxelGrid(np.zeros((0, 0, 0), dtype=np.uint8))) == []
