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


def test_ao_meshing_keeps_open_faces_merged_and_occluded_faces_separate():
    floor = solid((5, 1, 5))
    floor.data[2, 0, 2] = 1
    g = VoxelGrid.merge([floor, grid({(2, 1, 2): 2})])
    plain = greedy_mesh(g)
    shaded = greedy_mesh(g, ao=True)
    assert area(shaded) == area(plain) == exposed_faces(g)
    occluded = [q for q in shaded if q.ao is not None]
    # The 8 floor faces around the post's foot see it in their outward layer.
    assert len([q for q in occluded if q.axis == 1 and q.sign == 1]) == 8
    assert all((q.u1 - q.u0) * (q.v1 - q.v0) == 1 for q in occluded)
    assert all(q.ao is None for q in plain)


def test_ao_corner_values():
    g = VoxelGrid.merge([solid((3, 1, 3)), grid({(1, 1, 1): 2})])
    quads = greedy_mesh(g, ao=True)
    # Floor face at x=2, z=1 (just right of the post): its two corners at x=2 touch the post.
    face = next(q for q in quads if q.axis == 1 and q.sign == 1 and (q.u0, q.v0) == (2, 1))
    # corners ordered (u0,v0), (u1,v0), (u0,v1), (u1,v1); u = x, v = z
    assert face.ao == (2, 3, 2, 3)
