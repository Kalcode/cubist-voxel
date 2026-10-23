import numpy as np
import pytest

from cubist.core.grid import VoxelGrid


def make(points, origin=(0, 0, 0), shape=(4, 4, 4)):
    data = np.zeros(shape, dtype=np.uint8)
    for (x, y, z), c in points.items():
        data[x, y, z] = c
    return VoxelGrid(data, origin)


def test_count_and_colors_used():
    g = make({(0, 0, 0): 1, (1, 2, 3): 5, (2, 2, 2): 5})
    assert g.count() == 3
    assert g.colors_used() == {1: 1, 5: 2}


def test_bounds_are_world_space_half_open():
    g = make({(1, 0, 2): 1, (2, 3, 2): 1}, origin=(10, -5, 0))
    assert g.bounds() == ((11, -5, 2), (13, -1, 3))


def test_bounds_of_empty_grid_is_none():
    assert make({}).bounds() is None


def test_cropped_is_tight_and_keeps_world_position():
    g = make({(1, 0, 2): 3, (2, 3, 2): 4}, origin=(10, -5, 0))
    c = g.cropped()
    assert c.shape == (2, 4, 1)
    assert c.origin == (11, -5, 2)
    assert c.get(11, -5, 2) == 3
    assert c.get(12, -2, 2) == 4


def test_get_outside_grid_is_empty():
    g = make({(0, 0, 0): 1})
    assert g.get(-1, 0, 0) == 0
    assert g.get(100, 0, 0) == 0


def test_equality_compares_content_not_padding():
    a = make({(1, 1, 1): 2}, origin=(0, 0, 0))
    b = make({(0, 0, 0): 2}, origin=(1, 1, 1), shape=(1, 1, 1))
    assert a == b
    assert a != make({(1, 1, 1): 3})
    assert a != make({(1, 1, 2): 2})


def test_voxels_lists_world_coords_and_colors_sorted():
    g = make({(2, 0, 0): 1, (0, 1, 0): 2}, origin=(5, 0, 0))
    coords, colors = g.voxels()
    assert coords.tolist() == [[5, 1, 0], [7, 0, 0]]
    assert colors.tolist() == [2, 1]


def test_from_voxels_round_trips_voxels():
    g = make({(2, 0, 0): 1, (0, 1, 3): 2}, origin=(-3, 0, 0))
    assert VoxelGrid.from_voxels(*g.voxels()) == g


def test_data_must_be_3d_uint8():
    with pytest.raises(ValueError):
        VoxelGrid(np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError):
        VoxelGrid(np.zeros((2, 2, 2), dtype=np.int32))


def test_merge_places_grids_in_world_space_later_ones_win():
    a = make({(0, 0, 0): 1, (1, 0, 0): 1}, shape=(2, 1, 1))
    b = make({(0, 0, 0): 2}, origin=(1, 0, 0), shape=(1, 1, 1))
    c = make({(0, 0, 0): 3}, origin=(5, 2, -1), shape=(1, 1, 1))
    merged = VoxelGrid.merge([a, b, c])
    assert merged.get(0, 0, 0) == 1
    assert merged.get(1, 0, 0) == 2
    assert merged.get(5, 2, -1) == 3
    assert merged.count() == 3


def test_merge_of_nothing_is_empty():
    assert VoxelGrid.merge([]).count() == 0
