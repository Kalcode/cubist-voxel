import numpy as np
import pytest

from cubist.core.grid import VoxelGrid
from cubist.render.ascii import parse_slices, slices

NAMES = {1: "wood.dark", 2: "metal.mid"}


def grid(points):
    coords = np.array(list(points.keys()))
    colors = np.array(list(points.values()), dtype=np.uint8)
    return VoxelGrid.from_voxels(coords, colors)


G = grid({(0, 0, 0): 1, (1, 0, 0): 2, (1, 0, 1): 2, (2, 1, 0): 1})


def test_y_slices_read_like_the_top_view():
    assert slices(G, "y", 0, 1, NAMES) == (
        "legend: . = empty, a = wood.dark, b = metal.mid\n"
        "\n"
        "y=0  (rows: z, back to front; columns: x)\n"
        "   0\n"
        "0  ab.\n"
        "1  .b.\n"
        "\n"
        "y=1  (rows: z, back to front; columns: x)\n"
        "   0\n"
        "0  ..a\n"
        "1  ...\n"
    )


def test_z_slice_reads_like_the_front_view():
    out = slices(G, "z", 0, 0, NAMES)
    assert "z=0  (rows: y, top to bottom; columns: x)" in out
    assert out.endswith("1  ..a\n0  ab.\n")


def test_x_slice_columns_run_back_to_front():
    out = slices(G, "x", 1, 1, NAMES)
    assert "x=1  (rows: y, top to bottom; columns: z, back to front)" in out
    assert out.endswith("1  ..\n0  bb\n")


def test_legend_only_lists_colors_in_the_printed_layers():
    out = slices(G, "y", 1, 1, NAMES)
    assert out.startswith("legend: . = empty, a = wood.dark\n")


def test_unnamed_colors_fall_back_to_their_index():
    assert "a = color 1" in slices(G, "y", 1, 1, {})


def test_column_header_marks_every_fifth_world_coordinate():
    wide = grid({(x, 0, 0): 1 for x in range(-3, 9)})
    lines = slices(wide, "y", 0, 0, NAMES).splitlines()
    header, row = lines[3], lines[4]
    assert header.index("0") == row.index("a") + 3  # column x=0 is the 4th voxel
    assert header.index("5") == header.index("0") + 5


def test_parse_slices():
    assert parse_slices("y=10..14") == ("y", 10, 14)
    assert parse_slices("x=3") == ("x", 3, 3)
    assert parse_slices("z=-2..1") == ("z", -2, 1)
    with pytest.raises(ValueError, match="y=10..14"):
        parse_slices("w=1")
    with pytest.raises(ValueError):
        parse_slices("y=5..2")
