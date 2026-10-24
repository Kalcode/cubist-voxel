import time

import numpy as np
import pytest

from cubist import sphere
from cubist.core.grid import VoxelGrid
from cubist.core.palette import Palette
from cubist.render.contact_sheet import choose_scale, contact_sheet
from cubist.render.font import GLYPH_H, GLYPH_W, draw_text, text_width

P = Palette({"wood": ["#3b2414", "#8a5a32"], "eye": "#ffd23f"})


def block():
    coords = np.array([(x, y, z) for x in range(4) for y in range(6) for z in range(3)])
    colors = np.where(coords[:, 1] > 3, 3, 1).astype(np.uint8)
    return VoxelGrid.from_voxels(coords, colors)


@pytest.mark.parametrize(
    "dims, expected", [((12, 12, 12), 12), ((24, 24, 24), 10), ((64, 64, 64), 4), ((200, 8, 8), 2)]
)
def test_choose_scale_targets_256px_views_with_even_scale(dims, expected):
    assert choose_scale(dims) == expected


def test_text_width_and_drawing():
    assert text_width("AB", 2) == 2 * (GLYPH_W + 1) * 2 - 2
    canvas = np.zeros((20, 40, 3), dtype=np.uint8)
    draw_text(canvas, 1, 1, "1", (255, 255, 255))
    assert canvas.any()
    assert not canvas[:, 1 + GLYPH_W :].any()
    assert not canvas[1 + GLYPH_H :, :].any()


def test_unknown_characters_draw_as_question_marks():
    a = np.zeros((10, 10, 3), dtype=np.uint8)
    b = np.zeros((10, 10, 3), dtype=np.uint8)
    draw_text(a, 0, 0, "^", (255, 255, 255))
    draw_text(b, 0, 0, "?", (255, 255, 255))
    assert np.array_equal(a, b)


def test_contact_sheet_is_an_rgb_image_with_header_and_views():
    img = contact_sheet(block(), P, name="block", size=(4, 6, 3))
    assert img.mode == "RGB"
    assert img.width > 300 and img.height > 150


def test_more_views_make_a_bigger_sheet():
    small = contact_sheet(block(), P, name="block", views=("front",))
    full = contact_sheet(block(), P, name="block")
    assert full.width * full.height > small.width * small.height


def test_silhouette_view_is_solid_black():
    img = np.asarray(contact_sheet(block(), P, name="block", views=("silhouette",)))
    assert (np.all(img == 0, axis=-1)).sum() >= 4 * 6 * 12 * 12


def test_unknown_view_is_an_error():
    with pytest.raises(ValueError, match="sideways"):
        contact_sheet(block(), P, name="block", views=("sideways",))


def test_empty_grid_still_renders():
    empty = VoxelGrid(np.zeros((0, 0, 0), dtype=np.uint8))
    assert contact_sheet(empty, P, name="nothing").width > 0


def test_64_cubed_model_renders_in_under_a_second():
    shape = sphere(32.1, color=P.wood.mid).at(32, 32, 32)
    shape = shape - sphere(12, color=P.wood.dark).at(52, 44, 52)
    grid = shape.voxelize()
    assert max(grid.shape) == 64
    contact_sheet(grid, P, name="warmup")  # sprite caches, imports
    start = time.perf_counter()
    contact_sheet(grid, P, name="sphere", size=(64, 64, 64))
    assert time.perf_counter() - start < 1.0
