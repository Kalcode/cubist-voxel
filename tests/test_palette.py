import numpy as np
import pytest

from cubist.core.palette import Palette, PaletteError

RAMPS = {
    "wood": ["#3b2414", "#5e3a1f", "#8a5a32", "#b07d4a"],
    "metal": ["#2a2d34", "#4b505a", "#7d8491"],
    "cloth": {"green": "#3f6b2a", "red": "#8b2a2a"},
    "eye": "#ffd23f",
}


@pytest.fixture
def p():
    return Palette(RAMPS)


def test_indices_follow_definition_order_starting_at_one(p):
    assert [c.index for c in p.wood] == [1, 2, 3, 4]
    assert [c.index for c in p.metal] == [5, 6, 7]
    assert p.cloth.green.index == 8
    assert p.eye.index == 10


def test_list_ramp_shade_names(p):
    assert p.wood.dark.index == 1
    assert p.wood.mid.index == 3
    assert p.wood.light.index == 4
    assert p.metal.mid.index == 6
    assert p.wood[1].index == 2


def test_color_carries_name_and_rgba(p):
    c = p.wood.dark
    assert c.name == "wood.dark"
    assert c.rgba == (0x3B, 0x24, 0x14, 255)
    assert p.wood[1].name == "wood.1"
    assert p.eye.name == "eye"


def test_ramp_property_lists_colors_dark_to_light(p):
    assert [c.index for c in p.wood.ramp] == [1, 2, 3, 4]


def test_unknown_ramp_is_an_error_naming_close_matches(p):
    with pytest.raises(PaletteError, match="wod.*wood"):
        _ = p.wod


def test_unknown_shade_is_an_error(p):
    with pytest.raises(PaletteError, match="wood.medium"):
        _ = p.wood.medium


def test_rgba_table_is_indexed_by_color_index(p):
    table = p.rgba_table()
    assert table.shape == (256, 4)
    assert table.dtype == np.uint8
    assert tuple(table[0]) == (0, 0, 0, 0)
    assert tuple(table[1]) == (0x3B, 0x24, 0x14, 255)
    assert tuple(table[10]) == (0xFF, 0xD2, 0x3F, 255)


def test_name_of_index(p):
    assert p.name_of(3) == "wood.mid"
    assert p.name_of(8) == "cloth.green"


def test_more_than_255_colors_is_an_error():
    with pytest.raises(PaletteError, match="255"):
        Palette({"big": ["#000001"] * 256})


def test_bad_hex_is_an_error():
    with pytest.raises(PaletteError, match="#12345"):
        Palette({"bad": ["#12345"]})
