import textwrap

import pytest
from PIL import Image

from cubist.core.palette import Palette, PaletteError
from cubist.core.style import StyleError, load_style, read_palette_file

FULL = """
[project]
name = "goblin-dungeon"
voxel_size_m = 0.0625

[palette]
wood  = ["#3b2414", "#5e3a1f", "#8a5a32", "#b07d4a"]
metal = ["#2a2d34", "#4b505a", "#7d8491", "#c2c8d0"]

[scale]
humanoid_height = 24
door_height = 32

[budgets]
max_colors = 32
max_voxels = 40000
max_triangles = 6000
max_parts = 12

[rules]
no_pure_black = true
dither = "forbidden"
symmetry_tolerance = 2
notes = "Chunky silhouettes."
"""


def write(tmp_path, text, name="style.toml"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text))
    return path


def test_loads_every_section(tmp_path):
    s = load_style(write(tmp_path, FULL))
    assert s.name == "goblin-dungeon"
    assert s.voxel_size_m == 0.0625
    assert s.palette.wood.dark.index == 1
    assert s.scale == {"humanoid_height": 24, "door_height": 32}
    assert (s.budgets.max_colors, s.budgets.max_voxels) == (32, 40000)
    assert (s.budgets.max_triangles, s.budgets.max_parts) == (6000, 12)
    assert s.rules.no_pure_black is True
    assert s.rules.dither == "forbidden"
    assert s.rules.symmetry_tolerance == 2
    assert s.rules.notes == "Chunky silhouettes."


def test_missing_sections_get_defaults(tmp_path):
    s = load_style(write(tmp_path, '[palette]\nwood = ["#3b2414"]\n'))
    assert s.name == tmp_path.name
    assert s.voxel_size_m == 0.0625
    assert s.budgets.max_voxels is None
    assert s.rules.dither == "allowed"
    assert s.rules.symmetry_tolerance == 0


def test_unknown_section_is_an_error_with_suggestion(tmp_path):
    with pytest.raises(StyleError, match="budget.*budgets"):
        load_style(write(tmp_path, "[budget]\nmax_voxels = 1\n"))


def test_unknown_key_is_an_error(tmp_path):
    with pytest.raises(StyleError, match="max_voxel.*max_voxels"):
        load_style(write(tmp_path, "[budgets]\nmax_voxel = 1\n"))


def test_bad_dither_rule_is_an_error(tmp_path):
    with pytest.raises(StyleError, match="dither"):
        load_style(write(tmp_path, '[rules]\ndither = "sometimes"\n'))


def test_bad_value_type_is_an_error(tmp_path):
    with pytest.raises(StyleError, match="max_voxels"):
        load_style(write(tmp_path, '[budgets]\nmax_voxels = "lots"\n'))


def test_to_dict_lists_palette_names_and_hex(tmp_path):
    d = load_style(write(tmp_path, FULL)).to_dict()
    assert d["palette"][0] == {"index": 1, "name": "wood.dark", "hex": "#3b2414"}
    assert d["budgets"]["max_voxels"] == 40000


# --- palette source files ---------------------------------------------------


def test_hex_source_with_ramps_referencing_it_by_index(tmp_path):
    (tmp_path / "pal.hex").write_text("1a1c2c\n5d275d\n#b13e53\nef7d57\n")
    s = load_style(write(tmp_path, '[palette]\nsource = "pal.hex"\nwarm = [1, 2, 3]\n'))
    p = s.palette
    assert len(p) == 4
    assert p.warm.dark.index == p.source[1].index == 2
    assert p.warm.dark.name == "warm.dark"
    assert p.source[0].name == "source.0"
    assert p.warm.light.rgba == (0xEF, 0x7D, 0x57, 255)


def test_ramp_hex_matching_a_source_color_shares_its_index(tmp_path):
    (tmp_path / "pal.hex").write_text("1a1c2c\n5d275d\n")
    s = load_style(
        write(tmp_path, '[palette]\nsource = "pal.hex"\nink = ["#1A1C2C"]\nnew = "#ffffff"\n')
    )
    assert s.palette.ink.dark.index == 1
    assert s.palette.new.index == 3


def test_gpl_source(tmp_path):
    (tmp_path / "pal.gpl").write_text(
        "GIMP Palette\nName: test\nColumns: 2\n#\n 26  28  44\tnight\n255 255 255 white\n"
    )
    assert read_palette_file(tmp_path / "pal.gpl") == ["#1a1c2c", "#ffffff"]


def test_png_source_reads_distinct_colors_in_order(tmp_path):
    img = Image.new("RGB", (3, 2))
    pixels = [(10, 20, 30), (40, 50, 60), (10, 20, 30), (1, 2, 3), (40, 50, 60), (7, 8, 9)]
    img.putdata(pixels)
    img.save(tmp_path / "pal.png")
    assert read_palette_file(tmp_path / "pal.png") == ["#0a141e", "#28323c", "#010203", "#070809"]


def test_missing_source_file_is_an_error(tmp_path):
    with pytest.raises(StyleError, match="nope.hex"):
        load_style(write(tmp_path, '[palette]\nsource = "nope.hex"\n'))


def test_index_reference_without_source_is_an_error():
    with pytest.raises(PaletteError, match="source"):
        Palette({"warm": [1, 2]})


def test_index_reference_out_of_range_is_an_error():
    with pytest.raises(PaletteError, match="5"):
        Palette({"warm": [5]}, source=["#000001"])


# --- shade stepping ---------------------------------------------------------


def test_shift_steps_along_a_list_ramp_and_clamps():
    p = Palette(
        {
            "wood": ["#100000", "#200000", "#300000"],
            "eye": "#ffff00",
            "cloth": {"green": "#00ff00", "red": "#ff0000"},
        }
    )
    assert p.shift(p.wood.dark.index, 1) == p.wood[1].index
    assert p.shift(p.wood.dark.index, -1) == p.wood.dark.index
    assert p.shift(p.wood.light.index, 5) == p.wood.light.index
    assert p.shift(p.eye.index, 1) == p.eye.index
    assert p.shift(p.cloth.green.index, 1) == p.cloth.green.index


def test_shift_ignores_the_unordered_source_ramp():
    p = Palette({"warm": [1, 2]}, source=["#000001", "#000002", "#000003"])
    assert p.shift(p.source[0].index, 1) == p.source[0].index
    assert p.shift(p.warm.dark.index, 1) == p.warm.light.index


def test_to_dict_separates_ramps_from_named_colors(tmp_path):
    d = load_style(
        write(
            tmp_path,
            '[palette]\nwood = ["#100000", "#200000"]\n'
            'cloth = { white = "#ffffff" }\neye = "#ffff00"\n',
        )
    ).to_dict()
    assert d["ramps"] == ["wood"]
    assert d["named"] == ["cloth", "eye"]
