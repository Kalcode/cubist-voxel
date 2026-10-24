"""Glowing colors and color cycles, from the style bible to previews, the GLB and the manifest."""

import textwrap

import numpy as np
import pytest

from cubist import Model, Part, asset, box
from cubist.core.model import BuiltAsset
from cubist.core.style import StyleError, cycle_table, glow_array, load_style
from cubist.io.vox import decode_vox, encode_vox

STYLE = textwrap.dedent("""\
    [palette]
    wood  = ["#3b2414", "#8a5a32"]
    fire  = ["#7a1c00", "#d9480f", "#ffe066"]
    water = ["#123456", "#234567", "#345678", "#456789"]
    eye   = { yellow = "#ffd23f" }

    [glow]
    fire = 1.0
    "eye.yellow" = 2.5

    [cycles]
    shimmer = { ramp = "water", fps = 4 }
    """)


def style(tmp_path, text=STYLE):
    path = tmp_path / "style.toml"
    path.write_text(text)
    return load_style(path)


def test_glow_names_resolve_to_palette_indices(tmp_path):
    s = style(tmp_path)
    p = s.palette
    assert s.glow == {
        p.fire[0].index: 1.0,
        p.fire[1].index: 1.0,
        p.fire[2].index: 1.0,
        p.eye.yellow.index: 2.5,
    }


def test_glow_accepts_single_shades(tmp_path):
    s = style(tmp_path, STYLE.replace("fire = 1.0", '"fire.light" = 0.5'))
    assert s.glow[s.palette.fire.light.index] == 0.5 and s.palette.fire.dark.index not in s.glow


def test_unknown_glow_name_suggests(tmp_path):
    with pytest.raises(StyleError, match="fire"):
        style(tmp_path, STYLE.replace("fire = 1.0", "fyre = 1.0"))


def test_glow_strength_must_be_positive(tmp_path):
    with pytest.raises(StyleError, match="> 0"):
        style(tmp_path, STYLE.replace("fire = 1.0", "fire = 0"))


def test_cycle_needs_a_list_ramp(tmp_path):
    with pytest.raises(StyleError, match="list ramp"):
        style(tmp_path, STYLE.replace('ramp = "water"', 'ramp = "eye"'))


def test_cycle_table_shifts_by_ticks_and_wraps(tmp_path):
    s = style(tmp_path)
    [c] = s.cycles
    table = s.palette.rgba_table()
    w = [x.index for x in s.palette.water]
    assert c.name == "shimmer" and c.indices == tuple(w) and c.fps == 4
    shifted = cycle_table(table, s.cycles, 0.3)  # floor(0.3 * 4) = 1 tick
    assert np.array_equal(shifted[w[0]], table[w[1]])
    assert np.array_equal(shifted[w[3]], table[w[0]])  # wraps
    assert np.array_equal(cycle_table(table, s.cycles, 0.0), table)


def test_glow_array_is_indexed_by_color(tmp_path):
    s = style(tmp_path)
    g = glow_array(s.glow)
    assert g.shape == (256,) and g[s.palette.eye.yellow.index] == 2.5 and g[0] == 0


def test_glow_and_cycles_survive_summary_and_vox(tmp_path):
    s = style(tmp_path)

    def fn(p):
        return Model([Part("pond", box((2, 1, 2), color=p.water.mid))])

    a = asset(name="pond", size=(2, 1, 2))(fn).build(s.palette, glow=s.glow, cycles=s.cycles)
    summary = a.summary()
    again = BuiltAsset.from_vox(decode_vox(encode_vox(a.to_vox_scene())), summary, s.palette)
    assert again.glow == s.glow and again.cycles == s.cycles


def test_style_to_dict_lists_glow_and_cycles(tmp_path):
    d = style(tmp_path).to_dict()
    assert d["glow"]["eye.yellow"] == 2.5 and d["cycles"][0]["ramp"] == "water"


# --- glow in previews, the GLB and the manifest -------------------------------

from pygltflib import GLTF2  # noqa: E402

from cubist.io.gltf import encode_glb  # noqa: E402
from cubist.manifest import build_manifest  # noqa: E402
from cubist.render import raster  # noqa: E402
from cubist.render.posed import PosedScene  # noqa: E402


def lit(tmp_path, color="eye"):
    s = style(tmp_path)

    def fn(p):
        c = p.eye.yellow if color == "eye" else p.wood.light
        return Model([Part("orb", box((2, 2, 2), color=c))])

    return asset(name="orb", size=(2, 2, 2))(fn).build(s.palette, glow=s.glow), s


def test_glowing_voxels_render_at_full_color(tmp_path):
    a, s = lit(tmp_path)
    table, g = s.palette.rgba_table(), glow_array(a.glow)
    full = table[s.palette.eye.yellow.index, :3]
    region = ((0, 0, 0), (2, 2, 2))
    front = raster.ortho(a.merged_grid(), table, "front", 4, region, glow=g)
    assert np.abs(front.rgb[2:6, 2:6].reshape(-1, 3).astype(int) - full).max() <= 1
    iso = raster.iso(a.merged_grid(), table, 4, region, glow=g)
    posed = PosedScene(a, table, glow=g).render({}, "iso", 4, region)
    assert np.array_equal(iso.mask, posed.mask)
    assert np.abs(iso.rgb.astype(int) - posed.rgb.astype(int)).max() <= 1
    dull = raster.iso(a.merged_grid(), table, 4, region)
    assert iso.rgb[iso.mask].mean() > dull.rgb[dull.mask].mean()


def test_glb_emissive_only_when_a_glowing_color_is_used(tmp_path):
    a, s = lit(tmp_path)
    data = encode_glb(a, s.palette.rgba_table(), voxel_size_m=1)
    g = GLTF2.load_from_bytes(data)
    m = g.materials[0]
    assert m.emissiveTexture is not None and m.emissiveFactor == [1.0, 1.0, 1.0]
    assert m.extensions["KHR_materials_emissive_strength"]["emissiveStrength"] == 2.5
    assert "KHR_materials_emissive_strength" in g.extensionsUsed
    plain, _ = lit(tmp_path, color="wood")
    assert b"emissive" not in encode_glb(plain, s.palette.rgba_table(), voxel_size_m=1)


def test_manifest_lists_used_glowing_colors_by_name(tmp_path):
    a, _ = lit(tmp_path)
    assert build_manifest(a, voxel_size_m=1)["glow"] == {"eye.yellow": 2.5}
