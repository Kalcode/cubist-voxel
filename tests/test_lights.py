"""Point lights: where a torch or a campfire lights its surroundings."""

import json

import jsonschema
import numpy as np
import pytest
from pygltflib import GLTF2

from cubist import Model, ModelError, Part, asset, box, light
from cubist.core.model import BuiltAsset
from cubist.core.palette import Palette
from cubist.io.gltf import encode_glb, srgb_to_linear
from cubist.io.vox import decode_vox, encode_vox
from cubist.manifest import SCHEMA_PATH, build_manifest
from cubist.render.filmstrip import posed_sheet
from cubist.render.posed import PosedScene

P = Palette({"wood": ["#3b2414", "#8a5a32"], "fire": ["#d9480f", "#ffe066"]})


def torch(p, lights=None):
    stick = Part("stick", box((1, 6, 1), color=p.wood.dark), pivot=(0.5, 0, 0.5))
    lights = (
        lights
        if lights is not None
        else {"flame": light((0.5, 7, 0.5), p.fire.light, range=6, intensity=2, flicker=0.3)}
    )
    return Model([stick], lights=lights)


def built(fn=torch):
    return asset(name="torch", size=(1, 8, 1))(fn).build(P)


def test_light_round_trips_through_summary_and_vox():
    a = built()
    [(name, lt)] = a.lights.items()
    assert name == "flame" and lt.color == P.fire.light.index and lt.flicker == 0.3
    again = BuiltAsset.from_vox(decode_vox(encode_vox(a.to_vox_scene())), a.summary(), P)
    assert again.lights == a.lights


def test_glb_has_a_point_light_node_under_its_part():
    g = GLTF2.load_from_bytes(encode_glb(built(), P.rgba_table(), voxel_size_m=0.5))
    assert "KHR_lights_punctual" in g.extensionsUsed
    [lt] = g.extensions["KHR_lights_punctual"]["lights"]
    assert lt["type"] == "point" and lt["intensity"] == 2.0 and lt["range"] == 3.0
    want = srgb_to_linear(np.array([0xFF, 0xE0, 0x66]))
    assert np.allclose(lt["color"], want, atol=1e-6)
    names = [n.name for n in g.nodes]
    node = g.nodes[names.index("flame")]
    assert node.extensions == {"KHR_lights_punctual": {"light": 0}}
    stick = g.nodes[names.index("stick")]
    assert names.index("flame") in stick.children
    assert node.translation == [0.0, 3.5, 0.0]  # (0.5, 7, 0.5) - pivot (0.5, 0, 0.5), metres


def test_manifest_records_light_with_color_name_and_flicker():
    m = build_manifest(built(), voxel_size_m=1)
    jsonschema.validate(m, json.loads(SCHEMA_PATH.read_text()))
    assert m["lights"]["flame"] == {
        "position": [0.5, 7.0, 0.5], "part": "stick", "color": "fire.light", "rgb": "#ffe066",
        "range": 6.0, "intensity": 2.0, "flicker": 0.3,
    }  # fmt: skip
    assert m["colors"]["fire.light"] == {"index": P.fire.light.index, "rgb": "#ffe066"}


def test_bad_flicker_and_unknown_part_are_errors():
    with pytest.raises(ModelError, match="flicker"):
        light((0, 0, 0), P.fire.light, flicker=2)
    with pytest.raises(ModelError, match="unknown part"):
        built(lambda p: torch(p, {"x": light((0, 0, 0), p.fire.light, part="nope")}))
    with pytest.raises(ModelError, match="palette color"):
        light((0, 0, 0), "#ffffff")


def test_posed_views_mark_the_light():
    a = built()
    scene = PosedScene(a, P.rgba_table())
    region = ((-2, 0, -2), (3, 9, 3))
    plain = scene.render({}, "front", 8, region)
    marked = scene.render({}, "front", 8, region, markers=[((0.5, 7, 0.5), (255, 224, 102))])
    assert (marked.rgb != plain.rgb).any()
    rgb = np.array([255, 224, 102])
    assert np.any(np.all(marked.rgb == rgb, axis=-1))
    assert posed_sheet(a, P.rgba_table(), None, 0.0) is not None


def test_static_assets_have_no_lights_extension():
    from test_gltf_anim import P as GP
    from test_gltf_anim import build as gbuild
    from test_gltf_anim import static

    assert b"KHR_lights" not in encode_glb(gbuild(static), GP.rgba_table(), voxel_size_m=0.5)


# --- lint: lights that can't reach the floor ------------------------------------------

from cubist.core.style import StyleBible  # noqa: E402
from cubist.lint import lint  # noqa: E402


def post(height, range_, grounded=True):
    def fn(p):
        pole = Part("pole", box((1, height, 1), color=p.wood.dark), pivot=(0.5, 0, 0.5))
        lamp = light((0.5, height + 0.5, 0.5), p.fire.light, range=range_)
        return Model([pole], lights={"lamp": lamp})

    return asset(name="post", size=(1, height, 1), grounded=grounded)(fn).build(P)


def reach(a):
    style = StyleBible(path=None, name="t", voxel_size_m=0.0625, palette=P)
    return [f for f in lint(a, style).findings if f.check == "light_reach"]


def test_light_too_short_to_light_the_floor_warns_in_metres():
    # The lamp post that started this: 50 voxels up, range 48.
    [f] = reach(post(50, 48))
    assert f.severity == "warning"
    assert "lamp" in f.message and "3.2 m" in f.message and "3.0 m" in f.message
    assert f.details["min_range"] == 72  # ceil(sqrt(2) * 50.5)


def test_light_reaching_the_floor_around_it_is_fine():
    assert reach(post(50, 120)) == []
    assert reach(post(6, 14)) == []  # a campfire's light, low and wide


def test_light_reach_is_only_checked_on_grounded_assets():
    assert reach(post(50, 10, grounded=False)) == []  # held or hung: no known floor
