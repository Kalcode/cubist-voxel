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
