"""Parts shown and hidden by clips: muzzle flashes, sparks, anything that appears for a moment."""

import hashlib
import json
import textwrap

import jsonschema
import numpy as np
import pytest
from PIL import Image
from pygltflib import GLTF2

from cubist import Model, ModelError, Part, asset, box, clip, move, slide, visible
from cubist.build.runner import build_script
from cubist.core.palette import Palette
from cubist.io.gltf import encode_glb
from cubist.lint.motion import check_motion
from cubist.manifest import SCHEMA_PATH, build_manifest
from cubist.motion.pose import visibility, world_matrices
from cubist.render.filmstrip import filmstrip
from cubist.render.posed import PosedScene

P = Palette({"wood": ["#3b2414", "#5e3a1f", "#8a5a32"], "fire": ["#d9480f", "#ffe066"]})


def blaster(p, flash_keys=((0, 1), (0.1, 0)), extra=()):
    body = Part("body", box((2, 2, 6), color=p.wood.mid), pivot=(1, 0, 3))
    flash = Part(
        "flash", box((2, 2, 2), color=p.fire.light).at(0, 0, 6), parent="body", hidden=True
    )
    spark = Part("spark", box((1, 1, 1), color=p.fire.dark).at(0, 0, 8), parent="flash")
    return Model(
        [body, flash, spark, *extra],
        clips=[clip("fire", 0.3, [visible("flash", list(flash_keys))])],
    )


def built(fn=blaster, size=(2, 2, 9)):
    return asset(name="blaster", size=size, allow_floating=True)(fn).build(P)


def test_hidden_part_shows_only_while_its_track_says_so():
    a = built()
    c = a.clips[0]
    assert visibility(a, None, 0) == {"body": True, "flash": False, "spark": False}
    assert visibility(a, c, 0.05)["flash"] is True
    assert visibility(a, c, 0.2)["flash"] is False


def test_child_of_hidden_part_is_hidden():
    a = built()
    assert visibility(a, a.clips[0], 0.05)["spark"] is True
    assert visibility(a, a.clips[0], 0.2)["spark"] is False


def test_visible_track_is_always_step_and_values_are_0_or_1():
    c = clip("x", 1.0, [visible("flash", [(0, 1), (0.5, 0)])])
    assert c.track_ease(c.tracks[0]) == "step"
    with pytest.raises(ModelError, match="0 or 1"):
        clip("x", 1.0, [visible("flash", [(0, 0.5)])])


def test_render_skips_hidden_parts():
    a = built()
    scene = PosedScene(a, P.rgba_table())
    region = ((0, 0, 0), (2, 2, 9))
    shown = scene.render({}, "front", 4, region).mask.sum()
    region = ((0, 0, 0), (2, 2, 9))
    top_all = scene.render({}, "top", 4, region).mask.sum()
    top_body = scene.render({}, "top", 4, region, hidden=frozenset({"flash", "spark"})).mask.sum()
    assert shown > 0 and top_all > top_body


def test_filmstrip_frame_shows_flash_only_when_visible():
    a = built()
    strip = np.asarray(filmstrip(a, P.rgba_table(), a.clips, frames=3))
    flash = P.fire.light.index
    rgb = P.rgba_table()[flash, :3]
    # the flash color appears in the strip (the frame at t=0), lit somewhere
    assert np.any(np.all(np.abs(strip.astype(int) - rgb) <= 40, axis=-1))


def test_contact_sheet_leaves_out_hidden_parts_and_names_them(tmp_path):
    (tmp_path / "style.toml").write_text(
        '[project]\nvoxel_size_m = 0.25\n[palette]\nwood = ["#3b2414", "#8a5a32"]\n'
        'fire = ["#d9480f", "#ffe066"]\n'
    )
    (tmp_path / "assets").mkdir()
    script = tmp_path / "assets" / "b.py"
    script.write_text(
        textwrap.dedent("""\
        from cubist import *

        @asset(name="b", size=(2, 2, 8))
        def b(p):
            body = Part("body", box((2, 2, 6), color=p.wood.light), pivot=(1, 0, 3))
            flash = Part("flash", box((2, 2, 2), color=p.fire.light).at(0, 0, 6),
                         parent="body", hidden=True)
            return Model([body, flash], clips=[clip("fire", 0.2, [visible("flash", [1, 0])])])
        """)
    )
    [r] = build_script(script)
    sheet = np.asarray(Image.open(r.contact_path))
    fire = np.array([0xFF, 0xE0, 0x66])
    assert not np.any(np.all(np.abs(sheet.astype(int) - fire) <= 30, axis=-1))
    assert r.summary["parts"][1]["hidden"] is True


def test_glb_hidden_node_has_zero_scale_and_step_scale_channel():
    a = built()
    g = GLTF2.load_from_bytes(encode_glb(a, P.rgba_table(), voxel_size_m=1))
    names = [n.name for n in g.nodes]
    assert g.nodes[names.index("flash")].scale == [0.0, 0.0, 0.0]
    [anim] = g.animations
    [ch] = anim.channels
    assert names[ch.target.node] == "flash" and ch.target.path == "scale"
    assert anim.samplers[ch.sampler].interpolation == "STEP"


def test_visibility_and_motion_on_one_part_both_export():
    def fn(p):
        m = blaster(p)
        c = clip("fire", 0.3, [visible("flash", [(0, 1), (0.1, 0)]), move("flash", "z", [0, 2])])
        return Model(m.parts, clips=[c])

    g = GLTF2.load_from_bytes(encode_glb(built(fn, (2, 2, 11)), P.rgba_table(), voxel_size_m=1))
    paths = sorted(c.target.path for c in g.animations[0].channels)
    assert paths == ["scale", "translation"]


def test_collision_ignores_hidden_parts():
    def fn(p):
        body = Part("body", box((4, 4, 4), color=p.wood.mid), pivot=(2, 0, 2))
        ghost = Part(
            "ghost",
            box((4, 4, 4), color=p.fire.dark).at(0, 0, 6),
            parent="body",
            hidden=True,
            joint=slide("-z", 6),
        )
        return Model([body, ghost], clips=[clip("pass", 1, {"ghost": [0, 6]})])

    a = asset(name="g", size=(4, 4, 10), allow_floating=True)(fn).build(P)
    assert not [f for f in check_motion(a) if f.check == "motion_collision"]


def test_manifest_records_hidden():
    m = build_manifest(built(), voxel_size_m=1)
    jsonschema.validate(m, json.loads(SCHEMA_PATH.read_text()))
    assert [p["hidden"] for p in m["parts"]] == [False, True, False]


def test_static_glb_bytes_unchanged():
    from test_gltf_anim import STATIC_SHA256, build, static
    from test_gltf_anim import P as GP

    data = encode_glb(build(static), GP.rgba_table(), voxel_size_m=0.5)
    assert hashlib.sha256(data).hexdigest() == STATIC_SHA256


def test_world_matrices_ignore_visibility():
    a = built()
    m = world_matrices(a, a.clips[0], 0.2)
    assert np.allclose(m["flash"], np.eye(4))
