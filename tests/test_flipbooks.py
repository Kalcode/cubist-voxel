"""Flipbook parts: voxels that change shape over time (flames, water, a muzzle flash)."""

import json
import textwrap

import jsonschema
import numpy as np
import pytest
from pygltflib import GLTF2

from cubist import Model, ModelError, Part, asset, box, clip, frames, visible
from cubist.build.runner import build_script
from cubist.core.model import BuiltAsset
from cubist.core.palette import Palette
from cubist.export import export_asset
from cubist.io.gltf import encode_glb
from cubist.io.vox import decode_vox, encode_vox, read_vox, write_vox
from cubist.lint import lint
from cubist.manifest import SCHEMA_PATH, build_manifest
from cubist.motion.pose import frame_index, preview_clips
from cubist.render.filmstrip import clip_frames, motion_region
from cubist.render.posed import PosedScene

P = Palette({"wood": ["#3b2414", "#8a5a32"], "fire": ["#d9480f", "#f59f00", "#ffe066"]})


def flame_frames(p, n=4):
    # a column that grows and shrinks: frame k is 2 + k % 3 voxels tall
    return [box((1, 2 + k % 3, 1), color=p.fire.mid).at(0, 3, 0) for k in range(n)]


def torch(p, n=4, fps=8):
    stick = Part("stick", box((1, 3, 1), color=p.wood.dark), pivot=(0.5, 0, 0.5))
    flame = Part("flame", frames=flame_frames(p, n), fps=fps, parent="stick", pivot=(0.5, 3, 0.5))
    return Model([stick, flame])


def built(fn=torch, size=(1, 8, 1)):
    return asset(name="torch", size=size)(fn).build(P)


def test_frames_build_with_frame_zero_as_the_rest_look():
    a = built()
    flame = a.part("flame")
    assert len(flame.frames) == 4 and flame.fps == 8
    assert flame.grid == flame.frames[0] and flame.frames[1].count() == 3


def test_part_needs_a_shape_or_frames():
    with pytest.raises(ModelError, match="frames"):
        Model([Part("x")])
    with pytest.raises(ModelError, match="two"):
        Model([Part("x", frames=[box((1, 1, 1))])])


def test_vox_round_trips_frames_as_hidden_models():
    a = built()
    scene = decode_vox(encode_vox(a.to_vox_scene()))
    names = {p.name: p for p in scene.parts}
    assert names["flame@1"].hidden and not names["flame"].hidden
    again = BuiltAsset.from_vox(scene, a.summary(), P)
    assert [p.name for p in again.parts] == ["stick", "flame"]
    assert again.part("flame").frames == a.part("flame").frames


def test_frame_index_self_loops_and_a_clip_track_overrides():
    a = built()
    assert [frame_index(a, None, "flame", t) for t in (0, 0.13, 0.26, 0.4, 0.5)] == [0, 1, 2, 3, 0]

    def driven(p):
        m = torch(p)
        return Model(m.parts, clips=[clip("burst", 0.2, [frames("flame", [(0, 3), (0.1, 1)])])])

    b = built(driven)
    assert frame_index(b, b.clips[0], "flame", 0.05) == 3
    assert frame_index(b, b.clips[0], "flame", 0.15) == 1


def test_frame_track_needs_a_flipbook_part():
    def bad(p):
        m = torch(p)
        return Model(m.parts, clips=[clip("x", 1, [frames("stick", [0, 1])])])

    with pytest.raises(ModelError, match="flipbook"):
        built(bad)


def test_preview_clips_add_a_loop_per_free_running_flipbook():
    [c] = preview_clips(built())
    assert c.name == "flame_frames" and c.loop and c.length == pytest.approx(0.5)


def test_motion_region_covers_every_frame():
    a = built()
    scene = PosedScene(a, P.rgba_table())
    region = motion_region(a, preview_clips(a), scene)
    assert region[1][1] >= 3 + 4  # the tallest frame reaches y = 7


def test_gif_frames_differ_as_the_flipbook_plays():
    a = built()
    images, _ = clip_frames(a, P.rgba_table(), preview_clips(a)[0], fps=8)
    arrays = [np.asarray(i) for i in images]
    assert not np.array_equal(arrays[0], arrays[1])


def test_glb_frame_nodes_and_loop_animation():
    g = GLTF2.load_from_bytes(encode_glb(built(), P.rgba_table(), voxel_size_m=1))
    names = [n.name for n in g.nodes]
    flame = g.nodes[names.index("flame")]
    assert flame.mesh is None
    kids = [names[i] for i in flame.children]
    assert kids[:4] == ["flame@0", "flame@1", "flame@2", "flame@3"]
    assert g.nodes[names.index("flame@0")].scale is None
    assert g.nodes[names.index("flame@1")].scale == [0.0, 0.0, 0.0]
    [anim] = g.animations
    assert anim.name == "flame_frames" and anim.extras["loop"] is True
    targets = {names[c.target.node] for c in anim.channels}
    assert targets == {"flame@0", "flame@1", "flame@2", "flame@3"}
    assert all(anim.samplers[c.sampler].interpolation == "STEP" for c in anim.channels)


def test_driven_hidden_flipbook_exports_both_channels():
    def blaster(p):
        body = Part("body", box((1, 1, 4), color=p.wood.dark), pivot=(0.5, 0, 2))
        flash = Part(
            "flash",
            frames=[
                box((1, 1, 1), color=p.fire.light).at(0, 0, 4),
                box((1, 1, 2), color=p.fire.mid).at(0, 0, 4),
            ],
            parent="body",
            hidden=True,
            pivot=(0.5, 0.5, 4.5),
        )
        fire = clip(
            "fire",
            0.2,
            [visible("flash", [(0, 1), (0.1, 0)]), frames("flash", [(0, 0), (0.05, 1)])],
        )
        return Model([body, flash], clips=[fire])

    a = asset(name="b", size=(1, 1, 6))(blaster).build(P)
    g = GLTF2.load_from_bytes(encode_glb(a, P.rgba_table(), voxel_size_m=1))
    names = [n.name for n in g.nodes]
    [anim] = g.animations  # driven flipbooks get no loop of their own
    targets = {(names[c.target.node], c.target.path) for c in anim.channels}
    assert ("flash", "scale") in targets and ("flash@1", "scale") in targets


def test_lint_checks_palette_and_bounds_on_every_frame():
    from cubist.core.style import StyleBible

    def wild(p):
        m = torch(p)
        f = m.parts[1]
        tall = list(f.frames[:3]) + [box((1, 9, 1), color=p.fire.mid).at(0, 3, 0)]
        return Model([m.parts[0], Part("flame", frames=tall, parent="stick", pivot=(0.5, 3, 0.5))])

    a = built(wild)
    style = StyleBible(path=None, name="t", voxel_size_m=1, palette=P)
    report = lint(a, style)
    assert [f.check for f in report.findings if f.severity == "error"] == ["bounds"]


def test_manifest_lists_frames():
    m = build_manifest(built(), voxel_size_m=1)
    jsonschema.validate(m, json.loads(SCHEMA_PATH.read_text()))
    assert m["parts"][1]["frames"] == {
        "count": 4, "fps": 8.0, "vox_models": ["flame", "flame@1", "flame@2", "flame@3"],
    }  # fmt: skip
    assert m["parts"][0]["frames"] is None
    assert m["clips"] == []  # loops are implied by frames, not authored clips


@pytest.fixture
def project(tmp_path):
    (tmp_path / "style.toml").write_text(
        '[project]\nvoxel_size_m = 0.25\n[palette]\nwood = ["#3b2414", "#8a5a32"]\n'
        'fire = ["#d9480f", "#f59f00", "#ffe066"]\n'
    )
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "torch.py").write_text(
        textwrap.dedent("""\
        from cubist import *

        @asset(name="torch", size=(1, 8, 1), grounded=True)
        def torch(p):
            stick = Part("stick", box((1, 3, 1), color=p.wood.dark), pivot=(0.5, 0, 0.5))
            flames = [box((1, 2 + k % 3, 1), color=p.fire.mid).at(0, 3, 0) for k in range(4)]
            return Model([stick, Part("flame", frames=flames, fps=8, parent="stick")])
        """)
    )
    return tmp_path


def test_asset_with_only_a_flipbook_gets_motion_previews(project):
    [r] = build_script(project / "assets" / "torch.py")
    d = r.vox_path.parent
    assert (d / "motion.png").exists() and (d / "flame_frames.gif").exists()


def test_missing_frame_model_is_a_hand_edit(project):
    from cubist import export as ex

    [r] = build_script(project / "assets" / "torch.py")
    scene = read_vox(r.vox_path)
    scene.parts = [p for p in scene.parts if p.name != "flame@2"]
    write_vox(r.vox_path, scene)
    with pytest.raises(ex.ExportBlocked, match="flame"):
        export_asset(r.vox_path.parent)
