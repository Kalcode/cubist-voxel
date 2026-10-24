"""Fixes for the independent review of milestones 8-10."""

import runpy
from pathlib import Path

import numpy as np
import pytest
from pygltflib import GLTF2

from cubist import (
    Model,
    ModelError,
    Part,
    asset,
    box,
    clip,
    emitter,
    fracture,
    frames,
    light,
    turn,
    visible,
)
from cubist.build.runner import load_project_style
from cubist.core.model import AssetDef
from cubist.core.palette import Palette
from cubist.core.style import StyleError, load_style
from cubist.dsl.context import building
from cubist.io.gltf import encode_glb, nearest_part
from cubist.lint.motion import check_motion
from cubist.motion.particles import particles_at
from cubist.motion.pose import frame_index, preview_clips

DEMO = Path(__file__).parent.parent / "projects" / "demo"
P = Palette({"wood": ["#3b2414", "#8a5a32"], "fire": ["#d9480f", "#f59f00", "#ffe066"]})


def flames(p, n=3):
    return [box((1, 1 + k, 1), color=p.fire.mid).at(0, 2, 0) for k in range(n)]


def torch(p, clips):
    stick = Part("stick", box((1, 2, 1), color=p.wood.dark), pivot=(0.5, 0, 0.5))
    return Model([stick, Part("flame", frames=flames(p), parent="stick")], clips=clips)


def built(fn, size=(1, 6, 1), **kw):
    return asset(name="t", size=size, **kw)(fn).build(P)


def test_a_flipbook_driven_by_any_clip_rests_on_frame_zero_in_the_others():
    clips = [clip("flare", 0.3, [frames("flame", [(0, 2)])]),
             clip("sway", 1.0, [turn("stick", "z", [0, 10, 0])])]  # fmt: skip
    a = built(lambda p: torch(p, clips))
    assert [c.name for c in preview_clips(a)] == ["flare", "sway"]  # no flame_frames loop
    assert {frame_index(a, a.clips[1], "flame", t) for t in (0, 0.2, 0.4, 0.7)} == {0}
    info = [f for f in check_motion(a) if f.check == "flipbook_driven"]
    assert info and info[0].severity == "info" and "'flare'" in info[0].message


def test_frame_keys_past_the_last_frame_are_an_error():
    with pytest.raises(ModelError, match="frame 7"):
        built(lambda p: torch(p, [clip("x", 1, [frames("flame", [(0, 0), (0.5, 7)])])]))


def test_two_frames_tracks_on_one_part_are_an_error():
    with pytest.raises(ModelError, match="two frames"):
        clip("c", 0.4, [frames("flame", [(0, 0)]), frames("flame", [(0.2, 2)])])


def test_step_sampler_times_strictly_increase_in_float32():
    def fn(p):
        flash = Part("flash", box((1, 1, 1), color=p.fire.light), hidden=True)
        body = Part("body", box((1, 1, 1), color=p.wood.dark).at(1, 0, 0))
        return Model([body, flash], clips=[clip("fire", 0.1 * 3, [visible("flash",
                     [(0, 1), (0.3, 0)])])])  # fmt: skip

    a = built(fn, size=(2, 1, 1), allow_floating=True)
    g = GLTF2.load_from_bytes(encode_glb(a, P.rgba_table(), voxel_size_m=1))
    for anim in g.animations:
        for s in anim.samplers:
            acc = g.accessors[s.input]
            view = g.bufferViews[acc.bufferView]
            t = np.frombuffer(g.binary_blob(), np.float32, acc.count, view.byteOffset)
            assert np.all(np.diff(t) > 0)


def test_default_owners_skip_hidden_parts():
    def fn(p):
        body = Part("body", box((1, 1, 4), color=p.wood.dark), pivot=(0.5, 0, 2))
        flash = Part(
            "flash", box((1, 1, 1), color=p.fire.light).at(0, 0, 4), parent="body", hidden=True
        )
        return Model([body, flash],
                     emitters={"sparks": emitter((0.5, 0.5, 5.5), "spark", burst=3)},
                     lights={"glow": light((0.5, 0.5, 5.5), p.fire.light)})  # fmt: skip

    a = built(fn, size=(1, 1, 6))
    assert nearest_part(a, (0.5, 0.5, 5.5)) == "body"


def test_an_empty_flipbook_frame_is_a_clear_error():
    def fn(p):
        on = box((1, 2, 1), color=p.fire.mid)
        stick = Part("stick", box((1, 2, 1), color=p.wood.dark).at(0, 2, 0))
        return Model([stick, Part("flame", frames=[on, on - box((2, 3, 2))])])

    with pytest.raises(ModelError, match="empty"):
        built(fn)


def test_frames_may_be_any_iterable():
    a = built(lambda p: Model([Part("flame", frames=(f for f in flames(p)))]))
    assert len(a.part("flame").frames) == 3


def test_part_names_may_not_use_the_frame_separator():
    with pytest.raises(ModelError, match="@"):
        Model([Part("arm@2", box((1, 1, 1)))])


def test_particle_models_load_frame_zero_only(tmp_path):
    from cubist.io.vox import write_vox
    from cubist.motion.particles import load_particles

    def ember(p):
        return Model([Part("ember", frames=[box((1, 1, 1), color=p.fire.light),
                                            box((1, 5, 1), color=p.fire.light)])])  # fmt: skip

    e = asset(name="ember", size=(1, 5, 1))(ember).build(P)
    (tmp_path / "ember").mkdir()
    write_vox(tmp_path / "ember" / "ember.vox", e.to_vox_scene())
    user = built(lambda p: Model([Part("b", box((1, 1, 1), color=p.wood.dark))],
                                 emitters={"e": emitter((0.5, 1, 0.5), "ember", rate=1)}),
                 size=(1, 1, 1))  # fmt: skip
    assert load_particles(user, tmp_path)["ember"].count() == 1


def test_particle_previews_follow_the_build_seed():
    def fn(p):
        return Model([Part("b", box((1, 1, 1), color=p.wood.dark))],
                     emitters={"e": emitter((0.5, 1, 0.5), "x", rate=4, spread=60)})  # fmt: skip

    one = asset(name="t", size=(1, 1, 1))(fn).build(P, seed=1)
    two = asset(name="t", size=(1, 1, 1))(fn).build(P, seed=2)
    a = [p.position.tolist() for p in particles_at(one, None, 1.0)]
    b = [p.position.tolist() for p in particles_at(two, None, 1.0)]
    assert a != b


def test_fracture_merges_slivers_even_with_an_island_present():
    with building(P, 0):
        shape = box((12, 6, 6), color=P.wood.mid) | box((1, 1, 1), color=P.wood.mid).at(30, 0, 0)
        pieces = fracture(shape, 8, seed=3, min_voxels=30)
    sizes = sorted(p.voxelize().count() for p in pieces)
    assert sizes[0] == 1 and all(s >= 30 for s in sizes[1:])


def test_glow_errors_explain_dotted_keys(tmp_path):
    path = tmp_path / "style.toml"
    path.write_text('[palette]\nfire = ["#d9480f", "#ffe066"]\n[glow]\nfire.light = 1.5\n')
    with pytest.raises(StyleError, match="quote"):
        load_style(path)
    path.write_text('[palette]\nfire = ["#d9480f", "#ffe066"]\n[glow]\n"fire.light.x" = 1\n')
    with pytest.raises(StyleError, match="fire.light.x"):
        load_style(path)


def test_fast_slides_are_sampled_finely_enough_in_the_glb():
    from cubist import slide
    from cubist.io.gltf import animation_samples

    def fn(p):
        body = Part("body", box((2, 2, 2), color=p.wood.dark), pivot=(1, 0, 1))
        sl = Part(
            "slide",
            box((2, 1, 4), color=p.wood.light).at(0, 2, 0),
            parent="body",
            pivot=(1, 2, 2),
            joint=slide("-z", 3),
        )
        return Model([body, sl], clips=[clip("fire", 0.3, {"slide": [(0, 0), (0.04, 3),
                                                                      (0.3, 0)]})])  # fmt: skip

    a = built(fn, size=(2, 3, 4))
    s = animation_samples(a, a.clips[0], voxel_size_m=1)["slide"]
    steps = np.linalg.norm(np.diff(s["translation"], axis=0), axis=1)
    assert steps.max() <= 0.5 + 1e-9


def demo_glbs():
    style = load_project_style(DEMO / "style.toml")
    for script in sorted((DEMO / "assets").glob("*.py")):
        [a] = [v for v in runpy.run_path(str(script)).values() if isinstance(v, AssetDef)]
        b = a.build(style.palette, glow=style.glow, cycles=style.cycles)
        yield script.stem, GLTF2.load_from_bytes(encode_glb(b, style.palette.rgba_table()))


@pytest.mark.parametrize("name,g", list(demo_glbs()), ids=lambda v: v if isinstance(v, str) else "")
def test_demo_glbs_are_structurally_valid(name, g):
    for anim in g.animations:
        assert anim.channels and anim.samplers, name
        targets = [(c.target.node, c.target.path) for c in anim.channels]
        assert len(targets) == len(set(targets)), (name, anim.name)
        for s in anim.samplers:
            acc_in, acc_out = g.accessors[s.input], g.accessors[s.output]
            assert acc_in.count == acc_out.count and acc_in.count >= 1
            view = g.bufferViews[acc_in.bufferView]
            t = np.frombuffer(g.binary_blob(), np.float32, acc_in.count, view.byteOffset)
            assert np.all(np.diff(t) > 0), (name, anim.name)
            assert acc_in.min is not None and acc_in.max is not None


def test_step_keys_hold_their_values_exactly_at_their_times():
    """float32 rounds 0.06 s down to 0.0599999...; the value there must still be the step's."""
    from test_gltf_anim import accessor

    def fn(p):
        body = Part("body", box((1, 1, 1), color=p.wood.dark))
        flash = Part("flash", box((1, 1, 1), color=p.fire.light).at(0, 0, 1), parent="body",
                     hidden=True)  # fmt: skip
        return Model([body, flash], clips=[clip("fire", 0.35, [visible("flash",
                     [(0, 1), (0.06, 0)])])])  # fmt: skip

    a = built(fn, size=(1, 1, 2))
    g = GLTF2.load_from_bytes(encode_glb(a, P.rgba_table(), voxel_size_m=1))
    [ch] = g.animations[0].channels
    s = g.animations[0].samplers[ch.sampler]
    times = accessor(g, s.input)
    values = accessor(g, s.output)
    k = int(np.argmin(np.abs(times - 0.06)))
    assert values[k][0] == 0.0  # hidden from 0.06 s on
