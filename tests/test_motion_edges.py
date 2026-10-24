"""Edge cases from the M7 review: each test pins a bug found by an independent reviewer."""

import json
import math
import textwrap

import numpy as np
import pytest
from PIL import Image
from pygltflib import GLTF2

from cubist import Model, ModelError, Part, Track, asset, box, clip, hinge, spin
from cubist.build.runner import build_script
from cubist.core.palette import Palette
from cubist.export import export_asset
from cubist.io.gltf import encode_glb
from cubist.io.vox import read_vox, write_vox
from cubist.lint.motion import check_motion
from cubist.motion.pose import dense_times, track_value, world_matrices
from cubist.render.filmstrip import clip_frames, frame_times, motion_region
from cubist.render.posed import PosedScene

P = Palette({"wood": ["#3b2414", "#5e3a1f", "#8a5a32"]})


def build(fn, size=None, name="t"):
    return asset(name=name, size=size)(fn).build(P)


def chest(p, clips, extra=()):
    body = Part(
        "body", box((14, 7, 10), color=p.wood.mid) - box((12, 6, 8)).at(1, 1, 1), pivot=(7, 0, 5)
    )
    lid = Part(
        "lid",
        box((14, 5, 10), color=p.wood.light).at(0, 7, 0),
        parent="body",
        pivot=(7, 7, 0),
        joint=hinge("-x", (0, 120)),
    )
    return Model([body, lid, *extra], clips=clips)


def glb(a):
    return GLTF2.load_from_bytes(encode_glb(a, P.rgba_table(), voxel_size_m=1))


# 1: step-only clips must last the whole clip in glTF


def test_step_only_clip_lasts_its_length_in_glb():
    a = build(lambda p: chest(p, [clip("pop", 1.0, {"lid": [(0, 0), (0.25, 60)]}, ease="step")]))
    g = glb(a)
    for anim in g.animations:
        for s in anim.samplers:
            assert g.accessors[s.input].max == [1.0]


# 3: a renamed or removed part must give the hand-edit error, not a crash


@pytest.fixture
def project(tmp_path):
    (tmp_path / "style.toml").write_text(
        '[project]\nvoxel_size_m = 0.25\n[palette]\nwood = ["#3b2414", "#5e3a1f", "#8a5a32"]\n'
    )
    (tmp_path / "assets").mkdir()
    script = tmp_path / "assets" / "chest.py"
    script.write_text(
        textwrap.dedent("""\
        from cubist import *

        @asset(name="chest", size=(8, 7, 6))
        def chest(p):
            return Model(
                [Part("body", box((8, 4, 6), color=p.wood.mid), pivot=(4, 0, 3)),
                 Part("lid", box((8, 2, 6), color=p.wood.light).at(0, 4, 0), parent="body",
                      pivot=(4, 4, 0), joint=hinge("-x", (0, 120)))],
                clips=[clip("open", 0.5, {"lid": [0, 100]})],
            )
        """)
    )
    return tmp_path


def test_renamed_part_in_vox_gives_hand_edit_error_not_a_crash(project):
    from cubist.cli import main

    [r] = build_script(project / "assets" / "chest.py")
    scene = read_vox(r.vox_path)
    scene.parts[1].name = "lid_edited"
    write_vox(r.vox_path, scene)
    assert main(["lint", str(r.vox_path.parent)]) == 2
    out = export_asset(r.vox_path.parent, force=True)  # forced: exports without the lost motion
    assert "glb" in out


# 4: fast spins need denser samples for the canvas and for lint


def plate(p, turns=6):
    blade = Part(
        "plate", box((8, 8, 1), color=p.wood.mid).at(2, 2, 0), pivot=(6, 6, 0.5), joint=spin("z")
    )
    return Model([blade], clips=[clip("spin", 1, {"plate": [0, turns]}, loop=True, ease="linear")])


def test_dense_times_keep_turns_small():
    a = build(plate)
    ts = dense_times(a, a.clips[0], rate=24, max_turn=10)
    turns = np.diff([track_value(a.clips[0].tracks[0], t, a.clips[0]) for t in ts]) * 360
    assert np.abs(turns).max() <= 10 + 1e-9


def test_fast_spin_canvas_covers_every_gif_frame():
    a = build(plate)
    scene = PosedScene(a, P.rgba_table())
    region = motion_region(a, a.clips, scene)
    frames, _ = clip_frames(a, P.rgba_table(), a.clips[0], fps=20)
    big = ((-20, -20, -20), (40, 40, 40))
    for k in range(len(frames)):
        t = k * 0.05
        m = world_matrices(a, a.clips[0], t)
        assert (
            scene.render(m, "iso", 4, region).mask.sum()
            == scene.render(m, "iso", 4, big).mask.sum()
        )


def test_fast_spin_reach_is_reported():
    f = [x for x in check_motion(build(plate)) if x.check == "motion_bounds"][0]
    assert "stays in the rest box" not in f.message


# 5: parts that move together rigidly don't collide


def test_attached_child_of_moving_part_does_not_collide():
    def fn(p):
        trim = Part("trim", box((14, 1, 10), color=p.wood.dark).at(0, 12, 0), parent="lid")
        return chest(p, [clip("open", 0.6, {"lid": [0, 110]})], extra=[trim])

    a = build(fn, (14, 13, 10))
    assert not [f for f in check_motion(a) if f.check == "motion_collision"]


# 6: two-key smooth loops keep their speed through the seam


def test_two_key_smooth_loop_runs_at_constant_speed():
    c = clip("spin", 1.0, {"prop": [0, 2]}, loop=True)
    v = [track_value(c.tracks[0], t, c) for t in (0.05, 0.5, 0.95)]
    assert v == pytest.approx([0.1, 1.0, 1.9])


# 7: one joint track per part; mixed step tracks keep their jumps in glTF


def test_two_joint_tracks_on_one_part_are_an_error():
    with pytest.raises(ModelError, match="two joint tracks"):
        clip(
            "x",
            1,
            [Track("lid", "joint", (0, 1), (0, 100)), Track("lid", "joint", (0, 1), (0, 20))],
        )


def test_step_track_mixed_with_smooth_keeps_its_jump_in_glb():
    from cubist import turn

    c = clip(
        "x", 1.0, [turn("lid", "y", [(0, 0), (0.5, 30)], ease="step"), turn("lid", "z", [0, 10])]
    )
    a = build(lambda p: chest(p, [c]))
    g = glb(a)
    times = np.frombuffer(g.binary_blob(), np.float32,
                          count=g.accessors[g.animations[0].samplers[0].input].count,
                          offset=g.bufferViews[g.accessors[g.animations[0].samplers[0].input]
                                               .bufferView].byteOffset)  # fmt: skip
    gaps = np.diff(times)
    assert gaps.min() < 0.002  # a key just before the step at t = 0.5


# 8: manifests list parents before children, the same at build and export


def test_build_and_export_manifests_agree_parents_first(project):
    script = project / "assets" / "chest.py"
    script.write_text(
        script.read_text().replace(
            '[Part("body"',
            '[Part("knob", box((1, 1, 1), color=p.wood.dark).at(4, 6, 2), '
            'parent="lid"),\n                 Part("body"',
        )
    )
    [r] = build_script(script)
    built = json.loads((r.vox_path.parent / "manifest.json").read_text())
    names = [p["name"] for p in built["parts"]]
    assert names.index("body") < names.index("lid") < names.index("knob")
    exported = json.loads(open(export_asset(r.vox_path.parent)["manifest"]).read())
    exported.pop("files")
    assert exported == built


# 9: GIFs play at real speed


def test_gif_durations_add_up_to_the_clip(tmp_path):
    a = build(
        lambda p: chest(
            p,
            [
                clip("open", 0.6, {"lid": [0, 100]}),
                clip("spin", 0.35, {"lid": [0, 90, 0]}, loop=True),
            ],
        )
    )
    for c in a.clips:
        frames, durations = clip_frames(a, P.rgba_table(), c)
        assert all(d % 10 == 0 for d in durations)  # GIFs store centiseconds
        moving = sum(durations) - (0 if c.loop else durations[-1])
        assert moving == pytest.approx(c.length * 1000, abs=10)


# 10, 11: times stay strictly increasing in float32; non-finite values are refused


def test_glb_times_strictly_increase_in_float32():
    c = clip("long", 20.0, {"lid": [(0, 0), (10.0000001, 50), (20, 0)]})
    g = glb(build(lambda p: chest(p, [c])))
    acc = g.accessors[g.animations[0].samplers[0].input]
    view = g.bufferViews[acc.bufferView]
    t = np.frombuffer(g.binary_blob(), np.float32, count=acc.count, offset=view.byteOffset)
    assert np.all(np.diff(t) > 0)


def test_non_finite_keys_and_lengths_are_refused():
    with pytest.raises(ModelError, match="finite"):
        clip("a", math.inf, {"lid": [0, 1]})
    with pytest.raises(ModelError, match="finite"):
        clip("a", 1, {"lid": [(math.nan, 0)]})


def test_frame_times_unchanged():
    assert frame_times(clip("o", 0.6, {"p": [0, 1]}), 4) == pytest.approx([0, 0.2, 0.4, 0.6])
    assert Image  # PIL stays importable for the GIF tests


# 12: forced exports say what they forced past; MCP builds report GIF paths


def test_forced_export_warns_on_stderr(project, capsys):
    from cubist.cli import main

    [r] = build_script(project / "assets" / "chest.py")
    scene = read_vox(r.vox_path)
    scene.parts[1].grid.data[0, 0, 0] = 0
    write_vox(r.vox_path, scene)
    capsys.readouterr()
    assert main(["export", str(r.vox_path.parent), "--force"]) == 0
    assert "warning: exported despite 1 lint error" in capsys.readouterr().err


def test_mcp_build_lists_gif_paths(project):
    import anyio
    from mcp.client import Client

    from cubist.mcp_server import create_server

    async def go():
        async with Client(create_server(project)) as client:
            return await client.call_tool("build_asset", {"script_path": "assets/chest.py"})

    text = anyio.run(go).content[0].text
    assert "open.gif" in text
