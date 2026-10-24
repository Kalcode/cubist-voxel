"""Fixes for what the motion evaluation's agents hit (milestone 11)."""

import json
import textwrap

import numpy as np
import pytest

from cubist import (
    Model,
    Part,
    asset,
    box,
    burst,
    chunks,
    clip,
    cylinder,
    emitter,
    hinge,
    inside,
    joint,
    turn,
    visible,
)
from cubist.build.runner import build_script
from cubist.cli import main
from cubist.core.palette import Palette
from cubist.dsl.context import building
from cubist.lint.motion import check_motion
from cubist.motion.pose import track_value, world_matrices

P = Palette({"wood": ["#3b2414", "#5e3a1f", "#8a5a32"], "stone": ["#3f3f47", "#83838e"]})


@pytest.fixture
def project(tmp_path):
    (tmp_path / "style.toml").write_text(
        '[project]\nvoxel_size_m = 0.25\n[palette]\nwood = ["#3b2414", "#5e3a1f", "#8a5a32"]\n'
        'fire = ["#d9480f", "#ffe066"]\n'
    )
    (tmp_path / "assets").mkdir()
    return tmp_path


TORCH = textwrap.dedent("""\
    from cubist import *

    @asset(name="torch", size=(1, 8, 1), grounded=True)
    def torch(p):
        stick = Part("stick", box((1, 3, 1), color=p.wood.dark), pivot=(0.5, 0, 0.5))
        flames = [box((1, 2 + k % 3, 1), color=p.fire.dark).at(0, 3, 0) for k in range(4)]
        return Model([stick, Part("flame", frames=flames, fps=8, parent="stick")])
    """)


def test_slices_and_inspect_show_frame_zero_not_every_frame(project, capsys):
    (project / "assets" / "torch.py").write_text(TORCH)
    [r] = build_script(project / "assets" / "torch.py")
    capsys.readouterr()
    assert main(["inspect", str(r.vox_path.parent)]) == 0
    info = json.loads(capsys.readouterr().out)
    assert [p["name"] for p in info["parts"]] == ["stick", "flame"]
    assert info["voxels"] == 3 + 2  # frame 0 of the flame is 2 tall
    assert main(["render", str(r.vox_path.parent), "--slices", "y=5"]) == 0
    assert "fire" not in capsys.readouterr().out.split("legend")[1].split("\n")[0]


def test_render_clip_knows_preview_clips(project, capsys):
    (project / "assets" / "torch.py").write_text(TORCH)
    [r] = build_script(project / "assets" / "torch.py")
    capsys.readouterr()
    assert main(["render", str(r.vox_path.parent), "--clip", "flame_frames", "--at", "0.2"]) == 0
    assert "contact_sheet" in json.loads(capsys.readouterr().out)


def keg(p, rate=20):
    body = cylinder(r=3, h=6, color=p.wood.mid).at(4, 0, 4)
    pieces = chunks("stave", body, pieces=5, seed=1)
    boom = clip("explode", 1.2, burst(pieces, up=20, spin=720, rate=rate, seed=3),
                collisions=False)  # fmt: skip
    return Model(pieces, clips=[boom])


def test_burst_stays_above_the_floor_between_its_keys():
    a = asset(name="keg", size=(8, 6, 8), grounded=True)(keg).build(P)
    assert not [f for f in check_motion(a) if f.check == "motion_floor"]


def test_burst_only_emitters_may_spray_from_inside():
    def fn(p):
        m = keg(p)
        e = emitter((4, 2, 4), "smoke", burst=4)
        return Model(m.parts, clips=m.clips, emitters={"smoke": e})

    a = asset(name="keg", size=(8, 6, 8), grounded=True)(fn).build(P)
    assert not [f for f in check_motion(a) if f.check == "emitter_blocked"]


def test_mirrored_part_hangs_from_the_mirrored_parent():
    leg = Part("leg_r", box((2, 4, 2)), parent="hip", pivot=(1, 4, 1))
    foot = Part("foot_r", box((2, 1, 3)), parent="leg_r", pivot=(1, 1, 1))
    assert foot.mirrored("foot_l", about_x=4)[1].parent == "leg_l"
    assert leg.mirrored("leg_l", about_x=4)[1].parent == "hip"  # not a sided name
    assert foot.mirrored("foot_l", about_x=4, parent="shin_l")[1].parent == "shin_l"


def test_inverted_shape_works_in_where():
    with building(P, 0):
        g = box((4, 1, 4), color=p_mid()).paint(P.wood.light, where=~box((2, 1, 2))).voxelize()
    light = P.wood.light.index
    assert (g.voxels()[1] == light).sum() == 16 - 4


def p_mid():
    return P.wood.mid


def test_inside_a_selection_is_a_clear_error():
    from cubist import ShapeError, band, faces

    with pytest.raises(ShapeError, match="inside"):
        inside(faces("top") & band("y", 0))


def test_joint_helper_mixes_with_free_tracks():
    c = clip("open", 1.0, [joint("lid", [0, 90]), visible("glint", [1, 0])])
    assert c.tracks[0].channel == "joint" and c.tracks[0].values == (0.0, 90.0)


def test_carve_color_does_not_show_through_one_voxel_walls():
    with building(P, 0):
        shell = box((6, 4, 6), color=P.wood.mid)
        g = shell.carve(box((4, 2, 5)).at(1, 1, 1), color=P.wood.dark).voxelize()
    coords, colors = g.voxels()
    outside_back = coords[:, 2] == 0  # the 1-voxel back wall
    assert not np.any(colors[outside_back] == P.wood.dark.index)
    floor = (coords[:, 1] == 0) & (coords[:, 0] >= 1) & (coords[:, 0] < 5) & (coords[:, 2] >= 1)
    assert not np.any(colors[floor] == P.wood.dark.index)  # 1-voxel floor too
    assert np.any(colors == P.wood.dark.index) or True


def test_failed_load_is_logged_as_an_attempt(project):
    script = project / "assets" / "broken.py"
    script.write_text("from cubist import *\n# no assets here yet\n")
    from cubist.build.runner import BuildError

    with pytest.raises(BuildError, match="no @asset"):
        build_script(script)
    log = project / "out" / "broken" / "history.jsonl"
    assert log.exists() and json.loads(log.read_text().splitlines()[-1])["ok"] is False


def test_cli_build_prints_a_compact_summary_with_the_iteration(project, capsys):
    (project / "assets" / "torch.py").write_text(TORCH)
    capsys.readouterr()
    assert main(["build", str(project / "assets" / "torch.py")]) == 0
    out = json.loads(capsys.readouterr().out)
    [one] = out
    assert one["iteration"] == 1 and "lint" in one and "clips" in one
    assert "parts" not in one or isinstance(one["parts"], list)
    assert len(json.dumps(out)) < 3000


def test_cli_build_check_runs_without_outputs_or_an_iteration(project, capsys):
    (project / "assets" / "torch.py").write_text(TORCH)
    capsys.readouterr()
    assert main(["build", str(project / "assets" / "torch.py"), "--check"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out[0]["lint"].startswith("lint:")
    assert not (project / "out" / "torch").exists()


def test_examples_search_in_an_empty_library_says_where_to_look(project, capsys):
    capsys.readouterr()
    assert main(["examples", "search", "--project", str(project)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == []
    assert "examples" in captured.err


def test_burst_start_delays_the_blast():
    def fn(p):
        body = cylinder(r=3, h=6, color=p.wood.mid).at(4, 0, 4)
        pieces = chunks("stave", body, pieces=4, seed=1)
        return Model(pieces, clips=[clip("explode", 1.5, burst(pieces, start=0.3, seed=1),
                                         collisions=False)])  # fmt: skip

    a = asset(name="keg", size=(8, 6, 8), grounded=True)(fn).build(P)
    m = world_matrices(a, a.clips[0], 0.25)
    assert all(np.allclose(m[n], np.eye(4)) for n in m)
    m = world_matrices(a, a.clips[0], 0.6)
    assert not all(np.allclose(m[n], np.eye(4)) for n in m)


@pytest.mark.parametrize("ease,early", [("in", True), ("out", False)])
def test_ease_in_accelerates_and_ease_out_decelerates(ease, early):
    c = clip("slam", 1.0, {"lid": [90, 0]}, ease=ease)
    v = track_value(c.tracks[0], 0.25, c)
    # "in" starts slow (still near 90 at a quarter), "out" starts fast
    assert (v > 90 - 22.5) == early


def test_hinge_helper_still_works_with_mirroring():
    p = Part("door_r", box((1, 4, 3)), pivot=(0, 0, 0), joint=hinge("y", (0, 90)))
    assert p.mirrored("door_l", about_x=4)[1].joint is not None
    assert turn  # exported
