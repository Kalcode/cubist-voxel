"""Filmstrips, GIFs and posed contact sheets: the agent's eyes for motion."""

import numpy as np
import pytest
from PIL import Image

from cubist import Model, Part, box, clip, hinge, slide, spin
from cubist.core.model import AssetDef
from cubist.core.palette import Palette
from cubist.motion.pose import world_matrices
from cubist.render.contact_sheet import contact_sheet
from cubist.render.filmstrip import (
    clip_frames,
    filmstrip,
    frame_times,
    motion_region,
    motion_view,
    posed_sheet,
    save_gif,
)
from cubist.render.posed import PosedScene

P = Palette({"wood": ["#3b2414", "#5e3a1f", "#8a5a32"], "metal": ["#4b505a", "#c2c8d0"]})
TABLE = P.rgba_table()


def build(fn, size=None, name="t"):
    return AssetDef(fn, name, size, ()).build(P)


def chest(p, clips=None):
    body = Part("body", box((8, 4, 6), color=p.wood.mid), pivot=(4, 0, 3))
    lid = Part(
        "lid",
        box((8, 2, 6), color=p.wood.light).at(0, 4, 0),
        parent="body",
        pivot=(4, 4, 0),
        joint=hinge("-x", (0, 120)),
    )
    if clips is None:
        clips = [clip("open", 0.6, {"lid": [0, 110, 100]})]
    return Model([body, lid], clips=clips)


def door(p):
    leaf = box((8, 12, 1), color=p.wood.mid)
    part = Part("door", leaf, pivot=(0, 0, 0.5), joint=hinge("y", (-110, 0)))
    return Model([part], clips=[clip("open", 1.0, {"door": [0, -100]})])


def test_motion_view_follows_the_joint_axis():
    assert motion_view(build(chest), build(chest).clips[0]) == "right"

    def prop(p):
        blade = Part(
            "prop", box((6, 1, 1), color=p.wood.mid), pivot=(3, 0.5, 0.5), joint=spin("z", rps=4)
        )
        return Model([blade], clips=[clip("spin", 1.0, {"prop": [0, 4]}, ease="linear")])

    a = build(prop)
    assert motion_view(a, a.clips[0]) == "front"

    def drawer(p):
        d = Part("drawer", box((6, 2, 6), color=p.wood.mid), joint=slide("+z", 4))
        return Model([d], clips=[clip("open", 1.0, {"drawer": [0, 4]})])

    a = build(drawer)
    assert motion_view(a, a.clips[0]) == "right"


def test_motion_view_counts_free_tracks():
    from cubist import bob, swing

    def walker(p):
        body = Part("body", box((4, 6, 2), color=p.wood.mid))
        leg = Part("leg", box((1, 4, 1), color=p.wood.dark), parent="body")
        return Model(
            [body, leg], clips=[clip("walk", 1, [swing("leg", "x", 20), bob("body", 1)], loop=True)]
        )

    a = build(walker)
    assert motion_view(a, a.clips[0]) == "right"


def test_loop_frame_times_repeat_the_first_pose_at_the_end():
    c = clip("w", 0.8, {"p": [0, 1, 0]}, loop=True)
    assert frame_times(c, 4) == pytest.approx([0, 0.2, 0.4, 0.6, 0.8])
    once = clip("o", 0.6, {"p": [0, 1]})
    assert frame_times(once, 4) == pytest.approx([0, 0.2, 0.4, 0.6])


def test_filmstrip_has_one_block_per_clip():
    one = build(chest)
    two = build(
        lambda p: chest(
            p, [clip("open", 0.6, {"lid": [0, 100]}), clip("slam", 0.3, {"lid": [100, 0]})]
        )
    )
    a, b = filmstrip(one, TABLE, one.clips), filmstrip(two, TABLE, two.clips)
    assert a.width == b.width and b.height > a.height * 1.6


def test_filmstrip_is_deterministic():
    a = build(chest)
    x = np.asarray(filmstrip(a, TABLE, a.clips))
    y = np.asarray(filmstrip(a, TABLE, a.clips))
    assert np.array_equal(x, y)


def test_filmstrip_canvas_covers_negative_swing():
    a = build(door)
    scene = PosedScene(a, TABLE)
    region = motion_region(a, a.clips, scene)
    assert region[0][0] < 0
    big = ((-20, -20, -20), (20, 20, 20))
    for t in frame_times(a.clips[0], 8):
        m = world_matrices(a, a.clips[0], t)
        inside = scene.render(m, "iso", 4, region).mask.sum()
        assert inside == scene.render(m, "iso", 4, big).mask.sum()  # nothing cut off


def test_gif_is_deterministic_and_holds_last_frame(tmp_path):
    a = build(chest)
    frames, durations = clip_frames(a, TABLE, a.clips[0], fps=10)
    assert len(frames) == 7  # 0.6 s at 10 fps, both ends included
    assert durations[-1] == durations[0] + 600
    save_gif(frames, durations, tmp_path / "a.gif")
    save_gif(frames, durations, tmp_path / "b.gif")
    assert (tmp_path / "a.gif").read_bytes() == (tmp_path / "b.gif").read_bytes()
    with Image.open(tmp_path / "a.gif") as im:
        assert im.n_frames == 7 and im.info["loop"] == 0


def test_looping_gif_does_not_repeat_the_wrap_frame():
    def prop(p):
        blade = Part("prop", box((6, 1, 1), color=p.wood.mid), pivot=(3, 0.5, 0.5), joint=spin("z"))
        return Model([blade], clips=[clip("spin", 1.0, {"prop": [0, 1]}, loop=True, ease="linear")])

    a = build(prop)
    frames, durations = clip_frames(a, TABLE, a.clips[0], fps=10)
    assert len(frames) == 10 and len(set(durations)) == 1


def test_gif_frames_share_one_size():
    a = build(door)
    frames, _ = clip_frames(a, TABLE, a.clips[0], fps=6)
    assert len({f.size for f in frames}) == 1


def test_posed_sheet_at_zero_matches_rest_layout():
    a = build(chest, size=(8, 6, 6))
    rest = contact_sheet(a.merged_grid(), P, name=a.name, size=a.size)
    posed = posed_sheet(a, TABLE, a.clips[0], 0.0, names={c.index: c.name for c in P.colors()})
    assert posed.size[0] >= rest.size[0] - 2 and abs(posed.size[1] - rest.size[1]) <= 16


def test_posed_sheet_shows_open_lid_above_the_size_box():
    a = build(chest, size=(8, 6, 6))
    closed = posed_sheet(a, TABLE, a.clips[0], 0.0)
    opened = posed_sheet(a, TABLE, a.clips[0], 0.3)
    assert opened.size[1] > closed.size[1]  # the raised lid needs more room
