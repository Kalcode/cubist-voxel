"""The pose evaluator: clips -> per-part transforms, shared by previews, lint, and export."""

import numpy as np
import pytest

from cubist import Model, Part, box, clip, hinge, slide, spin
from cubist.core.model import AssetDef
from cubist.core.palette import Palette
from cubist.motion.pose import (
    joint_motion,
    local_motion,
    moving_parts,
    sample_times,
    track_value,
    world_matrices,
)

P = Palette({"wood": ["#3b2414", "#5e3a1f"]})


def built(fn, size=(16, 16, 16)):
    return AssetDef(fn, "t", size, ()).build(P)


def apply(m, point):
    return (m @ np.array([*point, 1.0]))[:3]


def test_linear_and_step_interpolation():
    c = clip("a", 1.0, {"lid": [0, 100]}, ease="linear")
    assert track_value(c.tracks[0], 0.25, c) == pytest.approx(25)
    s = clip("a", 1.0, {"lid": [0, 100]}, ease="step")
    assert track_value(s.tracks[0], 0.99, s) == 0
    assert track_value(s.tracks[0], 1.0, s) == 100


def test_smooth_eases_from_rest_and_never_overshoots_keys():
    c = clip("a", 1.0, {"lid": [0, 115, 105]})
    vs = [track_value(c.tracks[0], t, c) for t in np.linspace(0, 1, 201)]
    assert max(vs) == pytest.approx(115, abs=1e-9) and min(vs) >= 0
    assert track_value(c.tracks[0], 0.01, c) < 1.15  # slope 0 at the start
    assert track_value(c.tracks[0], 0.5, c) == pytest.approx(115)


def test_smooth_loop_keeps_speed_through_the_middle_key():
    # 0 -> 25 -> 0 -> -25 -> 0: fastest at the zero crossing, not stopped there
    c = clip("w", 1.0, {"leg": [0, 25, 0, -25, 0]}, loop=True)

    def v(t):
        return track_value(c.tracks[0], t, c)

    assert abs(v(0.51) - v(0.49)) > 1.0
    assert abs(v(0.01) - v(0.99)) > 1.0  # and through the wrap
    assert v(0.25) == pytest.approx(25)


def test_times_outside_keys_hold_first_and_last_values():
    c = clip("a", 2.0, {"lid": [(0.5, 10), (1.0, 20)]}, ease="linear")
    assert track_value(c.tracks[0], 0.0, c) == 10
    assert track_value(c.tracks[0], 2.0, c) == 20


def test_single_key_is_constant():
    c = clip("a", 1.0, {"lid": [45]})
    assert track_value(c.tracks[0], 0.7, c) == 45


def test_hinge_rotates_about_pivot_by_right_hand_rule():
    trans, rot = joint_motion(hinge("x", (0, 90)), 90)
    assert np.allclose(trans, 0)
    assert np.allclose(rot @ [0, 1, 0], [0, 0, 1])  # +Y toward +Z


def test_slide_moves_along_direction():
    trans, rot = joint_motion(slide("-z", 8), 3)
    assert np.allclose(trans, [0, 0, -3]) and np.allclose(rot, np.eye(3))


def test_spin_value_is_turns():
    _, rot = joint_motion(spin("z"), 0.25)
    assert np.allclose(rot @ [1, 0, 0], [0, 1, 0])


def chest(p):
    body = Part("body", box((6, 3, 4), color=p.wood.mid), pivot=(3, 0, 2))
    lid = Part(
        "lid",
        box((6, 2, 4), color=p.wood.mid).at(0, 3, 0),
        parent="body",
        pivot=(3, 3, 0),
        joint=hinge("-x", (0, 120)),
    )
    return Model([body, lid], clips=[clip("open", 1.0, {"lid": [0, 90]}, ease="linear")])


def test_world_matrix_turns_lid_about_its_pivot():
    a = built(chest)
    m = world_matrices(a, a.clips[0], 1.0)
    assert np.allclose(m["body"], np.eye(4))
    # hinge("-x") at 90: the lid's front edge (z = 4) swings up over the hinge line
    assert np.allclose(apply(m["lid"], (3, 3, 4)), (3, 7, 0))
    assert np.allclose(apply(m["lid"], (3, 3, 0)), (3, 3, 0))  # on the hinge line


def test_child_follows_moving_parent():
    def desk(p):
        top = Part("desk", box((8, 4, 8), color=p.wood.mid), pivot=(4, 0, 4))
        drawer = Part(
            "drawer",
            box((6, 2, 6), color=p.wood.dark).at(1, 1, 1),
            parent="desk",
            pivot=(4, 1, 7),
            joint=slide("+z", 4),
        )
        knob = Part(
            "knob",
            box((2, 1, 1), color=p.wood.dark).at(3, 1, 7),
            parent="drawer",
            pivot=(4, 1, 7),
            joint=hinge("x", (0, 90)),
        )
        return Model(
            [top, drawer, knob],
            clips=[clip("pull", 1.0, {"drawer": [0, 4], "knob": [0, 90]}, ease="linear")],
        )

    a = built(desk)
    c = a.clips[0]
    m = world_matrices(a, c, 1.0)
    local = local_motion(a, c, 1.0)
    assert np.allclose(local["drawer"][0], [0, 0, 4])
    # the knob turns 90 about x at its pivot, then rides the drawer 4 forward
    assert np.allclose(apply(m["knob"], (4, 2, 7)), (4, 1, 12))
    assert moving_parts(a, c) == ["drawer", "knob"]


def test_moving_parts_include_children_of_moving_parts():
    def m(p):
        base = Part("base", box((4, 2, 4), color=p.wood.mid))
        arm = Part(
            "arm",
            box((1, 4, 1), color=p.wood.mid).at(1, 2, 1),
            parent="base",
            pivot=(1, 2, 1),
            joint=hinge("z", (-90, 90)),
        )
        hand = Part("hand", box((1, 1, 1), color=p.wood.mid).at(1, 6, 1), parent="arm")
        return Model([base, arm, hand], clips=[clip("wave", 1, {"arm": [0, 30, 0]})])

    a = built(m)
    assert moving_parts(a, a.clips[0]) == ["arm", "hand"]


def test_sample_times_include_keys_and_end():
    c = clip("a", 0.5, {"lid": [(0, 0), (0.123, 10), (0.5, 0)]})
    ts = sample_times(c, rate=30)
    assert 0.123 in ts and ts[0] == 0 and ts[-1] == 0.5
    assert np.all(np.diff(ts) > 0)
