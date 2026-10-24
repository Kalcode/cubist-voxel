"""Generators turn intent ("swing the legs 25 degrees, opposite phase") into keys."""

import numpy as np
import pytest

from cubist import Model, Part, asset, bob, box, clip, shake, spin, spin_track, swing
from cubist.core.palette import Palette
from cubist.dsl.context import building
from cubist.lint.motion import check_motion

P = Palette({"wood": ["#3b2414", "#5e3a1f", "#8a5a32"]})


def test_swing_is_a_closed_sine():
    [t] = clip("w", 1.0, [swing("leg", "x", 25)], loop=True).tracks
    assert t.channel == "rotate_x" and len(t.values) == 9
    assert t.values[0] == pytest.approx(0) and t.values[-1] == pytest.approx(0, abs=1e-9)
    assert max(t.values) == pytest.approx(25) and min(t.values) == pytest.approx(-25)
    assert t.values[2] == pytest.approx(25)  # a quarter of the way in


def test_swing_phase_half_is_opposite():
    a, b = clip("w", 1.0, [swing("l", "x", 20), swing("r", "x", 20, phase=0.5)]).tracks
    assert np.allclose(a.values, -np.array(b.values), atol=1e-9)


def test_swing_cycles_add_samples_and_center_offsets():
    [t] = clip("w", 1.0, [swing("tail", "-y", 10, cycles=2, center=5)]).tracks
    assert t.channel == "rotate_y" and len(t.values) == 17
    assert t.values[0] == pytest.approx(-5)  # "-y" negates, center included


def test_bob_starts_at_rest_and_peaks_at_height():
    [t] = clip("w", 0.8, [bob("torso", -1)]).tracks
    assert t.channel == "move_y" and t.values[0] == pytest.approx(0)
    assert min(t.values) == pytest.approx(-1) and t.values[-1] == pytest.approx(0, abs=1e-9)
    # two cycles by default: lowest at a quarter and three quarters of the way
    assert len(t.values) == 17
    assert t.values[4] == pytest.approx(-1) and t.values[12] == pytest.approx(-1)


def test_shake_is_seeded_by_build():
    def keys(seed):
        with building(P, seed):
            return clip("fire", 0.5, [shake("gun", 4)]).tracks[0]

    a, b, c = keys(1), keys(1), keys(2)
    assert a == b and a.values != c.values
    assert a.values[0] == 0 and a.values[-1] == 0 and max(map(abs, a.values)) <= 4
    assert a.channel == "rotate_z" and a.ease == "linear" and len(a.times) == 7


def test_spin_track_on_joint_is_linear_turns():
    [t] = clip("s", 1.0, [spin_track("prop", 3)]).tracks
    assert t.channel == "joint" and t.values == (0.0, 3.0) and t.ease == "linear"
    [f] = clip("s", 1.0, [spin_track("top", 2, axis="y")]).tracks
    assert f.channel == "rotate_y" and f.values == (0.0, 720.0)


def test_generators_close_loops_without_lint_warning():
    def walker(p):
        torso = Part("torso", box((4, 4, 2), color=p.wood.mid).at(2, 5, 2), pivot=(4, 5, 3))
        leg = Part(
            "leg",
            box((1, 5, 1), color=p.wood.dark).at(3, 0, 2),
            parent="torso",
            pivot=(3.5, 5, 2.5),
        )
        prop = Part(
            "prop",
            box((3, 1, 1), color=p.wood.light).at(3, 9, 2),
            parent="torso",
            pivot=(4.5, 9.5, 2.5),
            joint=spin("y"),
        )
        walk = clip(
            "walk",
            0.8,
            [swing("leg", "x", 20), bob("torso", 0.5), spin_track("prop", 2)],
            loop=True,
        )
        return Model([torso, leg, prop], clips=[walk])

    a = asset(name="walker", size=(8, 10, 6))(walker).build(P)
    assert not [f for f in check_motion(a) if f.check == "loop_closure"]
