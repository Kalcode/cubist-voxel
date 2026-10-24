"""Joints and clips: how a script says what moves."""

import pytest

from cubist import Model, ModelError, Part, box, hinge, slide, spin
from cubist.motion.joints import Joint

# --- joints -----------------------------------------------------------------


def test_hinge_parses_signed_axis_and_limits():
    j = hinge("-x", limits=(0, 120))
    assert j.kind == "hinge" and j.axis == (-1.0, 0.0, 0.0) and j.limits == (0.0, 120.0)


def test_slide_limits_run_from_zero_to_distance():
    assert slide("+z", distance=8).limits == (0.0, 8.0)


def test_spin_has_no_limits():
    j = spin("z", rps=6)
    assert j.limits is None and j.rps == 6


def test_vector_axis_is_normalized():
    assert hinge((0, 3, 4), (0, 90)).axis == pytest.approx((0.0, 0.6, 0.8))


def test_joint_round_trips_through_dict():
    for j in (hinge((0, 1, 1), (-90, 90)), slide("y", 3), spin("x"), spin("-z", rps=2.5)):
        assert Joint.from_dict(j.to_dict()) == j


def test_bad_axis_says_what_is_allowed():
    with pytest.raises(ModelError, match=r"'x', '-y', '\+z' or a vector"):
        hinge("sideways", (0, 90))
    with pytest.raises(ModelError, match="non-zero"):
        spin((0, 0, 0))


def test_hinge_limits_must_be_increasing():
    with pytest.raises(ModelError, match="limits"):
        hinge("x", (90, 0))


def test_slide_distance_must_be_positive():
    with pytest.raises(ModelError, match="distance"):
        slide("z", 0)


def test_part_carries_its_joint():
    j = hinge("x", (0, 90))
    assert Part("lid", box((2, 2, 2)), joint=j).joint == j


def test_part_joint_must_be_a_joint():
    with pytest.raises(ModelError, match="hinge"):
        Model([Part("lid", box((2, 2, 2)), joint="x")])


def test_mirrored_part_reflects_joint_as_rotation_axis():
    # A rotation axis is a pseudovector: mirroring across x keeps an x axis and flips y and z,
    # so mirrored legs swing forward together and mirrored turns go opposite ways.
    arm = Part("arm_r", box((2, 2, 2)), pivot=(1, 1, 1), joint=hinge("x", (-45, 45)))
    _, twin = arm.mirrored("arm_l", about_x=8)
    assert twin.joint.axis == (1.0, 0.0, 0.0)
    turn = Part("t", box((2, 2, 2)), joint=hinge("y", (0, 90))).mirrored("u", about_x=8)[1]
    assert turn.joint.axis == (0.0, -1.0, 0.0)


def test_mirrored_part_reflects_slide_as_direction():
    p = Part("d", box((2, 2, 2)), joint=slide("+x", 4)).mirrored("e", about_x=8)[1]
    assert p.joint.axis == (-1.0, 0.0, 0.0)
