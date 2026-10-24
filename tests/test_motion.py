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


# --- clips ------------------------------------------------------------------

from cubist import Track, clip  # noqa: E402
from cubist.core.palette import Palette  # noqa: E402
from cubist.io.vox import decode_vox, encode_vox  # noqa: E402
from cubist.motion.clips import Clip  # noqa: E402


def test_clip_spreads_plain_values_evenly():
    c = clip("open", 0.6, {"lid": [0, 115, 105]})
    [t] = c.tracks
    assert t.part == "lid" and t.channel == "joint"
    assert t.times == pytest.approx((0.0, 0.3, 0.6)) and t.values == (0, 115, 105)


def test_one_value_is_a_constant_at_zero():
    [t] = clip("hold", 1.0, {"lid": [45]}).tracks
    assert t.times == (0.0,) and t.values == (45.0,)


def test_clip_accepts_time_value_pairs():
    [t] = clip("x", 1.0, {"lid": [(0, 0), (0.25, 90)]}).tracks
    assert t.times == (0.0, 0.25) and t.values == (0.0, 90.0)


def test_clip_accepts_track_objects():
    tr = Track("lid", "joint", (0.0, 1.0), (0.0, 90.0), ease="linear")
    c = clip("x", 1.0, [tr])
    assert c.tracks == (tr,) and c.track_ease(tr) == "linear"


def test_track_ease_defaults_to_the_clips():
    c = clip("x", 1.0, {"lid": [0, 1]}, ease="step")
    assert c.track_ease(c.tracks[0]) == "step"


def test_clip_name_must_be_identifier():
    with pytest.raises(ModelError, match="letters, digits"):
        clip("open lid!", 1.0, {"lid": [0, 90]})


def test_clip_rejects_nonpositive_length_and_unknown_ease():
    with pytest.raises(ModelError, match="length"):
        clip("a", 0, {"lid": [0]})
    with pytest.raises(ModelError, match="ease"):
        clip("a", 1, {"lid": [0, 1]}, ease="bouncy")


def test_keys_must_be_in_order_and_inside_the_clip():
    with pytest.raises(ModelError, match="order"):
        clip("a", 1, {"lid": [(0.5, 0), (0.2, 1)]})
    with pytest.raises(ModelError, match="inside"):
        clip("a", 1, {"lid": [(0, 0), (1.5, 1)]})
    with pytest.raises(ModelError, match="no keys"):
        clip("a", 1, {"lid": []})


def test_events_are_sorted_time_name_pairs():
    c = clip("walk", 1.0, {"lid": [0, 1]}, events=[(0.5, "step"), (0.0, "step")])
    assert c.events == ((0.0, "step"), (0.5, "step"))
    with pytest.raises(ModelError, match="event"):
        clip("walk", 1.0, {"lid": [0, 1]}, events=[(2.0, "late")])


def test_clip_round_trips_through_dict():
    c = clip("open", 0.6, {"lid": [0, 115, 105]}, loop=True, events=[(0.1, "creak")])
    assert Clip.from_dict(c.to_dict()) == c


def _chest_model(clips, color=None):
    body = Part("body", box((6, 3, 4), color=color), pivot=(3, 0, 2))
    lid = Part(
        "lid",
        box((6, 2, 4), color=color).at(0, 3, 0),
        parent="body",
        pivot=(3, 3, 0),
        joint=hinge("-x", (0, 120)),
    )
    return Model([body, lid], clips=clips)


def test_model_rejects_track_on_unknown_part_or_part_without_joint():
    body = Part("body", box((4, 4, 4)))
    with pytest.raises(ModelError, match="unknown part 'lid'"):
        Model([body], clips=[clip("a", 1, {"lid": [0, 1]})])
    with pytest.raises(ModelError, match="has no joint"):
        Model([body], clips=[clip("a", 1, {"body": [0, 1]})])


def test_model_rejects_duplicate_clip_names():
    with pytest.raises(ModelError, match="duplicate clip"):
        _chest_model([clip("open", 1, {"lid": [0, 90]}), clip("open", 1, {"lid": [0, 45]})])


def test_model_clips_must_be_clips():
    with pytest.raises(ModelError, match="clip"):
        _chest_model(["open"])


def _built(clips):
    from cubist.core.model import AssetDef

    palette = Palette({"wood": ["#3b2414", "#5e3a1f"]})
    a = AssetDef(lambda p: _chest_model(clips, p.wood.mid), "chest", (6, 5, 4), ())
    return a.build(palette), palette


def test_built_asset_carries_joints_and_clips_through_summary_and_vox():
    from cubist.core.model import BuiltAsset

    clips = [clip("open", 0.6, {"lid": [0, 115, 105]})]
    built, palette = _built(clips)
    assert built.parts[1].joint == hinge("-x", (0, 120)) and built.clips == clips
    assert built.is_animated()
    summary = built.summary()
    assert summary["parts"][1]["joint"] == {
        "type": "hinge",
        "axis": [-1.0, 0.0, 0.0],
        "limits": [0.0, 120.0],
    }
    assert len(summary["parts"][0]["voxel_sha256"]) == 64
    scene = decode_vox(encode_vox(built.to_vox_scene()))
    again = BuiltAsset.from_vox(scene, summary, palette)
    assert again.parts[1].joint == built.parts[1].joint and again.clips == built.clips


def test_static_asset_summary_has_empty_clips_and_no_joints():
    built, _ = _built(None)
    assert built.summary()["clips"] == []
    built.parts[1].joint = None
    built.clips = []
    assert not built.is_animated()
