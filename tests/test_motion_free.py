"""Free rotate/move tracks on any part, track specs, travel, and the floor check."""

import json

import jsonschema
import numpy as np
import pytest
from pygltflib import GLTF2

from cubist import Model, Part, TrackSpec, asset, box, clip, hinge, move, turn
from cubist.core.palette import Palette
from cubist.io.gltf import encode_glb
from cubist.lint.motion import check_motion
from cubist.manifest import SCHEMA_PATH, build_manifest
from cubist.motion.pose import local_motion, world_matrices

P = Palette({"wood": ["#3b2414", "#5e3a1f", "#8a5a32"]})


def build(fn, size=None, name="t", **kw):
    return asset(name=name, size=size, **kw)(fn).build(P)


def apply(m, point):
    return (m @ np.array([*point, 1.0]))[:3]


def test_turn_and_move_resolve_against_clip_length():
    c = clip("w", 0.8, [turn("leg", "x", [0, 25, 0]), move("torso", "-y", [0, 1])])
    assert c.tracks[0].channel == "rotate_x"
    assert c.tracks[0].times == pytest.approx((0, 0.4, 0.8))
    assert c.tracks[1].channel == "move_y" and c.tracks[1].values == (0.0, -1.0)


def test_track_lists_are_flattened():
    c = clip("w", 1.0, [[turn("a", "x", [0, 1])], turn("b", "y", [0, 2])])
    assert [t.part for t in c.tracks] == ["a", "b"]


def test_callable_keys_sample_evenly():
    [t] = clip("a", 1.0, [TrackSpec("p", "rotate_z", lambda u: 10 * u, samples=5)]).tracks
    assert t.times == pytest.approx((0, 0.25, 0.5, 0.75, 1))
    assert t.values[-1] == pytest.approx(10)


def test_bad_free_axis_is_an_error():
    from cubist import ModelError

    with pytest.raises(ModelError, match="axis"):
        turn("leg", "w", [0, 1])


def arm(p, clips, joint=None):
    torso = Part("torso", box((4, 6, 2), color=p.wood.mid).at(2, 0, 2), pivot=(4, 0, 3))
    limb = Part(
        "arm",
        box((1, 4, 1), color=p.wood.light).at(1, 1, 2),
        parent="torso",
        pivot=(1.5, 5, 2.5),
        joint=joint,
    )
    return Model([torso, limb], clips=clips)


def test_free_tracks_allowed_on_parts_without_joints():
    a = build(lambda p: arm(p, [clip("wave", 1, [turn("arm", "z", [0, 90])])]))
    m = world_matrices(a, a.clips[0], 1.0)
    # rotate_z 90 about the shoulder (1.5, 5, 2.5): the hand end (1.5, 1, 2.5) swings to +x
    assert np.allclose(apply(m["arm"], (1.5, 1, 2.5)), (5.5, 5, 2.5))


def test_move_tracks_translate_in_parent_axes():
    a = build(lambda p: arm(p, [clip("lift", 1, [move("torso", "y", [0, 2])])]))
    m = world_matrices(a, a.clips[0], 1.0)
    assert np.allclose(m["arm"][:3, 3], (0, 2, 0))  # the child rides along


def test_joint_and_free_rotation_compose():
    # R = R_joint @ R_y: rotate_y 90 first, then the hinge("x") 90
    clips = [clip("c", 1, [turn("arm", "y", [0, 90]), TrackSpec("arm", "joint", [0, 90])])]
    a = build(lambda p: arm(p, clips, joint=hinge("x", (0, 90))))
    _, rot = local_motion(a, a.clips[0], 1.0)["arm"]
    ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)
    rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    assert np.allclose(rot, rx @ ry)


def test_glb_free_tracks_match_pose():
    clips = [clip("c", 1, [turn("arm", "z", [0, 60, 0]), move("arm", "y", [0, 1, 0])])]
    a = build(lambda p: arm(p, clips), size=(8, 8, 6))
    g = GLTF2.load_from_bytes(encode_glb(a, P.rgba_table(), voxel_size_m=1.0))
    [anim] = g.animations
    assert sorted(c.target.path for c in anim.channels) == ["rotation", "translation"]


def test_fast_free_spin_keys_stay_under_90_degrees():
    from test_gltf_anim import accessor

    clips = [clip("s", 1, [turn("arm", "y", [0, 3600])], ease="linear", loop=True)]
    a = build(lambda p: arm(p, clips), size=(8, 8, 6))
    g = GLTF2.load_from_bytes(encode_glb(a, P.rgba_table(), voxel_size_m=1.0))
    q = accessor(g, g.animations[0].samplers[0].output).astype(float)
    angles = 2 * np.degrees(np.arccos(np.clip(np.sum(q[1:] * q[:-1], axis=1), -1, 1)))
    assert angles.max() < 90


def test_collision_forgives_near_free_rotation_pivot():
    clips = [clip("swing", 0.8, [turn("arm", "x", [0, 20, 0, -20, 0])], loop=True)]
    findings = check_motion(build(lambda p: arm(p, clips)))
    assert not [f for f in findings if f.check == "motion_collision"]


def test_manifest_schema_accepts_free_channels():
    clips = [clip("c", 1, [turn("arm", "z", [0, 60]), move("torso", "y", [0, 1])])]
    m = build_manifest(build(lambda p: arm(p, clips)), voxel_size_m=1)
    jsonschema.validate(m, json.loads(SCHEMA_PATH.read_text()))
