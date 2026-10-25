"""Clips exported as glTF animations, sampled from the same pose evaluator as the previews."""

import hashlib

import numpy as np
import pytest
from pygltflib import GLTF2

from cubist import Model, Part, asset, box, clip, hinge, slide, spin
from cubist.core.palette import Palette
from cubist.io.gltf import animation_samples, encode_glb
from cubist.motion.pose import local_motion

P = Palette({"wood": ["#3b2414", "#8a5a32"], "metal": "#7d8491"})
S = 0.5


def accessor(g: GLTF2, index: int) -> np.ndarray:
    acc = g.accessors[index]
    view = g.bufferViews[acc.bufferView]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc.type]
    arr = np.frombuffer(
        g.binary_blob(), dtype=np.float32, count=acc.count * width, offset=view.byteOffset
    )
    return arr.reshape(acc.count, width) if width > 1 else arr


def static(p):
    return Model(
        parts=[
            Part("body", box((4, 4, 4), color=p.wood.dark), pivot=(2, 0, 2)),
            Part(
                "lid",
                box((4, 1, 4), color=p.wood.light).at(0, 4, 0),
                parent="body",
                pivot=(0, 4, 0),
            ),
        ],
        sockets={"handle": (2, 5, 2)},
    )


def chest(p, clips=None, joint=None):
    m = static(p)
    m.parts[1] = Part(
        m.parts[1].name,
        m.parts[1].shape,
        pivot=(0, 4, 0),
        parent="body",
        joint=joint or hinge("-x", (0, 120)),
    )
    return Model(
        m.parts,
        sockets=m.sockets,
        clips=clips if clips is not None else [clip("open", 0.5, {"lid": [0, 100]})],
    )


def build(fn):
    return asset(name="chest", size=(4, 5, 4))(fn).build(P)


def quat_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def test_static_glb_bytes_unchanged():
    data = encode_glb(build(static), P.rgba_table(), voxel_size_m=S)
    assert b"animations" not in data
    assert hashlib.sha256(data).hexdigest() == STATIC_SHA256


def test_clip_becomes_named_animation_with_loop_extra():
    g = GLTF2.load_from_bytes(encode_glb(build(chest), P.rgba_table(), voxel_size_m=S))
    [anim] = g.animations
    assert anim.name == "open" and anim.extras == {"loop": False, "length": 0.5}
    [channel] = anim.channels
    lid = [n.name for n in g.nodes].index("lid")
    assert channel.target.node == lid and channel.target.path == "rotation"
    sampler = anim.samplers[channel.sampler]
    assert sampler.interpolation == "LINEAR"
    times = g.accessors[sampler.input]
    assert times.min == [0.0] and times.max == [0.5]


def test_glb_keys_match_pose_at_sample_times():
    a = build(chest)
    g = GLTF2.load_from_bytes(encode_glb(a, P.rgba_table(), voxel_size_m=S))
    [anim] = g.animations
    sampler = anim.samplers[anim.channels[0].sampler]
    times, quats = accessor(g, sampler.input), accessor(g, sampler.output)
    for t, q in zip(times, quats, strict=True):
        _, rot = local_motion(a, a.clips[0], float(t))["lid"]
        assert np.allclose(quat_to_matrix(q), rot, atol=1e-5)


def test_samples_report_translation_and_rotation_per_part():
    a = build(chest)
    s = animation_samples(a, a.clips[0], voxel_size_m=S)
    assert set(s) == {"lid"}
    lid = s["lid"]
    assert lid["translation"].shape == (len(lid["times"]), 3)
    assert lid["rotation"].shape == (len(lid["times"]), 4)
    assert np.allclose(lid["translation"], [-1.0, 2.0, -1.0])  # pivot offset from body, metres


def test_glb_nested_motion_matches_pose():
    def desk(p):
        top = Part("desk", box((8, 4, 8), color=p.wood.dark), pivot=(4, 0, 4))
        drawer = Part(
            "drawer",
            box((6, 2, 6), color=p.wood.light).at(1, 1, 1),
            parent="desk",
            pivot=(4, 1, 7),
            joint=slide("+z", 4),
        )
        knob = Part(
            "knob",
            box((2, 1, 1), color=p.metal).at(3, 1, 7),
            parent="drawer",
            pivot=(4, 1, 7),
            joint=hinge("x", (0, 90)),
        )
        return Model(
            [top, drawer, knob], clips=[clip("pull", 1.0, {"drawer": [0, 4], "knob": [0, 90]})]
        )

    a = asset(name="desk", size=(8, 4, 12))(desk).build(P)
    g = GLTF2.load_from_bytes(encode_glb(a, P.rgba_table(), voxel_size_m=S))
    [anim] = g.animations
    names = [n.name for n in g.nodes]
    paths = {(names[c.target.node], c.target.path) for c in anim.channels}
    assert paths == {("drawer", "translation"), ("knob", "rotation")}
    for c in anim.channels:
        sampler = anim.samplers[c.sampler]
        times, out = accessor(g, sampler.input), accessor(g, sampler.output)
        node = names[c.target.node]
        for t, v in zip(times, out, strict=True):
            trans, rot = local_motion(a, a.clips[0], float(t))[node]
            if c.target.path == "translation":
                # drawer pivot (4,1,7) minus desk pivot (4,0,4), plus the slide, in metres
                assert np.allclose(v, (np.array([0, 1, 3]) + trans) * S, atol=1e-5)
            else:
                assert np.allclose(quat_to_matrix(v), rot, atol=1e-5)


def test_fast_spin_keys_stay_under_90_degrees():
    def prop(p):
        blade = Part(
            "prop", box((4, 1, 1), color=p.wood.dark), pivot=(2, 0.5, 0.5), joint=spin("z", rps=20)
        )
        return Model(
            [blade], clips=[clip("spin", 1.0, {"prop": [0, 20]}, loop=True, ease="linear")]
        )

    a = asset(name="prop", size=(4, 1, 1))(prop).build(P)
    g = GLTF2.load_from_bytes(encode_glb(a, P.rgba_table(), voxel_size_m=S))
    sampler = g.animations[0].samplers[0]
    q = accessor(g, sampler.output).astype(np.float64)
    dots = np.sum(q[1:] * q[:-1], axis=1)
    assert np.all(dots > 0)
    angles = 2 * np.degrees(np.arccos(np.clip(dots, -1, 1)))
    assert angles.max() < 90


def test_step_track_uses_step_interpolation_at_key_times_only():
    a = build(lambda p: chest(p, [clip("snap", 1.0, {"lid": [0, 45, 90]}, ease="step")]))
    g = GLTF2.load_from_bytes(encode_glb(a, P.rgba_table(), voxel_size_m=S))
    sampler = g.animations[0].samplers[0]
    assert sampler.interpolation == "STEP"
    assert np.allclose(accessor(g, sampler.input), [0, 0.5, 1.0])


def test_glb_constant_track_is_valid():
    a = build(lambda p: chest(p, [clip("hold", 1.0, {"lid": [30]})]))
    g = GLTF2.load_from_bytes(encode_glb(a, P.rgba_table(), voxel_size_m=S))
    [anim] = g.animations
    assert anim.name == "hold" and len(anim.channels) == 1
    sampler = anim.samplers[0]
    times = g.accessors[sampler.input]
    assert times.min is not None and times.max is not None


def test_clip_that_moves_nothing_still_exports():
    a = build(lambda p: chest(p, [clip("rest", 1.0, {"lid": [0]})]))
    g = GLTF2.load_from_bytes(encode_glb(a, P.rgba_table(), voxel_size_m=S))
    assert [x.name for x in g.animations] == ["rest"] and len(g.animations[0].channels) == 1


def test_translation_channel_only_when_part_slides():
    a = build(lambda p: chest(p, joint=slide("+y", 3), clips=[clip("lift", 1.0, {"lid": [0, 3]})]))
    g = GLTF2.load_from_bytes(encode_glb(a, P.rgba_table(), voxel_size_m=S))
    assert [c.target.path for c in g.animations[0].channels] == ["translation"]


@pytest.mark.parametrize("ease", ["smooth", "linear"])
def test_sampled_rate_is_30_per_second(ease):
    a = build(lambda p: chest(p, [clip("open", 1.0, {"lid": [0, 100]}, ease=ease)]))
    s = animation_samples(a, a.clips[0], voxel_size_m=S)["lid"]
    assert len(s["times"]) == 31


# encode_glb of the static model above. Recorded before animations existed; re-recorded when the
# palette PNG switched to stored deflate blocks so it's the same on every machine (same pixels,
# and every other byte unchanged).
STATIC_SHA256 = "af2f4e1c17c50083ab8bb22243d4d93d9bc22a3b08ce26af7e709f21b9e5dd58"
