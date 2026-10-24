"""burst(...): chunks thrown apart, tumbling, landing on the floor - a baked explosion."""

import numpy as np
from pygltflib import GLTF2

from cubist import Model, asset, burst, chunks, clip, cylinder
from cubist.core.palette import Palette
from cubist.io.gltf import animation_samples, encode_glb
from cubist.lint.motion import check_motion
from cubist.motion.pose import world_matrices

P = Palette({"wood": ["#3b2414", "#5e3a1f", "#8a5a32"]})


def barrel(p, seed=0, **kw):
    body = cylinder(r=4, h=10, color=p.wood.mid).at(5, 0, 5)
    pieces = chunks("barrel", body, pieces=6, seed=2)
    boom = clip("explode", 1.5, burst(pieces, seed=seed, **kw), collisions=False)
    return Model(pieces, clips=[boom])


def built(seed=0, **kw):
    return asset(name="barrel", size=(10, 10, 10), grounded=True)(
        lambda p: barrel(p, seed, **kw)
    ).build(P)


def lowest(a, t):
    out = np.inf
    for p in a.parts:
        m = world_matrices(a, a.clips[0], t)[p.name]
        c = p.grid.voxels()[0] + 0.5
        y = c @ m[1, :3] + m[1, 3] - 0.5 * np.abs(m[1, :3]).sum()
        out = min(out, float(y.min()))
    return out


def test_chunks_fly_away_from_the_center():
    a = built()
    c = a.clips[0]
    center = np.array([5.0, 0, 5.0])
    for p in a.parts:
        start = np.array(p.pivot)
        m = world_matrices(a, c, 0.3)[p.name]
        moved = m[:3, :3] @ start + m[:3, 3]
        assert np.linalg.norm((moved - center)[[0, 2]]) > np.linalg.norm((start - center)[[0, 2]])


def test_chunks_land_on_the_floor_and_stay():
    a = built()
    assert lowest(a, 1.5) > -0.05
    assert min(lowest(a, t) for t in np.linspace(0, 1.5, 61)) > -0.25
    end = world_matrices(a, a.clips[0], 1.5)
    later = world_matrices(a, a.clips[0], 1.45)
    assert all(np.allclose(end[n], later[n], atol=1e-6) for n in end)  # at rest
    assert not [f for f in check_motion(a) if f.check == "motion_floor"]


def test_burst_is_seeded():
    a, b, c = built(1), built(1), built(2)
    ka = [t.values for t in a.clips[0].tracks]
    assert ka == [t.values for t in b.clips[0].tracks]
    assert ka != [t.values for t in c.clips[0].tracks]


def test_burst_is_plain_move_and_rotate_tracks():
    a = built()
    channels = {t.channel for t in a.clips[0].tracks}
    assert channels <= {"move_x", "move_y", "move_z", "rotate_x", "rotate_y", "rotate_z"}
    assert all(a.clips[0].track_ease(t) == "linear" for t in a.clips[0].tracks)


def test_collisions_off_skips_the_collision_check_only():
    a = built()
    assert a.clips[0].collisions is False
    assert not [f for f in check_motion(a) if f.check == "motion_collision"]


def test_glb_burst_matches_pose():
    a = built()
    g = GLTF2.load_from_bytes(encode_glb(a, P.rgba_table(), voxel_size_m=1))
    [anim] = g.animations
    assert len({c.target.node for c in anim.channels}) == len(a.parts)
    samples = animation_samples(a, a.clips[0], voxel_size_m=1)
    assert set(samples) == {p.name for p in a.parts}
