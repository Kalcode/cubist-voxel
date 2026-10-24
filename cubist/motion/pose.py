"""The pose evaluator: (asset, clip, time) -> how every part sits.

Previews, motion lint, and the GLB exporter all read poses from here, so what the agent sees is
what the engine plays. Matrices are in rest-world voxel coordinates: a part's world matrix maps
its voxels as modeled to where they are at time ``t``.
"""

from __future__ import annotations

import numpy as np

from cubist.core.model import BuiltAsset
from cubist.dsl.ops import rotation
from cubist.motion.clips import Clip, Track
from cubist.motion.joints import Joint


def _tangents(times: np.ndarray, values: np.ndarray, loop: bool, length: float) -> np.ndarray:
    """Monotone cubic (PCHIP, Fritsch-Butland) tangents; flat ends unless the clip loops."""
    n = len(times)
    h = np.diff(times)
    d = np.diff(values) / h
    m = np.zeros(n)

    def blend(d0: float, d1: float, h0: float, h1: float) -> float:
        if d0 == 0 or d1 == 0 or (d0 > 0) != (d1 > 0):
            return 0.0
        w1, w2 = 2 * h1 + h0, h1 + 2 * h0
        return (w1 + w2) / (w1 / d0 + w2 / d1)

    for k in range(1, n - 1):
        m[k] = blend(d[k - 1], d[k], h[k - 1], h[k])
    if loop and n >= 3 and times[0] == 0 and abs(times[-1] - length) < 1e-9:
        # The first and last keys are the same moment in a loop: blend across the wrap.
        m[0] = m[-1] = blend(d[-1], d[0], h[-1], h[0])
    return m


def track_value(track: Track, t: float, clip: Clip) -> float:
    """The track's value at time ``t`` (held flat before the first and after the last key)."""
    times = np.asarray(track.times, dtype=np.float64)
    values = np.asarray(track.values, dtype=np.float64)
    if len(times) == 1 or t <= times[0]:
        return float(values[0])
    if t >= times[-1]:
        return float(values[-1])
    k = int(np.searchsorted(times, t, side="right")) - 1
    ease = clip.track_ease(track)
    if ease == "step":
        return float(values[k])
    h = times[k + 1] - times[k]
    s = (t - times[k]) / h
    if ease == "linear":
        return float(values[k] + s * (values[k + 1] - values[k]))
    m = _tangents(times, values, clip.loop, clip.length)
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    return float(h00 * values[k] + h10 * h * m[k] + h01 * values[k + 1] + h11 * h * m[k + 1])


def joint_motion(joint: Joint, value: float) -> tuple[np.ndarray, np.ndarray]:
    """(translation, 3x3 rotation) in the parent's frame for a joint at ``value``."""
    axis = np.asarray(joint.axis, dtype=np.float64)
    if joint.kind == "slide":
        return axis * value, np.eye(3)
    degrees = value * 360.0 if joint.kind == "spin" else value
    return np.zeros(3), rotation(degrees, tuple(axis), np.zeros(3))[:3, :3]


def local_motion(
    asset: BuiltAsset, clip: Clip, t: float
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Every part's (translation, rotation) relative to rest at ``t``; identity if still."""
    out = {p.name: (np.zeros(3), np.eye(3)) for p in asset.parts}
    for track in clip.tracks:
        joint = asset.part(track.part).joint
        if track.channel == "joint" and joint is not None:
            out[track.part] = joint_motion(joint, track_value(track, t, clip))
    return out


def _about(pivot: np.ndarray, trans: np.ndarray, rot: np.ndarray) -> np.ndarray:
    """T(pivot) @ T(trans) @ R @ T(-pivot)."""
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = pivot + trans - rot @ pivot
    return m


def world_matrices(asset: BuiltAsset, clip: Clip | None, t: float) -> dict[str, np.ndarray]:
    """Each part's 4x4 matrix from rest-world to posed-world voxel coordinates."""
    if clip is None:
        return {p.name: np.eye(4) for p in asset.parts}
    local = local_motion(asset, clip, t)
    parts = {p.name: p for p in asset.parts}
    out: dict[str, np.ndarray] = {}

    def world(name: str) -> np.ndarray:
        if name not in out:
            p = parts[name]
            trans, rot = local[name]
            own = _about(np.asarray(p.pivot, dtype=np.float64), trans, rot)
            out[name] = (world(p.parent) if p.parent in parts else np.eye(4)) @ own
        return out[name]

    for name in parts:
        world(name)
    return out


def moving_parts(asset: BuiltAsset, clip: Clip) -> list[str]:
    """Parts the clip drives, plus everything hanging from them, in model order."""
    driven = {t.part for t in clip.tracks}
    parents = {p.name: p.parent for p in asset.parts}

    def moves(name: str | None) -> bool:
        while name is not None:
            if name in driven:
                return True
            name = parents.get(name)
        return False

    return [p.name for p in asset.parts if moves(p.name)]


def sample_times(clip: Clip, rate: float = 30.0) -> np.ndarray:
    """Sorted, distinct times: a regular grid at ``rate`` per second, every key, and the end."""
    grid = np.arange(0.0, clip.length, 1.0 / rate)
    keys = [t for track in clip.tracks for t in track.times]
    ts = np.unique(np.round(np.concatenate([grid, keys, [0.0, clip.length]]), 9))
    return ts[(ts >= 0) & (ts <= clip.length)]
