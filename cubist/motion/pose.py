"""The pose evaluator: (asset, clip, time) -> how every part sits.

Previews, motion lint, and the GLB exporter all read poses from here, so what the agent sees is
what the engine plays. Matrices are in rest-world voxel coordinates: a part's world matrix maps
its voxels as modeled to where they are at time ``t``.
"""

from __future__ import annotations

from functools import lru_cache

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
    if loop and n >= 2 and times[0] == 0 and abs(times[-1] - length) < 1e-9:
        # The first and last keys are the same moment in a loop: blend across the wrap.
        m[0] = m[-1] = blend(d[-1], d[0], h[-1], h[0])
    return m


@lru_cache(maxsize=4096)
def _tangents_for(times: tuple, values: tuple, loop: bool, length: float) -> tuple:
    return tuple(_tangents(np.array(times), np.array(values), loop, length))


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
    if ease == "in":  # accelerating into each key
        return float(values[k] + s * s * (values[k + 1] - values[k]))
    if ease == "out":  # decelerating into each key
        return float(values[k] + (1 - (1 - s) ** 2) * (values[k + 1] - values[k]))
    m = _tangents_for(track.times, track.values, clip.loop, clip.length)
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
    """Every part's (translation, rotation) relative to rest at ``t``; identity if still.

    Translation is the joint's slide plus the ``move_*`` tracks; rotation is
    ``R_joint @ R_z @ R_y @ R_x`` (free rotations about the parent's axes, x first, then the
    joint). Cached on the asset: the same clip and time always give the same motion."""
    cache = asset.__dict__.setdefault("_local_motion", {})
    key = (clip, float(t))
    if key not in cache:
        cache[key] = _local_motion(asset, clip, t)
    return cache[key]


def _local_motion(
    asset: BuiltAsset, clip: Clip, t: float
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {p.name: (np.zeros(3), np.eye(3)) for p in asset.parts}
    joint_motion_of: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    turns: dict[str, np.ndarray] = {}
    moves: dict[str, np.ndarray] = {}
    for track in clip.tracks:
        if track.channel == "joint":
            joint = asset.part(track.part).joint
            if joint is not None:
                joint_motion_of[track.part] = joint_motion(joint, track_value(track, t, clip))
        elif track.channel.startswith(("rotate_", "move_")):
            kind, axis = track.channel.split("_")
            store = turns if kind == "rotate" else moves
            vec = store.setdefault(track.part, np.zeros(3))
            vec["xyz".index(axis)] += track_value(track, t, clip)
    for name in {*joint_motion_of, *turns, *moves}:
        trans, rot = joint_motion_of.get(name, (np.zeros(3), np.eye(3)))
        if name in turns:
            free = np.eye(3)
            for i, axis in enumerate("xyz"):
                if turns[name][i]:
                    free = rotation(turns[name][i], axis, np.zeros(3))[:3, :3] @ free
            rot = rot @ free
        if name in moves:
            trans = trans + moves[name]
        out[name] = (trans, rot)
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


def part_bounds(part) -> tuple | None:
    """A part's voxel bounds, across all frames of a flipbook."""
    boxes = [g.bounds() for g in (part.frames or [part.grid])]
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    lo = tuple(min(b[0][i] for b in boxes) for i in range(3))
    hi = tuple(max(b[1][i] for b in boxes) for i in range(3))
    return lo, hi


def posed_bounds(asset: BuiltAsset, matrices: dict[str, np.ndarray]) -> tuple:
    """The whole-voxel box around every part's posed voxel bounds (all flipbook frames)."""
    pts = []
    for p in asset.parts:
        b = part_bounds(p)
        if b is None:
            continue
        lo, hi = np.array(b[0], float), np.array(b[1], float)
        corners = np.array([[(hi if i & k else lo)[j] for j, k in enumerate((1, 2, 4))]
                            for i in range(8)])  # fmt: skip
        m = matrices.get(p.name, np.eye(4))
        pts.append(corners @ m[:3, :3].T + m[:3, 3])
    if not pts:
        return (0, 0, 0), (1, 1, 1)
    allp = np.round(np.concatenate(pts), 6)
    lo = tuple(int(v) for v in np.floor(allp.min(axis=0)))
    hi = tuple(int(v) for v in np.ceil(allp.max(axis=0)))
    return lo, hi


def visibility(asset: BuiltAsset, clip: Clip | None, t: float) -> dict[str, bool]:
    """Whether each part shows at ``t``: its own ``visible`` track (else its rest state), and
    every ancestor showing too."""
    own = {p.name: not p.hidden for p in asset.parts}
    if clip is not None:
        for track in clip.tracks:
            if track.channel == "visible":
                own[track.part] = track_value(track, t, clip) > 0.5
    parents = {p.name: p.parent for p in asset.parts}
    out: dict[str, bool] = {}

    def shows(name: str) -> bool:
        if name not in out:
            parent = parents[name]
            out[name] = own[name] and (parent not in parents or shows(parent))
        return out[name]

    for name in own:
        shows(name)
    return out


def frame_index(asset: BuiltAsset, clip: Clip | None, name: str, t: float) -> int:
    """Which frame of a flipbook part shows at ``t``: the clip's ``frame`` track if it has
    one; frame 0 if some other clip drives this flipbook; else the flipbook looping on its own
    at its fps. 0 for other parts."""
    part = asset.part(name)
    if not part.frames:
        return 0
    n = len(part.frames)
    if clip is not None:
        for track in clip.tracks:
            if track.part == name and track.channel == "frame":
                return int(round(track_value(track, t, clip))) % n
    if any(tr.part == name and tr.channel == "frame" for c in asset.clips for tr in c.tracks):
        return 0  # clips play this flipbook; between them it rests on its first frame
    return int(np.floor(t * (part.fps or 8.0) + 1e-9)) % n


def frame_indices(asset: BuiltAsset, clip: Clip | None, t: float) -> dict[str, int]:
    return {p.name: frame_index(asset, clip, p.name, t) for p in asset.parts if p.frames}


def used_cycles(asset: BuiltAsset) -> list:
    """The style's color cycles whose colors this asset uses."""
    used: set[int] = set()
    for p in asset.parts:
        for g in p.frames or [p.grid]:
            used.update(g.colors_used())
    return [c for c in asset.cycles if used & set(c.indices)]


def preview_clips(asset: BuiltAsset, with_previews: bool = False) -> list[Clip]:
    """The clips worth previewing and exporting: the authored ones, plus a looping
    ``<part>_frames`` clip for each flipbook that no clip drives (it plays on its own). With
    ``with_previews``, when nothing else would show them, previews also get a ``colors`` clip
    (one full color cycle) or a ``particles`` clip (always-on emitters, one particle life);
    these have no tracks and aren't exported."""
    out = list(asset.clips)
    driven = {t.part for c in asset.clips for t in c.tracks if t.channel == "frame"}
    names = {c.name for c in out}
    for p in asset.parts:
        if not p.frames or p.name in driven:
            continue
        n, fps = len(p.frames), p.fps or 8.0
        name = f"{p.name}_frames"
        while name in names:
            name += "_"
        track = Track(
            p.name, "frame", tuple(k / fps for k in range(n)), tuple(map(float, range(n)))
        )
        out.append(Clip(name, n / fps, (track,), loop=True, ease="step"))
        names.add(name)
    cycles = used_cycles(asset)
    if with_previews and cycles and not out:
        c = cycles[0]
        out.append(Clip("colors", len(c.indices) / c.fps, (), loop=True, ease="step"))
    lives = [e.life for e in asset.emitters.values() if e.rate]
    if with_previews and lives and not out:
        out.append(Clip("particles", max(lives), (), loop=True, ease="step"))
    return out


def hidden_at(asset: BuiltAsset, clip: Clip | None, t: float) -> frozenset[str]:
    return frozenset(n for n, shown in visibility(asset, clip, t).items() if not shown)


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


def turn_degrees(asset: BuiltAsset, clip: Clip, part: str, t0: float, t1: float) -> float:
    """Degrees a part's own rotating tracks travel between two times."""
    total = 0.0
    for track in clip.tracks:
        if track.part != part:
            continue
        dv = abs(track_value(track, t1, clip) - track_value(track, t0, clip))
        if track.channel.startswith("rotate_"):
            total += dv
        elif track.channel == "joint":
            joint = asset.part(track.part).joint
            if joint is not None and joint.kind != "slide":
                total += dv * (360.0 if joint.kind == "spin" else 1.0)
    return total


def dense_times(
    asset: BuiltAsset,
    clip: Clip,
    rate: float = 30.0,
    max_turn: float = 10.0,
    parts: list[str] | None = None,
    max_move: float | None = None,
) -> np.ndarray:
    """``sample_times``, subdivided wherever a part turns more than ``max_turn`` degrees
    between samples, counting the turns of the parts it hangs from (or, with ``max_move``,
    moves more than that many voxels), so fast spins and snaps are sampled finely enough to
    see and to play back. Results are cached on the asset."""
    names = tuple(parts if parts is not None else [p.name for p in asset.parts])
    cache = asset.__dict__.setdefault("_dense_times", {})
    key = (clip, rate, max_turn, names, max_move)
    if key in cache:
        return cache[key]
    base = sample_times(clip, rate)
    turning: dict[str, list[tuple[Track, float]]] = {}
    for track in clip.tracks:
        factor = 0.0
        if track.channel.startswith("rotate_"):
            factor = 1.0
        elif track.channel == "joint":
            joint = asset.part(track.part).joint
            if joint is not None and joint.kind != "slide":
                factor = 360.0 if joint.kind == "spin" else 1.0
        if factor:
            turning.setdefault(track.part, []).append((track, factor))
    parents = {p.name: p.parent for p in asset.parts}
    chains = {}
    for n in names:  # the part and everything it hangs from, as far as they turn
        chain, m = [], n
        while m is not None and m in parents:
            if m in turning:
                chain.append(m)
            m = parents[m]
        chains[n] = chain
    values: dict[tuple[int, float], float] = {}

    def value(track: Track, t: float) -> float:
        k = (id(track), t)
        if k not in values:
            values[k] = track_value(track, t, clip)
        return values[k]

    motions: dict[float, dict] = {}

    def motion(t: float) -> dict:
        if t not in motions:
            motions[t] = local_motion(asset, clip, t)
        return motions[t]

    def too_far(t0: float, t1: float) -> bool:
        own = {
            n: sum(f * abs(value(tr, t1) - value(tr, t0)) for tr, f in turning[n]) for n in turning
        }
        if any(sum(own[m] for m in chains[n]) > max_turn + 1e-9 for n in names):
            return True
        if max_move:
            a, b = motion(t0), motion(t1)
            return max(float(np.linalg.norm(b[n][0] - a[n][0])) for n in names) > max_move + 1e-9
        return False

    def split(t0: float, t1: float, depth: int = 0) -> list[float]:
        """Times after t0 up to t1, halving any step that turns or moves too far."""
        if depth >= 10 or not too_far(t0, t1):
            return [t1]
        mid = (t0 + t1) / 2
        return split(t0, mid, depth + 1) + split(mid, t1, depth + 1)

    out = [float(base[0])]
    for t0, t1 in zip(base, base[1:], strict=False):
        out.extend(split(float(t0), float(t1)))
    cache[key] = np.array(out)
    return cache[key]


def sample_times(clip: Clip, rate: float = 30.0) -> np.ndarray:
    """Sorted, distinct times: a regular grid at ``rate`` per second, every key, and the end."""
    grid = np.arange(0.0, clip.length, 1.0 / rate)
    keys = [t for track in clip.tracks for t in track.times]
    ts = np.unique(np.round(np.concatenate([grid, keys, [0.0, clip.length]]), 9))
    return ts[(ts >= 0) & (ts <= clip.length)]
