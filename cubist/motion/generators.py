"""Generators: common motion as track specs, so an agent writes intent instead of keys.

Each returns a ``TrackSpec`` that ``clip()`` resolves against the clip's length. Curves are
sampled 8 times per cycle; with the default smooth easing that traces a sine closely, and whole
cycles close a loop exactly.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from cubist.core.errors import ModelError
from cubist.dsl.noise import hash_unit, mix_seed
from cubist.motion.clips import CHANNELS, TrackSpec, _free

SAMPLES_PER_CYCLE = 8


def swing(
    part: str, axis: str, amount: float, phase: float = 0.0, cycles: int = 1, center: float = 0.0
) -> TrackSpec:
    """Rock ``part`` about its pivot: ``center + amount * sin(2π(cycles·u + phase))`` degrees.
    ``phase`` is in cycles, so 0.5 swings the opposite way (the other leg)."""

    def curve(u: float) -> float:
        return center + amount * math.sin(2 * math.pi * (cycles * u + phase))

    spec = _free("rotate", part, axis, curve, None)
    return replace(spec, samples=SAMPLES_PER_CYCLE * max(1, int(cycles)) + 1)


def bob(
    part: str, height: float, cycles: int = 2, phase: float = 0.0, axis: str = "y"
) -> TrackSpec:
    """Move ``part`` from rest to ``height`` voxels and back, ``cycles`` times (two steps per
    walk cycle by default). A negative height dips instead."""

    def curve(u: float) -> float:
        return height * (1 - math.cos(2 * math.pi * (cycles * u + phase))) / 2

    spec = _free("move", part, axis, curve, None)
    return replace(spec, samples=SAMPLES_PER_CYCLE * max(1, int(cycles)) + 1)


def shake(
    part: str, amount: float, channel: str = "rotate_z", rate: float = 12, seed: int = 0
) -> TrackSpec:
    """Seeded jitter in [-amount, amount] on a free channel, ``rate`` keys a second, starting
    and ending at rest: recoil, impacts, a flickering flame. Follows the build seed."""
    if channel not in CHANNELS or not channel.startswith(("rotate_", "move_")):
        raise ModelError(f"shake channel must be rotate_x|y|z or move_x|y|z, got {channel!r}")

    def make(length: float):
        n = max(2, math.ceil(length * rate)) + 1
        times = np.linspace(0.0, length, n)
        noise = hash_unit(np.array([[i, 0, 0] for i in range(n)]), mix_seed(seed))
        values = amount * (2 * noise - 1)
        values[0] = values[-1] = 0.0
        return times.tolist(), values.tolist()

    return TrackSpec(part, channel, ease="linear", make=make)


def spin_track(part: str, turns: float, axis: str | None = None) -> TrackSpec:
    """Turn at a constant speed: ``turns`` full turns over the clip, on the part's spin joint,
    or about ``axis`` ('y', '-z', ...) on any part."""
    if axis is None:
        return TrackSpec(part, "joint", [0.0, float(turns)], ease="linear")
    return _free("rotate", part, axis, [0.0, 360.0 * float(turns)], "linear")


def _rot(angles) -> np.ndarray:
    """R_z @ R_y @ R_x, the order free rotate tracks compose in."""
    from cubist.dsl.ops import rotation

    out = np.eye(3)
    for deg, axis in zip(angles, "xyz", strict=True):
        if deg:
            out = rotation(deg, axis, np.zeros(3))[:3, :3] @ out
    return out


def _lowest(height: float, centers: np.ndarray, pos_y: float, ang) -> float:
    """World y of a chunk's lowest corner: pivot ``height`` + ``pos_y``, turned by ``ang``."""
    r = _rot(ang)
    return height + pos_y + float((centers @ r[1]).min()) - 0.5 * np.abs(r[1]).sum()


def _fly(height, centers, v, omega, gravity, times, start=0.0):
    """One chunk's flight sampled at ``times``: (moves (n, 3), turns (n, 3)). It waits until
    ``start``; while rising it rolls up off the floor; once falling, it stops where its lowest
    corner touches down. Keys are then lifted wherever the straight line between two keys (how
    the clip plays them) would dip below the floor."""
    moves, turns = _flight(height, centers, v, omega, gravity, np.maximum(times - start, 0.0))
    for _ in range(4):
        dipped = False
        for k in range(1, len(times)):
            for f in (0.25, 0.5, 0.75):
                pos = moves[k - 1] + f * (moves[k] - moves[k - 1])
                ang = turns[k - 1] + f * (turns[k] - turns[k - 1])
                low = _lowest(height, centers, pos[1], ang)
                if low < -0.05:
                    moves[k - 1 : k + 1, 1] -= low
                    dipped = True
        if not dipped:
            break
    return moves, turns


def _flight(height, centers, v, omega, gravity, times):
    """The ballistic flight itself (see ``_fly``)."""

    def at(t):
        pos = v * t
        pos[1] -= 0.5 * gravity * t * t
        return pos, omega * t

    apex = v[1] / gravity if gravity > 0 else np.inf
    moves, turns, landed = [], [], None
    for k, t in enumerate(times):
        if landed is None:
            pos, ang = at(t)
            if 0 < t <= apex:
                pos[1] -= min(0.0, _lowest(height, centers, pos[1], ang))
            elif t > apex and _lowest(height, centers, pos[1], ang) <= 0:
                lo_t, hi_t = max(times[k - 1], apex), t
                for _ in range(30):  # bisect the moment it touches down
                    mid = (lo_t + hi_t) / 2
                    p_mid, a_mid = at(mid)
                    if _lowest(height, centers, p_mid[1], a_mid) <= 0:
                        hi_t = mid
                    else:
                        lo_t = mid
                pos, ang = at(hi_t)
                pos[1] -= _lowest(height, centers, pos[1], ang)  # resting on the floor exactly
                landed = (pos, ang)
        if landed is not None:
            pos, ang = landed
        moves.append(np.array(pos, dtype=np.float64))
        turns.append(np.array(ang, dtype=np.float64))
    return np.array(moves), np.array(turns)


def burst(
    parts,
    origin=None,
    speed: float = 6.0,
    up: float = 6.0,
    spin: float = 540.0,
    gravity: float = 40.0,
    rate: float = 20.0,
    seed: int = 0,
    start: float = 0.0,
) -> list[TrackSpec]:
    """Throw ``parts`` (usually from ``chunks(...)``) apart: each flies out from ``origin``
    (default: their center, at y = 0) at about ``speed`` voxels/s plus ``up``, tumbles at up to
    ``spin`` degrees/s, falls under ``gravity`` (voxels/s²) and comes to rest when its lowest
    corner reaches the floor. ``start`` delays the blast (seconds) for a wind-up; ``rate`` is
    keys per second. Seeded; follows the build seed. Use with ``clip(..., collisions=False)``:
    debris passes through debris."""
    for part in parts:
        if part.pivot is None or part.frames:
            raise ModelError(
                f"burst needs plain parts with pivots, like the ones chunks(...) makes; part "
                f"{part.name!r} has {'frames' if part.frames else 'no pivot'}"
            )
    parts = list(parts)
    pivots = np.array([p.pivot for p in parts], dtype=np.float64)
    o = pivots.mean(axis=0) if origin is None else np.asarray(origin, dtype=np.float64)
    if origin is None:
        o[1] = 0.0
    s = mix_seed(seed)
    cache: dict[float, dict] = {}

    def simulate(length: float) -> dict:
        if length in cache:
            return cache[length]
        n = max(2, math.ceil(length * rate)) + 1
        times = np.unique(np.concatenate([np.linspace(0.0, length, n), [min(start, length)]]))
        out = {}
        for i, part in enumerate(parts):
            rnd = hash_unit(np.array([[i, k, 7] for k in range(6)]), s)
            centers = part.shape.voxelize().voxels()[0] + 0.5 - pivots[i]
            out_dir = (pivots[i] - o) * np.array([1.0, 0.0, 1.0])
            norm = np.linalg.norm(out_dir)
            if norm < 1e-6:
                a = 2 * math.pi * rnd[0]
                out_dir = np.array([math.cos(a), 0.0, math.sin(a)])
            else:
                out_dir = out_dir / norm
            v = out_dir * speed * (0.7 + 0.6 * rnd[1])
            v[1] = up * (0.6 + 0.8 * rnd[2])
            omega = spin * (2 * rnd[3:6] - 1)

            moves, turns = _fly(pivots[i][1], centers, v, omega, gravity, times, start)
            out[part.name] = (times, np.array(moves), np.array(turns))
        cache[length] = out
        return out

    specs = []
    for part in parts:
        for k, axis in enumerate("xyz"):
            for kind, idx in (("move", 1), ("rotate", 2)):

                def make(length, name=part.name, k=k, idx=idx):
                    run = simulate(length)[name]
                    return run[0].tolist(), run[idx][:, k].tolist()

                specs.append(TrackSpec(part.name, f"{kind}_{axis}", ease="linear", make=make))
    return specs
