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
