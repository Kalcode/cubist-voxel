"""Kiln's goblin, ported from Kiln's ``demos/Diorama/Goblin.cs`` (``Tick``'s moves, ``MoveAxis``)
and ``src/Kiln.Voxel/World/VoxelCollision.cs`` (``Blocked``, ``GroundBelow``), so the walk lint's
walker can be tested against the character it stands for. Voxels are a numpy bool array indexed
[x, y, z]: the world's bounds, outside which it is empty.

Ported: the acceleration toward the stick's speed, ``MoveAxis`` (a blocked axis loses its
speed), gravity onto ``GroundBelow``, the clamp to the world's bounds and the drop back in from
the top after falling out. Left out, as they don't move it: the turn toward its heading and the
walk clip's blend. Kiln's walking speed is the walk clip's travel (19 voxels a second without a
clip); ``walk`` takes it as ``speed``."""

from __future__ import annotations

import math

import numpy as np

HALF_WIDTH = 4.0  # Goblin.cs: HalfWidth, Height, StepUp, Gravity
HEIGHT = 22.0
STEP_UP = 6.0
GRAVITY = 400.0
ACCEL = 260.0
WALK_SPEED = 19.0  # voxels a second, its walk with no clip
DT = 1 / 60


def blocked(solid: np.ndarray, lo, hi) -> bool:
    """VoxelCollision.Blocked: a solid voxel overlaps the box (touching doesn't count)."""
    a = [max(math.floor(lo[i]), 0) for i in range(3)]
    b = [min(math.ceil(hi[i]) - 1, solid.shape[i] - 1) for i in range(3)]
    if any(a[i] > b[i] for i in range(3)):
        return False
    return bool(solid[a[0] : b[0] + 1, a[1] : b[1] + 1, a[2] : b[2] + 1].any())


def ground_below(solid: np.ndarray, x, z, half, from_y, max_drop) -> float | None:
    """VoxelCollision.GroundBelow: the top of the highest voxel under the footprint at or below
    from_y, no more than max_drop down."""
    x0, x1 = max(0, math.floor(x - half)), min(solid.shape[0] - 1, math.ceil(x + half) - 1)
    z0, z1 = max(0, math.floor(z - half)), min(solid.shape[2] - 1, math.ceil(z + half) - 1)
    top = min(solid.shape[1] - 1, math.floor(from_y) - 1)
    bottom = max(0, math.ceil(from_y - max_drop) - 1)
    for y in range(top, bottom - 1, -1):
        if solid[x0 : x1 + 1, y, z0 : z1 + 1].any():
            return float(y + 1)
    return None


def _box(feet):
    x, y, z = feet
    return (x - HALF_WIDTH, y + 0.05, z - HALF_WIDTH), (x + HALF_WIDTH, y + HEIGHT, z + HALF_WIDTH)


def walk(
    solid: np.ndarray, start, move, seconds: float, speed: float = WALK_SPEED, run: bool = False
) -> tuple[list[float], float]:
    """Goblin.Tick at 60 a second for ``seconds``, from rest at ``start`` (feet, standing), the
    stick held at ``move`` (x, z, length 1 at most). Where the feet end up, and the lowest they
    went."""
    pos, vel, grounded = [float(v) for v in start], [0.0, 0.0, 0.0], True
    lowest = pos[1]
    target = np.array([move[0], move[1]], float) * speed * (2 if run else 1)
    for _ in range(round(seconds / DT)):
        horizontal = np.array([vel[0], vel[2]])
        delta = target - horizontal
        step = ACCEL * DT
        n = float(np.linalg.norm(delta))
        horizontal = target if n <= step else horizontal + delta / n * step
        vel[0], vel[2] = float(horizontal[0]), float(horizontal[1])
        for axis in (0, 2):  # MoveAxis, x then z
            d = vel[axis] * DT
            if d == 0:
                continue
            nxt = list(pos)
            nxt[axis] += d
            if not blocked(solid, *_box(nxt)):
                pos = nxt
                continue
            raised = [nxt[0], nxt[1] + STEP_UP, nxt[2]]
            if grounded and not blocked(solid, *_box(raised)):
                top = ground_below(solid, nxt[0], nxt[2], HALF_WIDTH, raised[1], STEP_UP + 0.5)
                if top is not None:
                    pos = [nxt[0], top, nxt[2]]
                    continue
            vel[axis] = 0.0
        ground = ground_below(solid, pos[0], pos[2], HALF_WIDTH, pos[1] + 0.5, 256)
        vel[1] -= GRAVITY * DT
        y = pos[1] + vel[1] * DT
        if ground is not None and y <= ground:
            pos[1], vel[1], grounded = ground, 0.0, True
        else:
            pos[1], grounded = y, False
        lowest = min(lowest, pos[1])
        pos[0] = min(max(pos[0], HALF_WIDTH), solid.shape[0] - HALF_WIDTH)
        pos[2] = min(max(pos[2], HALF_WIDTH), solid.shape[2] - HALF_WIDTH)
        if pos[1] < -64:
            pos[1] = float(solid.shape[1])  # fell out of the world: drop back in from the top
    return pos, lowest
