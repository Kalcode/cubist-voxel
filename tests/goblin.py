"""Kiln's goblin, ported from Kiln's ``demos/Diorama/Goblin.cs`` (``Update``'s moves and gravity,
``MoveAxis``) and ``src/Kiln.Voxel/World/VoxelCollision.cs`` (``Blocked``, ``GroundBelow``), so the
walk lint's walker can be tested against the character it stands for. Voxels are a numpy bool
array indexed [x, y, z]; outside it is empty."""

from __future__ import annotations

import math

import numpy as np

HALF_WIDTH = 4.0  # Goblin.cs: HalfWidth, Height, StepUp, Gravity
HEIGHT = 22.0
STEP_UP = 6.0
GRAVITY = 400.0
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


def walk(solid: np.ndarray, start, direction, seconds: float) -> list[float]:
    """Walk from ``start`` (feet, standing) at walking speed along ``direction`` (a unit x or z
    vector) for ``seconds``; where the feet end up."""
    pos, fall, grounded = [float(v) for v in start], 0.0, True
    for _ in range(round(seconds / DT)):
        for axis in (0, 2):  # MoveAxis, x then z
            if not direction[axis]:
                continue
            nxt = list(pos)
            nxt[axis] += direction[axis] * WALK_SPEED * DT
            if not blocked(solid, *_box(nxt)):
                pos = nxt
                continue
            raised = [nxt[0], nxt[1] + STEP_UP, nxt[2]]
            if grounded and not blocked(solid, *_box(raised)):
                top = ground_below(solid, nxt[0], nxt[2], HALF_WIDTH, raised[1], STEP_UP + 0.5)
                if top is not None:
                    pos = [nxt[0], top, nxt[2]]
        ground = ground_below(solid, pos[0], pos[2], HALF_WIDTH, pos[1] + 0.5, 256)
        fall -= GRAVITY * DT
        y = pos[1] + fall * DT
        if ground is not None and y <= ground:
            pos[1], fall, grounded = ground, 0.0, True
        else:
            pos[1], grounded = y, False
    return pos
