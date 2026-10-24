"""Particle previews: a simple, seeded simulation of what an asset's emitters spray at time t.

This is for seeing density, direction and reach in filmstrips and GIFs, not for playing:
the engine's particle system plays emitters for real from the manifest. A continuous emitter
is shown as if it had been running forever; a burst appears at each ``emit:<name>`` clip event.
Particles fly from where the emitter was when they were born.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cubist.core.grid import VoxelGrid
from cubist.core.model import BuiltAsset
from cubist.dsl.noise import hash_unit, mix_seed
from cubist.motion.clips import Clip
from cubist.motion.pose import world_matrices

FADE_FROM = 0.7  # particles shrink away over the last 30% of their life


@dataclass(frozen=True)
class Particle:
    emitter: str
    particle: str
    position: np.ndarray
    scale: float

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Particle)
            and (self.emitter, self.particle, self.scale)
            == (other.emitter, other.particle, other.scale)
            and np.array_equal(self.position, other.position)
        )


def _owner(asset: BuiltAsset, position) -> str | None:
    from cubist.io.gltf import nearest_part

    return nearest_part(asset, position)


def _basis(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    a = np.cross(d, helper)
    a /= np.linalg.norm(a)
    return a, np.cross(d, a)


def particles_at(asset: BuiltAsset, clip: Clip | None, t: float) -> list[Particle]:
    out: list[Particle] = []
    for e_index, (name, e) in enumerate(asset.emitters.items()):
        owner = e.part or _owner(asset, e.position)
        length = clip.length if clip is not None else 0.0
        cache: dict[float, np.ndarray] = {}

        def frame(s: float, owner=owner, length=length, cache=cache) -> np.ndarray:
            s = min(max(s, 0.0), length)
            if s not in cache:
                cache[s] = world_matrices(asset, clip, s).get(owner, np.eye(4))
            return cache[s]

        births: list[tuple[int, float]] = []
        if e.rate:
            first = math.ceil((t - e.life) * e.rate - 1e-9)
            last = math.floor(t * e.rate + 1e-9)
            births += [(k, k / e.rate) for k in range(first, last + 1) if t - k / e.rate < e.life]
        if e.burst and clip is not None:
            for j, (when, label) in enumerate(clip.events):
                if label == f"emit:{name}" and 0 <= t - when < e.life:
                    births += [(1_000_000 * (j + 1) + i, when) for i in range(e.burst)]
        seed = mix_seed(e.seed + 7919 * e_index)
        for k, born in births:
            age = t - born
            r = hash_unit(np.array([[k, 0, 1], [k, 0, 2], [k, 0, 3]]), seed)
            m = frame(born)
            origin = m[:3, :3] @ np.array(e.position) + m[:3, 3]
            d = m[:3, :3] @ np.array(e.direction)
            d /= np.linalg.norm(d)
            a, b = _basis(d)
            theta = math.radians(e.spread) * math.sqrt(r[0])
            phi = 2 * math.pi * r[1]
            v = d * math.cos(theta) + (a * math.cos(phi) + b * math.sin(phi)) * math.sin(theta)
            pos = origin + v * e.speed * (0.75 + 0.5 * r[2]) * age
            pos[1] -= 0.5 * e.gravity * age * age
            life = age / e.life
            scale = e.grow[0] + (e.grow[1] - e.grow[0]) * life
            if e.fade and life > FADE_FROM:
                scale *= 1 - (life - FADE_FROM) / (1 - FADE_FROM)
            if scale > 1e-3:
                out.append(Particle(name, e.particle, pos, float(scale)))
    return out


def instance_matrix(grid: VoxelGrid, p: Particle) -> np.ndarray:
    """Places a particle model's voxels, centered and scaled, at the particle."""
    b = grid.bounds()
    center = (np.array(b[0]) + np.array(b[1])) / 2 if b else np.zeros(3)
    m = np.eye(4)
    m[:3, :3] *= p.scale
    m[:3, 3] = p.position - p.scale * center
    return m


def stand_in(table: np.ndarray) -> VoxelGrid:
    """One voxel in the palette's lightest color, drawn for particles not built yet."""
    used = [i for i in range(1, 256) if table[i, 3] or table[i, :3].any()]
    light = max(used or [1], key=lambda i: int(table[i, :3].astype(int).sum()))
    return VoxelGrid.from_voxels(np.array([[0, 0, 0]]), np.array([light], dtype=np.uint8))


def load_particles(asset: BuiltAsset, out_root: str | Path) -> dict[str, VoxelGrid | None]:
    """Each emitter's particle model from ``out/<particle>/<particle>.vox`` (None if it hasn't
    been built), attached to the asset for previews and lint, and returned."""
    from cubist.io.vox import VoxError, read_vox

    models: dict[str, VoxelGrid | None] = {}
    for e in asset.emitters.values():
        path = Path(out_root) / e.particle / f"{e.particle}.vox"
        try:
            models[e.particle] = VoxelGrid.merge([p.grid for p in read_vox(path).parts])
        except (OSError, VoxError):
            models[e.particle] = None
    asset.particle_models = models
    return models
