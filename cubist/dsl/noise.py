"""Deterministic integer hashing and 3D noise, identical on every platform.

All randomness in a build flows from here: an op's own seed mixed with the build seed.
"""

from __future__ import annotations

import numpy as np

from cubist.dsl.context import current

_M = 0xFFFFFFFF
_GRADS = np.array(
    [[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
     [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
     [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]],
    dtype=np.float64,
)  # fmt: skip


def _lowbias32(x: np.ndarray) -> np.ndarray:
    """Chris Wellons' lowbias32 integer hash on uint32 arrays."""
    x = x ^ (x >> np.uint32(16))
    x = x * np.uint32(0x7FEB352D)
    x = x ^ (x >> np.uint32(15))
    x = x * np.uint32(0x846CA68B)
    return x ^ (x >> np.uint32(16))


def _u32(values) -> np.ndarray:
    return (np.asarray(values, dtype=np.int64) & _M).astype(np.uint32)


def mix_seed(op_seed: int, build_seed: int | None = None) -> int:
    """An op's seed combined with the build seed (default: the current build's)."""
    build = current().seed if build_seed is None else build_seed
    h = _lowbias32(_u32([op_seed]) ^ _lowbias32(_u32([build]) + np.uint32(0x9E3779B9)))
    return int(h[0])


def hash3(ijk: np.ndarray, seed: int) -> np.ndarray:
    """uint32 hash of integer coordinates (N, 3)."""
    ijk = _u32(ijk)
    h = np.uint32((int(seed) * 0x9E3779B1) & _M)
    h = _lowbias32(h ^ ijk[..., 0])
    h = _lowbias32(h ^ (ijk[..., 1] * np.uint32(0x85EBCA77)))
    return _lowbias32(h ^ (ijk[..., 2] * np.uint32(0xC2B2AE3D)))


def hash_unit(ijk: np.ndarray, seed: int) -> np.ndarray:
    """Uniform floats in [0, 1) per integer coordinate."""
    return hash3(ijk, seed).astype(np.float64) / 4294967296.0


def _fade(t: np.ndarray) -> np.ndarray:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def perlin(p: np.ndarray, seed: int) -> np.ndarray:
    """Gradient noise in roughly [-1, 1]; zero at integer lattice points."""
    p = np.asarray(p, dtype=np.float64)
    base = np.floor(p)
    f = p - base
    base = base.astype(np.int64)
    u = _fade(f)
    total = np.zeros(len(p))
    for cx in (0, 1):
        for cy in (0, 1):
            for cz in (0, 1):
                corner = np.array([cx, cy, cz])
                g = _GRADS[hash3(base + corner, seed) % 12]
                dot = np.sum(g * (f - corner), axis=-1)
                w = (
                    (u[:, 0] if cx else 1 - u[:, 0])
                    * (u[:, 1] if cy else 1 - u[:, 1])
                    * (u[:, 2] if cz else 1 - u[:, 2])
                )
                total += w * dot
    return total


def value_noise(p: np.ndarray, seed: int) -> np.ndarray:
    """Smoothly interpolated lattice values in [-1, 1]."""
    p = np.asarray(p, dtype=np.float64)
    base = np.floor(p)
    u = _fade(p - base)
    base = base.astype(np.int64)
    total = np.zeros(len(p))
    for cx in (0, 1):
        for cy in (0, 1):
            for cz in (0, 1):
                corner = np.array([cx, cy, cz])
                v = hash_unit(base + corner, seed) * 2.0 - 1.0
                w = (
                    (u[:, 0] if cx else 1 - u[:, 0])
                    * (u[:, 1] if cy else 1 - u[:, 1])
                    * (u[:, 2] if cz else 1 - u[:, 2])
                )
                total += w * v
    return total


NOISES = {"perlin": perlin, "value": value_noise}


def fbm(p: np.ndarray, seed: int, octaves: int = 1, kind: str = "perlin") -> np.ndarray:
    """Sum of octaves at doubling frequency and halving amplitude, normalized to ~[-1, 1]."""
    fn = NOISES[kind]
    total, amp, norm = np.zeros(len(p)), 1.0, 0.0
    for o in range(octaves):
        total += amp * fn(p * (2**o), seed + 1013 * o)
        norm += amp
        amp *= 0.5
    return total / norm
