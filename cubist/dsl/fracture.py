"""Fracture: split a shape into connected chunks, for things that break apart.

Seeds are spread over the shape by seeded farthest-point sampling; each voxel joins its nearest
seed (with a little seeded jitter, so cuts aren't perfectly straight). Pieces that come out
disconnected or too small are merged into the neighbor they share the most faces with, so every
chunk is one face-connected piece and no chunk is a sliver.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from cubist.core.errors import ModelError
from cubist.core.grid import VoxelGrid
from cubist.core.part import Part
from cubist.core.topology import SIX
from cubist.dsl.gridops import GridShape
from cubist.dsl.noise import hash_unit, mix_seed
from cubist.dsl.shape import Shape, ShapeError

_FACES = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


class FixedGrid(GridShape):
    """Voxels that are already decided (a chunk)."""

    def __init__(self, grid: VoxelGrid):
        self.grid = grid

    def _grid(self) -> VoxelGrid:
        return self.grid


def _neighbors(labels: np.ndarray, d) -> np.ndarray:
    """``out[v] = labels[v + d]``, -1 beyond the array."""
    out = np.full_like(labels, -1)
    src = tuple(slice(max(k, 0), labels.shape[i] + min(k, 0)) for i, k in enumerate(d))
    dst = tuple(slice(max(-k, 0), labels.shape[i] + min(-k, 0)) for i, k in enumerate(d))
    out[dst] = labels[src]
    return out


def _touching(labels: np.ndarray, mask: np.ndarray) -> dict[int, int]:
    """Shared faces between the ``mask`` cells and each label outside it."""
    counts: dict[int, int] = {}
    for d in _FACES:
        n = _neighbors(labels, d)[mask]
        for lab in n[(n >= 0)]:
            counts[int(lab)] = counts.get(int(lab), 0) + 1
    return counts


def fracture(shape: Shape, pieces: int = 8, seed: int = 0, min_voxels: int | None = None):
    """``shape`` split into about ``pieces`` connected chunks (fewer when small pieces merge),
    as voxel shapes that keep their colors. Seeded, and follows the build seed."""
    if isinstance(pieces, bool) or not isinstance(pieces, int) or pieces < 2:
        raise ModelError(f"fracture pieces must be a whole number >= 2, got {pieces!r}")
    grid = shape.voxelize().cropped()
    coords, colors = grid.voxels()
    n = len(coords)
    if n == 0:
        raise ShapeError("fracture: the shape has no voxels")
    pieces = min(pieces, n)
    min_voxels = max(4, n // (pieces * 4)) if min_voxels is None else int(min_voxels)
    s = mix_seed(seed)
    centers = coords + 0.5
    rnd = hash_unit(coords, s)

    # Seeded farthest-point sampling spreads the seeds over the shape.
    chosen = [int(np.argmax(rnd))]
    dist = np.linalg.norm(centers - centers[chosen[0]], axis=1)
    for _ in range(pieces - 1):
        nxt = int(np.argmax(dist * (0.75 + 0.5 * rnd)))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(centers - centers[nxt], axis=1))
    seeds = centers[chosen]
    jitter = np.stack([hash_unit(coords, s + 1 + j) for j in range(len(seeds))], axis=1)
    d = np.linalg.norm(centers[:, None, :] - seeds[None, :, :], axis=2) * (1 + 0.3 * jitter)
    label_of = np.argmin(d, axis=1)

    origin = np.array(grid.origin)
    local = coords - origin
    labels = np.full(grid.shape, -1, dtype=np.int64)
    labels[tuple(local.T)] = label_of

    # Every chunk one connected piece: stray bits join the neighbor they touch most.
    next_label = len(seeds)
    for lab in range(len(seeds)):
        comps, k = ndimage.label(labels == lab, structure=SIX)
        if k <= 1:
            continue
        sizes = np.bincount(comps.ravel())[1:]
        keep = int(np.argmax(sizes)) + 1
        for c in range(1, k + 1):
            if c == keep:
                continue
            mask = comps == c
            labels[mask] = -2  # out of the way while we look at its neighbors
            touching = _touching(labels, mask)
            if touching:
                labels[mask] = max(sorted(touching), key=lambda t: touching[t])
            else:
                labels[mask] = next_label
                next_label += 1

    # No slivers: the smallest chunks merge into their best neighbor until all are big enough.
    while True:
        present = np.unique(labels[labels >= 0])
        sizes = {int(lab): int((labels == lab).sum()) for lab in present}
        small = sorted((size, lab) for lab, size in sizes.items() if size < min_voxels)
        if len(sizes) == 1:
            break
        merged = False
        for _, lab in small:
            mask = labels == lab
            touching = _touching(np.where(mask, -1, labels), mask)
            if touching:  # islands on their own stay as they are
                labels[mask] = max(sorted(touching), key=lambda t: touching[t])
                merged = True
                break
        if not merged:
            break

    out = []
    final = labels[tuple(local.T)]
    for lab in sorted(set(final.tolist()), key=lambda v: int(np.argmax(final == v))):
        keep = final == lab
        out.append(FixedGrid(VoxelGrid.from_voxels(coords[keep], colors[keep])))
    return out


def chunks(
    name: str,
    shape: Shape,
    pieces: int = 8,
    seed: int = 0,
    parent: str | None = None,
    min_voxels: int | None = None,
) -> list[Part]:
    """``shape`` fractured into parts ``<name>_0``, ``<name>_1``, ..., each pivoted at its
    center of mass: the asset is the chunks together, ready for ``burst(...)``."""
    out = []
    for i, piece in enumerate(fracture(shape, pieces, seed, min_voxels)):
        centers = piece.voxelize().voxels()[0] + 0.5
        pivot = tuple(float(v) for v in centers.mean(axis=0))
        out.append(Part(f"{name}_{i}", piece, pivot=pivot, parent=parent))
    return out
