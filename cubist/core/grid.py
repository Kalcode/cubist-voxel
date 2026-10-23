"""Dense voxel grids.

Coordinates are Y-up, right-handed integers: +X right, +Y up, +Z toward the front.
A grid stores palette color indices (0 = empty) in a ``uint8`` array indexed
``[x, y, z]``, placed in the world at ``origin`` (the world coord of ``data[0, 0, 0]``).
"""

from __future__ import annotations

import numpy as np

Vec3i = tuple[int, int, int]


class VoxelGrid:
    def __init__(self, data: np.ndarray, origin: Vec3i = (0, 0, 0)):
        if data.ndim != 3 or data.dtype != np.uint8:
            raise ValueError(f"grid data must be a 3D uint8 array, got {data.ndim}D {data.dtype}")
        self.data = data
        self.origin: Vec3i = tuple(int(v) for v in origin)  # type: ignore[assignment]

    @classmethod
    def from_voxels(cls, coords: np.ndarray, colors: np.ndarray) -> VoxelGrid:
        coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
        if len(coords) == 0:
            return cls(np.zeros((0, 0, 0), dtype=np.uint8))
        lo = coords.min(axis=0)
        data = np.zeros(tuple(coords.max(axis=0) - lo + 1), dtype=np.uint8)
        local = coords - lo
        data[local[:, 0], local[:, 1], local[:, 2]] = colors
        return cls(data, tuple(lo))

    @property
    def shape(self) -> Vec3i:
        return self.data.shape  # type: ignore[return-value]

    def count(self) -> int:
        return int(np.count_nonzero(self.data))

    def colors_used(self) -> dict[int, int]:
        values, counts = np.unique(self.data[self.data != 0], return_counts=True)
        return {int(v): int(c) for v, c in zip(values, counts, strict=True)}

    def bounds(self) -> tuple[Vec3i, Vec3i] | None:
        """World-space ``(lo, hi)`` of occupied voxels, half-open; None if empty."""
        filled = np.argwhere(self.data)
        if len(filled) == 0:
            return None
        o = np.array(self.origin)
        lo = filled.min(axis=0) + o
        hi = filled.max(axis=0) + 1 + o
        return tuple(int(v) for v in lo), tuple(int(v) for v in hi)  # type: ignore[return-value]

    def cropped(self) -> VoxelGrid:
        b = self.bounds()
        if b is None:
            return VoxelGrid(np.zeros((0, 0, 0), dtype=np.uint8))
        lo, hi = b
        o = self.origin
        sl = tuple(slice(lo[i] - o[i], hi[i] - o[i]) for i in range(3))
        return VoxelGrid(self.data[sl].copy(), lo)

    def get(self, x: int, y: int, z: int) -> int:
        local = (x - self.origin[0], y - self.origin[1], z - self.origin[2])
        if any(v < 0 or v >= s for v, s in zip(local, self.data.shape, strict=True)):
            return 0
        return int(self.data[local])

    def voxels(self) -> tuple[np.ndarray, np.ndarray]:
        """World coords (N, 3) and color indices (N,), sorted by x, then y, then z."""
        filled = np.argwhere(self.data)
        colors = self.data[tuple(filled.T)] if len(filled) else np.zeros(0, dtype=np.uint8)
        return filled + np.array(self.origin), colors

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VoxelGrid):
            return NotImplemented
        a, b = self.cropped(), other.cropped()
        return a.origin == b.origin and np.array_equal(a.data, b.data)

    def __repr__(self) -> str:
        return f"VoxelGrid(shape={self.shape}, origin={self.origin}, voxels={self.count()})"
