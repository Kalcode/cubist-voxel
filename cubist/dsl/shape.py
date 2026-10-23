"""The Shape base class: a signed distance field with colors, sampled into a grid on demand.

A voxel is filled when its center lies inside the shape or exactly on its surface.
Closed sampling keeps exact ties symmetric, so a shape mirrored about a plane samples
to a mirrored set of voxels.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cubist.core.grid import VoxelGrid

# Distance at or below which a sample counts as inside. Absorbs float noise from
# transforms without visibly growing shapes.
EPS = 1e-9
MAX_SAMPLE_AXIS = 1024
MAX_SAMPLE_POINTS = 64 * 1024 * 1024
_CHUNK_POINTS = 1 << 20

Vec3 = Sequence[float]


class ShapeError(ValueError):
    """A shape cannot be built as written."""


class Shape:
    """Base for all shapes. Subclasses implement ``_eval`` and ``bounds``."""

    def _eval(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Distances (N,) and color indices (N,) for points (N, 3). Color 0 = none."""
        raise NotImplementedError

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Conservative world-space AABB ``(lo, hi)`` of the shape's inside."""
        raise NotImplementedError

    def sdf(self, points: np.ndarray) -> np.ndarray:
        return self._eval(np.asarray(points, dtype=np.float64).reshape(-1, 3))[0]

    # --- CSG ---------------------------------------------------------------

    def union(self, *others: Shape) -> Shape:
        """Combine shapes. Where shapes overlap, the later one's color wins."""
        from cubist.dsl.ops import Union

        return Union([self, *others])

    def subtract(self, *others: Shape) -> Shape:
        """Carve ``others`` out of this shape; the result keeps this shape's colors."""
        from cubist.dsl.ops import Subtract

        return Subtract(self, others[0].union(*others[1:]) if len(others) > 1 else others[0])

    def intersect(self, other: Shape) -> Shape:
        """Keep only the overlap; the result keeps this shape's colors."""
        from cubist.dsl.ops import Intersect

        return Intersect(self, other)

    def smooth_union(self, other: Shape, k: float) -> Shape:
        """Union with a fillet of blend radius ``k`` voxels where the shapes meet."""
        from cubist.dsl.ops import SmoothUnion

        return SmoothUnion(self, other, k)

    __or__ = union
    __sub__ = subtract
    __and__ = intersect

    # --- transforms ----------------------------------------------------------

    def at(self, x: float, y: float, z: float) -> Shape:
        """Move the shape by (x, y, z). Primitives start at their local origin,
        so ``box(...).at(x, y, z)`` puts the box's min corner at (x, y, z)."""
        from cubist.dsl.ops import transformed, translation

        return transformed(self, translation((x, y, z)))

    translate = at

    def rotate(self, degrees: float, axis: str | Vec3 = "y", about: Vec3 | None = None) -> Shape:
        """Rotate by any angle (right-hand rule) about ``about``, default the bounds center."""
        from cubist.dsl.ops import rotation, transformed

        return transformed(self, rotation(degrees, axis, self._pivot(about)))

    def scale(self, factor: float | Vec3, about: Vec3 | None = None) -> Shape:
        """Scale about ``about``, default the bounds center."""
        from cubist.dsl.ops import scaling, transformed

        return transformed(self, scaling(factor, self._pivot(about)))

    def reflect(self, axis: str, about: float | None = None) -> Shape:
        """Replace the shape with its reflection across the plane ``axis = about``."""
        from cubist.dsl.ops import reflection, transformed

        return transformed(self, reflection(axis, self._plane(axis, about)))

    def mirror(self, axis: str, about: float | None = None) -> Shape:
        """Keep the shape and add its reflection across the plane ``axis = about``.

        The result is exactly symmetric in both form and color.
        """
        from cubist.dsl.ops import Mirror

        return Mirror(self, axis, self._plane(axis, about))

    def _pivot(self, about: Vec3 | None) -> np.ndarray:
        if about is not None:
            return np.asarray(about, dtype=np.float64)
        lo, hi = self.bounds()
        return (lo + hi) / 2.0

    def _plane(self, axis: str, about: float | None) -> float:
        from cubist.dsl.ops import axis_index

        i = axis_index(axis)
        return float(self._pivot(None)[i]) if about is None else float(about)

    # --- sampling ------------------------------------------------------------

    def voxelize(self) -> VoxelGrid:
        """Sample the shape at voxel centers into a tight grid."""
        lo, hi = self.bounds()
        i0 = np.floor(lo).astype(np.int64) - 1
        i1 = np.ceil(hi).astype(np.int64) + 1
        size = i1 - i0
        if np.any(size > MAX_SAMPLE_AXIS) or np.prod(size) > MAX_SAMPLE_POINTS:
            raise ShapeError(
                f"shape is too large to sample: bounds {lo.tolist()} to {hi.tolist()} "
                f"(limit {MAX_SAMPLE_AXIS} voxels per axis)"
            )
        data = np.zeros(tuple(size), dtype=np.uint8)
        ys = np.arange(i0[1], i1[1]) + 0.5
        zs = np.arange(i0[2], i1[2]) + 0.5
        slab = max(1, _CHUNK_POINTS // max(1, size[1] * size[2]))
        for xa in range(0, size[0], slab):
            xs = np.arange(i0[0] + xa, i0[0] + min(xa + slab, size[0])) + 0.5
            pts = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
            d, c = self._eval(pts)
            inside = d <= EPS
            if np.any(inside & (c <= 0)):
                raise ShapeError(
                    "shape has voxels with no color; pass color=p.<ramp>.<shade> to the primitive"
                )
            data[xa : xa + len(xs)] = (
                np.where(inside, c, 0).astype(np.uint8).reshape(len(xs), size[1], size[2])
            )
        return VoxelGrid(data, tuple(i0)).cropped()
