"""CSG, mirror, affine transforms, taper, and displace.

Each node works on distance fields when all its inputs are SDF shapes, and on voxels
otherwise, with the same color rules either way.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from cubist.core.grid import VoxelGrid
from cubist.dsl import noise, sdf
from cubist.dsl.shape import EPS, Shape, ShapeError, Vec3, require_sdf

_AXES = {"x": 0, "y": 1, "z": 2}


def axis_index(axis: str) -> int:
    if axis not in _AXES:
        raise ShapeError(f"axis must be 'x', 'y' or 'z', got {axis!r}")
    return _AXES[axis]


def _union_color(da, ca, db, cb):
    # Later shape wins where it is inside; elsewhere the nearer shape's color.
    return np.where(db <= EPS, cb, np.where(da <= db, ca, cb))


def _cells(grid: VoxelGrid) -> np.ndarray:
    """World coords of every cell in the grid's array, in array order."""
    idx = np.indices(grid.shape).reshape(3, -1).T
    return idx + np.array(grid.origin)


class Union(Shape):
    def __init__(self, shapes: Sequence[Shape]):
        self.shapes = list(shapes)

    @property
    def is_sdf(self):
        return all(s.is_sdf for s in self.shapes)

    def _eval(self, p):
        d, c = self.shapes[0]._eval(p)
        for shape in self.shapes[1:]:
            db, cb = shape._eval(p)
            c = _union_color(d, c, db, cb)
            d = np.minimum(d, db)
        return d, c

    def bounds(self):
        los, his = zip(*(s.bounds() for s in self.shapes), strict=True)
        return np.min(los, axis=0), np.max(his, axis=0)

    def voxelize(self):
        if self.is_sdf:
            return super().voxelize()
        return VoxelGrid.merge([s.voxelize() for s in self.shapes])


class SmoothUnion(Shape):
    def __init__(self, a: Shape, b: Shape, k: float):
        require_sdf(a, "smooth_union")
        require_sdf(b, "smooth_union")
        if k <= 0:
            raise ShapeError(f"smooth_union k must be > 0, got {k}")
        self.a, self.b, self.k = a, b, float(k)

    def _eval(self, p):
        da, ca = self.a._eval(p)
        db, cb = self.b._eval(p)
        return sdf.smooth_min(da, db, self.k), _union_color(da, ca, db, cb)

    def bounds(self):
        (alo, ahi), (blo, bhi) = self.a.bounds(), self.b.bounds()
        return np.minimum(alo, blo) - self.k, np.maximum(ahi, bhi) + self.k


class Subtract(Shape):
    def __init__(self, a: Shape, b: Shape):
        self.a, self.b = a, b

    @property
    def is_sdf(self):
        return self.a.is_sdf and self.b.is_sdf

    def _eval(self, p):
        da, ca = self.a._eval(p)
        db, _ = self.b._eval(p)
        return np.maximum(da, -db), ca

    def bounds(self):
        return self.a.bounds()

    def voxelize(self):
        if self.is_sdf:
            return super().voxelize()
        ga, gb = self.a.voxelize(), self.b.voxelize()
        keep = gb.lookup(_cells(ga)).reshape(ga.shape) == 0
        return VoxelGrid(np.where(keep, ga.data, 0).astype(np.uint8), ga.origin).cropped()


class Intersect(Shape):
    def __init__(self, a: Shape, b: Shape):
        self.a, self.b = a, b

    @property
    def is_sdf(self):
        return self.a.is_sdf and self.b.is_sdf

    def _eval(self, p):
        da, ca = self.a._eval(p)
        db, _ = self.b._eval(p)
        return np.maximum(da, db), ca

    def bounds(self):
        (alo, ahi), (blo, bhi) = self.a.bounds(), self.b.bounds()
        return np.maximum(alo, blo), np.minimum(ahi, bhi)

    def voxelize(self):
        if self.is_sdf:
            return super().voxelize()
        ga, gb = self.a.voxelize(), self.b.voxelize()
        keep = gb.lookup(_cells(ga)).reshape(ga.shape) > 0
        return VoxelGrid(np.where(keep, ga.data, 0).astype(np.uint8), ga.origin).cropped()


class Mirror(Shape):
    """Symmetric union of a shape and its reflection.

    Colors come from a canonical point per mirrored pair (the one on the low side of
    the plane), so both halves always match.
    """

    def __init__(self, shape: Shape, axis: str, about: float):
        self.shape, self.axis, self.about = shape, axis_index(axis), about
        if not shape.is_sdf and abs(2 * about - round(2 * about)) > 1e-9:
            raise ShapeError(
                f"mirroring a painted shape needs a plane on the voxel grid (a whole or half "
                f"number), got {about}"
            )

    @property
    def is_sdf(self):
        return self.shape.is_sdf

    def _reflect_points(self, p):
        q = p.copy()
        q[:, self.axis] = 2.0 * self.about - q[:, self.axis]
        return q

    def _eval(self, p):
        da, ca = self.shape._eval(p)
        db, cb = self.shape._eval(self._reflect_points(p))
        low = p[:, self.axis] <= self.about
        # At the canonical point, the original shape's color wins if it is inside there.
        c_low = np.where(da <= EPS, ca, np.where(db <= EPS, cb, np.where(da <= db, ca, cb)))
        c_high = np.where(db <= EPS, cb, np.where(da <= EPS, ca, np.where(db <= da, cb, ca)))
        return np.minimum(da, db), np.where(low, c_low, c_high)

    def bounds(self):
        lo, hi = self.shape.bounds()
        rlo, rhi = lo.copy(), hi.copy()
        rlo[self.axis] = 2.0 * self.about - hi[self.axis]
        rhi[self.axis] = 2.0 * self.about - lo[self.axis]
        return np.minimum(lo, rlo), np.maximum(hi, rhi)

    def voxelize(self):
        if self.is_sdf:
            return super().voxelize()
        g = self.shape.voxelize()
        if g.count() == 0:
            return g
        a = self.axis
        flip = int(round(2 * self.about)) - 1  # cell i reflects to cell flip - i
        coords, colors = g.voxels()
        mirrored = coords.copy()
        mirrored[:, a] = flip - coords[:, a]
        frame = VoxelGrid.merge([g, VoxelGrid.from_voxels(mirrored, colors)])
        cells = _cells(frame)
        reflected = cells.copy()
        reflected[:, a] = flip - cells[:, a]
        orig, refl = g.lookup(cells), g.lookup(reflected)
        low = cells[:, a] + 0.5 <= self.about
        out = np.where(low, np.where(orig > 0, orig, refl), np.where(refl > 0, refl, orig))
        return VoxelGrid(out.astype(np.uint8).reshape(frame.shape), frame.origin).cropped()


# --- affine transforms ------------------------------------------------------


class Transform(Shape):
    """A child shape placed by a 4x4 affine matrix (local -> world)."""

    def __init__(self, shape: Shape, matrix: np.ndarray):
        linear = matrix[:3, :3]
        if abs(np.linalg.det(linear)) < 1e-12:
            raise ShapeError("transform is degenerate (zero scale?)")
        self.shape = shape
        self.matrix = matrix
        self.inverse = np.linalg.inv(matrix)
        # Distances shrink by at most the smallest singular value; using it keeps the
        # sign exact and the distance a lower bound under non-uniform scale.
        self.dist_scale = float(np.linalg.svd(linear, compute_uv=False).min())

    @property
    def is_sdf(self):
        return self.shape.is_sdf

    def _eval(self, p):
        local = p @ self.inverse[:3, :3].T + self.inverse[:3, 3]
        d, c = self.shape._eval(local)
        return d * self.dist_scale, c

    def bounds(self):
        lo, hi = self.shape.bounds()
        corners = np.array(
            [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        )
        world = corners @ self.matrix[:3, :3].T + self.matrix[:3, 3]
        return world.min(axis=0), world.max(axis=0)

    def voxelize(self):
        if self.is_sdf:
            return super().voxelize()
        src = self.shape.voxelize()
        if src.count() == 0:
            return src
        shift = self.matrix[:3, 3]
        if np.allclose(self.matrix[:3, :3], np.eye(3)) and np.allclose(shift, np.round(shift)):
            return VoxelGrid(src.data, tuple(np.add(src.origin, np.round(shift).astype(int))))
        # Resample: each target voxel center maps back to the source cell it lands in.
        lo, hi = self.bounds()
        i0 = np.floor(lo).astype(np.int64) - 1
        i1 = np.ceil(hi).astype(np.int64) + 1
        frame = VoxelGrid(np.zeros(tuple(i1 - i0), dtype=np.uint8), tuple(i0))
        centers = _cells(frame) + 0.5
        local = centers @ self.inverse[:3, :3].T + self.inverse[:3, 3]
        colors = src.lookup(np.floor(local + 1e-9).astype(np.int64))
        return VoxelGrid(colors.reshape(frame.shape), frame.origin).cropped()


def transformed(shape: Shape, matrix: np.ndarray) -> Shape:
    if isinstance(shape, Transform):
        return Transform(shape.shape, matrix @ shape.matrix)
    return Transform(shape, matrix)


def translation(offset: Vec3) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = offset
    return m


def _about(linear: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = linear
    m[:3, 3] = pivot - linear @ pivot
    return m


def _exact_cos_sin(degrees: float) -> tuple[float, float]:
    if float(degrees) % 90 == 0:
        return [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)][int(degrees // 90) % 4]
    rad = math.radians(degrees)
    return math.cos(rad), math.sin(rad)


def rotation(degrees: float, axis: str | Vec3, pivot: np.ndarray) -> np.ndarray:
    c, s = _exact_cos_sin(degrees)
    if isinstance(axis, str):
        u = np.zeros(3)
        u[axis_index(axis)] = 1.0
    else:
        u = np.asarray(axis, dtype=np.float64)
        n = np.linalg.norm(u)
        if n == 0:
            raise ShapeError("rotation axis must be non-zero")
        u = u / n
    k = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    linear = c * np.eye(3) + s * k + (1 - c) * np.outer(u, u)
    return _about(linear, pivot)


def scaling(factor: float | Vec3, pivot: np.ndarray) -> np.ndarray:
    f = np.broadcast_to(np.asarray(factor, dtype=np.float64), (3,))
    return _about(np.diag(f), pivot)


def reflection(axis: str, about: float) -> np.ndarray:
    f = np.ones(3)
    f[axis_index(axis)] = -1.0
    pivot = np.zeros(3)
    pivot[axis_index(axis)] = about
    return _about(np.diag(f), pivot)


# --- field modifiers --------------------------------------------------------


class Taper(Shape):
    """Cross-section scaled linearly along an axis: 1 at the bounds' bottom, ``top`` at the top."""

    def __init__(self, shape: Shape, top: float, axis: str):
        require_sdf(shape, "taper")
        if top <= 0:
            raise ShapeError(f"taper top must be > 0, got {top}")
        self.shape, self.top, self.axis = shape, float(top), axis_index(axis)
        self.lo, self.hi = shape.bounds()
        self.center = (self.lo + self.hi) / 2.0

    def _factor(self, h: np.ndarray) -> np.ndarray:
        span = max(self.hi[self.axis] - self.lo[self.axis], 1e-9)
        t = np.clip((h - self.lo[self.axis]) / span, 0.0, 1.0)
        return 1.0 + (self.top - 1.0) * t

    def _eval(self, p):
        s = self._factor(p[:, self.axis])
        q = p.copy()
        for i in range(3):
            if i != self.axis:
                q[:, i] = self.center[i] + (p[:, i] - self.center[i]) / s
        d, c = self.shape._eval(q)
        return d * np.minimum(s, 1.0), c

    def bounds(self):
        grow = max(1.0, self.top)
        lo, hi = self.lo.copy(), self.hi.copy()
        for i in range(3):
            if i != self.axis:
                lo[i] = self.center[i] + (lo[i] - self.center[i]) * grow
                hi[i] = self.center[i] + (hi[i] - self.center[i]) * grow
        return lo, hi


class Displace(Shape):
    def __init__(self, shape: Shape, amp: float, scale: float, seed: int, kind: str, octaves: int):
        require_sdf(shape, "displace")
        if kind not in noise.NOISES:
            raise ShapeError(f"noise must be one of {', '.join(noise.NOISES)}, got {kind!r}")
        if scale <= 0:
            raise ShapeError(f"displace scale must be > 0, got {scale}")
        self.shape, self.amp, self.scale = shape, float(amp), float(scale)
        self.seed, self.kind, self.octaves = int(seed), kind, max(1, int(octaves))

    def _eval(self, p):
        d, c = self.shape._eval(p)
        if self.amp == 0:
            return d, c
        n = noise.fbm(p / self.scale, noise.mix_seed(self.seed), self.octaves, self.kind)
        return d + self.amp * n, c

    def bounds(self):
        lo, hi = self.shape.bounds()
        return lo - abs(self.amp), hi + abs(self.amp)
