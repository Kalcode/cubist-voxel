"""Grid-stage shapes: voxel-level morphology (hollow, erode, dilate, round, chamfer) and
heightmaps. Distances are exact Euclidean distances between voxel centers."""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from cubist.core.grid import VoxelGrid
from cubist.dsl.context import active
from cubist.dsl.shape import Shape, ShapeError, _occupancy_only, check_finite

AXIS_DIRS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


class GridShape(Shape):
    """A shape that exists as voxels. Subclasses implement ``_grid``."""

    is_sdf = False

    def _grid(self) -> VoxelGrid:
        raise NotImplementedError

    def voxelize(self) -> VoxelGrid:
        ctx = active()
        if ctx is None:
            return self._grid()
        key = (id(ctx), _occupancy_only.get())
        if getattr(self, "_cache_key", None) != key:
            self._cache = self._grid()
            self._cache_key = key
        return self._cache

    def bounds(self):
        b = self.voxelize().bounds()
        if b is None:
            return np.zeros(3), np.zeros(3)
        return np.array(b[0], dtype=np.float64), np.array(b[1], dtype=np.float64)

    def _eval(self, p):
        raise ShapeError(
            f"{type(self).__name__} is made of voxels and has no distance field; "
            "use it in union/subtract/intersect, transforms, or paint ops"
        )


def padded(grid: VoxelGrid, pad: int) -> tuple[np.ndarray, tuple[int, int, int]]:
    """The grid's colors with ``pad`` empty cells on every side, and the new origin."""
    g = grid.cropped()
    data = np.pad(g.data, pad)
    return data, tuple(int(o) - pad for o in g.origin)


def neighbor(mask: np.ndarray, d) -> np.ndarray:
    """``out[v] = mask[v + d]``, False beyond the array."""
    out = np.zeros_like(mask)
    src = tuple(slice(max(k, 0), mask.shape[i] + min(k, 0)) for i, k in enumerate(d))
    dst = tuple(slice(max(-k, 0), mask.shape[i] + min(-k, 0)) for i, k in enumerate(d))
    out[dst] = mask[src]
    return out


def exposed(filled: np.ndarray, d) -> np.ndarray:
    """Filled cells whose neighbor in direction d is empty."""
    return filled & ~neighbor(filled, d)


class _Morph(GridShape):
    pad = 1

    def __init__(self, shape: Shape):
        self.shape = shape

    def _color_sources(self):
        return [self.shape]

    def _grid(self) -> VoxelGrid:
        src = self.shape.voxelize()
        if src.count() == 0:
            return src
        data, origin = padded(src, self.pad)
        return VoxelGrid(self._apply(data), origin).cropped()

    def _apply(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def _depth(filled: np.ndarray) -> np.ndarray:
    """Distance from each filled cell's center to the nearest empty cell's center."""
    return ndimage.distance_transform_edt(filled)


class Hollow(_Morph):
    """Keep voxels within ``t`` of empty space in any of the 26 directions (chessboard
    distance), so the shell stays face-connected and watertight on curved surfaces."""

    def __init__(self, shape: Shape, t: float):
        super().__init__(shape)
        if t <= 0:
            raise ShapeError(f"hollow thickness must be > 0, got {t}")
        self.t = float(t)

    def _apply(self, data):
        filled = data > 0
        depth = ndimage.distance_transform_cdt(filled, metric="chessboard")
        return np.where(filled & (depth <= self.t), data, 0).astype(np.uint8)


class Erode(_Morph):
    def __init__(self, shape: Shape, r: float):
        super().__init__(shape)
        self.r = float(r)

    def _apply(self, data):
        filled = data > 0
        return np.where(filled & (_depth(filled) > self.r), data, 0).astype(np.uint8)


class Dilate(_Morph):
    def __init__(self, shape: Shape, r: float, color):
        super().__init__(shape)
        self.r = float(r)
        self.pad = math.ceil(self.r) + 1
        self.color = color

    def _apply(self, data):
        filled = data > 0
        dist, idx = ndimage.distance_transform_edt(~filled, return_indices=True)
        grow = ~filled & (dist <= self.r)
        out = data.copy()
        if self.color is not None:
            out[grow] = self.color.index
        else:
            out[grow] = data[idx[0][grow], idx[1][grow], idx[2][grow]]
        return out


class Opening(_Morph):
    """Round convex features: erode by r, grow back by r, never past the original."""

    def __init__(self, shape: Shape, r: float):
        super().__init__(shape)
        if r <= 0:
            raise ShapeError(f"round radius must be > 0, got {r}")
        self.r = float(r)

    def _apply(self, data):
        filled = data > 0
        core = filled & (_depth(filled) > self.r)
        if not core.any():
            return np.zeros_like(data)
        reach = ndimage.distance_transform_edt(~core) <= self.r
        return np.where(filled & reach, data, 0).astype(np.uint8)


class Chamfer(_Morph):
    def __init__(self, shape: Shape, c: int):
        super().__init__(shape)
        if int(c) != c or c < 1:
            raise ShapeError(f"chamfer takes a whole number of voxels >= 1, got {c}")
        self.c = int(c)

    def _apply(self, data):
        data = data.copy()
        for _ in range(self.c):
            filled = data > 0
            axes = sum(
                (exposed(filled, d) | exposed(filled, tuple(-k for k in d))).astype(int)
                for d in AXIS_DIRS[::2]
            )
            data[axes >= 2] = 0
        return data


class Heightmap(GridShape):
    def __init__(self, heights, color):
        h = np.asarray(heights, dtype=np.float64)
        if h.ndim != 2:
            raise ShapeError(f"heightmap needs a 2D array indexed [x][z], got {h.ndim}D")
        check_finite("heightmap heights", h)
        if np.any(h < 0):
            raise ShapeError("heightmap heights must be >= 0")
        self.heights = np.rint(h).astype(np.int64)
        self.color = color

    def _grid(self) -> VoxelGrid:
        from cubist.dsl.primitives import color_index

        c = color_index(self.color)
        if c == 0:
            raise ShapeError("heightmap needs color=p.<ramp>.<shade>")
        nx, nz = self.heights.shape
        ny = int(self.heights.max(initial=0))
        ys = np.arange(ny)[None, :, None]
        filled = ys < self.heights[:, None, :]
        return VoxelGrid(np.where(filled, c, 0).astype(np.uint8)).cropped()


def _walk6(p0: np.ndarray, p1: np.ndarray) -> list[tuple[int, int, int]]:
    """Every cell the segment passes through, stepping one axis at a time (Amanatides-Woo);
    exact corner crossings step x, then y, then z, so the path stays face-connected."""
    cell = np.floor(p0).astype(np.int64)
    end = np.floor(p1).astype(np.int64)
    d = p1 - p0
    step = np.sign(d).astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        boundary = cell + (step > 0)
        t_max = np.where(d != 0, (boundary - p0) / d, np.inf)
        t_delta = np.where(d != 0, 1.0 / np.abs(d), np.inf)
    cells = [tuple(int(v) for v in cell)]
    for _ in range(int(np.abs(end - cell).sum())):
        axis = int(np.argmin(t_max))
        cell[axis] += step[axis]
        t_max[axis] += t_delta[axis]
        cells.append(tuple(int(v) for v in cell))
    return cells


def _walk26(p0: np.ndarray, p1: np.ndarray) -> list[tuple[int, int, int]]:
    """One cell per step along the dominant axis (3D Bresenham between end cells)."""
    a = np.floor(p0).astype(np.int64)
    b = np.floor(p1).astype(np.int64)
    n = int(np.abs(b - a).max())
    if n == 0:
        return [tuple(int(v) for v in a)]
    t = np.arange(n + 1)[:, None] / n
    cells = np.floor(a + (b - a) * t + 0.5).astype(np.int64)
    return [tuple(int(v) for v in c) for c in cells]


class VoxelPath(GridShape):
    """Voxels along a polyline: face-connected (connect=6) or one per step (connect=26)."""

    kind = "line"

    def __init__(self, points, color, connect: int):
        from cubist.dsl.primitives import color_index

        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
            raise ShapeError("a line needs at least two (x, y, z) points")
        check_finite("line points", points)
        if connect not in (6, 26):
            raise ShapeError(f"connect must be 6 or 26, got {connect}")
        self.points, self.connect = pts, connect
        self.color = color
        self._c = color_index(color)

    def _grid(self) -> VoxelGrid:
        if self._c == 0 and not _occupancy_only.get():
            raise self._missing_color()
        walk = _walk6 if self.connect == 6 else _walk26
        cells = {
            c for a, b in zip(self.points[:-1], self.points[1:], strict=True) for c in walk(a, b)
        }
        coords = np.array(sorted(cells))
        color = self._c or 255
        return VoxelGrid.from_voxels(coords, np.full(len(coords), color, dtype=np.uint8))
