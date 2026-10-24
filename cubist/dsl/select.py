"""Selections choose which voxels a paint op touches. Combine with ``&``, ``|``, ``-``, ``~``.

Any shape also works as a selection: the voxels whose centers it contains.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cubist.core.grid import VoxelGrid
from cubist.dsl.gridops import exposed, neighbor
from cubist.dsl.shape import EPS, Shape, ShapeError

DIRECTIONS = {
    "top": (0, 1, 0),
    "bottom": (0, -1, 0),
    "right": (1, 0, 0),
    "left": (-1, 0, 0),
    "front": (0, 0, 1),
    "back": (0, 0, -1),
}
_ALIASES = {"+y": "top", "-y": "bottom", "+x": "right", "-x": "left", "+z": "front", "-z": "back"}


def direction(name: str) -> tuple[int, int, int]:
    key = _ALIASES.get(name, name)
    if key not in DIRECTIONS:
        raise ShapeError(
            f"unknown face direction {name!r}; use {', '.join(DIRECTIONS)} (or +y, -x, ...)"
        )
    return DIRECTIONS[key]


def _coords(grid: VoxelGrid) -> np.ndarray:
    return np.indices(grid.shape).transpose(1, 2, 3, 0) + np.array(grid.origin)


class Selection:
    def mask(self, grid: VoxelGrid) -> np.ndarray:
        """Boolean array shaped like ``grid.data``."""
        raise NotImplementedError

    def __and__(self, other):
        return _Combine(self, as_selection(other), np.logical_and)

    def __or__(self, other):
        return _Combine(self, as_selection(other), np.logical_or)

    def __sub__(self, other):
        return _Combine(self, ~as_selection(other), np.logical_and)

    def __invert__(self):
        return _Not(self)


class _Combine(Selection):
    def __init__(self, a, b, op):
        self.a, self.b, self.op = a, b, op

    def mask(self, grid):
        return self.op(self.a.mask(grid), self.b.mask(grid))


class _Not(Selection):
    def __init__(self, a):
        self.a = a

    def mask(self, grid):
        return ~self.a.mask(grid)


class _All(Selection):
    def mask(self, grid):
        return np.ones(grid.shape, dtype=bool)


class _Band(Selection):
    def __init__(self, axis: str, at: float, height: float):
        if axis not in ("x", "y", "z"):
            raise ShapeError(f"band axis must be x, y or z, got {axis!r}")
        self.axis, self.at, self.height = "xyz".index(axis), at, height

    def mask(self, grid):
        c = _coords(grid)[..., self.axis]
        return (c >= self.at) & (c < self.at + self.height)


class _Region(Selection):
    def __init__(self, lo, hi):
        self.lo, self.hi = np.asarray(lo), np.asarray(hi)

    def mask(self, grid):
        c = _coords(grid)
        return np.all((c >= self.lo) & (c < self.hi), axis=-1)


class _Colored(Selection):
    def __init__(self, colors):
        self.indices = [c.index for c in colors]

    def mask(self, grid):
        return np.isin(grid.data, self.indices)


class _Faces(Selection):
    def __init__(self, name: str):
        self.d = direction(name)

    def mask(self, grid):
        return exposed(grid.data > 0, self.d)


class _Surface(Selection):
    def mask(self, grid):
        filled = grid.data > 0
        hidden = filled.copy()
        for d in DIRECTIONS.values():
            hidden &= neighbor(filled, d)
        return filled & ~hidden


class _Inside(Selection):
    def __init__(self, shape: Shape):
        self.shape = shape

    def mask(self, grid):
        cells = _coords(grid).reshape(-1, 3)
        if self.shape.is_sdf:
            d, _ = self.shape._eval(cells + 0.5)
            inside = d <= EPS
        else:
            inside = self.shape.voxelize().lookup(cells) > 0
        return inside.reshape(grid.shape)


def band(axis: str, at: float, height: float = 1) -> Selection:
    """Voxels whose ``axis`` coordinate is in ``[at, at + height)``."""
    return _Band(axis, at, height)


def region(lo: Sequence[float], hi: Sequence[float]) -> Selection:
    """Voxels inside the box ``[lo, hi)`` in world coordinates."""
    return _Region(lo, hi)


def colored(*colors) -> Selection:
    """Voxels currently painted any of ``colors``."""
    return _Colored(colors)


def faces(name: str) -> Selection:
    """Voxels whose face toward ``name`` (top, bottom, left, right, front, back) is exposed."""
    return _Faces(name)


def surface() -> Selection:
    """Voxels with at least one exposed face."""
    return _Surface()


def inside(shape: Shape) -> Selection:
    """Voxels whose centers lie inside ``shape``."""
    return _Inside(shape)


def as_selection(where) -> Selection:
    if where is None:
        return _All()
    if isinstance(where, Selection):
        return where
    if isinstance(where, Shape):
        return _Inside(where)
    raise ShapeError(f"where= takes a selection or a shape, got {where!r}")
