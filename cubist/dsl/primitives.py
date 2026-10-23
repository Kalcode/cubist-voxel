"""Primitive shapes. Each starts at its local origin; place it with ``.at(x, y, z)``.

* ``box``: min corner at the origin.
* ``sphere`` / ``ellipsoid``: centered on the origin.
* ``cylinder`` / ``cone``: base centered on the origin, extending ``h`` along ``axis``.
* ``capsule``: between two absolute points.
"""

from __future__ import annotations

import numpy as np

from cubist.core.palette import Color
from cubist.dsl import sdf
from cubist.dsl.ops import axis_index
from cubist.dsl.shape import Shape, ShapeError, Vec3


def _index(color: Color | None) -> int:
    if color is None:
        return 0
    if not isinstance(color, Color):
        raise ShapeError(f"color must be a palette color like p.wood.dark, got {color!r}")
    return color.index


def _positive(name: str, *values: float) -> None:
    if any(v <= 0 for v in values):
        raise ShapeError(f"{name} must be > 0, got {values if len(values) > 1 else values[0]}")


class Primitive(Shape):
    def __init__(self, color: Color | None):
        self.color = color
        self._c = _index(color)

    def _eval(self, p):
        d = self._distance(p)
        return d, np.full(len(d), self._c, dtype=np.int16)

    def _distance(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Box(Primitive):
    def __init__(self, size: Vec3, color: Color | None):
        super().__init__(color)
        self.size = np.asarray(size, dtype=np.float64)
        _positive("box size", *self.size)

    def _distance(self, p):
        half = self.size / 2.0
        return sdf.box(p - half, half)

    def bounds(self):
        return np.zeros(3), self.size.copy()


class Ellipsoid(Primitive):
    def __init__(self, radii: Vec3, color: Color | None):
        super().__init__(color)
        self.radii = np.asarray(radii, dtype=np.float64)
        _positive("ellipsoid radii", *self.radii)

    def _distance(self, p):
        if np.all(self.radii == self.radii[0]):
            return sdf.sphere(p, self.radii[0])
        return sdf.ellipsoid(p, self.radii)

    def bounds(self):
        return -self.radii.copy(), self.radii.copy()


class _Axial(Primitive):
    """Shapes of revolution standing on the origin along an axis."""

    def __init__(self, r: float, h: float, axis: str, color: Color | None):
        super().__init__(color)
        _positive("r and h", r, h)
        self.r, self.h, self.axis = float(r), float(h), axis_index(axis)

    def _split(self, p):
        axial = p[:, self.axis] - self.h / 2.0
        others = [i for i in range(3) if i != self.axis]
        radial = np.hypot(p[:, others[0]], p[:, others[1]])
        return radial, axial

    def bounds(self):
        lo, hi = np.full(3, -self.r), np.full(3, self.r)
        lo[self.axis], hi[self.axis] = 0.0, self.h
        return lo, hi


class Cylinder(_Axial):
    def _distance(self, p):
        radial, axial = self._split(p)
        return sdf.capped_cylinder(radial, axial, self.r, self.h / 2.0)


class Cone(_Axial):
    def __init__(self, r: float, h: float, axis: str, r_top: float, color: Color | None):
        super().__init__(r, h, axis, color)
        if r_top < 0:
            raise ShapeError(f"r_top must be >= 0, got {r_top}")
        self.r_top = float(r_top)

    def _distance(self, p):
        radial, axial = self._split(p)
        return sdf.capped_cone(radial, axial, self.h / 2.0, self.r, self.r_top)

    def bounds(self):
        lo, hi = super().bounds()
        reach = max(self.r, self.r_top)
        for i in range(3):
            if i != self.axis:
                lo[i], hi[i] = -reach, reach
        return lo, hi


class Capsule(Primitive):
    def __init__(self, p0: Vec3, p1: Vec3, r: float, color: Color | None):
        super().__init__(color)
        _positive("capsule r", r)
        self.p0 = np.asarray(p0, dtype=np.float64)
        self.p1 = np.asarray(p1, dtype=np.float64)
        self.r = float(r)

    def _distance(self, p):
        return sdf.segment(p, self.p0, self.p1, self.r)

    def bounds(self):
        return np.minimum(self.p0, self.p1) - self.r, np.maximum(self.p0, self.p1) + self.r


def box(size: Vec3, color: Color | None = None) -> Shape:
    return Box(size, color)


def ellipsoid(radii: Vec3, color: Color | None = None) -> Shape:
    return Ellipsoid(radii, color)


def sphere(r: float, color: Color | None = None) -> Shape:
    return Ellipsoid((r, r, r), color)


def cylinder(r: float, h: float, axis: str = "y", color: Color | None = None) -> Shape:
    return Cylinder(r, h, axis, color)


def cone(
    r: float, h: float, axis: str = "y", r_top: float = 0.0, color: Color | None = None
) -> Shape:
    return Cone(r, h, axis, r_top, color)


def capsule(p0: Vec3, p1: Vec3, r: float, color: Color | None = None) -> Shape:
    return Capsule(p0, p1, r, color)
