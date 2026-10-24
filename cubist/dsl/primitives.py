"""Primitive shapes. Each starts at its local origin; place it with ``.at(x, y, z)``.

* ``box``: min corner at the origin.
* ``sphere`` / ``ellipsoid``: centered on the origin.
* ``cylinder`` / ``cone``: base centered on the origin, extending ``h`` along ``axis``.
* ``capsule`` / ``line`` / ``polyline`` / ``panel``: through absolute points.
* ``pyramid``: like ``box``, narrowing to its top center.
* ``torus``: centered on the origin, ring around ``axis``.
* ``wedge``: like ``box``, sloping up along ``rise``.
* ``lathe``: a profile of (height, radius) points revolved around ``axis`` at the origin.
* ``extrude``: a 2D polygon pushed ``depth`` voxels along ``axis`` from the origin.
* ``heightmap``: columns rising from y = 0, one per array cell.
"""

from __future__ import annotations

import numpy as np

from cubist.core.palette import Color
from cubist.dsl import sdf
from cubist.dsl.ops import axis_index
from cubist.dsl.shape import Shape, ShapeError, Vec3


def color_index(color: Color | None) -> int:
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
        self._c = color_index(color)

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
    def __init__(
        self, p0: Vec3, p1: Vec3, r: float, color: Color | None, r_end: float | None = None
    ):
        super().__init__(color)
        _positive("capsule r", r)
        self.p0 = np.asarray(p0, dtype=np.float64)
        self.p1 = np.asarray(p1, dtype=np.float64)
        self.r = float(r)
        self.r_end = self.r if r_end is None else float(r_end)
        if self.r_end < 0:
            raise ShapeError(f"capsule r_end must be >= 0, got {r_end}")

    def _distance(self, p):
        if self.r_end == self.r or np.allclose(self.p0, self.p1):
            return sdf.segment(p, self.p0, self.p1, self.r)
        return sdf.round_cone(p, self.p0, self.p1, self.r, self.r_end)

    def bounds(self):
        reach = max(self.r, self.r_end)
        return np.minimum(self.p0, self.p1) - reach, np.maximum(self.p0, self.p1) + reach


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


def capsule(
    p0: Vec3, p1: Vec3, r: float, color: Color | None = None, r_end: float | None = None
) -> Shape:
    """Round-ended tube from p0 (radius r) to p1 (radius r_end, default r): tails, horns,
    limbs, necks."""
    return Capsule(p0, p1, r, color, r_end)


class Torus(Primitive):
    def __init__(self, major: float, minor: float, axis: str, color: Color | None):
        super().__init__(color)
        _positive("torus R and r", major, minor)
        self.major, self.minor, self.axis = float(major), float(minor), axis_index(axis)

    def _distance(self, p):
        others = [i for i in range(3) if i != self.axis]
        radial = np.hypot(p[:, others[0]], p[:, others[1]])
        return sdf.torus(radial, p[:, self.axis], self.major, self.minor)

    def bounds(self):
        reach = self.major + self.minor
        lo, hi = np.full(3, -reach), np.full(3, reach)
        lo[self.axis], hi[self.axis] = -self.minor, self.minor
        return lo, hi


_RISES = {"+x": (0, 1), "-x": (0, -1), "+z": (2, 1), "-z": (2, -1)}


class Wedge(Box):
    """A box cut diagonally: height 0 at the low end of ``rise``, full height at the high end."""

    def __init__(self, size: Vec3, rise: str, color: Color | None):
        super().__init__(size, color)
        if rise not in _RISES:
            raise ShapeError(f"wedge rise must be one of {', '.join(_RISES)}, got {rise!r}")
        self.rise = rise

    def _distance(self, p):
        axis, sign = _RISES[self.rise]
        run, height = self.size[axis], self.size[1]
        along = p[:, axis] if sign > 0 else run - p[:, axis]
        # Inside the slope when y * run <= along * height.
        plane = (p[:, 1] * run - along * height) / np.hypot(run, height)
        return np.maximum(super()._distance(p), plane)


class Line(Primitive):
    def __init__(self, p0: Vec3, p1: Vec3, r: float, color: Color | None):
        super().__init__(color)
        _positive("line r", r)
        self.p0 = np.asarray(p0, dtype=np.float64)
        self.p1 = np.asarray(p1, dtype=np.float64)
        self.r = float(r)

    def _distance(self, p):
        return sdf.segment_chebyshev(p, self.p0, self.p1, self.r)

    def bounds(self):
        return np.minimum(self.p0, self.p1) - self.r, np.maximum(self.p0, self.p1) + self.r


class Lathe(Primitive):
    def __init__(self, profile, axis: str, color: Color | None):
        super().__init__(color)
        pts = np.asarray(profile, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
            raise ShapeError("lathe profile needs at least two (height, radius) points")
        if np.any(np.diff(pts[:, 0]) < 0) or pts[-1, 0] <= pts[0, 0]:
            raise ShapeError("lathe profile heights must increase from first to last point")
        if np.any(pts[:, 1] < 0):
            raise ShapeError("lathe profile radii must be >= 0")
        self.axis = axis_index(axis)
        h0, h1 = pts[0, 0], pts[-1, 0]
        # Closed outline in the (radius, height) half-plane, along the axis and back.
        outline = [(0.0, h0)] + [(r, h) for h, r in pts] + [(0.0, h1)]
        self.outline = np.array(outline)
        self.h0, self.h1, self.rmax = h0, h1, float(pts[:, 1].max())

    def _distance(self, p):
        others = [i for i in range(3) if i != self.axis]
        radial = np.hypot(p[:, others[0]], p[:, others[1]])
        return sdf.polygon(np.stack([radial, p[:, self.axis]], -1), self.outline)

    def bounds(self):
        lo, hi = np.full(3, -self.rmax), np.full(3, self.rmax)
        lo[self.axis], hi[self.axis] = self.h0, self.h1
        return lo, hi


# extrusion axis -> (polygon u axis, polygon v axis)
_EXTRUDE_PLANES = {"z": (0, 1), "y": (0, 2), "x": (2, 1)}


class Extrude(Primitive):
    def __init__(self, polygon, depth: float, axis: str, color: Color | None):
        super().__init__(color)
        verts = np.asarray(polygon, dtype=np.float64)
        if verts.ndim != 2 or verts.shape[1] != 2 or len(verts) < 3:
            raise ShapeError("extrude needs a polygon of at least three (u, v) points")
        _positive("extrude depth", depth)
        if axis not in _EXTRUDE_PLANES:
            raise ShapeError(f"extrude axis must be x, y or z, got {axis!r}")
        self.verts, self.depth = verts, float(depth)
        self.axis = axis_index(axis)
        self.u, self.v = _EXTRUDE_PLANES[axis]

    def _distance(self, p):
        d2 = sdf.polygon(np.stack([p[:, self.u], p[:, self.v]], -1), self.verts)
        dz = np.abs(p[:, self.axis] - self.depth / 2.0) - self.depth / 2.0
        outside = np.hypot(np.maximum(d2, 0.0), np.maximum(dz, 0.0))
        return outside + np.minimum(np.maximum(d2, dz), 0.0)

    def bounds(self):
        lo, hi = np.zeros(3), np.zeros(3)
        lo[self.u], hi[self.u] = self.verts[:, 0].min(), self.verts[:, 0].max()
        lo[self.v], hi[self.v] = self.verts[:, 1].min(), self.verts[:, 1].max()
        lo[self.axis], hi[self.axis] = 0.0, self.depth
        return lo, hi


class Panel(Primitive):
    def __init__(self, points, color: Color | None, thickness: float | None):
        super().__init__(color)
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
            raise ShapeError("panel needs at least three (x, y, z) points")
        u = pts[1] - pts[0]
        normal = None
        for q in pts[2:]:
            n = np.cross(u, q - pts[0])
            if np.linalg.norm(n) > 1e-9:
                normal = n / np.linalg.norm(n)
                break
        if normal is None or np.linalg.norm(u) < 1e-9:
            raise ShapeError("panel points are all on one line; give a flat polygon")
        off_plane = np.abs((pts - pts[0]) @ normal).max()
        if off_plane > 1e-6:
            raise ShapeError(f"panel points must be coplanar (one is {off_plane:.3g} off)")
        self.origin_pt, self.normal = pts[0], normal
        self.u = u / np.linalg.norm(u)
        self.v = np.cross(normal, self.u)
        local = pts - pts[0]
        self.poly = np.stack([local @ self.u, local @ self.v], axis=1)
        if thickness is None:
            # A standard digital plane: thin as possible without face gaps. The tiny offset
            # breaks exact ties to one side so 45-degree plates come out 2 wide, not 3.
            self.half, self.bias = 0.5 * float(np.abs(normal).sum()) - 5e-5, 1e-4
        else:
            _positive("panel thickness", thickness)
            self.half, self.bias = thickness / 2.0, 0.0
        self.pts = pts

    def _distance(self, p):
        local = p - self.origin_pt
        d2 = sdf.polygon(np.stack([local @ self.u, local @ self.v], axis=1), self.poly)
        dn = np.abs(local @ self.normal + self.bias) - self.half
        outside = np.hypot(np.maximum(d2, 0.0), np.maximum(dn, 0.0))
        return outside + np.minimum(np.maximum(d2, dn), 0.0)

    def bounds(self):
        pad = self.half + 1.0
        return self.pts.min(axis=0) - pad, self.pts.max(axis=0) + pad


class Pyramid(Box):
    def _distance(self, p):
        sx, sy, sz = self.size
        d = super()._distance(p)
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        for along, run in ((x, sx), (z, sz)):
            norm = np.hypot(2.0 * sy, run)
            d = np.maximum(d, (y * run + 2.0 * sy * along - 2.0 * sy * run) / norm)
            d = np.maximum(d, (y * run - 2.0 * sy * along) / norm)
        return d


def torus(R: float, r: float, axis: str = "y", color: Color | None = None) -> Shape:  # noqa: N803
    return Torus(R, r, axis, color)


def wedge(size: Vec3, rise: str = "+x", color: Color | None = None) -> Shape:
    return Wedge(size, rise, color)


def line(
    p0: Vec3,
    p1: Vec3,
    color: Color | None = None,
    connect: int = 6,
    r: float | None = None,
) -> Shape:
    """The voxels a segment passes through: ropes, spokes, braces, whiskers.

    ``connect=6`` (default) gives a face-connected path (a 2-wide staircase on diagonals,
    so it never counts as floating); ``connect=26`` one voxel per step, joined at edges and
    corners. With ``r``, a thick line of max-norm radius ``r`` instead (a distance field).
    Give endpoints directly rather than rotating a line afterwards."""
    if r is not None:
        return Line(p0, p1, r, color)
    from cubist.dsl.gridops import VoxelPath

    return VoxelPath([p0, p1], color, connect)


def polyline(points, color: Color | None = None, connect: int = 6) -> Shape:
    """``line`` through a list of points, joined end to end (bows, sagging ropes, cracks)."""
    from cubist.dsl.gridops import VoxelPath

    return VoxelPath(points, color, connect)


def panel(points, color: Color | None = None, thickness: float | None = None) -> Shape:
    """A flat plate through 3+ coplanar points at any angle: wings, sails, roof planes.

    The default thickness is the thinnest that stays face-connected and watertight at the
    plate's tilt (one voxel when axis-aligned, a 2-wide staircase at 45 degrees)."""
    return Panel(points, color, thickness)


def pyramid(size: Vec3, color: Color | None = None) -> Shape:
    """A box narrowing to its top center: hip roofs, spires, tent tops. Odd widths keep a
    one-voxel tip (the apex needs a voxel column)."""
    return Pyramid(size, color)


def lathe(profile, axis: str = "y", color: Color | None = None) -> Shape:
    return Lathe(profile, axis, color)


def extrude(polygon, depth: float, axis: str = "z", color: Color | None = None) -> Shape:
    return Extrude(polygon, depth, axis, color)


def heightmap(heights, color: Color | None = None) -> Shape:
    """Columns from y = 0 up to ``heights[x][z]`` voxels (rounded)."""
    from cubist.dsl.gridops import Heightmap

    return Heightmap(heights, color)
