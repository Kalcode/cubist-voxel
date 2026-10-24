"""The Shape base class and every chainable method asset scripts call.

Shapes come in two stages:

* **SDF shapes** (``is_sdf``): primitives and the CSG, transforms, taper, and displace built
  on them. They are signed distance fields sampled at voxel centers on demand. A voxel is
  filled when its center lies inside the shape or exactly on its surface; closed sampling
  keeps exact ties symmetric, so a shape mirrored about a plane samples to mirrored voxels.
* **Grid shapes**: anything painted or reshaped on voxels (paint ops, hollow, erode, dilate,
  generic round and chamfer, heightmaps). CSG, transforms, and mirror still work on them,
  with the same color rules; ops that need a distance field (smooth_union, taper, displace)
  must come before painting.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

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


_PACKAGE_DIR = str(Path(__file__).resolve().parent.parent)
# While sampling a cutter (the right side of - or &), only occupancy matters.
_occupancy_only: ContextVar[bool] = ContextVar("cubist_occupancy_only", default=False)
_CUTTER_PLACEHOLDER = 255


@contextmanager
def occupancy_only() -> Iterator[None]:
    """Sample shapes for their occupancy alone: uncolored voxels are allowed."""
    token = _occupancy_only.set(True)
    try:
        yield
    finally:
        _occupancy_only.reset(token)


def _caller_origin() -> str | None:
    """'script.py:12' for the first stack frame outside the cubist package."""
    frame = sys._getframe(2)
    while frame is not None:
        filename = frame.f_code.co_filename
        if not filename.startswith(_PACKAGE_DIR) and not filename.startswith("<"):
            return f"{Path(filename).name}:{frame.f_lineno}"
        frame = frame.f_back
    return None


def describe(shape: Shape) -> str:
    """'box (crate.py:6)' — how errors name a shape."""
    origin = getattr(shape, "origin", None)
    name = getattr(shape, "kind", None) or type(shape).__name__.lower()
    return f"{name} ({origin})" if origin else name


def _check_operand(other, op: str) -> None:
    if not isinstance(other, Shape):
        kind = type(other).__name__.lstrip("_").lower()
        if hasattr(other, "mask"):
            raise ShapeError(
                f"{op} takes shapes, but got a {kind} selection; selections like band() and "
                "faces() only go in where=. To cut, use a shape such as box(...)"
            )
        raise ShapeError(f"{op} takes shapes, got {other!r}")


def require_sdf(shape: Shape, op: str) -> None:
    if not shape.is_sdf:
        raise ShapeError(
            f"{op} needs an unpainted shape; apply {op} before painting, hollowing, or "
            "other voxel-level ops"
        )


class Shape:
    """Base for all shapes. SDF subclasses implement ``_eval`` and ``bounds``."""

    is_sdf = True

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        obj.origin = _caller_origin()  # where the script created it, for error messages
        return obj

    def _color_sources(self) -> list[Shape]:
        """The shapes whose colors this shape shows (cutters excluded)."""
        return []

    def _eval(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Distances (N,) and color indices (N,) for points (N, 3). Color 0 = none."""
        raise NotImplementedError

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Conservative world-space AABB ``(lo, hi)`` of the shape's inside."""
        raise NotImplementedError

    def sdf(self, points: np.ndarray) -> np.ndarray:
        require_sdf(self, "sdf")
        return self._eval(np.asarray(points, dtype=np.float64).reshape(-1, 3))[0]

    # --- CSG ---------------------------------------------------------------

    def union(self, *others: Shape) -> Shape:
        """Combine shapes. Where shapes overlap, the later one's color wins."""
        from cubist.dsl.ops import Union

        for other in others:
            _check_operand(other, "union (|)")
        return Union([self, *others])

    def subtract(self, *others: Shape) -> Shape:
        """Carve ``others`` out of this shape; the result keeps this shape's colors, so the
        cutters need no color."""
        from cubist.dsl.ops import Subtract

        for other in others:
            _check_operand(other, "subtract (-)")
        return Subtract(self, others[0].union(*others[1:]) if len(others) > 1 else others[0])

    def intersect(self, other: Shape) -> Shape:
        """Keep only the overlap; the result keeps this shape's colors."""
        from cubist.dsl.ops import Intersect

        _check_operand(other, "intersect (&)")
        return Intersect(self, other)

    def smooth_union(self, other: Shape, k: float) -> Shape:
        """Union with a fillet of blend radius ``k`` voxels where the shapes meet."""
        from cubist.dsl.ops import SmoothUnion

        _check_operand(other, "smooth_union")
        return SmoothUnion(self, other, k)

    def occupancy(self) -> VoxelGrid:
        """Which voxels the shape fills, ignoring color (for cutters and selections)."""
        with occupancy_only():
            return self.voxelize()

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

    def array(self, n: int, step: Vec3) -> Shape:
        """``n`` copies, each moved ``step`` further than the last."""
        if n < 1:
            raise ShapeError(f"array needs n >= 1, got {n}")
        s = np.asarray(step, dtype=np.float64)
        return self.union(*(self.at(*(s * k)) for k in range(1, n)))

    def radial(self, n: int, axis: str = "y", about: Vec3 = (0, 0, 0)) -> Shape:
        """``n`` copies spaced evenly around ``axis`` through ``about``."""
        if n < 1:
            raise ShapeError(f"radial needs n >= 1, got {n}")
        return self.union(*(self.rotate(360.0 * k / n, axis, about) for k in range(1, n)))

    def _pivot(self, about: Vec3 | None) -> np.ndarray:
        if about is not None:
            return np.asarray(about, dtype=np.float64)
        lo, hi = self.bounds()
        return (lo + hi) / 2.0

    def _plane(self, axis: str, about: float | None) -> float:
        from cubist.dsl.ops import axis_index

        i = axis_index(axis)
        return float(self._pivot(None)[i]) if about is None else float(about)

    # --- shape modifiers -----------------------------------------------------

    def taper(self, top: float = 1.0, axis: str = "y") -> Shape:
        """Scale the cross-section linearly along ``axis``: 1 at the bottom, ``top`` at the top."""
        from cubist.dsl.ops import Taper

        return Taper(self, top, axis)

    def displace(
        self,
        amp: float,
        scale: float = 4.0,
        seed: int = 0,
        noise: str = "perlin",
        octaves: int = 1,
    ) -> Shape:
        """Push the surface in and out by noise of amplitude ``amp`` voxels and feature size
        ``scale`` voxels. Seeded: same seed, same result."""
        from cubist.dsl.ops import Displace

        return Displace(self, amp, scale, seed, noise, octaves)

    def round(self, r: float = 1) -> Shape:
        """Round convex edges and corners with radius ``r`` voxels, never growing the shape.

        ``round(1)`` takes one voxel off every edge; larger radii curve further."""
        from cubist.dsl.gridops import Opening

        return Opening(self, r)

    def chamfer(self, c: int = 1) -> Shape:
        """Cut ``c`` voxel layers off every convex edge, at 45 degrees."""
        from cubist.dsl.gridops import Chamfer

        return Chamfer(self, c)

    def hollow(self, t: float = 1) -> Shape:
        """Keep only a shell ``t`` voxels thick."""
        from cubist.dsl.gridops import Hollow

        return Hollow(self, t)

    def erode(self, r: float = 1) -> Shape:
        """Shrink by ``r`` voxels."""
        from cubist.dsl.gridops import Erode

        return Erode(self, r)

    def dilate(self, r: float = 1, color=None) -> Shape:
        """Grow by ``r`` voxels; new voxels take the nearest voxel's color or ``color``."""
        from cubist.dsl.gridops import Dilate

        return Dilate(self, r, color)

    # --- paint ---------------------------------------------------------------

    def paint(self, color, where=None) -> Shape:
        """Recolor the selected voxels (all by default). ``where`` is a selection or a shape."""
        from cubist.dsl.paint import paint

        return paint(self, color, where)

    def paint_faces(self, direction: str, color, depth: int = 1, where=None) -> Shape:
        """Recolor voxels whose face toward ``direction`` (top, bottom, left, right, front,
        back) is exposed, ``depth`` voxels deep."""
        from cubist.dsl.paint import paint_faces

        return paint_faces(self, direction, color, depth, where)

    def gradient(
        self,
        axis: str,
        ramp,
        where=None,
        reverse: bool = False,
        lo: float | None = None,
        hi: float | None = None,
    ) -> Shape:
        """Color the selection in equal bands along ``axis``, dark to light through ``ramp``.

        The bands span the selected voxels' extent on that axis (hidden ones included), or
        ``[lo, hi)`` in world coordinates when given."""
        from cubist.dsl.paint import gradient

        return gradient(self, axis, ramp, where, reverse, lo, hi)

    def dither(self, ramp, seed: int = 0, amount: float = 0.5, where=None, mirror=None) -> Shape:
        """Nudge a share (``amount``) of voxels one shade up or down ``ramp``. Seeded.
        ``mirror=("x", 8)`` makes the pattern symmetric across that plane."""
        from cubist.dsl.paint import dither

        return dither(self, ramp, seed, amount, where, mirror)

    def darken_crevices(self, steps: int = 1, where=None) -> Shape:
        """Darken surface voxels in concave corners by ``steps`` shades of their list ramp.

        Colors from named tables or single colors have no order and stay unchanged."""
        from cubist.dsl.paint import darken_crevices

        return darken_crevices(self, steps, where)

    def highlight_edges(self, steps: int = 1, where=None, facing=None) -> Shape:
        """Lighten voxels on convex edges by ``steps`` shades of their list ramp. With
        ``facing=("top", "left")``, only edges open toward the light.

        Colors from named tables or single colors have no order and stay unchanged."""
        from cubist.dsl.paint import highlight_edges

        return highlight_edges(self, steps, where, facing)

    def masonry(
        self, ramp, block=(4, 2, 4), mortar=None, seed: int = 0, stagger: bool = True, where=None
    ) -> Shape:
        """Blocks of ``block`` voxels, each a seeded shade of ``ramp``, in staggered courses,
        with ``mortar`` joints if given: stone, brick, shingles, flagstones."""
        from cubist.dsl.paint import masonry

        return masonry(self, ramp, block, mortar, seed, stagger, where)

    def carve(self, cutter: Shape, color=None) -> Shape:
        """Cut ``cutter`` out and, with ``color``, color the walls and back of the cut."""
        from cubist.dsl.paint import carve

        return carve(self, cutter, color)

    def decal(self, rows, colors: dict, face: str = "front", at=(0, 0), where=None) -> Shape:
        """Paint a character grid (rows top to bottom) onto the surface seen from ``face``;
        ``at`` is the top-left cell in that view's ruler coordinates."""
        from cubist.dsl.paint import decal

        return decal(self, rows, colors, face, at, where)

    # --- sampling ------------------------------------------------------------

    def voxelize(self) -> VoxelGrid:
        """Sample the shape at voxel centers into a tight grid."""
        lo, hi = self.bounds()
        i0 = np.floor(lo).astype(np.int64) - 1
        i1 = np.ceil(hi).astype(np.int64) + 1
        size = i1 - i0
        if np.any(size > MAX_SAMPLE_AXIS) or np.prod(size) > MAX_SAMPLE_POINTS:
            raise ShapeError(
                f"{describe(self)} is too large to sample: bounds {lo.tolist()} to "
                f"{hi.tolist()} (limit {MAX_SAMPLE_AXIS} voxels per axis)"
            )
        occupancy = _occupancy_only.get()
        data = np.zeros(tuple(size), dtype=np.uint8)
        ys = np.arange(i0[1], i1[1]) + 0.5
        zs = np.arange(i0[2], i1[2]) + 0.5
        slab = max(1, _CHUNK_POINTS // max(1, size[1] * size[2]))
        for xa in range(0, size[0], slab):
            xs = np.arange(i0[0] + xa, i0[0] + min(xa + slab, size[0])) + 0.5
            pts = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
            d, c = self._eval(pts)
            inside = d <= EPS
            uncolored = inside & (c <= 0)
            if np.any(uncolored):
                if not occupancy:
                    raise self._missing_color()
                c = np.where(uncolored, _CUTTER_PLACEHOLDER, c)
            data[xa : xa + len(xs)] = (
                np.where(inside, c, 0).astype(np.uint8).reshape(len(xs), size[1], size[2])
            )
        return VoxelGrid(data, tuple(i0)).cropped()

    def _missing_color(self) -> ShapeError:
        culprits = []
        stack = [self]
        while stack:
            node = stack.pop()
            if getattr(node, "_c", None) == 0:
                culprits.append(describe(node))
            stack.extend(node._color_sources())
        names = ", ".join(culprits) or describe(self)
        return ShapeError(
            f"{names} has no color: pass color=p.<ramp>.<shade>. Shapes used only as cutters "
            "(a - b, a & b) or in where= don't need one."
        )
