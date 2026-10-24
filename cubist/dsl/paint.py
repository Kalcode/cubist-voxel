"""Paint ops: recolor voxels after sampling. Each returns a grid-stage shape."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from cubist.core.grid import VoxelGrid
from cubist.core.topology import outside_air
from cubist.dsl import noise
from cubist.dsl.context import BuildContext, current
from cubist.dsl.gridops import GridShape, exposed, neighbor
from cubist.dsl.primitives import color_index
from cubist.dsl.select import as_selection, direction
from cubist.dsl.shape import Shape, ShapeError


class PaintOp(GridShape):
    def __init__(self, shape: Shape, name: str, fn: Callable, where=None):
        self.shape, self.name, self.fn = shape, name, fn
        self.where = as_selection(where)

    def _color_sources(self):
        return [self.shape]

    def _grid(self) -> VoxelGrid:
        src = self.shape.voxelize().cropped()
        ctx = current()
        ctx.ops_used.add(self.name)
        if src.count() == 0:
            return src
        selected = self.where.mask(src) & (src.data > 0)
        data = self.fn(src, selected, ctx)
        return VoxelGrid(data.astype(np.uint8), src.origin).cropped()


def _ramp_indices(ramp) -> list[int]:
    indices = [color_index(c) for c in ramp]
    if not indices or 0 in indices:
        raise ShapeError("ramp must be palette colors, like p.wood or p.wood.ramp")
    return indices


def _shift_table(ctx: BuildContext, steps: int, op: str) -> np.ndarray:
    if ctx.palette is None:
        raise ShapeError(f"{op} needs the project palette; build through @asset (or `building`)")
    table = np.arange(256, dtype=np.uint8)
    for c in ctx.palette.colors():
        table[c.index] = ctx.palette.shift(c.index, steps)
    return table


def paint(shape: Shape, color, where=None) -> Shape:
    index = color_index(color)
    if index == 0:
        raise ShapeError("paint needs a palette color like p.wood.dark")

    def fn(grid, selected, ctx):
        return np.where(selected, index, grid.data)

    return PaintOp(shape, "paint", fn, where)


def paint_faces(shape: Shape, face: str, color, depth: int = 1, where=None) -> Shape:
    d = direction(face)
    index = color_index(color)
    if index == 0:
        raise ShapeError("paint_faces needs a palette color like p.grass.mid")
    if depth < 1:
        raise ShapeError(f"paint_faces depth must be >= 1, got {depth}")

    def fn(grid, selected, ctx):
        filled = np.pad(grid.data > 0, 1)
        layer = exposed(filled, d)
        painted = layer.copy()
        for _ in range(depth - 1):
            layer = filled & neighbor(layer, d)
            painted |= layer
        painted = painted[1:-1, 1:-1, 1:-1] & selected
        return np.where(painted, index, grid.data)

    return PaintOp(shape, "paint_faces", fn, where)


def gradient(
    shape: Shape,
    axis: str,
    ramp,
    where=None,
    reverse: bool = False,
    lo: float | None = None,
    hi: float | None = None,
) -> Shape:
    if axis not in ("x", "y", "z"):
        raise ShapeError(f"gradient axis must be x, y or z, got {axis!r}")
    a = "xyz".index(axis)
    indices = _ramp_indices(ramp)
    if reverse:
        indices = indices[::-1]
    table = np.array(indices, dtype=np.uint8)

    def fn(grid, selected, ctx):
        if not selected.any():
            return grid.data
        coord = np.indices(grid.shape)[a] + grid.origin[a]
        start = coord[selected].min() if lo is None else lo
        stop = coord[selected].max() + 1 if hi is None else hi
        t = (coord + 0.5 - start) / max(stop - start, 1e-9)
        k = np.clip(np.floor(t * len(table)).astype(int), 0, len(table) - 1)
        return np.where(selected, table[k], grid.data)

    return PaintOp(shape, "gradient", fn, where)


def dither(
    shape: Shape, ramp, seed: int = 0, amount: float = 0.5, where=None, mirror=None
) -> Shape:
    indices = _ramp_indices(ramp)
    if not 0 <= amount <= 1:
        raise ShapeError(f"dither amount must be between 0 and 1, got {amount}")
    position = {c: i for i, c in enumerate(indices)}

    def fn(grid, selected, ctx):
        on_ramp = selected & np.isin(grid.data, indices)
        if not on_ramp.any() or amount == 0:
            return grid.data
        cells = np.argwhere(on_ramp) + np.array(grid.origin)
        if mirror is not None:
            # Hash the canonical cell of each mirrored pair so both halves match.
            a = "xyz".index(mirror[0])
            flip = int(round(2 * mirror[1])) - 1
            cells[:, a] = np.minimum(cells[:, a], flip - cells[:, a])
        s = noise.mix_seed(seed)
        pick = noise.hash_unit(cells, s) < amount
        step = np.where(noise.hash_unit(cells, s ^ 0x5BD1E995) < 0.5, -1, 1)
        pos = np.array([position[int(c)] for c in grid.data[on_ramp]])
        new_pos = np.clip(pos + np.where(pick, step, 0), 0, len(indices) - 1)
        out = grid.data.copy()
        out[on_ramp] = np.array(indices, dtype=np.uint8)[new_pos]
        return out

    return PaintOp(shape, "dither", fn, where)


def _in_plane(d, k: int = 1) -> list[tuple[int, int, int]]:
    """The 8 offsets around ``k * d`` in the plane perpendicular to d."""
    axis = int(np.flatnonzero(d)[0])
    others = [i for i in range(3) if i != axis]
    out = []
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            if a == 0 and b == 0:
                continue
            off = [k * v for v in d]
            off[others[0]] += a
            off[others[1]] += b
            out.append(tuple(off))
    return out


def darken_crevices(shape: Shape, steps: int = 1, where=None) -> Shape:
    # A crevice has solid material beside the open face one AND two layers out, like a
    # floor meeting a wall. Bevel steps and one-voxel curbs only reach one layer out.
    def fn(grid, selected, ctx):
        table = _shift_table(ctx, -steps, "darken_crevices")
        filled = np.pad(grid.data > 0, 2)
        air = outside_air(filled)
        crevice = np.zeros_like(filled)
        for d in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            near = sum(neighbor(filled, off).astype(int) for off in _in_plane(d, 1))
            far = sum(neighbor(filled, off).astype(int) for off in _in_plane(d, 2))
            hit = exposed(filled, d) & (near >= 2) & (far >= 2)
            if d[1] == 0:
                # Skip one-voxel walls whose other side faces outside air: darkening them
                # would show through. Walls of sealed rooms keep their shading, and floors
                # are exempt (their undersides rarely show).
                hit &= ~neighbor(air, tuple(-k for k in d))
            crevice |= hit
        mask = crevice[2:-2, 2:-2, 2:-2] & selected
        return np.where(mask, table[grid.data], grid.data)

    return PaintOp(shape, "darken_crevices", fn, where)


def highlight_edges(shape: Shape, steps: int = 1, where=None, facing=None) -> Shape:
    """Edges are voxels open on one side along two different axes; an axis open on both
    sides (a thin plate) doesn't count, so thin parts don't light up entirely. With
    ``facing`` (e.g. ("top", "left")), only edges open toward those directions lighten."""
    wanted = [direction(f) for f in facing] if facing else None

    def fn(grid, selected, ctx):
        table = _shift_table(ctx, steps, "highlight_edges")
        filled = np.pad(grid.data > 0, 1)
        axes = np.zeros(filled.shape, dtype=int)
        for d in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            plus, minus = exposed(filled, d), exposed(filled, tuple(-k for k in d))
            axes += (plus ^ minus).astype(int)
        mask = axes >= 2
        if wanted is not None:
            mask &= np.logical_or.reduce([exposed(filled, d) for d in wanted])
        mask = mask[1:-1, 1:-1, 1:-1] & selected
        return np.where(mask, table[grid.data], grid.data)

    return PaintOp(shape, "highlight_edges", fn, where)


def masonry(
    shape: Shape,
    ramp,
    block=(4, 2, 4),
    mortar=None,
    seed: int = 0,
    stagger: bool = True,
    where=None,
) -> Shape:
    """Blocks of ``block`` voxels, each a seeded shade from ``ramp``, in staggered courses,
    with ``mortar`` joints if given: stone walls, bricks, shingles, flagstones."""
    indices = _ramp_indices(ramp)
    bx, by, bz = (int(v) for v in block)
    if min(bx, by, bz) < 1:
        raise ShapeError(f"masonry block sizes must be >= 1, got {block}")
    mortar_index = color_index(mortar)

    def fn(grid, selected, ctx):
        x, y, z = (np.indices(grid.shape)[i] + grid.origin[i] for i in range(3))
        row = np.floor_divide(y, by)
        shift = (row % 2) if stagger else 0
        xs = x + shift * (bx // 2)
        zs = z + shift * (bz // 2)
        ids = np.stack([np.floor_divide(xs, bx), row, np.floor_divide(zs, bz)], axis=-1)
        pick = noise.hash3(ids.reshape(-1, 3), noise.mix_seed(seed)) % len(indices)
        colors = np.array(indices, dtype=np.uint8)[pick].reshape(grid.shape)
        out = np.where(selected, colors, grid.data)
        if mortar_index:
            joint = (y % by == 0) | (xs % bx == 0) | (zs % bz == 0)
            out = np.where(selected & joint, mortar_index, out)
        return out

    return PaintOp(shape, "masonry", fn, where)


class Carve(GridShape):
    """Cut ``cutter`` out of ``shape``; with ``color``, paint the walls of the cut."""

    def __init__(self, shape: Shape, cutter: Shape, color):
        from cubist.dsl.shape import _check_operand

        _check_operand(cutter, "carve")
        self.shape, self.cutter, self.color = shape, cutter, color
        self._paint = color_index(color)

    def _color_sources(self):
        return [self.shape]

    def _grid(self) -> VoxelGrid:
        cut = (self.shape - self.cutter).voxelize()
        if not self._paint or cut.count() == 0:
            return cut
        data = np.pad(cut.data, 1)
        origin = np.array(cut.origin) - 1
        cells = np.indices(data.shape).reshape(3, -1).T + origin
        hole = self.cutter.occupancy().lookup(cells).reshape(data.shape) > 0
        air = (data == 0) & ~hole  # empty space outside the cut
        touching = np.zeros_like(hole)
        for d in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            back = tuple(-k for k in d)
            # A one-voxel wall with outside air behind it stays unpainted, so the cut's color
            # never shows through to the outside.
            touching |= neighbor(hole, d) & ~neighbor(air, back)
        data = np.where((data > 0) & touching, self._paint, data)
        return VoxelGrid(data.astype(np.uint8), tuple(origin)).cropped()


def carve(shape: Shape, cutter: Shape, color=None) -> Shape:
    """Cut ``cutter`` out of ``shape`` and, with ``color``, color the cut's walls and back:
    windows, arrow slits, eye sockets, keyholes."""
    return Carve(shape, cutter, color)


# face -> (u axis, u sign, v axis, v sign, depth axis, depth looks toward +?)
_DECAL_FACES = {
    "front": (0, 1, 1, -1, 2, True),
    "back": (0, -1, 1, -1, 2, False),
    "right": (2, -1, 1, -1, 0, True),
    "left": (2, 1, 1, -1, 0, False),
    "top": (0, 1, 2, 1, 1, True),
    "bottom": (0, 1, 2, -1, 1, False),
}


def decal(shape: Shape, rows, colors: dict, face: str = "front", at=(0, 0), where=None) -> Shape:
    """Paint a character grid onto the first surface voxel seen from ``face``, the way that
    view shows it: ``rows`` top to bottom, ``colors`` maps characters to palette colors, any
    other character (like '.') is transparent. ``at`` is the (u, v) cell of the top-left
    character in that view's ruler coordinates. Eyes, emblems, signs, faces."""
    if face not in _DECAL_FACES:
        raise ShapeError(f"decal face must be one of {', '.join(_DECAL_FACES)}, got {face!r}")
    lookup = {ch: color_index(c) for ch, c in colors.items()}
    if 0 in lookup.values():
        raise ShapeError("decal colors must be palette colors like p.eye.dark")
    u_axis, u_sign, v_axis, v_sign, depth_axis, from_high = _DECAL_FACES[face]

    def fn(grid, selected, ctx):
        out = grid.data.copy()
        filled = (grid.data > 0) & selected
        origin = np.array(grid.origin)
        for i, row in enumerate(rows):
            for j, ch in enumerate(row):
                if ch not in lookup:
                    continue
                cell = [0, 0, 0]
                cell[u_axis] = at[0] + u_sign * j - origin[u_axis]
                cell[v_axis] = at[1] + v_sign * i - origin[v_axis]
                if not (
                    0 <= cell[u_axis] < grid.shape[u_axis]
                    and 0 <= cell[v_axis] < grid.shape[v_axis]
                ):
                    continue
                index = [slice(None)] * 3
                index[u_axis], index[v_axis] = cell[u_axis], cell[v_axis]
                column = np.nonzero(filled[tuple(index)])[0]
                if len(column) == 0:
                    continue
                cell[depth_axis] = column.max() if from_high else column.min()
                out[tuple(cell)] = lookup[ch]
        return out

    return PaintOp(shape, "decal", fn, where)
