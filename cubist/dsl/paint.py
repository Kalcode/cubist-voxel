"""Paint ops: recolor voxels after sampling. Each returns a grid-stage shape."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from cubist.core.grid import VoxelGrid
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


def gradient(shape: Shape, axis: str, ramp, where=None, reverse: bool = False) -> Shape:
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
        lo, hi = coord[selected].min(), coord[selected].max() + 1
        t = (coord + 0.5 - lo) / (hi - lo)
        k = np.clip(np.floor(t * len(table)).astype(int), 0, len(table) - 1)
        return np.where(selected, table[k], grid.data)

    return PaintOp(shape, "gradient", fn, where)


def dither(shape: Shape, ramp, seed: int = 0, amount: float = 0.5, where=None) -> Shape:
    indices = _ramp_indices(ramp)
    if not 0 <= amount <= 1:
        raise ShapeError(f"dither amount must be between 0 and 1, got {amount}")
    position = {c: i for i, c in enumerate(indices)}

    def fn(grid, selected, ctx):
        on_ramp = selected & np.isin(grid.data, indices)
        if not on_ramp.any() or amount == 0:
            return grid.data
        cells = np.argwhere(on_ramp) + np.array(grid.origin)
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
        crevice = np.zeros_like(filled)
        for d in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
            near = sum(neighbor(filled, off).astype(int) for off in _in_plane(d, 1))
            far = sum(neighbor(filled, off).astype(int) for off in _in_plane(d, 2))
            crevice |= exposed(filled, d) & (near >= 2) & (far >= 2)
        mask = crevice[2:-2, 2:-2, 2:-2] & selected
        return np.where(mask, table[grid.data], grid.data)

    return PaintOp(shape, "darken_crevices", fn, where)


def highlight_edges(shape: Shape, steps: int = 1, where=None) -> Shape:
    def fn(grid, selected, ctx):
        table = _shift_table(ctx, steps, "highlight_edges")
        filled = np.pad(grid.data > 0, 1)
        axes = sum(
            (exposed(filled, d) | exposed(filled, tuple(-k for k in d))).astype(int)
            for d in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        )
        mask = (axes >= 2)[1:-1, 1:-1, 1:-1] & selected
        return np.where(mask, table[grid.data], grid.data)

    return PaintOp(shape, "highlight_edges", fn, where)
