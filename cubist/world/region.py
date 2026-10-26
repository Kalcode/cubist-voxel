"""``cubist world render``: a close look at part of a world in voxels (design.md, "Worlds").

The area's chunks are built as the bake builds them (caves and water included) at the finest level
that fits the asset renderer (256 cells a side), and drawn as iso and top views. ``cut_m`` removes
everything above a height, to look into caves from above. Props aren't drawn yet.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from cubist.core.grid import VoxelGrid
from cubist.render.contact_sheet import contact_sheet
from cubist.world.bake import VOXELS_PER_M, N, chunk_row, layers_of
from cubist.world.field import WorldError
from cubist.world.model import LoadedWorld

MAX_CELLS = 256


def region_grid(loaded: LoadedWorld, at, size_m: float = 64.0, cut_m: float | None = None):
    """(the area's cells as a ``VoxelGrid`` [x, y, z], level used, its min corner in metres)."""
    if size_m <= 0:
        raise WorldError(f"region size_m must be positive, got {size_m}")
    level = 0
    while size_m * VOXELS_PER_M / (1 << level) > MAX_CELLS:
        level += 1
    cell_m = (1 << level) / VOXELS_PER_M
    n = int(round(size_m / cell_m))
    x0 = int(math.floor((at[0] - size_m / 2) / cell_m))
    z0 = int(math.floor((at[1] - size_m / 2) / cell_m))
    cx0, cz0 = x0 // N * N, z0 // N * N
    cols = (x0 + n - cx0 + N - 1) // N
    rows = (z0 + n - cz0 + N - 1) // N
    layers = layers_of(loaded)
    built = [chunk_row(loaded, level, cx0, cz0 + r * N, cols, layers) for r in range(rows)]
    present = [c for row in built for c in row if c is not None]
    if not present:
        raise WorldError(f"({at[0]:g}, {at[1]:g}) is outside the world")
    top = max(c[1] for c in present) * N
    if cut_m is not None:
        top = min(top, int(math.floor(cut_m / cell_m)))
    bottom = max(min(c[0] for c in present) * N, top - MAX_CELLS)
    data = np.zeros((cols * N, top - bottom, rows * N), dtype=np.uint8)
    for r, row in enumerate(built):
        for c, column in enumerate(row):
            if column is None:
                continue
            min_cy, _, cells = column
            stack = np.concatenate(list(cells), axis=1)  # [z, y, x] from min_cy up
            y0 = min_cy * N
            lo, hi = max(bottom, y0), min(top, y0 + stack.shape[1])
            if lo >= hi:
                continue
            part = stack[:, lo - y0 : hi - y0, :].transpose(2, 1, 0)  # [x, y, z]
            data[c * N : (c + 1) * N, lo - bottom : hi - bottom, r * N : (r + 1) * N] = part
    # Cells under the stored bounds are all rock: fill them in below each column.
    for r, row in enumerate(built):
        for c, column in enumerate(row):
            if column is not None and column[0] * N > bottom:
                data[c * N : (c + 1) * N, : column[0] * N - bottom, r * N : (r + 1) * N] = (
                    layers.rest
                )
    data = data[x0 - cx0 : x0 - cx0 + n, :, z0 - cz0 : z0 - cz0 + n]
    return VoxelGrid(np.ascontiguousarray(data)), level, (x0 * cell_m, bottom * cell_m, z0 * cell_m)


def render_region(
    loaded: LoadedWorld, at, size_m: float = 64.0, cut_m: float | None = None
) -> Image.Image:
    grid, level, corner = region_grid(loaded, at, size_m, cut_m)
    cell = (1 << level) / VOXELS_PER_M
    note = f"LEVEL {level} ({cell:g} M CELLS) FROM ({corner[0]:g}, {corner[2]:g})"
    if cut_m is not None:
        note += f" CUT AT {cut_m:g} M"
    title = f"{loaded.name} at {at[0]:g},{at[1]:g}"
    return contact_sheet(grid, loaded.palette, title, views=("iso", "top"), note=note)
