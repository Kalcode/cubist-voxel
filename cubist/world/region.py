"""``cubist world render``: a close look at part of a world in voxels (design.md, "Worlds").

The area's chunks are built as the bake builds them (caves and water included) at the finest level
that fits the asset renderer (256 cells a side), and drawn as iso and top views. ``cut_m`` removes
everything above a height, to look into caves from above. Props are drawn from the project's
exports, placed and turned as Kiln stamps them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from cubist.core.grid import VoxelGrid
from cubist.render.contact_sheet import contact_sheet
from cubist.world.bake import VOXELS_PER_M, N, chunk_row, layers_of
from cubist.world.field import WorldError
from cubist.world.model import LoadedWorld

MAX_CELLS = 256


def region_grid(
    loaded: LoadedWorld,
    at,
    size_m: float = 64.0,
    cut_m: float | None = None,
    export: Path | None = None,
):
    """(the area's cells as a ``VoxelGrid`` [x, y, z], level used, its min corner in metres).

    With ``export`` (the project's exports), the world's props are drawn too."""
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
    props = _prop_voxels(loaded, export, level, (x0, z0, x0 + n, z0 + n)) if export else None
    if props is not None and len(props[0]):
        top = max(top, int(props[0][:, 1].max()) + 1)
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
    if props is not None:
        cells, colors = props
        cells = cells - np.array([cx0, bottom, cz0])
        inside = ((cells >= 0) & (cells < data.shape)).all(axis=1)
        data[tuple(cells[inside].T)] = colors[inside]
    data = data[x0 - cx0 : x0 - cx0 + n, :, z0 - cz0 : z0 - cz0 + n]
    return VoxelGrid(np.ascontiguousarray(data)), level, (x0 * cell_m, bottom * cell_m, z0 * cell_m)


# Kiln's quarter turns about y (its VoxelRotation.AboutY): continuous points, then voxels.
_TURN_POINT = {
    0: lambda x, y, z: (x, y, z),
    90: lambda x, y, z: (z, y, -x),
    180: lambda x, y, z: (-x, y, -z),
    270: lambda x, y, z: (-z, y, x),
}
_TURN_VOXEL = {
    0: lambda x, y, z: (x, y, z),
    90: lambda x, y, z: (z, y, -x - 1),
    180: lambda x, y, z: (-x - 1, y, -z - 1),
    270: lambda x, y, z: (-z - 1, y, x),
}


def _asset(export: Path, name: str):
    """An exported asset's visible voxels (asset space, [k, 3]), their colors, its glb_origin and
    reach (voxels from the origin to its farthest corner)."""
    from cubist.io.vox import read_vox

    scene = read_vox(export / f"{name}.vox")
    from cubist.core.grid import VoxelGrid

    grid = VoxelGrid.merge([p.grid for p in scene.parts if not p.hidden and "@" not in p.name])
    local = np.argwhere(grid.data != 0)
    coords = local + np.array(grid.origin)
    colors = grid.data[tuple(local.T)]
    manifest = json.loads((export / f"{name}.cubist.json").read_text())
    sx, _, sz = manifest.get("size") or grid.shape
    origin = np.array(manifest.get("glb_origin") or (sx / 2, 0, sz / 2), dtype=np.float64)
    reach = float(np.abs(coords - origin).max()) + 1 if len(coords) else 0.0
    return coords, colors, origin, reach


def _prop_voxels(loaded: LoadedWorld, export: Path, level: int, area):
    """The props' voxels over ``area`` (x0, z0, x1, z1 in cells of ``level``) as cells [k, 3] and
    colors, each placed as Kiln stamps it: asset voxel v lands at R(v) + floor(at − R(origin) + ½).
    """
    from cubist.world.props import placements

    size = 1 << level
    x0, z0, x1, z1 = (v * size for v in area)  # level-0 voxels
    assets: dict = {}
    cells, colors = [], []
    for p in placements(loaded, export):
        if p.asset not in assets:
            assets[p.asset] = _asset(export, p.asset)
        coords, color, origin, reach = assets[p.asset]
        ax, _, az = p.at
        if ax + reach < x0 or ax - reach > x1 or az + reach < z0 or az - reach > z1:
            continue
        o = np.floor(np.asarray(p.at) - np.array(_TURN_POINT[p.rotate_y](*origin)) + 0.5)
        turned = np.stack(_TURN_VOXEL[p.rotate_y](*coords.T), axis=1) + o.astype(np.int64)
        cells.append(np.floor_divide(turned, size))
        colors.append(color)
    if not cells:
        return np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.uint8)
    return np.concatenate(cells), np.concatenate(colors)


def render_region(
    loaded: LoadedWorld,
    at,
    size_m: float = 64.0,
    cut_m: float | None = None,
    export: Path | None = None,
) -> Image.Image:
    grid, level, corner = region_grid(loaded, at, size_m, cut_m, export)
    cell = (1 << level) / VOXELS_PER_M
    note = f"LEVEL {level} ({cell:g} M CELLS) FROM ({corner[0]:g}, {corner[2]:g})"
    if cut_m is not None:
        note += f" CUT AT {cut_m:g} M"
    title = f"{loaded.name} at {at[0]:g},{at[1]:g}"
    return contact_sheet(grid, loaded.palette, title, views=("iso", "top"), note=note)
