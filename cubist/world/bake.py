"""``cubist world bake``: a world in Kiln's map format.

See design.md, "Worlds", and Kiln's formats.md, "Baked maps". Per level of detail k (cells of
2^k voxels), the world's columns are grouped into tiles of 64 × 64 columns. Each column stores its
bounds (chunks below ``MinCy`` are all the deepest subsoil color, chunks from ``MaxCy`` up are air)
and its chunks in between as chunk payload v1, deflate-compressed. Far-field grids sample the
ground beyond the voxels. Tiles bake in parallel processes.
"""

from __future__ import annotations

import json
import math
import os
import struct
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cubist.build.runner import find_style, load_project_style
from cubist.world.caves import Worms
from cubist.world.field import WorldError
from cubist.world.model import LoadedWorld, Rule
from cubist.world.sample import heights, paint, water, world_context

N = 32  # cells per chunk side
TILE_COLUMNS = 64
VOXELS_PER_M = 16
LATTICE_M = 0.5  # fields are sampled this far apart and interpolated per column (levels 0–2)
FAR_LEVELS, FAR_CELLS, FAR_FIRST_SPACING = 6, 128, 128
DEFLATE_LEVEL = 6
MAP_FORMAT = 1
_ENTRY = struct.Struct("<iiHHIII")  # MinCy, MaxCy, Below, 0, offset, length, raw length
_HEADER = struct.Struct("<4sHBBii")


# ---- Chunk payload v1 ------------------------------------------------------------------------


def encode_chunk(cells: np.ndarray) -> bytes:
    """Chunk payload v1 (Kiln's formats.md) for 32³ cells indexed ``[z, y, x]``."""
    return encode_chunks(cells[None])[0]


def encode_chunks(cells: np.ndarray) -> list[bytes]:
    """Chunk payloads v1 for a stack of chunks ``[chunk, z, y, x]``, all sharing one palette.

    A uniform chunk is 4 bytes. Otherwise the palette is every value in the stack (unused entries
    are allowed), uniform 8³ bricks are table entries, and every other brick lists the whole palette
    as its local entries with 1, 2, 4 or 8 bits per voxel. Kiln re-encodes compactly when it loads a
    chunk.
    """
    k = cells.shape[0]
    counts = np.bincount(cells.reshape(-1), minlength=1)
    values = np.flatnonzero(counts).astype(np.uint16)
    n = len(values)
    lut = np.zeros(len(counts), dtype=np.uint16)
    lut[values] = np.arange(n, dtype=np.uint16)
    # Bricks numbered x fastest, then y, then z; voxels within a brick likewise.
    bricks = (
        lut[cells].reshape(k, 4, 8, 4, 8, 4, 8).transpose(0, 1, 3, 5, 2, 4, 6).reshape(k, 64, 512)
    )
    lo, hi = bricks.min(axis=2), bricks.max(axis=2)
    uniform = lo == hi
    whole = uniform.all(axis=1) & (lo == lo[:, :1]).all(axis=1)
    solid = (cells.reshape(k, -1) != 0).sum(axis=1)
    if n > 256:
        raise WorldError(f"a chunk column uses {n} materials; at most 256 fit one brick palette")
    bits = next(b for b in (1, 2, 4, 8) if (1 << b) >= max(n, 2))
    per = 8 // bits
    record_len = (1 + n + 512 * bits // 8 + 3) & ~3
    palette = values.astype("<u2").tobytes()
    local = bytes([n - 1]) + np.arange(n, dtype=np.uint8).tobytes()  # h = m − 1, then the m entries
    shifts = (np.arange(per, dtype=np.uint16) * bits).astype(np.uint16)
    out = []
    for c in range(k):
        if whole[c]:
            out.append(struct.pack("<BBH", 1, 1, int(values[lo[c, 0]])))
            continue
        table = np.where(uniform[c], 0x8000 | lo[c], 0).astype(np.uint16)
        mixed = np.flatnonzero(~uniform[c])
        table[mixed] = np.arange(len(mixed), dtype=np.uint16) * (record_len // 4)
        head = struct.pack("<HH", n, int(solid[c])) + table.astype("<u2").tobytes() + palette
        head += b"\0" * (-len(head) % 4)
        records = np.zeros((len(mixed), record_len), dtype=np.uint8)
        if len(mixed):  # every brick may be uniform, just not all the same
            packed = (bricks[c, mixed].reshape(len(mixed), -1, per) << shifts).sum(axis=2)
            records[:, : len(local)] = np.frombuffer(local, dtype=np.uint8)
            records[:, len(local) : len(local) + packed.shape[1]] = packed.astype(np.uint8)
        out.append(bytes([1, 0]) + head + records.tobytes())
    return out


def decode_chunk(payload: bytes) -> np.ndarray:
    """The 32³ cells ``[z, y, x]`` of a payload v1, raw or bit-packed (for tests and tools)."""
    if payload[0] != 1:
        raise WorldError(f"chunk payload version {payload[0]} is not 1")
    if payload[1] & 1:
        return np.full((N, N, N), struct.unpack_from("<H", payload, 2)[0], dtype=np.uint16)
    d = payload[2:]
    n = struct.unpack_from("<H", d, 0)[0]
    table = np.frombuffer(d, dtype="<u2", count=64, offset=4)
    palette = np.frombuffer(d, dtype="<u2", count=n, offset=132)
    start = (132 + 2 * n + 3) & ~3
    bricks = np.zeros((64, 512), dtype=np.uint16)
    for b, e in enumerate(table):
        if e & 0x8000:
            bricks[b] = palette[e & 0x7FFF]
            continue
        o = start + 4 * int(e)
        h = d[o]
        if h == 0:
            bricks[b] = palette[np.frombuffer(d, dtype="<u2", count=512, offset=o + 1)]
            continue
        m = h + 1
        wide = n > 256
        local = np.frombuffer(d, dtype="<u2" if wide else np.uint8, count=m, offset=o + 1)
        bits = next(b for b in (1, 2, 4, 8) if (1 << b) >= m)
        raw = np.frombuffer(
            d, dtype=np.uint8, count=512 * bits // 8, offset=o + 1 + m * (2 if wide else 1)
        )
        idx = np.unpackbits(raw, bitorder="little").reshape(512, bits)
        li = (idx * (1 << np.arange(bits))).sum(axis=1)
        bricks[b] = palette[local[li]]
    return bricks.reshape(4, 4, 4, 8, 8, 8).transpose(0, 3, 1, 4, 2, 5).reshape(N, N, N)


# ---- Sampling columns -------------------------------------------------------------------------


@dataclass
class Layers:
    """The ground's material bands, in voxels below the surface."""

    surface_depth: float
    bands: list[tuple[float, int]]  # (bottom depth below the surface, palette index)
    rest: int

    @property
    def deepest(self) -> float:
        return self.bands[-1][0] if self.bands else self.surface_depth


def layers_of(loaded: LoadedWorld) -> Layers:
    depth = loaded.world.surface_depth_m * VOXELS_PER_M
    bands, rest = loaded.subsoil_layers()
    out, bottom = [], depth
    for band_m, color in bands:
        bottom += band_m * VOXELS_PER_M
        out.append((bottom, color.index))
    return Layers(depth, out, rest.index)


def _bilinear(grid: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``grid[v, u]`` at fractional lattice coordinates."""
    i0 = np.clip(np.floor(u).astype(int), 0, grid.shape[1] - 2)
    j0 = np.clip(np.floor(v).astype(int), 0, grid.shape[0] - 2)
    fu, fv = u - i0, v - j0
    top = grid[j0, i0] * (1 - fu) + grid[j0, i0 + 1] * fu
    bottom = grid[j0 + 1, i0] * (1 - fu) + grid[j0 + 1, i0 + 1] * fu
    return top * (1 - fv) + bottom * fv


def sample_columns(loaded: LoadedWorld, level: int, x0: int, z0: int, nx: int, nz: int):
    """Ground at the centers of cells ``[x0, x0 + nx) × [z0, z0 + nz)`` of level ``level``.

    Returns heights (level-0 voxels), slopes (degrees), surface palette indices, and the water
    surface (level-0 voxels, -inf for none) with its color, each [nz, nx]. Levels whose cells are
    finer than ``LATTICE_M`` interpolate fields sampled on that lattice.
    """
    cell_m = (1 << level) / VOXELS_PER_M
    cx = (x0 + np.arange(nx) + 0.5) * cell_m
    cz = (z0 + np.arange(nz) + 0.5) * cell_m
    step = max(LATTICE_M, cell_m)
    if cell_m < LATTICE_M:
        lx0, lz0 = math.floor(cx[0] / step) - 1, math.floor(cz[0] / step) - 1
        lnx, lnz = math.ceil(cx[-1] / step) + 2 - lx0, math.ceil(cz[-1] / step) + 2 - lz0
        gx, gz = np.meshgrid((lx0 + np.arange(lnx)) * step, (lz0 + np.arange(lnz)) * step)
        u = np.broadcast_to(cx / step - lx0, (nz, nx))
        v = np.broadcast_to((cz / step - lz0)[:, None], (nz, nx))
    else:
        gx, gz = np.meshgrid(
            np.concatenate([[cx[0] - step], cx, [cx[-1] + step]]),
            np.concatenate([[cz[0] - step], cz, [cz[-1] + step]]),
        )
        u = np.broadcast_to(np.arange(nx) + 1.0, (nz, nx))
        v = np.broadcast_to((np.arange(nz) + 1.0)[:, None], (nz, nx))
    grid = heights(loaded, gx, gz)
    dz, dx = np.gradient(grid, step)
    slope_grid = np.degrees(np.arctan(np.hypot(dx, dz)))
    h = _bilinear(grid, u, v)
    if loaded.world.step_m:  # the world's ground steps: heights snap to them
        h = np.floor(h / loaded.world.step_m + 0.5) * loaded.world.step_m
    s = _bilinear(slope_grid, u, v)
    out = np.full((nz, nx), loaded.default_surface().index, dtype=np.int16)
    decided = np.zeros((nz, nx), dtype=bool)
    with world_context(loaded):
        for rule in loaded.world.surface:
            if not isinstance(rule, Rule):
                break
            if rule.kind == "where":
                hit = _bilinear(rule.mask.sample(gx, gz, loaded.seed), u, v) > rule.value
            else:
                hit = rule.matches(h, s, None, None, loaded.seed)
            hit &= ~decided
            out[hit] = rule.color.index
            decided |= hit
    if loaded.world.rivers or loaded.world.roads:
        px, pz = np.meshgrid(cx, cz)
        paint(loaded, px, pz, out)
        level, color = water(loaded, px, pz)
    else:
        level, color = np.full((nz, nx), -np.inf), np.zeros((nz, nx), dtype=np.int16)
    return h * VOXELS_PER_M, s, out, level * VOXELS_PER_M, color


def chunk_cells(
    top: np.ndarray, surface: np.ndarray, cy: int, level: int, layers: Layers
) -> np.ndarray:
    """A chunk's cells ``[z, y, x]`` (see ``column_cells``)."""
    return column_cells(top, surface, cy, cy + 1, level, layers)[0]


def column_cells(
    top, surface, min_cy: int, max_cy: int, level: int, layers: Layers, water=None, water_color=None
) -> np.ndarray:
    """Chunks ``[min_cy, max_cy)`` of a chunk column as ``[chunk, z, y, x]``.

    From its 32 × 32 columns' heights (level-0 voxels) and surface colors: a cell is solid when its
    center lies below the ground, and its depth picks the band. Air under a column's water surface
    is water.
    """
    size = 1 << level
    count = max_cy - min_cy
    y = (min_cy * N + np.arange(N * count) + 0.5) * size  # cell centers, level-0 voxels
    depth = top[:, None, :] - y[None, :, None]  # [z, y, x]
    out = np.where(depth > 0, surface[:, None, :], 0).astype(np.uint16)
    # A coarse cell exposed to air shows the surface even when the band is thinner than the cell (as
    # Kiln's downsampling does): the surface band is at least one cell deep; the bands below move.
    shift = max(size - layers.surface_depth, 0.0)
    above = layers.surface_depth + shift
    for bottom, color in layers.bands:
        out[(depth > above) & (depth <= bottom + shift)] = color
        above = bottom + shift
    out[depth > above] = layers.rest
    if water is not None:
        wet = (depth <= 0) & (y[None, :, None] < water[:, None, :])
        out[wet] = np.broadcast_to(water_color[:, None, :], out.shape)[wet]
    return out.reshape(N, count, N, N).transpose(1, 0, 2, 3)


# ---- Tiles ------------------------------------------------------------------------------------


@dataclass
class TileStats:
    level: int
    raw: int
    packed: int
    chunks: int
    seconds: float = 0.0


WORM_STEP = 16  # worm noise is sampled every 16 cells (1 m at level 0) and interpolated


def _upsample(v: np.ndarray, n: int, step: int) -> np.ndarray:
    """Values on a lattice every ``step`` cells (``n / step + 1`` points per axis, last three axes)
    at the ``n`` cell centers between, trilinearly."""
    idx = (np.arange(n) + 0.5) / step
    i0 = np.minimum(np.floor(idx).astype(int), v.shape[-1] - 2)
    f = idx - i0
    for axis in (-1, -2, -3):
        a = np.take(v, i0, axis=axis)
        b = np.take(v, i0 + 1, axis=axis)
        shape = [1] * v.ndim
        shape[axis] = n
        v = a * (1 - f.reshape(shape)) + b * f.reshape(shape)
    return v


class WormLattice:
    """A worm carver's noise for a row of chunk columns, sampled in one go.

    Points every ``WORM_STEP`` cells (aligned to them); each column slices its part.
    """

    def __init__(
        self, carver: Worms, level: int, x0: int, z0: int, nx: int, nz: int, top, seed: int
    ):
        cell_m = (1 << level) / VOXELS_PER_M
        inside = top > -1e8
        y0, y1 = carver.band(
            float(top[inside].min()) / VOXELS_PER_M, float(top[inside].max()) / VOXELS_PER_M
        )
        self.ly0 = math.floor(y0 / cell_m / WORM_STEP) - 1
        ly1 = math.ceil(y1 / cell_m / WORM_STEP) + 1
        self.lx0, self.lz0 = x0 // WORM_STEP, z0 // WORM_STEP
        lx = (self.lx0 + np.arange(nx // WORM_STEP + 1)) * WORM_STEP * cell_m
        lz = (self.lz0 + np.arange(nz // WORM_STEP + 1)) * WORM_STEP * cell_m
        ly = (self.ly0 + np.arange(ly1 - self.ly0 + 1)) * WORM_STEP * cell_m
        self.values = carver.value(
            lx.reshape(1, 1, -1), ly.reshape(1, -1, 1), lz.reshape(-1, 1, 1), seed
        )

    def chunk_values(self, cx0: int, cz0: int, rows: np.ndarray) -> np.ndarray:
        """The lattice over chunks starting at cell rows ``rows`` of the column at (cx0, cz0).

        ``[chunk, z, y, x]``, ``N / WORM_STEP + 1`` points a side; unsampled points are far above 1.
        """
        k = N // WORM_STEP
        xs = slice(cx0 // WORM_STEP - self.lx0, cx0 // WORM_STEP - self.lx0 + k + 1)
        zs = slice(cz0 // WORM_STEP - self.lz0, cz0 // WORM_STEP - self.lz0 + k + 1)
        out = np.full((len(rows), k + 1, k + 1, k + 1), 9.0)
        for i, r in enumerate(rows):
            y = r // WORM_STEP - self.ly0
            lo, hi = max(y, 0), min(y + k + 1, self.values.shape[1])
            if lo < hi:
                out[i, :, lo - y : hi - y, :] = self.values[zs, lo:hi, xs]
        return out


def carve_column(
    cells, caves, level: int, min_cy: int, cx0: int, cz0: int, top, seed: int, lattices=None
) -> None:
    """Hollows a chunk column's cells ``[chunk, z, y, x]`` where its caves carve (in place).
    ``lattices`` holds each worm carver's noise for the column's row (``WormLattice``)."""
    cell_m = (1 << level) / VOXELS_PER_M
    count = cells.shape[0]
    x = ((cx0 + np.arange(N) + 0.5) * cell_m).reshape(1, 1, 1, N)
    z = ((cz0 + np.arange(N) + 0.5) * cell_m).reshape(1, N, 1, 1)
    ground = (top / VOXELS_PER_M).reshape(1, N, 1, N)
    for carver, (_, y0, _, _, y1, _) in caves:
        if isinstance(carver, Worms):  # this column's own band under its ground
            y0, y1 = carver.band(float(ground.min()), float(ground.max()))
        chunk_lo = (min_cy + np.arange(count)) * N * cell_m
        sub = np.flatnonzero((chunk_lo < y1) & (chunk_lo + N * cell_m > y0))
        if len(sub) == 0:
            continue
        rows = (min_cy + sub)[:, None] * N
        y = ((rows + np.arange(N) + 0.5) * cell_m).reshape(len(sub), 1, N, 1)
        if isinstance(carver, Worms):
            inside = carver.inside(x, y, z, ground)
            if not inside.any():
                continue
            # The noise is smooth: sampled on a coarse lattice per row of columns, interpolated.
            if lattices is not None and carver in lattices:
                coarse = lattices[carver].chunk_values(cx0, cz0, rows[:, 0])
            else:
                lattice = np.arange(0, N + 1, WORM_STEP)
                lx = ((cx0 + lattice) * cell_m).reshape(1, 1, 1, -1)
                lz = ((cz0 + lattice) * cell_m).reshape(1, -1, 1, 1)
                ly = ((rows + lattice) * cell_m).reshape(len(sub), 1, -1, 1)
                coarse = carver.value(lx, ly, lz, seed)
            value = _upsample(coarse, N, WORM_STEP)
            hollow = inside & (value < carver.thickness)
            if carver.floors and hollow.any():
                hollow &= ~carver.floor_rock(x, y, z)
        else:
            hollow = carver.carve(x, y, z, ground, seed)
        block = cells[sub]
        block[np.broadcast_to(hollow, block.shape)] = 0
        cells[sub] = block


def chunk_row(
    loaded: LoadedWorld, level: int, x0: int, cz0: int, columns: int, layers: Layers, caves=None
):
    """The chunk columns ``[x0, x0 + 32 · columns)`` × ``[cz0, cz0 + 32)`` (that level's cells).

    For each, ``(min_cy, max_cy, cells [chunk, z, y, x])``, or None past the world's edge.
    """
    size_cells = [int(round(s * VOXELS_PER_M)) >> level for s in loaded.size_m]
    caves = loaded.cave_bounds() if caves is None else caves
    if cz0 >= size_cells[1] or cz0 < 0:
        return [None] * columns
    h, _, surf, wet, wet_color = sample_columns(loaded, level, x0, cz0, columns * N, N)
    cell = 1 << level
    # Worm noise for the whole row at once (one call instead of one per column).
    row_x, row_z = x0 * cell / VOXELS_PER_M, cz0 * cell / VOXELS_PER_M
    row_w, row_d = columns * N * cell / VOXELS_PER_M, N * cell / VOXELS_PER_M
    lattices = {
        c: WormLattice(c, level, x0, cz0, columns * N, N, h, loaded.seed)
        for c, b in caves
        if isinstance(c, Worms)
        and level <= c.max_level
        and b[0] < row_x + row_w
        and b[3] > row_x
        and b[2] < row_z + row_d
        and b[5] > row_z
    }
    out = []
    for col in range(columns):
        cx0 = x0 + col * N
        if cx0 >= size_cells[0] or cx0 < 0:
            out.append(None)
            continue
        top = h[:, col * N : (col + 1) * N].copy()
        colors = surf[:, col * N : (col + 1) * N]
        water_top = wet[:, col * N : (col + 1) * N]
        water_color = wet_color[:, col * N : (col + 1) * N]
        # Columns past the world's edge (a chunk column straddling it) are air.
        outside_x = cx0 + np.arange(N) >= size_cells[0]
        outside_z = cz0 + np.arange(N) >= size_cells[1]
        top[outside_z[:, None] | outside_x[None, :]] = -1e9
        inside = top > -1e8
        hi = max(top[inside].max(), water_top[inside].max()) / cell
        lo = (top[inside].min() - layers.deepest - cell) / cell
        # Caves under this chunk column reach the bounds down to their floor.
        x_lo, z_lo = cx0 * cell / VOXELS_PER_M, cz0 * cell / VOXELS_PER_M
        x_hi, z_hi = x_lo + N * cell / VOXELS_PER_M, z_lo + N * cell / VOXELS_PER_M
        here = [
            (c, b)
            for c, b in caves
            if level <= c.max_level and b[0] < x_hi and b[3] > x_lo and b[2] < z_hi and b[5] > z_lo
        ]
        if here:
            floor_m = min(
                c.band(top[inside].min() / VOXELS_PER_M, 0)[0] if isinstance(c, Worms) else b[1]
                for c, b in here
            )
            lo = min(lo, floor_m * VOXELS_PER_M / cell - 1)
        # Below min_cy every cell is deeper than the last band (one cell of slack for ties).
        min_cy = math.floor((lo - 1) / N)
        max_cy = max(math.ceil(hi / N), min_cy + 1)
        cells = column_cells(top, colors, min_cy, max_cy, level, layers, water_top, water_color)
        if here:
            carve_column(cells, here, level, min_cy, cx0, cz0, top, loaded.seed, lattices)
            # Chunks at the bottom the caves missed are all rock again: back under the bounds.
            rock = (cells.reshape(len(cells), -1) == layers.rest).all(axis=1)
            keep = int(np.argmin(rock[:-1])) if not rock[:-1].all() else len(cells) - 1
            cells, min_cy = cells[keep:], min_cy + keep
        out.append((min_cy, max_cy, cells))
    return out


def bake_tile(
    loaded: LoadedWorld, out: Path, level: int, tx: int, tz: int, layers: Layers
) -> TileStats:
    """Writes ``L{level}/{tx}_{tz}.tile``: 64 × 64 columns, those outside the world left empty."""
    started = time.perf_counter()
    caves = loaded.cave_bounds()
    entries = []
    blobs: list[bytes] = []
    raw_total = chunks = 0
    offset = _HEADER.size + _ENTRY.size * TILE_COLUMNS * TILE_COLUMNS
    for row in range(TILE_COLUMNS):
        cz0 = (tz * TILE_COLUMNS + row) * N
        for column in chunk_row(
            loaded, level, tx * TILE_COLUMNS * N, cz0, TILE_COLUMNS, layers, caves
        ):
            if column is None:
                entries.append((0, 0, 0, 0, 0, 0, 0))
                continue
            min_cy, max_cy, cells = column
            payloads = encode_chunks(cells)
            raw = b"".join(struct.pack("<I", len(p)) + p for p in payloads)
            blob = zlib.compress(raw, DEFLATE_LEVEL)
            entries.append((min_cy, max_cy, layers.rest, 0, offset, len(blob), len(raw)))
            blobs.append(blob)
            offset += len(blob)
            raw_total += len(raw)
            chunks += len(payloads)
    path = out / f"L{level}" / f"{tx}_{tz}.tile"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(_HEADER.pack(b"KTIL", 1, level, 0, tx, tz))
        for e in entries:
            f.write(_ENTRY.pack(*e))
        for blob in blobs:
            f.write(blob)
    return TileStats(level, raw_total, offset, chunks, time.perf_counter() - started)


# ---- The world --------------------------------------------------------------------------------

_worker: tuple[LoadedWorld, Layers, Path] | None = None


def _init_worker(script: str, seed: int, style: str | None, out: str) -> None:
    from cubist.world.build import load_script

    global _worker
    loaded = load_script(script, seed, style)
    _worker = (loaded, layers_of(loaded), Path(out))


def _bake_task(task: tuple[int, int, int]) -> TileStats:
    loaded, layers, out = _worker  # type: ignore[misc]
    return bake_tile(loaded, out, *task, layers)


def tiles_for(loaded: LoadedWorld, levels: int, region=None) -> list[tuple[int, int, int]]:
    """Every (level, tx, tz) covering the world, or only those over ``region`` ((x0, z0), (x1, z1)
    metres)."""
    tasks = []
    for k in range(levels):
        span = TILE_COLUMNS * N << k  # voxels
        nx = math.ceil(loaded.size_m[0] * VOXELS_PER_M / span)
        nz = math.ceil(loaded.size_m[1] * VOXELS_PER_M / span)
        tasks += [(k, tx, tz) for tz in range(nz) for tx in range(nx)]
    if region is not None:
        (x0, z0), (x1, z1) = region

        def over(task) -> bool:
            k, tx, tz = task
            m = (TILE_COLUMNS * N << k) / VOXELS_PER_M
            return tx * m < x1 and (tx + 1) * m > x0 and tz * m < z1 and (tz + 1) * m > z0

        tasks = [t for t in tasks if over(t)]
    return tasks


def far_grids(loaded: LoadedWorld, out: Path) -> list[dict]:
    """The far field's height grids, reaching past the world as far as a ring around it can."""
    grids = []
    sx, sz = (s * VOXELS_PER_M for s in loaded.size_m)
    for k in range(FAR_LEVELS):
        spacing = FAR_FIRST_SPACING << k
        margin = (FAR_CELLS // 2 + 8) * spacing
        x0 = math.floor(-margin / spacing) * spacing
        z0 = x0
        count_x = int((sx + margin - x0) // spacing) + 2
        count_z = int((sz + margin - z0) // spacing) + 2
        # Heights at the grid points (in voxels) and the surface color there by the world's rules;
        # water shows as its surface.
        h, _, surf = sample_columns_at(loaded, x0, z0, spacing, count_x, count_z)
        gx, gz = np.meshgrid(
            (x0 + np.arange(count_x) * spacing) / VOXELS_PER_M,
            (z0 + np.arange(count_z) * spacing) / VOXELS_PER_M,
        )
        level, color = water(loaded, gx, gz)
        wet = level * VOXELS_PER_M > h
        h = np.where(wet, level * VOXELS_PER_M, h)
        surf = np.where(wet, color, surf)
        payload = h.astype("<f4").tobytes() + surf.astype("<u2").tobytes()
        name = f"far-L{k}.bin"
        (out / name).write_bytes(
            b"KFAR" + struct.pack("<HH", 1, 0) + zlib.compress(payload, DEFLATE_LEVEL)
        )
        grids.append(
            {
                "level": k,
                "spacing": spacing,
                "x0": x0,
                "z0": z0,
                "count_x": count_x,
                "count_z": count_z,
                "file": name,
            }
        )
    return grids


def sample_columns_at(
    loaded: LoadedWorld, x0: int, z0: int, spacing: int, count_x: int, count_z: int
):
    """Like ``sample_columns`` at grid points ``(x0 + i · spacing, z0 + j · spacing)`` (voxels)."""
    xs = (x0 + np.arange(count_x) * spacing) / VOXELS_PER_M
    zs = (z0 + np.arange(count_z) * spacing) / VOXELS_PER_M
    gx, gz = np.meshgrid(xs, zs)
    h = heights(loaded, gx, gz)
    dz, dx = np.gradient(h, spacing / VOXELS_PER_M)
    s = np.degrees(np.arctan(np.hypot(dx, dz)))
    from cubist.world.sample import surface

    return h * VOXELS_PER_M, s, surface(loaded, gx, gz, h, s)


def materials_of(loaded: LoadedWorld, glow: dict[str, float]) -> list[dict]:
    """The material registry for the engine: one entry per palette color, its index as the id."""
    out = []
    for c in loaded.palette.colors():
        r, g, b, _ = c.rgba
        strength = glow.get(c.name, glow.get(c.name.split(".")[0], 0.0))
        out.append(
            {"id": c.index, "name": c.name, "rgb": f"#{r:02x}{g:02x}{b:02x}", "glow": strength}
        )
    return out


@dataclass
class BakeSummary:
    name: str
    out: Path
    seconds: float
    tiles: int
    chunks: int
    raw_bytes: int
    disk_bytes: int
    level0_bytes: int
    km2: float
    level_seconds: dict[int, tuple[int, float]] = field(default_factory=dict)  # level: (tiles, s)

    def levels_line(self) -> str:
        """Tiles and worker time per level (summed over processes)."""
        return "  by level: " + ", ".join(
            f"L{k} {n} tiles {s:.0f} s ({s / n:.2f} s each)"
            for k, (n, s) in sorted(self.level_seconds.items())
        )

    def line(self) -> str:
        return (
            f"baked {self.name}: {self.tiles} tiles, {self.chunks:,} chunks in "
            f"{self.seconds:.0f} s; {self.disk_bytes / 2**20:.0f} MiB on disk "
            f"({self.raw_bytes / 2**20:.0f} MiB raw), level 0 "
            f"{self.level0_bytes / 2**20 / self.km2:.0f} MiB/km², "
            f"{self.seconds / self.km2:.0f} s/km²"
        )


def bake_world(
    script,
    out_dir=None,
    levels: int = 7,
    workers: int | None = None,
    seed: int = 0,
    style=None,
    progress=None,
    region=None,
) -> BakeSummary:
    """Bakes every tile of every level, the far grids, ``map.json`` and ``world.json``.

    ``progress(done, total, seconds)`` is called as tiles finish (about every 5%). ``region``
    ((x0, z0), (x1, z1) metres) re-bakes only the tiles over it into an existing bake (and the far
    grids and JSON): quick while working on one part of a big world, but tiles elsewhere keep what
    they had, so re-bake everything once the change reaches past it.
    """
    from cubist.world.build import load_script

    started = time.perf_counter()
    script = Path(script)
    style_path = Path(style) if style else find_style(script)
    loaded = load_script(script, seed, style_path)
    out = Path(out_dir) if out_dir else style_path.parent / "export" / "worlds" / loaded.name
    km2 = loaded.size_m[0] * loaded.size_m[1] / 1e6
    if region is not None:
        if not (out / "map.json").is_file():
            raise WorldError(f"--region re-bakes part of an existing bake; {out} has none yet")
        baked = json.loads((out / "map.json").read_text())["levels"]
        if baked != levels:
            raise WorldError(f"{out} was baked with {baked} levels; bake the region with as many")
    out.mkdir(parents=True, exist_ok=True)
    tasks = tiles_for(loaded, levels, region)
    if region is not None:  # per km² of the level-0 tiles baked
        m = TILE_COLUMNS * N / VOXELS_PER_M
        km2 = (
            sum(
                max(0.0, min((tx + 1) * m, loaded.size_m[0]) - tx * m)
                * max(0.0, min((tz + 1) * m, loaded.size_m[1]) - tz * m)
                for k, tx, tz in tasks
                if k == 0
            )
            / 1e6
        )
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    stats: list[TileStats] = []
    step = max(1, len(tasks) // 20)

    def done(stat: TileStats) -> None:
        stats.append(stat)
        if progress and (len(stats) % step == 0 or len(stats) == len(tasks)):
            progress(len(stats), len(tasks), time.perf_counter() - started)

    if workers == 1:
        _init_worker(str(script), seed, str(style_path), str(out))
        for t in tasks:
            done(_bake_task(t))
    else:
        init = (str(script), seed, str(style_path), str(out))
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=init) as pool:
            for stat in pool.map(_bake_task, tasks, chunksize=1):
                done(stat)
    grids = far_grids(loaded, out)
    sx, sz = (int(round(s * VOXELS_PER_M)) for s in loaded.size_m)
    style_bible = load_project_style(style_path)
    info = {
        "format": MAP_FORMAT,
        "version": 1,
        "source": f"cubist world {loaded.name} (seed {seed})",
        "levels": levels,
        "tile_columns": TILE_COLUMNS,
        "min_x": 0,
        "min_z": 0,
        "max_x": sx,
        "max_z": sz,
        "materials": len(loaded.palette),
        "compression": "deflate",
        "voxel_size_m": 1 / VOXELS_PER_M,
        "material_registry": materials_of(loaded, dict(style_bible.glow)),
        "far": grids,
    }
    (out / "map.json").write_text(json.dumps(info, indent=2) + "\n")
    landmarks = {}
    for name, (x, z, y) in loaded.landmark_points().items():
        if y is None:
            y = float(heights(loaded, np.array([x]), np.array([z]))[0])
        landmarks[name] = [x * VOXELS_PER_M, y * VOXELS_PER_M, z * VOXELS_PER_M]
    from cubist.world.props import placements

    export = style_path.parent / "export"
    props = [p.json(out, export) for p in placements(loaded, export)]
    world_json = {"voxel_size_m": 1 / VOXELS_PER_M, "landmarks": landmarks, "props": props}
    (out / "world.json").write_text(json.dumps(world_json, indent=2) + "\n")
    disk = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    level0 = sum(s.packed for s in stats if s.level == 0)
    chunks, raw = sum(s.chunks for s in stats), sum(s.raw for s in stats)
    seconds = time.perf_counter() - started
    by_level: dict[int, tuple[int, float]] = {}
    for st in stats:
        n, t = by_level.get(st.level, (0, 0.0))
        by_level[st.level] = (n + 1, t + st.seconds)
    return BakeSummary(
        loaded.name, out, seconds, len(tasks), chunks, raw, disk, level0, km2, by_level
    )
