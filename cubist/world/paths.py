"""Paths, rivers and roads: edits that follow a line across the ground (design.md, "Worlds").

A river cuts a channel whose water level only falls from its first point to its last, fills it with
water, and gives its bed a color; a road levels a strip to its own smoothed profile and paints it.
Both ease into the ground over their banks. Their profiles come from the world's ground before any
river or road is applied.
"""

from __future__ import annotations

import math

import numpy as np

from cubist.core.palette import Color
from cubist.world.field import Field, WorldError

_BLOCK = 64  # grids are searched in blocks this many columns on a side (each block culls segments)
_BUCKET_M = 64.0  # scattered points are searched in buckets of this many metres on a side
_TILE = 64  # a coarse distance grid is measured, and kept, in tiles of this many points a side
_TILE_MIN_POINTS = 256  # fewer queries than this in a tile not yet measured get their own corners


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


class Path:
    """A line across the ground through ``points`` (x, z metres), smoothed into a curve."""

    def __init__(self, points, smooth: bool = True, spacing_m: float = 4.0):
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
            raise WorldError(f"a path needs at least two (x, z) points, got {points!r}")
        if smooth and len(pts) >= 3:
            pts = _catmull_rom(pts, spacing_m)
        self.points = pts
        self.a, self.b = pts[:-1], pts[1:]
        self.ab = self.b - self.a
        self.len2 = np.maximum((self.ab**2).sum(axis=1), 1e-12)
        self.seg_len = np.sqrt(self.len2)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])  # arc length at each point
        self.lo = np.minimum(self.a, self.b)
        self.hi = np.maximum(self.a, self.b)

    @property
    def length(self) -> float:
        return float(self.cum[-1])

    def distance(self, max_m: float = 1000.0, step_m: float | None = None) -> Field:
        """A field: the distance (metres) to this path, capped at ``max_m`` (shape a valley around a
        river, say). Farther than ``max_m`` it reads ``max_m``.

        ``step_m`` samples it on a grid that far apart and interpolates: much cheaper for a far
        reach (a valley's walls), but off by up to about a third of a step where the nearest part of
        the path switches and half a step next to it, so leave it unset for masks near a path.
        """
        return _PathDistance(self, float(max_m), step_m)

    def nearest(self, x, z, reach: float) -> tuple[np.ndarray, np.ndarray]:
        """(distance, arc length of the nearest point) to the path, per (x, z); distance is inf past
        ``reach``. Grids are searched block by block, scattered points bucket by bucket, each
        against the segments near it."""
        x, z = np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64)
        d = np.full(x.shape, np.inf)
        t = np.zeros(x.shape)
        if x.ndim == 2:
            for j in range(0, x.shape[0], _BLOCK):
                for i in range(0, x.shape[1], _BLOCK):
                    block = (slice(j, j + _BLOCK), slice(i, i + _BLOCK))
                    d[block], t[block] = self._nearest(x[block], z[block], reach)
        elif x.size > _BLOCK * _BLOCK:
            fx, fz = x.ravel(), z.ravel()
            key = np.floor(fx / _BUCKET_M) * 1e6 + np.floor(fz / _BUCKET_M)
            order = np.argsort(key, kind="stable")
            edges = np.flatnonzero(np.diff(key[order])) + 1
            fd, ft = d.reshape(-1), t.reshape(-1)
            for idx in np.split(order, edges):
                fd[idx], ft[idx] = self._nearest(fx[idx], fz[idx], reach)
        else:
            d[...], t[...] = self._nearest(x, z, reach)
        return d, t

    def _nearest(self, x, z, reach):
        d = np.full(x.shape, np.inf)
        t = np.zeros(x.shape)
        if x.size == 0:
            return d, t
        near = (
            (self.lo[:, 0] <= x.max() + reach)
            & (self.hi[:, 0] >= x.min() - reach)
            & (self.lo[:, 1] <= z.max() + reach)
            & (self.hi[:, 1] >= z.min() - reach)
        )
        segs = np.flatnonzero(near)
        if len(segs) == 0:
            return d, t
        px, pz = x.reshape(-1, 1), z.reshape(-1, 1)
        ax, az = self.a[segs, 0], self.a[segs, 1]
        bx, bz = self.ab[segs, 0], self.ab[segs, 1]
        u = np.clip(((px - ax) * bx + (pz - az) * bz) / self.len2[segs], 0.0, 1.0)
        dist = np.hypot(px - (ax + u * bx), pz - (az + u * bz))
        k = dist.argmin(axis=1)
        best = dist[np.arange(len(k)), k]
        arc = self.cum[segs[k]] + u[np.arange(len(k)), k] * self.seg_len[segs[k]]
        best = np.where(best <= reach, best, np.inf)
        return best.reshape(x.shape), arc.reshape(x.shape)


class _PathDistance(Field):
    name = "distance"

    def __init__(self, line: Path, cap: float, step: float | None):
        self.path, self.cap, self.step = line, cap, step
        self._tiles: dict[tuple[int, int], np.ndarray] = {}

    def _exact(self, x, z):
        d, _ = self.path.nearest(x, z, self.cap)
        return np.minimum(d, self.cap)

    def _tile(self, ti: int, tj: int) -> np.ndarray:
        """Distances at grid points ``[ti·T, ti·T + T] × [tj·T, tj·T + T]`` ([z, x]); one point past
        the tile, so every cell's corners are in one tile. Measured once, then kept."""
        grid = self._tiles.get((ti, tj))
        if grid is None:
            span = np.arange(_TILE + 1)
            gx, gz = np.meshgrid((ti * _TILE + span) * self.step, (tj * _TILE + span) * self.step)
            grid = self._tiles[(ti, tj)] = self._exact(gx, gz)
        return grid

    def sample(self, x, z, seed):
        x, z = np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64)
        s = self.step
        if s is None or x.size == 0:
            return self._exact(x, z)
        # Bilinear between the distances at the corners of a fixed step_m grid: the same value
        # for a point however it's batched (props stand on what the bake carves).
        u, v = x.ravel() / s, z.ravel() / s
        i, j = np.floor(u), np.floor(v)
        fu, fv = u - i, v - j
        i, j = i.astype(np.int64), j.astype(np.int64)
        ti, tj = np.floor_divide(i, _TILE), np.floor_divide(j, _TILE)
        li, lj = i - ti * _TILE, j - tj * _TILE
        c00, c10, c01, c11 = (np.empty(u.shape) for _ in range(4))
        tiles = [
            (a, b) for a in range(ti.min(), ti.max() + 1) for b in range(tj.min(), tj.max() + 1)
        ]
        if len(tiles) > 16:  # scattered queries: only the tiles they touch
            width = int(tj.max() - tj.min()) + 1
            touched = np.unique((ti - ti.min()) * width + (tj - tj.min())).tolist()
            tiles = [(int(ti.min()) + k // width, int(tj.min()) + k % width) for k in touched]
        for a, b in tiles:
            m = (ti == a) & (tj == b)
            count = int(m.sum())
            if count == 0:
                continue
            if (a, b) not in self._tiles and count < _TILE_MIN_POINTS:
                # A few points in a tile not measured yet (far-field samples kilometres
                # apart): their own corners, the values the tile would hold, without the tile.
                ci, cj = i[m], j[m]
                cx = np.stack([ci, ci + 1, ci, ci + 1]) * s
                cz = np.stack([cj, cj, cj + 1, cj + 1]) * s
                c00[m], c10[m], c01[m], c11[m] = self._exact(cx, cz)
                continue
            grid = self._tile(int(a), int(b))
            gi, gj = li[m], lj[m]
            c00[m], c10[m] = grid[gj, gi], grid[gj, gi + 1]
            c01[m], c11[m] = grid[gj + 1, gi], grid[gj + 1, gi + 1]
        top = c00 * (1 - fu) + c10 * fu
        bottom = c01 * (1 - fu) + c11 * fu
        return (top * (1 - fv) + bottom * fv).reshape(x.shape)


def _catmull_rom(pts: np.ndarray, spacing: float) -> np.ndarray:
    """A uniform Catmull-Rom curve through ``pts``, sampled about every ``spacing`` metres."""
    padded = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    out = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        n = max(1, math.ceil(np.linalg.norm(p2 - p1) / spacing))
        s = np.arange(n)[:, None] / n
        out.append(
            0.5
            * (
                2 * p1
                + (-p0 + p2) * s
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s**2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * s**3
            )
        )
    out.append(pts[-1:])
    return np.vstack(out)


def path(points, smooth: bool = True) -> Path:
    """A line through (x, z) points in metres; ``smooth`` turns it into a curve through them."""
    return Path(points, smooth)


class PathEdit:
    """A river or road: lowers or levels the ground along a path and paints a strip of it."""

    name: str | None = None  # its line's name in world.json

    def __init__(self, line: Path, width_m: float, bank_m: float | None):
        if not isinstance(line, Path):
            raise WorldError(f"expected a path(...), got {line!r}")
        if width_m <= 0:
            raise WorldError(f"width_m must be positive, got {width_m}")
        self.path = line
        self.half = width_m / 2
        self.bank = width_m if bank_m is None else float(bank_m)
        self._profiles: dict[int, np.ndarray] = {}

    @property
    def reach(self) -> float:
        return self.half + self.bank

    def profile(self, ground: Field, seed: int) -> np.ndarray:
        """The level (metres) at each of the path's points, cached per seed."""
        if seed not in self._profiles:
            pts = self.path.points
            self._profiles[seed] = self._shape(ground.sample(pts[:, 0], pts[:, 1], seed))
        return self._profiles[seed]

    def level(self, ground: Field, seed: int, arc: np.ndarray) -> np.ndarray:
        return np.interp(arc, self.path.cum, self.profile(ground, seed))

    def _shape(self, along: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply(self, h: np.ndarray, x, z, seed: int, ground: Field) -> np.ndarray:
        raise NotImplementedError

    def strip(self, x, z) -> np.ndarray:
        """Where this edit paints its color."""
        d, _ = self.path.nearest(x, z, self.half)
        return d <= self.half


class River(PathEdit):
    def __init__(self, line, width_m, depth_m, bank_m, bed, water):
        super().__init__(line, width_m, bank_m)
        if depth_m <= 0:
            raise WorldError(f"river depth_m must be positive, got {depth_m}")
        self.depth, self.bed, self.water = float(depth_m), bed, water

    def _shape(self, along):
        # Water never flows uphill: the level only falls from the first point to the last, and
        # sits a little under the bank so the channel holds it.
        return np.minimum.accumulate(along) - 0.3

    def apply(self, h, x, z, seed, ground):
        d, arc = self.path.nearest(x, z, self.reach)
        near = np.isfinite(d)
        if not near.any():
            return h
        out = h.copy()
        w = self.level(ground, seed, arc[near])
        dn, hn = d[near], h[near]
        bed = w - self.depth * (1 - (np.minimum(dn, self.half) / self.half) ** 2)
        bank = w + (hn - w) * _smoothstep((dn - self.half) / max(self.bank, 1e-9))
        out[near] = np.minimum(hn, np.where(dn < self.half, bed, bank))
        return out

    def water_level(self, x, z, seed, ground) -> np.ndarray:
        """The water surface (metres) over the channel, -inf elsewhere."""
        d, arc = self.path.nearest(x, z, self.half)
        level = np.full(np.shape(d), -np.inf)
        inside = np.isfinite(d)
        level[inside] = self.level(ground, seed, arc[inside])
        return level


class Road(PathEdit):
    def __init__(self, line, width_m, color, bank_m, smooth_m, bridges=False):
        super().__init__(line, width_m, bank_m)
        self.color, self.smooth, self.bridges = color, float(smooth_m), bool(bridges)
        self.rivers: list[
            River
        ] = []  # the world's rivers, set when it loads (a bridge crosses them)

    def in_channel(self, x, z) -> np.ndarray:
        """Where a bridged road leaves the ground alone: inside any river's channel."""
        inside = np.zeros(np.shape(x), dtype=bool)
        if self.bridges:
            for r in self.rivers:
                inside |= np.isfinite(r.path.nearest(x, z, r.half)[0])
        return inside

    def crossings(self) -> list[tuple[float, float]]:
        """(entry, exit) arc lengths of each stretch of a bridged road over a river's channel,
        found to a few millimetres. A road that starts or ends in the water has no bank to
        carry a deck from, so that stretch isn't a crossing."""
        if not self.bridges or not self.rivers:
            return []
        pts = self.path.points

        def over(s):
            return self.in_channel(
                np.interp(s, self.path.cum, pts[:, 0]), np.interp(s, self.path.cum, pts[:, 1])
            )

        s = np.arange(0.0, self.path.length + 0.25, 0.25).clip(max=self.path.length)
        wet = over(s)
        out = []
        edges = np.flatnonzero(np.diff(wet.astype(np.int8)))
        enter = None
        for i in edges:
            lo, hi = float(s[i]), float(s[i + 1])
            for _ in range(24):  # bisect the edge between a dry and a wet sample
                mid = (lo + hi) / 2
                if bool(over(mid)) == bool(wet[i + 1]):
                    hi = mid
                else:
                    lo = mid
            edge = (lo + hi) / 2
            if wet[i + 1]:
                enter = edge
            elif enter is not None:
                out.append((enter, edge))
                enter = None
        return out

    def at(self, arc: float) -> tuple[float, float]:
        """The (x, z) at an arc length along the road."""
        pts = self.path.points
        return (
            float(np.interp(arc, self.path.cum, pts[:, 0])),
            float(np.interp(arc, self.path.cum, pts[:, 1])),
        )

    def strip(self, x, z) -> np.ndarray:
        return super().strip(x, z) & ~self.in_channel(x, z)

    def _shape(self, along):
        # A moving average over about `smooth_m` of road, so it doesn't follow every bump.
        spacing = max(self.path.length / max(len(along) - 1, 1), 1e-9)
        n = max(1, int(round(self.smooth / spacing)))
        if n <= 1:
            return along
        kernel = np.ones(n) / n
        padded = np.concatenate([np.full(n, along[0]), along, np.full(n, along[-1])])
        return np.convolve(padded, kernel, mode="same")[n:-n]

    def apply(self, h, x, z, seed, ground):
        d, arc = self.path.nearest(x, z, self.reach)
        near = np.isfinite(d)
        if not near.any():
            return h
        out = h.copy()
        r = self.level(ground, seed, arc[near])
        dn, hn = d[near], h[near]
        out[near] = r + (hn - r) * _smoothstep((dn - self.half) / max(self.bank, 1e-9))
        if self.bridges:  # the river's channel stays as it was cut, under the deck
            dry = ~self.in_channel(x, z)
            out = np.where(dry, out, h)
        return out


def river(
    line: Path,
    width_m: float,
    depth_m: float = 1.5,
    bank_m: float | None = None,
    bed: Color | None = None,
    water: Color | None = None,
    name: str | None = None,
) -> River:
    """A river along ``line`` (source first).

    A channel ``width_m`` wide and ``depth_m`` deep whose water level only falls downstream, banks
    easing out over ``bank_m`` (default: the width), its bed painted ``bed`` and filled with
    ``water`` (solid voxels of that color). ``name`` names its line in the bake's ``world.json``
    (default ``river1``, ...).
    """
    for c in (bed, water):
        if c is not None and not isinstance(c, Color):
            raise WorldError(f"river colors are palette colors such as p.water.mid, got {c!r}")
    r = River(line, width_m, depth_m, bank_m, bed, water)
    r.name = name
    return r


def road(
    line: Path,
    width_m: float,
    color: Color,
    bank_m: float | None = None,
    smooth_m: float = 30.0,
    name: str | None = None,
    bridges: bool = False,
) -> Road:
    """A road along ``line``.

    ``width_m`` of ground levelled to the road's own profile (averaged over ``smooth_m``) and
    painted ``color``, cut and filled into the ground over ``bank_m``. ``name`` names its line in
    the bake's ``world.json`` (default ``road1``, ...): routes for walks and caravans.

    Where it crosses a river it fills the channel (a ford), unless ``bridges``: then the road
    leaves the channel, its water and bed alone, and its ``world.json`` line crosses at deck height,
    straight between the road's heights at the two banks. The deck itself is a prop: ``place()`` a
    bridge asset over each crossing (lint warns of one with none).
    """
    if not isinstance(color, Color):
        raise WorldError(f"road color must be a palette color such as p.stone.light, got {color!r}")
    r = Road(line, width_m, color, bank_m, smooth_m, bridges)
    r.name = name
    return r


class EditedGround(Field):
    """The world's ground with its rivers, then its roads, applied."""

    name = "edited"

    def __init__(self, ground: Field, edits: list[PathEdit]):
        self.ground, self.edits = ground, edits

    def sample(self, x, z, seed):
        h = self.ground.sample(x, z, seed)
        for edit in self.edits:
            h = edit.apply(h, x, z, seed, self.ground)
        return h
