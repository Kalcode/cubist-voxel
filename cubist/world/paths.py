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

    def nearest(self, x, z, reach: float) -> tuple[np.ndarray, np.ndarray]:
        """(distance, arc length of the nearest point) to the path, per (x, z); distance is inf past
        ``reach``. Grids are searched block by block, each against the segments near it."""
        x, z = np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64)
        d = np.full(x.shape, np.inf)
        t = np.zeros(x.shape)
        if x.ndim == 2:
            for j in range(0, x.shape[0], _BLOCK):
                for i in range(0, x.shape[1], _BLOCK):
                    block = (slice(j, j + _BLOCK), slice(i, i + _BLOCK))
                    d[block], t[block] = self._nearest(x[block], z[block], reach)
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
    def __init__(self, line, width_m, color, bank_m, smooth_m):
        super().__init__(line, width_m, bank_m)
        self.color, self.smooth = color, float(smooth_m)

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
        return out


def river(
    line: Path,
    width_m: float,
    depth_m: float = 1.5,
    bank_m: float | None = None,
    bed: Color | None = None,
    water: Color | None = None,
) -> River:
    """A river along ``line`` (source first).

    A channel ``width_m`` wide and ``depth_m`` deep whose water level only falls downstream, banks
    easing out over ``bank_m`` (default: the width), its bed painted ``bed`` and filled with
    ``water`` (solid voxels of that color).
    """
    for c in (bed, water):
        if c is not None and not isinstance(c, Color):
            raise WorldError(f"river colors are palette colors such as p.water.mid, got {c!r}")
    return River(line, width_m, depth_m, bank_m, bed, water)


def road(
    line: Path, width_m: float, color: Color, bank_m: float | None = None, smooth_m: float = 30.0
) -> Road:
    """A road along ``line``.

    ``width_m`` of ground levelled to the road's own profile (averaged over ``smooth_m``) and
    painted ``color``, cut and filled into the ground over ``bank_m``.
    """
    if not isinstance(color, Color):
        raise WorldError(f"road color must be a palette color such as p.stone.light, got {color!r}")
    return Road(line, width_m, color, bank_m, smooth_m)


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
