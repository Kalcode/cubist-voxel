"""Caves: carvers that hollow out the ground (design.md, "Worlds").

A carver says which points (x, y, z in metres, y up) are air. Tunnels follow a 3D path, caverns are
ellipsoids with optionally rough walls, and worms are noise tunnels in a depth band under a region.
Carvers only remove ground; they apply up to their ``max_level`` of detail (tunnels and caverns to
level 2, so a cave mouth shows from a distance; worms at level 0 only). Worms leave the rock under
tunnels' and caverns' floors, so the way through stays walkable.
"""

from __future__ import annotations

import numpy as np

from cubist.dsl.noise import fbm, mix_seed
from cubist.world.field import WorldError
from cubist.world.paths import _catmull_rom

FLOOR_M = 2.0  # rock worms leave under a tunnel's or cavern's floor


class Carver:
    max_level = 2
    name = "carver"

    def resolve(self, ground_at) -> None:
        """Turns depths below the ground into heights once the world's ground is known;
        ``ground_at(x, z)`` gives heights (metres) for arrays."""

    def bounds(self, ground_range) -> tuple[float, float, float, float, float, float]:
        """(x0, y0, z0, x1, y1, z1) in metres that holds everything it carves.

        ``ground_range(x0, z0, x1, z1)`` gives the ground's (lowest, highest) height over an area,
        for carvers that follow it.
        """
        raise NotImplementedError

    def carve(self, x, y, z, ground, seed: int) -> np.ndarray:
        """Where (x, y, z) is air.

        Arrays broadcastable together; ``ground`` is the ground height (metres) over each (x, z).
        """
        raise NotImplementedError


class Tunnel(Carver):
    name = "tunnel"

    def __init__(self, points, radius_m: float, smooth: bool, below_ground: bool = False):
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
            raise WorldError(f"a tunnel needs at least two (x, y, z) points, got {points!r}")
        if radius_m <= 0:
            raise WorldError(f"tunnel radius_m must be positive, got {radius_m}")
        self.radius, self.below_ground, self._smooth = float(radius_m), below_ground, smooth
        self._authored = pts.copy()
        self._set(pts)

    def _set(self, pts: np.ndarray) -> None:
        if self._smooth and len(pts) >= 3:
            pts = _catmull_rom(pts, 2.0)
        self.points = pts
        self.a, self.ab = pts[:-1], pts[1:] - pts[:-1]
        self.len2 = np.maximum((self.ab**2).sum(axis=1), 1e-12)

    def resolve(self, ground_at) -> None:
        if not self.below_ground:
            return
        # Depths at the authored points become heights; the curve then runs through those.
        pts = self._authored
        heights = ground_at(pts[:, 0], pts[:, 2]) - pts[:, 1]
        self._set(np.stack([pts[:, 0], heights, pts[:, 2]], axis=-1))
        self.below_ground = False

    def bounds(self, ground_range):
        lo, hi = self.points.min(axis=0) - self.radius, self.points.max(axis=0) + self.radius
        return (lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])

    def carve(self, x, y, z, ground, seed):
        return self._near(x, y, z, self.radius, under=False)

    def under_floor(self, x, y, z, margin_m: float) -> np.ndarray:
        """Where (x, y, z) is rock under this tunnel's floor, within ``margin_m`` of it."""
        return self._near(x, y, z, self.radius + margin_m, under=True)

    def _near(self, x, y, z, reach: float, under: bool) -> np.ndarray:
        # Within `reach` of the centerline; with `under`, only below its lower half.
        p = np.stack(np.broadcast_arrays(x, y, z), axis=-1)
        shape = p.shape[:-1]
        p = p.reshape(-1, 3)
        lo, hi = p.min(axis=0) - reach, p.max(axis=0) + reach
        seg_lo = np.minimum(self.a, self.a + self.ab)
        seg_hi = np.maximum(self.a, self.a + self.ab)
        near = np.flatnonzero(((seg_lo <= hi) & (seg_hi >= lo)).all(axis=1))
        out = np.zeros(len(p), dtype=bool)
        for k in near:  # usually a handful of segments per chunk
            ap = p - self.a[k]
            u = np.clip(ap @ self.ab[k] / self.len2[k], 0.0, 1.0)
            off = ap - u[:, None] * self.ab[k]
            hit = (off**2).sum(axis=1) <= reach**2
            if under:
                hit &= off[:, 1] < -0.5 * self.radius
            out |= hit
        return out.reshape(shape)


class Cavern(Carver):
    name = "cavern"

    def __init__(self, center, radius_m, roughness_m: float, seed: int, below_ground: bool = False):
        self.below_ground = below_ground
        self.c = np.asarray(center, dtype=np.float64)
        r = np.broadcast_to(np.asarray(radius_m, dtype=np.float64), (3,)).copy()
        if self.c.shape != (3,) or (r <= 0).any():
            raise WorldError(
                f"a cavern needs a center (x, y, z) and positive radii, "
                f"got {center!r}, {radius_m!r}"
            )
        self.r, self.rough, self.seed = r, float(roughness_m), int(seed)

    def resolve(self, ground_at) -> None:
        if self.below_ground:
            ground = float(ground_at(np.array([self.c[0]]), np.array([self.c[2]]))[0])
            self.c = np.array([self.c[0], ground - self.c[1], self.c[2]])
            self.below_ground = False

    def bounds(self, ground_range):
        lo, hi = self.c - self.r - self.rough, self.c + self.r + self.rough
        return (lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])

    def under_floor(self, x, y, z, margin_m: float) -> np.ndarray:
        """Where (x, y, z) is rock under this cavern's floor, within ``margin_m`` of it."""
        x, y, z = np.broadcast_arrays(x, y, z)
        r = self.r + self.rough
        q = (
            ((x - self.c[0]) / r[0]) ** 2
            + ((y - self.c[1]) / r[1]) ** 2
            + ((z - self.c[2]) / r[2]) ** 2
        )
        # The ellipsoid grown by the margin over its shortest radius: at least margin_m everywhere.
        return (q <= (1 + margin_m / r.min()) ** 2) & (y < self.c[1] - 0.5 * self.r[1])

    def carve(self, x, y, z, ground, seed):
        x, y, z = np.broadcast_arrays(x, y, z)
        q = (
            ((x - self.c[0]) / self.r[0]) ** 2
            + ((y - self.c[1]) / self.r[1]) ** 2
            + ((z - self.c[2]) / self.r[2]) ** 2
        )
        if self.rough <= 0:
            return q <= 1.0
        # Rough walls: the radius wobbles by up to roughness_m (noise with 4 m features).
        pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1) / 4.0
        wobble = fbm(pts, mix_seed(self.seed, seed), octaves=2).reshape(x.shape)
        scale = 1.0 + wobble * self.rough / self.r.min()
        return q <= scale * scale


class Worms(Carver):
    name = "worms"
    max_level = 0

    def __init__(
        self, region, depth_m, scale_m: float, thickness: float, seed: int, floors: bool = True
    ):
        (x0, z0), (x1, z1) = region
        self.x0, self.z0, self.x1, self.z1 = min(x0, x1), min(z0, z1), max(x0, x1), max(z0, z1)
        self.dmin, self.dmax = (float(depth_m[0]), float(depth_m[1]))
        if self.dmin < 0 or self.dmax <= self.dmin:
            raise WorldError(
                f"worms depth_m is (shallowest, deepest) metres under the ground, got {depth_m!r}"
            )
        if scale_m <= 0 or not 0 < thickness < 1:
            raise WorldError(
                f"worms need scale_m > 0 and 0 < thickness < 1, got {scale_m}, {thickness}"
            )
        self.scale, self.thickness, self.seed = float(scale_m), float(thickness), int(seed)
        self.keeps_floors = floors
        self.floors: tuple = ()  # tunnels and caverns whose floors it leaves (set by load_world)

    def keep_floors(self, caves) -> None:
        """Leaves FLOOR_M of rock under these caves' tunnels' and caverns' floors."""
        if self.keeps_floors:
            self.floors = tuple(c for c in caves if isinstance(c, (Tunnel, Cavern)))

    def floor_rock(self, x, y, z) -> np.ndarray:
        """Where (x, y, z) is rock it leaves under a floor."""
        x, y, z = np.broadcast_arrays(x, y, z)
        out = np.zeros(x.shape, dtype=bool)
        for cave in self.floors:
            x0, y0, z0, x1, y1, z1 = cave.bounds(None)
            m = FLOOR_M
            if (x.max() < x0 - m or x.min() > x1 + m or y.max() < y0 - m or y.min() > y1
                    or z.max() < z0 - m or z.min() > z1 + m):  # fmt: skip
                continue
            out |= cave.under_floor(x, y, z, m)
        return out

    def bounds(self, ground_range):
        lo, hi = ground_range(self.x0, self.z0, self.x1, self.z1)
        y0, y1 = self.band(lo, hi)
        return (self.x0, y0, self.z0, self.x1, y1, self.z1)

    def band(self, ground_lo: float, ground_hi: float) -> tuple[float, float]:
        """The heights (lowest, highest) tunnels can reach under ground between these heights."""
        return ground_lo - self.dmax, ground_hi - self.dmin

    def value(self, x, y, z, seed) -> np.ndarray:
        """max(|n1|, |n2|): tunnels run where both noise fields cross zero (under ``thickness``)."""
        x, y, z = np.broadcast_arrays(x, y, z)
        pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1) / self.scale
        s = mix_seed(self.seed, seed)
        a = np.abs(fbm(pts, s, octaves=2))
        b = np.abs(fbm(pts + 17.31, s + 1, octaves=2))
        return np.maximum(a, b).reshape(x.shape)

    def inside(self, x, y, z, ground) -> np.ndarray:
        x, y, z, ground = np.broadcast_arrays(x, y, z, ground)
        depth = ground - y
        return (
            (x >= self.x0)
            & (x < self.x1)
            & (z >= self.z0)
            & (z < self.z1)
            & (depth >= self.dmin)
            & (depth <= self.dmax)
        )

    def carve(self, x, y, z, ground, seed):
        hollow = self.inside(x, y, z, ground) & (self.value(x, y, z, seed) < self.thickness)
        return hollow & ~self.floor_rock(x, y, z) if self.floors and hollow.any() else hollow


def tunnel(
    points, radius_m: float = 2.5, smooth: bool = True, below_ground: bool = False
) -> Tunnel:
    """A tunnel through (x, y, z) points in metres (y up), ``radius_m`` round.

    ``smooth`` curves it through the points. With ``below_ground``, each y is a depth under the
    ground at that point instead (negative: above it), so a mouth opens wherever the ground is.
    """
    return Tunnel(points, radius_m, smooth, below_ground)


def cavern(
    center, radius_m, roughness_m: float = 0.0, seed: int = 0, below_ground: bool = False
) -> Cavern:
    """A cavern: an ellipsoid at ``center`` (x, y, z metres) with ``radius_m`` (one number, or
    (x, y, z) radii), its walls wobbling by up to ``roughness_m``. With ``below_ground``, the
    center's y is a depth under the ground there."""
    return Cavern(center, radius_m, roughness_m, seed, below_ground)


def worms(
    region,
    depth_m=(4.0, 30.0),
    scale_m: float = 24.0,
    thickness: float = 0.08,
    seed: int = 0,
    floors: bool = True,
) -> Worms:
    """Winding noise tunnels under ``region`` ((x0, z0), (x1, z1) metres), ``depth_m`` (shallowest,
    deepest) under the ground. ``scale_m`` spaces them; ``thickness`` (0–1) widens them.

    They open into the world's tunnels and caverns through walls and ceilings, but leave 2 m of
    rock under their floors (no pits in the way through); ``floors=False`` cuts those too.
    """
    return Worms(region, depth_m, scale_m, thickness, seed, floors)
