"""Caves: carvers that hollow out the ground (design.md, "Worlds").

A carver says which points (x, y, z in metres, y up) are air. Tunnels follow a 3D path, caverns are
ellipsoids with optionally rough walls, and worms are noise tunnels in a depth band under a region.
Carvers only remove ground; they apply up to their ``max_level`` of detail (tunnels and caverns to
level 2, so a cave mouth shows from a distance; worms at level 0 only).
"""

from __future__ import annotations

import numpy as np

from cubist.dsl.noise import fbm, mix_seed
from cubist.world.field import WorldError
from cubist.world.paths import _catmull_rom


class Carver:
    max_level = 2
    name = "carver"

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

    def __init__(self, points, radius_m: float, smooth: bool):
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
            raise WorldError(f"a tunnel needs at least two (x, y, z) points, got {points!r}")
        if radius_m <= 0:
            raise WorldError(f"tunnel radius_m must be positive, got {radius_m}")
        if smooth and len(pts) >= 3:
            pts = _catmull_rom(pts, 2.0)
        self.points, self.radius = pts, float(radius_m)
        self.a, self.ab = pts[:-1], pts[1:] - pts[:-1]
        self.len2 = np.maximum((self.ab**2).sum(axis=1), 1e-12)

    def bounds(self, ground_range):
        lo, hi = self.points.min(axis=0) - self.radius, self.points.max(axis=0) + self.radius
        return (lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])

    def carve(self, x, y, z, ground, seed):
        p = np.stack(np.broadcast_arrays(x, y, z), axis=-1)
        shape = p.shape[:-1]
        p = p.reshape(-1, 3)
        lo, hi = p.min(axis=0) - self.radius, p.max(axis=0) + self.radius
        seg_lo = np.minimum(self.a, self.a + self.ab)
        seg_hi = np.maximum(self.a, self.a + self.ab)
        near = np.flatnonzero(((seg_lo <= hi) & (seg_hi >= lo)).all(axis=1))
        out = np.zeros(len(p), dtype=bool)
        for k in near:  # usually a handful of segments per chunk
            ap = p - self.a[k]
            u = np.clip(ap @ self.ab[k] / self.len2[k], 0.0, 1.0)
            d2 = ((ap - u[:, None] * self.ab[k]) ** 2).sum(axis=1)
            out |= d2 <= self.radius**2
        return out.reshape(shape)


class Cavern(Carver):
    name = "cavern"

    def __init__(self, center, radius_m, roughness_m: float, seed: int):
        self.c = np.asarray(center, dtype=np.float64)
        r = np.broadcast_to(np.asarray(radius_m, dtype=np.float64), (3,)).copy()
        if self.c.shape != (3,) or (r <= 0).any():
            raise WorldError(
                f"a cavern needs a center (x, y, z) and positive radii, "
                f"got {center!r}, {radius_m!r}"
            )
        self.r, self.rough, self.seed = r, float(roughness_m), int(seed)

    def bounds(self, ground_range):
        lo, hi = self.c - self.r - self.rough, self.c + self.r + self.rough
        return (lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])

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

    def __init__(self, region, depth_m, scale_m: float, thickness: float, seed: int):
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
        return self.inside(x, y, z, ground) & (self.value(x, y, z, seed) < self.thickness)


def tunnel(points, radius_m: float = 2.5, smooth: bool = True) -> Tunnel:
    """A tunnel along (x, y, z) points in metres (y up), ``radius_m`` round, curved by ``smooth``."""
    return Tunnel(points, radius_m, smooth)


def cavern(center, radius_m, roughness_m: float = 0.0, seed: int = 0) -> Cavern:
    """A cavern: an ellipsoid at ``center`` (x, y, z metres) with ``radius_m`` (one number, or
    (x, y, z) radii), its walls wobbling by up to ``roughness_m``."""
    return Cavern(center, radius_m, roughness_m, seed)


def worms(
    region, depth_m=(4.0, 30.0), scale_m: float = 24.0, thickness: float = 0.08, seed: int = 0
) -> Worms:
    """Winding noise tunnels under ``region`` ((x0, z0), (x1, z1) metres), ``depth_m`` (shallowest,
    deepest) under the ground. ``scale_m`` spaces them; ``thickness`` (0–1) widens them."""
    return Worms(region, depth_m, scale_m, thickness, seed)
