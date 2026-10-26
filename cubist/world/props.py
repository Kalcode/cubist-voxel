"""Props: built assets placed in a world (design.md, "Worlds").

``place`` puts one asset at a point; ``scatter`` spreads many over a region by density, with rules
on slope and height. They stay placements: the engine stamps exported assets
(``export/<asset>.vox`` and its manifest) where the bake says, standing on the ground, so moving one
doesn't re-bake terrain.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cubist.dsl.noise import hash_unit, mix_seed
from cubist.world.field import Field, WorldError, rect

VOXELS_PER_M = 16


@dataclass(frozen=True)
class Place:
    asset: str
    at: tuple
    rotate_y: int
    name: str | None
    sink_m: float
    below_ground_m: float | None = None
    carve: bool = False


@dataclass(frozen=True)
class Scatter:
    asset: str
    region: Field
    per_ha: float
    seed: int
    spacing_m: float
    rotate: bool
    slope_max: float | None
    height_m: tuple[float, float] | None
    avoid: Field | None
    sink_m: float


def _rotation(rotate_y: int) -> int:
    if rotate_y not in (0, 90, 180, 270):
        raise WorldError(f"rotate_y is 0, 90, 180 or 270 degrees, got {rotate_y}")
    return rotate_y


def place(
    asset: str,
    at,
    rotate_y: int = 0,
    name: str | None = None,
    sink_m: float = 0.0,
    below_ground_m: float | None = None,
    carve: bool = False,
) -> Place:
    """One ``asset`` (a built, exported asset's name) at ``at``.

    (x, z) metres stands it on the ground (its footprint's highest point); (x, y, z) puts its base
    at that height. ``sink_m`` lowers it. ``below_ground_m`` puts its base that far under the
    ground at (x, z) instead: on a cave floor, say. ``carve`` digs it into the rock: Kiln empties
    the world wherever the asset's box is empty (a crypt or a room underground).
    """
    if len(at) not in (2, 3):
        raise WorldError(f"place {asset}: at is (x, z) or (x, y, z) metres, got {at!r}")
    below = None if below_ground_m is None else float(below_ground_m)
    at = tuple(float(v) for v in at)
    return Place(asset, at, _rotation(rotate_y), name, float(sink_m), below, bool(carve))


def scatter(
    asset: str,
    region,
    per_ha: float,
    seed: int = 0,
    spacing_m: float | None = None,
    rotate: bool = True,
    slope_max: float | None = None,
    height_m: tuple[float, float] | None = None,
    avoid: Field | None = None,
    sink_m: float = 0.25,
) -> Scatter:
    """``asset`` spread over ``region`` (a mask, or ``((x0, z0), (x1, z1))``).

    About ``per_ha`` per hectare, at least ``spacing_m`` apart (default: from the density), turned
    at random by quarter turns. Points steeper than ``slope_max``°, outside ``height_m`` (low,
    high), or where ``avoid`` is over 0.5 are skipped. Each sinks ``sink_m`` into the ground, so
    trunks on slopes don't float.
    """
    if per_ha <= 0:
        raise WorldError(f"scatter {asset}: per_ha must be positive, got {per_ha}")
    if not isinstance(region, Field):
        (x0, z0), (x1, z1) = region
        region = rect((x0, z0), (x1, z1), edge_m=0)
    spacing = spacing_m if spacing_m is not None else math.sqrt(10000.0 / per_ha) * 0.6
    return Scatter(asset, region, float(per_ha), int(seed), float(spacing), rotate, slope_max,
                   height_m, avoid, float(sink_m))  # fmt: skip


@dataclass
class Placement:
    """One prop as the engine stamps it: the asset file, its base point (voxels) and turn."""

    asset: str
    at: tuple[float, float, float]
    rotate_y: int
    name: str | None
    carve: bool = False

    def json(self, relative_to: Path, export: Path) -> dict:
        import os

        path = (export / f"{self.asset}.vox").resolve()
        asset = os.path.relpath(path, Path(relative_to).resolve())
        if asset.count("..") > 3:  # the bake is far from the project: an absolute path reads better
            asset = str(path)
        out = {"asset": asset, "at": [round(v, 3) for v in self.at], "rotate_y": self.rotate_y}
        if self.name:
            out["name"] = self.name
        if self.carve:
            out["carve"] = True
        return out


def footprint(export: Path, asset: str, rotate_y: int) -> tuple[float, float]:
    """The asset's footprint (x, z metres) in its turn, from its exported manifest."""
    manifest = export / f"{asset}.cubist.json"
    if not manifest.is_file():
        raise WorldError(
            f"prop {asset} isn't exported: no {manifest}. "
            f"Build it, then run `cubist export out/{asset}`."
        )
    sx, _, sz = json.loads(manifest.read_text())["size"]
    w, d = sx / VOXELS_PER_M, sz / VOXELS_PER_M
    return (d, w) if rotate_y in (90, 270) else (w, d)


def _stand(loaded, xs, zs, half_w, half_d) -> np.ndarray:
    """The highest ground (metres) under each footprint: 3 × 3 samples over it."""
    from cubist.world.sample import heights

    u = np.array([-1.0, 0.0, 1.0])
    ox, oz = np.meshgrid(u, u)
    px = np.asarray(xs)[:, None] + ox.ravel()[None, :] * np.asarray(half_w)[..., None]
    pz = np.asarray(zs)[:, None] + oz.ravel()[None, :] * np.asarray(half_d)[..., None]
    return heights(loaded, px, pz).max(axis=1)


def scatter_points(loaded, s: Scatter) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, z, rotate_y) of a scatter: a jittered grid ``spacing_m`` apart, kept with probability
    ``per_ha`` × cell area, then filtered by region and rules. Deterministic for a seed; worked out
    once per loaded world (the arrays are read-only)."""
    if loaded._scatters is None:
        loaded._scatters = {}
    if id(s) not in loaded._scatters:
        points = _scatter_points(loaded, s)
        for a in points:
            a.setflags(write=False)
        loaded._scatters[id(s)] = (s, points)  # the scatter too, so its id stays its own
    return loaded._scatters[id(s)][1]


def _scatter_points(loaded, s: Scatter) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from cubist.world.sample import heights, slopes, world_context

    sx, sz = loaded.size_m
    cell = s.spacing_m
    nx, nz = int(sx // cell), int(sz // cell)
    j, i = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
    ij = np.stack([i.ravel(), np.zeros(i.size, dtype=np.int64), j.ravel()], axis=-1)
    seed = mix_seed(s.seed, loaded.seed)
    keep = hash_unit(ij, seed) < s.per_ha * cell * cell / 10000.0
    jitter_x = hash_unit(ij, seed + 1) * 0.8 + 0.1
    jitter_z = hash_unit(ij, seed + 2) * 0.8 + 0.1
    x = (ij[:, 0] + jitter_x) * cell
    z = (ij[:, 2] + jitter_z) * cell
    turn = (
        (np.floor(hash_unit(ij, seed + 3) * 4).astype(int) * 90)
        if s.rotate
        else np.zeros(len(x), dtype=int)
    )
    x, z, turn = x[keep], z[keep], turn[keep]
    with world_context(loaded):
        ok = s.region.sample(x, z, loaded.seed) > 0.5
        if s.avoid is not None:
            ok &= s.avoid.sample(x, z, loaded.seed) <= 0.5
    if s.slope_max is not None and ok.any():
        ok[ok] &= slopes(loaded, x[ok], z[ok]) <= s.slope_max
    if s.height_m is not None and ok.any():
        h = heights(loaded, x[ok], z[ok])
        ok[ok] &= (h >= s.height_m[0]) & (h <= s.height_m[1])
    return x[ok], z[ok], turn[ok]


def placements(loaded, export: Path) -> list[Placement]:
    """Every prop of the world, standing on its ground, in script order (scatters in grid order)."""
    out: list[Placement] = []
    for prop in loaded.world.props:
        if isinstance(prop, Place):
            if prop.below_ground_m is not None:
                x, z = prop.at[0], prop.at[-1]
                from cubist.world.sample import heights as ground

                y = float(ground(loaded, [x], [z])[0]) - prop.below_ground_m
            elif len(prop.at) == 3:
                x, y, z = prop.at
            else:
                x, z = prop.at
                w, d = footprint(export, prop.asset, prop.rotate_y)
                y = float(_stand(loaded, [x], [z], w / 2, d / 2)[0])
            base = (x * VOXELS_PER_M, (y - prop.sink_m) * VOXELS_PER_M, z * VOXELS_PER_M)
            out.append(Placement(prop.asset, base, prop.rotate_y, prop.name, prop.carve))
        elif isinstance(prop, Scatter):
            x, z, turn = scatter_points(loaded, prop)
            if len(x) == 0:
                continue
            w0, d0 = footprint(export, prop.asset, 0)
            half_w = np.where(np.isin(turn, (90, 270)), d0, w0) / 2
            half_d = np.where(np.isin(turn, (90, 270)), w0, d0) / 2
            y = _stand(loaded, x, z, half_w, half_d) - prop.sink_m
            for xi, yi, zi, ti in zip(x, y, z, turn, strict=True):
                out.append(
                    Placement(
                        prop.asset,
                        (xi * VOXELS_PER_M, yi * VOXELS_PER_M, zi * VOXELS_PER_M),
                        int(ti),
                        None,
                    )
                )
        else:
            raise WorldError(
                f"world {loaded.name}: props take place(...) or scatter(...), got {prop!r}"
            )
    return out
