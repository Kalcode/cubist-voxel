"""Sampling a loaded world: heights, slopes and surface colors per column."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

from cubist.world.field import script_dir, world_size
from cubist.world.model import LoadedWorld, Rule

# Slopes come from central differences this far apart (metres): the bake's sampling lattice.
SLOPE_STEP_M = 0.5


@contextmanager
def world_context(loaded: LoadedWorld) -> Iterator[None]:
    """Fields resolve painted maps against the world script while sampling."""
    token_dir, token_size = script_dir.set(loaded.script_dir), world_size.set(loaded.size_m)
    try:
        yield
    finally:
        script_dir.reset(token_dir)
        world_size.reset(token_size)


def heights(loaded: LoadedWorld, x, z) -> np.ndarray:
    """Ground heights in metres at (x, z)."""
    with world_context(loaded):
        return loaded.world.ground.sample(
            np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64), loaded.seed
        )


def slopes(loaded: LoadedWorld, x, z) -> np.ndarray:
    """Ground slope in degrees at (x, z), from central differences ``SLOPE_STEP_M`` apart."""
    x, z, d = np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64), SLOPE_STEP_M
    gx = (heights(loaded, x + d, z) - heights(loaded, x - d, z)) / (2 * d)
    gz = (heights(loaded, x, z + d) - heights(loaded, x, z - d)) / (2 * d)
    return np.degrees(np.arctan(np.hypot(gx, gz)))


def surface(loaded: LoadedWorld, x, z, h: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Palette index of the surface per column: the first matching rule, else the default color."""
    out = np.full(np.shape(h), loaded.default_surface().index, dtype=np.int16)
    decided = np.zeros(np.shape(h), dtype=bool)
    with world_context(loaded):
        for rule in loaded.world.surface:
            if not isinstance(rule, Rule):
                break
            hit = rule.matches(h, s, x, z, loaded.seed) & ~decided
            out[hit] = rule.color.index
            decided |= hit
    return out


def columns(loaded: LoadedWorld, x, z, slope: np.ndarray | None = None):
    """(heights m, slopes °, surface palette indices) at (x, z); pass ``slope`` if already known."""
    h = heights(loaded, x, z)
    s = slopes(loaded, x, z) if slope is None else slope
    return h, s, surface(loaded, x, z, h, s)


def surface_names(loaded: LoadedWorld, indices) -> list[str]:
    """Palette names for surface indices."""
    return [loaded.palette.name_of(int(i)) for i in np.ravel(indices)]
