"""Worlds: terrain for premade game worlds (design.md, "Worlds"). World scripts start with
``from cubist.world import *``."""

from cubist.world.build import WorldBuild, build_world, load_script
from cubist.world.caves import cavern, tunnel, worms
from cubist.world.field import (
    Field,
    WorldError,
    circle,
    flat,
    height_map,
    mask_map,
    noise,
    rect,
    ridges,
)
from cubist.world.model import (
    LoadedWorld,
    Rule,
    World,
    WorldDef,
    height_over,
    height_under,
    load_world,
    slope_over,
    where,
    world,
)
from cubist.world.paths import Path, path, river, road
from cubist.world.props import place, scatter

__all__ = [
    "Field",
    "LoadedWorld",
    "Rule",
    "World",
    "WorldBuild",
    "WorldDef",
    "WorldError",
    "build_world",
    "cavern",
    "circle",
    "flat",
    "height_map",
    "height_over",
    "height_under",
    "load_script",
    "load_world",
    "mask_map",
    "noise",
    "Path",
    "path",
    "place",
    "rect",
    "ridges",
    "river",
    "road",
    "scatter",
    "slope_over",
    "tunnel",
    "where",
    "world",
    "worms",
]
