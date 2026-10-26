"""World scripts: ``@world`` functions returning a ``World`` (design.md, "Worlds")."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cubist.core.palette import Color, Palette
from cubist.dsl.context import building
from cubist.dsl.noise import mix_seed
from cubist.world.field import Field, WorldError, script_dir, world_size

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class Rule:
    """A surface rule: where it matches (by height, slope or a mask), the surface is ``color``."""

    color: Color
    kind: str
    value: float
    mask: Field | None = None

    def matches(self, heights: np.ndarray, slopes: np.ndarray, x, z, seed: int) -> np.ndarray:
        if self.kind == "height_over":
            return heights > self.value
        if self.kind == "height_under":
            return heights < self.value
        if self.kind == "slope_over":
            return slopes > self.value
        return self.mask.sample(x, z, seed) > self.value  # "where"


def height_over(m: float, color: Color) -> Rule:
    """Surface ``color`` above ``m`` metres (snow on peaks)."""
    return Rule(_color(color), "height_over", float(m))


def height_under(m: float, color: Color) -> Rule:
    """Surface ``color`` below ``m`` metres (sand along a shore)."""
    return Rule(_color(color), "height_under", float(m))


def slope_over(degrees: float, color: Color) -> Rule:
    """Surface ``color`` where the ground is steeper than ``degrees`` (rock on cliffs)."""
    return Rule(_color(color), "slope_over", float(degrees))


def where(mask: Field, color: Color, over: float = 0.5) -> Rule:
    """Surface ``color`` where ``mask`` is above ``over`` (a painted biome, a road)."""
    if not isinstance(mask, Field):
        raise WorldError(f"where() needs a mask field (circle, rect, mask_map, ...), got {mask!r}")
    return Rule(_color(color), "where", float(over), mask)


def _color(color) -> Color:
    if not isinstance(color, Color):
        raise WorldError(f"expected a palette color such as p.stone.mid, got {color!r}")
    return color


@dataclass
class World:
    """What a ``@world`` function returns.

    ``ground`` is the height field in metres. ``surface`` is a list of rules, first match wins,
    ending in a plain color (the default); the surface is the top ``surface_depth_m`` of ground.
    ``subsoil`` is ``(depth_m, color)`` layers under it, ending in the color that fills the rest.
    ``rivers`` and ``roads`` cut and level the ground along paths (rivers first); ``caves`` hollow
    it out. ``props`` places built assets. ``landmarks`` names points for the engine: (x, z) on
    the ground, or the name of a placed prop to sit at its base (on a cave floor, say). ``step_m``
    snaps the baked ground's heights to multiples of it: on slopes, fewer and taller steps than one
    voxel each (fewer quads for the engine).
    """

    ground: Field
    surface: Sequence[Rule | Color]
    subsoil: Sequence[tuple[float, Color] | Color] = ()
    surface_depth_m: float = 0.25
    rivers: Sequence = ()
    roads: Sequence = ()
    caves: Sequence = ()
    props: Sequence = ()
    landmarks: dict[str, tuple[float, float] | str] = field(default_factory=dict)
    step_m: float | None = None


@dataclass(frozen=True)
class WorldDef:
    """A world script function plus its metadata, produced by ``@world``."""

    fn: Callable
    name: str
    size_m: tuple[float, float]
    seed: int


def world(name: str, size_m, seed: int = 0) -> Callable[[Callable], WorldDef]:
    """Declare a world: ``size_m`` (x, z) metres from the origin, and its own seed.

    The world's seed mixes with the build's.
    """
    if not isinstance(name, str) or not _NAME.match(name):
        raise WorldError(
            f"world name {name!r}: use lowercase letters, digits and underscores, "
            "starting with a letter"
        )
    try:
        size = (float(size_m[0]), float(size_m[1]))
    except (TypeError, IndexError, ValueError):
        raise WorldError(f"world {name}: size_m must be (x, z) metres, got {size_m!r}") from None
    if size[0] <= 0 or size[1] <= 0:
        raise WorldError(f"world {name}: size_m must be positive on both axes, got {size_m!r}")

    def wrap(fn: Callable) -> WorldDef:
        return WorldDef(fn, name, size, int(seed))

    return wrap


@dataclass
class LoadedWorld:
    """A world script run against a palette: its definition, layers and effective seed."""

    defn: WorldDef
    world: World
    palette: Palette
    seed: int
    script_dir: Path | None
    ground: Field = None  # type: ignore[assignment]  # the ground with rivers and roads applied
    _caves: list | None = None

    @property
    def name(self) -> str:
        return self.defn.name

    def ground_range(self, x0: float, z0: float, x1: float, z1: float) -> tuple[float, float]:
        """The ground's (lowest, highest) height over an area, sampled about every 4 m."""
        nx = int(min(max((x1 - x0) / 4.0, 2), 512))
        nz = int(min(max((z1 - z0) / 4.0, 2), 512))
        gx, gz = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(z0, z1, nz))
        h = self.ground.sample(gx, gz, self.seed)
        return float(h.min()), float(h.max())

    def cave_bounds(self) -> list:
        """Each cave with the box (x0, y0, z0, x1, y1, z1 metres) holding what it carves; cached."""
        if self._caves is None:
            from cubist.world.sample import world_context

            with world_context(self):
                self._caves = [(c, c.bounds(self.ground_range)) for c in self.world.caves]
        return self._caves

    @property
    def size_m(self) -> tuple[float, float]:
        return self.defn.size_m

    def subsoil_layers(self) -> tuple[list[tuple[float, Color]], Color]:
        """(depth under the surface layer, color) bands, and the color filling the rest below."""
        layers, rest = [], None
        for entry in self.world.subsoil:
            if isinstance(entry, Color):
                rest = entry
            else:
                depth, color = entry
                layers.append((float(depth), _color(color)))
        if rest is None:
            rest = layers[-1][1] if layers else self.default_surface()
        return layers, rest

    def landmark_points(self) -> dict[str, tuple[float, float, float | None]]:
        """Each landmark's (x, z, y) in metres; y is None for one on the ground."""
        from cubist.world.props import Place

        named = {p.name: p for p in self.world.props if isinstance(p, Place) and p.name}
        out = {}
        for name, at in self.world.landmarks.items():
            if not isinstance(at, str):
                out[name] = (float(at[0]), float(at[-1]), None)
                continue
            prop = named[at]
            x, z = prop.at[0], prop.at[-1]
            if prop.below_ground_m is not None:
                from cubist.world.sample import heights

                y = float(heights(self, [x], [z])[0]) - prop.below_ground_m - prop.sink_m
            else:
                y = prop.at[1] - prop.sink_m if len(prop.at) == 3 else None
            out[name] = (x, z, y)
        return out

    def default_surface(self) -> Color:
        last = self.world.surface[-1] if self.world.surface else None
        if not isinstance(last, Color):
            raise WorldError(
                f"world {self.name}: end the surface rules with a plain color (the default surface)"
            )
        return last


def load_world(
    defn: WorldDef, palette: Palette, seed: int = 0, script: Path | None = None
) -> LoadedWorld:
    """Runs the world function with the palette; the world's seed mixes with the build ``seed``."""
    directory = None if script is None else Path(script).resolve().parent
    token_dir, token_size = script_dir.set(directory), world_size.set(defn.size_m)
    try:
        with building(palette, seed):
            result = defn.fn(palette)
    finally:
        script_dir.reset(token_dir)
        world_size.reset(token_size)
    if not isinstance(result, World):
        raise WorldError(
            f"world {defn.name}: the @world function must return a World(ground=..., "
            f"surface=[...]), got {result!r}"
        )
    if not isinstance(result.ground, Field):
        raise WorldError(
            f"world {defn.name}: ground must be a field (noise, flat, height_map, ...), "
            f"got {result.ground!r}"
        )
    from cubist.world.caves import Carver, Worms
    from cubist.world.paths import EditedGround, River, Road
    from cubist.world.props import Place, Scatter

    kinds = (
        ("rivers", result.rivers, (River,), "river(...)"),
        ("roads", result.roads, (Road,), "road(...)"),
        ("caves", result.caves, (Carver,), "tunnel, cavern or worms"),
        ("props", result.props, (Place, Scatter), "place(...) or scatter(...)"),
    )
    for kind, items, classes, wanted in kinds:
        for item in items:
            if not isinstance(item, classes):
                raise WorldError(f"world {defn.name}: {kind} take {wanted}, got {item!r}")
    edits = [*result.rivers, *result.roads]
    ground = EditedGround(result.ground, edits) if edits else result.ground
    loaded = LoadedWorld(defn, result, palette, mix_seed(defn.seed, seed), directory, ground)
    from cubist.world.sample import heights

    for cave in result.caves:
        cave.resolve(lambda x, z: heights(loaded, x, z))
    for cave in result.caves:
        if isinstance(cave, Worms):
            cave.keep_floors(result.caves)
    if result.step_m is not None and result.step_m <= 0:
        raise WorldError(f"world {defn.name}: step_m must be positive, got {result.step_m}")
    named = {p.name for p in result.props if isinstance(p, Place) and p.name}
    for name, at in result.landmarks.items():
        if isinstance(at, str) and at not in named:
            raise WorldError(
                f"world {defn.name}: landmark {name} names {at!r}, but there's no prop named "
                f"that; give a place(...) name={at!r}, or the landmark (x, z)"
            )
    loaded.default_surface()
    for rule in result.surface[:-1]:
        if not isinstance(rule, Rule):
            raise WorldError(
                f"world {defn.name}: only the last surface entry may be a plain color; "
                f"got {rule!r} before it"
            )
    loaded.subsoil_layers()
    return loaded
