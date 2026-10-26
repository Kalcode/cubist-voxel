"""``cubist world build``: run a world script, render its map, write stats (design.md, "Worlds")."""

from __future__ import annotations

import json
import runpy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from cubist.build.runner import find_style, load_project_style
from cubist.world.field import WorldError
from cubist.world.map import WorldMap, render_map, surface_shares
from cubist.world.model import LoadedWorld, WorldDef, load_world

SLOPE_BANDS = ((0, 15), (15, 30), (30, 45), (45, 91))


@dataclass
class WorldBuild:
    name: str
    loaded: LoadedWorld
    map: WorldMap
    stats: dict
    map_path: Path | None

    @property
    def map_image(self) -> Image.Image:
        return self.map.image


def load_script(script: str | Path, seed: int = 0, style: str | Path | None = None) -> LoadedWorld:
    """Runs a world script and loads its one ``@world``.

    In this process for now; asset builds use a subprocess with a timeout.
    """
    script = Path(script)
    if not script.is_file():
        raise WorldError(f"no such world script: {script}")
    style_path = Path(style) if style else find_style(script)
    palette = load_project_style(style_path).palette
    namespace = runpy.run_path(str(script), run_name="__cubist_world__")
    defs = list({id(v): v for v in namespace.values() if isinstance(v, WorldDef)}.values())
    if len(defs) != 1:
        raise WorldError(f"{script}: expected one @world function, found {len(defs)}")
    return load_world(defs[0], palette, seed, script)


def _prop_counts(loaded: LoadedWorld) -> dict[str, int]:
    from cubist.world.props import Place, scatter_points

    counts: dict[str, int] = {}
    for prop in loaded.world.props:
        n = 1 if isinstance(prop, Place) else len(scatter_points(loaded, prop)[0])
        counts[prop.asset] = counts.get(prop.asset, 0) + n
    return counts


def stats_of(loaded: LoadedWorld, m: WorldMap) -> dict:
    slopes = m.slopes
    return {
        "name": loaded.name,
        "size_m": list(loaded.size_m),
        "seed": loaded.seed,
        "metres_per_pixel": m.metres_per_pixel,
        "height_m": {
            "min": round(float(m.heights.min()), 2),
            "max": round(float(m.heights.max()), 2),
            "mean": round(float(m.heights.mean()), 2),
        },  # fmt: skip
        "slope_pct": {
            f"{lo}-{hi}" if hi < 91 else f"{lo}+": round(
                100.0 * float(np.mean((slopes >= lo) & (slopes < hi))), 2
            )
            for lo, hi in SLOPE_BANDS
        },  # fmt: skip
        "surface_pct": {
            k: round(float(v), 2) for k, v in surface_shares(loaded, m.surface).items()
        },
        "landmarks": {k: list(v) for k, v in loaded.world.landmarks.items()},
        "props": _prop_counts(loaded),
    }


def build_world(
    script: str | Path,
    out_dir: str | Path | None = None,
    seed: int = 0,
    style: str | Path | None = None,
    write: bool = True,
) -> WorldBuild:
    """Builds the map and stats; with ``write``, into ``<out_dir>/<name>/``.

    ``out_dir`` defaults to the project's ``out/worlds``.
    """
    loaded = load_script(script, seed, style)
    m = render_map(loaded)
    stats = stats_of(loaded, m)
    map_path = None
    if write:
        base = Path(out_dir) if out_dir else find_style(Path(script)).parent / "out" / "worlds"
        target = base / loaded.name
        target.mkdir(parents=True, exist_ok=True)
        map_path = target / "map.png"
        m.image.save(map_path, optimize=False)
        (target / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return WorldBuild(loaded.name, loaded, m, stats, map_path)
