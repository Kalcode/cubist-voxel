# Cubist

Agent-native voxel asset authoring: an agent writes a Python script, Cubist samples it into
voxels and writes MagicaVoxel `.vox` files. Renders, lint, and glTF export come next.

Status: milestone 1 of the [design](docs/design.md) (core grid, palette, primitives, CSG,
mirror, `.vox` read/write).

## Setup

```sh
uv sync
uv run pytest
```

## Build an asset

```sh
uv run cubist build projects/demo/assets/goblin.py   # -> projects/demo/out/goblin_scout/
uv run cubist inspect projects/demo/out/goblin_scout/goblin_scout.vox
```

An asset script:

```python
from cubist import *

@asset(name="potion", size=(8, 12, 8), tags=["prop"])
def potion(p):
    bottle = sphere(3.6, color=p.glass.light).at(4, 3.5, 4)
    cork = cylinder(r=1.2, h=2, color=p.cork).at(4, 10, 4)
    return bottle | cork
```

Colors come from the project's `style.toml` by name (`p.<ramp>.dark|mid|light`,
`p.<ramp>[i]`, or named shades); an unknown name is a build error.

Conventions: Y-up, right-handed, +Z is the front, integer voxel coordinates. A voxel is
filled when its center is inside or on a shape. Primitives start at their local origin
(`box` at its min corner, `sphere` at its center, `cylinder`/`cone` at the center of their
base) and `.at()` moves them. Union gives overlaps the later shape's color; subtract and
intersect keep the left shape's colors.

## Viewing output

Open `.vox` files in [MagicaVoxel](https://ephtracy.github.io/)
(`brew install --cask magicavoxel`) or [Goxel](https://goxel.xyz/)
(`brew install --cask goxel`).

## Layout

```text
cubist/
  core/    grid, palette, parts, models, @asset
  dsl/     SDF primitives, CSG and transforms
  build/   subprocess runner with timeout
  io/      .vox reader/writer
  cli.py
projects/demo/   style.toml and five demo asset scripts
tests/
```
