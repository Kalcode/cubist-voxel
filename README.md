# Cubist

Agent-native voxel asset authoring: an agent writes a Python script, Cubist samples it into
voxels, writes MagicaVoxel `.vox` files, and renders a contact sheet to critique. Lint and glTF
export come next.

Status: milestones 1–2 of the [design](docs/design.md): core grid, palette, primitives, CSG,
mirror, `.vox` read/write, contact sheets, and ASCII slices.

## Setup

```sh
uv sync
uv run pytest
```

## Build an asset

```sh
uv run cubist build projects/demo/assets/goblin.py   # -> projects/demo/out/goblin_scout/
uv run cubist render projects/demo/out/goblin_scout --slices y=18..19
uv run cubist render projects/demo/out/goblin_scout --views front,iso,silhouette
uv run cubist inspect projects/demo/out/goblin_scout/goblin_scout.vox
```

A build writes `<name>.vox`, `contact.png` (front, right, back, top, and iso views with
voxel-coordinate rulers and a color legend), and `build.json`.

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
  render/  rasterizer, contact sheet, ASCII slices, bitmap font
  cli.py
projects/demo/   style.toml and five demo asset scripts
tests/           golden/ holds reviewed contact sheets
```
