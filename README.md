# Cubist

Agent-native voxel asset authoring: an agent writes a Python script, Cubist samples it into
voxels, writes MagicaVoxel `.vox` files, renders a contact sheet to critique, and lints the
result against the project's style bible, and exports `.glb` with parts as nodes for game
engines.

Status: milestones 1–4 of the [design](docs/design.md): the full shape DSL (primitives, CSG,
transforms, morphology, paint ops, selections), `.vox` read/write with part hierarchy, contact
sheets, ASCII slices, lint, `style.toml`, and GLB export verified in Godot 4.

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
uv run cubist lint projects/demo/out/goblin_scout          # exit 2 on errors
uv run cubist export projects/demo/out/goblin_scout --ao  # -> projects/demo/export/
uv run cubist inspect projects/demo/out/goblin_scout/goblin_scout.vox
uv run cubist style --style projects/demo/style.toml
```

A build writes `<name>.vox`, `contact.png` (front, right, back, top, and iso views with
voxel-coordinate rulers and a color legend), `lint.json`, and `build.json`. `--seed N` reshuffles
every seeded op; `--check-determinism` builds twice in separate processes and compares.
`cubist export` re-lints, refuses on errors unless `--force`, and writes `.glb` (one node per
part at its pivot, sockets as empty nodes, metres from `voxel_size_m`) and `.vox`; other formats
go through `--via-vengi FORMAT` when `vengi-voxconvert` is installed.

An asset script (the barrel from the design doc):

```python
from cubist import *


@asset(name="barrel", size=(12, 16, 12), tags=["prop", "wood"], grounded=True)
def barrel(p):
    body = lathe(profile=[(0, 5), (3, 6), (13, 6), (16, 5)], color=p.wood.mid).at(6, 0, 6)
    body = body.hollow(1).union(cylinder(r=5, h=1, color=p.wood.dark).at(6, 15, 6))
    for y in (2, 13):
        body = body.paint(p.metal.dark, where=band(axis="y", at=y, height=1))
    body = body.dither(p.wood.ramp, seed=1).darken_crevices()
    return Model(parts=[Part("body", body, pivot=(6, 0, 6))])
```

The DSL reference (every primitive, op, selection, and its conventions) is in
[docs/design.md](docs/design.md#the-dsl).

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
  core/    grid, palette, style bible, parts, models, @asset
  dsl/     SDF primitives, CSG, transforms, voxel morphology, paint, selections, noise
  build/   subprocess runner with timeout
  io/      .vox reader/writer, greedy meshing, GLB writer, vengi bridge
  export.py  lint-gated export
  lint/    lint checks and report
  render/  rasterizer, contact sheet, ASCII slices, bitmap font
  cli.py
projects/demo/   style.toml and six demo asset scripts
tests/           golden/ holds reviewed contact sheets
```
