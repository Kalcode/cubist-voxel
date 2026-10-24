# Getting started

## Install

You need Python 3.11+ and [uv](https://docs.astral.sh/uv/) (`brew install uv`).

```sh
git clone ssh://git@192.168.50.249/kalcode/voxel-auther-system.git
cd voxel-auther-system
uv sync
uv run pytest -q        # optional: the full test suite
```

Optional tools:

| Tool | For | Install (macOS) |
| --- | --- | --- |
| [MagicaVoxel](https://ephtracy.github.io/) | Viewing and hand-editing `.vox` files, nice renders | `brew install --cask magicavoxel` |
| [Goxel](https://goxel.xyz/) | Open-source `.vox` editor | `brew install --cask goxel` |
| [Godot 4](https://godotengine.org/) | Using the exported `.glb`; also runs the GLB import test | `brew install --cask godot` |
| [vengi](https://vengi-voxel.github.io/vengi/) | `vengi-voxconvert` for OBJ, Qubicle, schematics, and more | see the vengi site |
| [Claude Code](https://claude.com/claude-code) | An agent that authors assets for you | see the Claude Code docs |

## Your first asset in five minutes

Everything happens inside a project: a directory with a `style.toml`. The demo project is ready
to use.

```sh
cd projects/demo
uv run cubist new lantern --category prop --tags metal,light \
    --description "A hanging iron lantern" --must-have "glowing glass panes"
```

That writes `briefs/lantern.toml` and a buildable stub in `assets/lantern.py`. Replace the stub
with something real:

```python
from cubist import *


@asset(name="lantern", size=(7, 13, 7), tags=["prop", "metal", "light"], grounded=True)
def lantern(p):
    C = 3.5  # odd width, centered on a voxel center, so the roof keeps its tip
    base = box((7, 1, 7), color=p.metal.dark)
    glass = box((5, 6, 5), color=p.glass.light).at(1, 1, 1)
    posts = [box((1, 6, 1), color=p.metal.dark).at(x, 1, z) for x in (1, 5) for z in (1, 5)]
    roof = cone(r=3.6, h=3, color=p.metal.mid).at(C, 7, C)
    ring = (box((3, 3, 1), color=p.metal.mid) - box((1, 1, 1)).at(1, 1, 0)).at(2, 10, 3)
    return base.union(glass, *posts, roof, ring).highlight_edges()
```

Build it and look:

```sh
uv run cubist build assets/lantern.py     # JSON summary with the lint report
open out/lantern/contact.png               # five views with rulers and a color legend
```

Change something, rebuild, and compare:

```sh
uv run cubist build assets/lantern.py
uv run cubist diff out/lantern             # before / after / changes
uv run cubist render out/lantern --slices y=6..7   # exact layers as text
```

When it looks right, export it:

```sh
uv run cubist export out/lantern --ao      # export/lantern.glb and export/lantern.vox
```

Open `export/lantern.vox` in MagicaVoxel or drag `export/lantern.glb` into Godot.

## Commands

| Command | Does |
| --- | --- |
| `cubist new <name>` | Scaffold a brief and a script stub |
| `cubist build <script>` | Build: `.vox`, contact sheet, lint report, and for moving assets a filmstrip and GIFs; prints the iteration (`--seed N`, `--check-determinism`, `--full` for the whole summary, `--check` to run and lint without outputs or counting an iteration) |
| `cubist render <asset>` | Re-render views (`--views front,iso,silhouette`) or print `--slices y=a..b`; `--clip NAME` draws a filmstrip (`--frames`, `--view`) or, with `--at T`, a posed contact sheet |
| `cubist diff <asset> [<other>]` | Before/after/changes sheet (default: against the previous build) |
| `cubist lint <asset>` | Lint only (exit 2 on errors) |
| `cubist inspect <file.vox>` | Parts, bounds, and color counts as JSON |
| `cubist export <asset>` | `.glb`, `.vox` and the manifest `<name>.cubist.json` (blocked by lint errors unless `--force`); `--via-vengi FORMAT` |
| `cubist export --all` | The same for every built asset in `out/`, say after a style change; a game that loads `export/` should commit it |
| `cubist style` | The parsed style bible with every palette name |
| `cubist palette import <file>` | Use a palette file (.png, .gif, .gpl, .hex) as the palette source |
| `cubist examples list \| search \| promote` | The example library |
| `cubist eval` | Report on every brief: lint, iterations, scores, and a gallery |
| `cubist mcp` | Run the MCP server on stdio |

Every command prints JSON (slices print text) and takes `--help`.

## Start your own project

```sh
mkdir -p projects/mygame && cp projects/demo/style.toml projects/mygame/
```

Edit the palette, scale, budgets, and rules (see the `voxel-project` skill, or ask Claude to set
it up). Then `cubist new` your first brief.

Next: [working with Claude](working-with-claude.md), [how it works](how-it-works.md), and
the [DSL reference](dsl.md).
