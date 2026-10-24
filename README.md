<p align="center">
  <img src="docs/images/icon-256.png" width="128" alt="Cubist logo: a voxel cube with one block lifted out">
</p>

<h1 align="center">Cubist</h1>

<p align="center"><b>Voxel assets, written as code by agents.</b></p>

<p align="center">
  <img src="docs/images/banner.png" alt="Voxel assets built with Cubist: a goblin, chest, barrel, torch, hut, tree, mushroom, potion, and sword">
</p>

Cubist is a headless voxel modeling toolkit for AI agents. An agent writes a short Python
script, Cubist turns it into voxels, renders a contact sheet the agent can look at, checks it
against your game's style bible, and exports MagicaVoxel `.vox` and engine-ready `.glb` files.
The agent never places voxels by hand: it writes code, looks, and revises.

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

## What you get

- **A shape DSL built for agents.** Primitives (box, sphere, cylinder, cone, pyramid, capsule,
  torus, wedge, lathe, extrude, panel, line, heightmap), CSG with smooth blends and carving,
  transforms, mirroring, voxel morphology (round, chamfer, hollow), paint ops (gradients,
  dithering, masonry, decals, crevice shading, edge highlights), and selections. Colors come
  from named palette ramps, never indices. See the [DSL reference](docs/dsl.md).
- **Eyes for the agent.** Every build renders front, side, back, top, and iso views with
  voxel-coordinate rulers and a palette legend, so feedback can be exact. ASCII slices and
  before/after diffs cover the details.

  ![A contact sheet](docs/images/contact-goblin.png)

- **A style bible.** One `style.toml` per game fixes the palette, scale, budgets, and art rules,
  and the linter holds every asset to it: off-palette colors, out-of-bounds voxels, floating
  pieces, broken symmetry, budgets, thin walls, hidden interiors.
- **Motion.** Hinges, slides and spins on parts, and clips written as code: the chest lid
  opens, the drawer slides, the propeller spins. Every build with clips renders a filmstrip for
  the agent and a GIF for you, lints the motion (limits, loops that hitch, parts passing through
  each other), and exports glTF animations plus a manifest for your engine.

  ![The chest opening](docs/images/chest-open.gif)

- **Game-ready export.** `.glb` with one node per part at its pivot (lids open, arms swing),
  sockets as empty nodes, a palette texture or vertex colors, and optional baked AO, tested by
  importing into Godot 4. `.vox` keeps the part hierarchy for touch-ups in MagicaVoxel or Goxel.
  Everything else goes through vengi.
- **Agent integration.** A `CLAUDE.md` workflow and Claude Code skills, an MCP server for other
  clients, a library of approved examples for few-shot references, and an evaluation harness.
- **Deterministic.** The same script and seed give the same bytes, every time, on any machine.

## Install

Python 3.11+ and [uv](https://docs.astral.sh/uv/):

```sh
git clone ssh://git@192.168.50.249/kalcode/voxel-auther-system.git
cd voxel-auther-system
uv sync
```

Optional: [MagicaVoxel](https://ephtracy.github.io/) or [Goxel](https://goxel.xyz/) to view and
edit `.vox` (`brew install --cask magicavoxel goxel`), [Godot 4](https://godotengine.org/) for the
`.glb`, and [vengi](https://vengi-voxel.github.io/vengi/) for other formats.

## Run it

```sh
cd projects/demo
uv run cubist build assets/goblin.py            # out/goblin_scout/: .vox, contact.png, lint
uv run cubist diff out/goblin_scout             # what changed since the last build
uv run cubist render out/goblin_scout --slices y=18..19
uv run cubist export out/goblin_scout --ao      # export/goblin_scout.glb and .vox
```

The full walkthrough is in [docs/getting-started.md](docs/getting-started.md).

## Let Claude make the assets

Open [Claude Code](https://claude.com/claude-code) in the repo and ask like an art director.
The `CLAUDE.md` workflow and the skills in `.claude/skills/` take it from there: read the brief
and similar examples, write the script, build, look, critique, revise (at most five times), and
show you the result.

| You want to | Say something like | Skill |
| --- | --- | --- |
| Create | "Make a wooden signpost with two arrows and moss at the base, about goblin height." | `voxel-create` |
| Edit or refine | "The goblin's ears look like nubs; make them longer and pointier." | `voxel-refine` |
| Animate | "Make the chest lid open with a little bounce." / "Give the goblin a walk cycle." | `voxel-animate` |
| Get ideas | "Suggest ten props for a blacksmith's yard and write briefs for the best five." | `voxel-ideas` |
| Start a game | "Set up a snowy mountain project from this Lospec palette." | `voxel-project` |

Every revision comes with a diff sheet: added voxels green, removed red, recolored orange.

![A diff sheet](docs/images/diff-goblin.png)

More in [docs/working-with-claude.md](docs/working-with-claude.md). Other MCP clients can use the
same tools: `claude mcp add cubist -- uv run --directory "$PWD" cubist mcp --project
projects/demo` ([docs/mcp.md](docs/mcp.md)).

## Does it work?

We gave four fresh Claude agents nothing but `CLAUDE.md`, the docs, and twenty briefs (props,
buildings, characters, creatures) in a style bible they had never seen. They built all twenty:

| | |
| --- | --- |
| Briefs built | 20 / 20 |
| Final builds lint-clean (no errors, no warnings) | 20 / 20 |
| Finished within the 5-iteration budget | 20 / 20 (mean 3.95) |
| Agent self-score | 3.7 / 5 on average |

![Twenty assets built by agents from briefs](docs/images/eval-gallery.png)

Their reports (bugs, confusing errors, doc gaps, wished-for features) drove the next round of
fixes and the `carve`, `decal`, `masonry`, `panel`, and `pyramid` additions. The full record is in
[projects/eval/eval/agent-notes.md](projects/eval/eval/agent-notes.md); human scores are still to
come.

## Docs

| | |
| --- | --- |
| [Getting started](docs/getting-started.md) | Install, first asset, commands, new projects |
| [Working with Claude](docs/working-with-claude.md) | Creating, editing, refining, and brainstorming with an agent |
| [DSL reference](docs/dsl.md) | Every primitive, op, selection, and paint call, with conventions |
| [How it works](docs/how-it-works.md) | Architecture, build pipeline, shape stages, lint, export (with diagrams) |
| [MCP server](docs/mcp.md) | Setup and tool reference |
| [Manifest](docs/manifest.md) | The engine contract: parts, joints, clips, units |
| [Design](docs/design.md) | The design of record: decisions and milestones |

## Project layout

```text
cubist/            the package: core, dsl, build, render, lint, io, export, mcp_server, cli
projects/demo/     a sample game: style.toml, briefs, asset scripts, examples/ (the library)
projects/eval/     20 briefs used to evaluate agents end to end
.claude/skills/    Claude Code skills: voxel-create, voxel-refine, voxel-animate, voxel-ideas, voxel-project
docs/              guides, the design doc, and images (scripts/make_images.py regenerates them)
tests/             the test suite; golden/ holds reviewed contact sheets
```

## Status

Milestones 1–8 of the [design](docs/design.md) are done (7: hinges, slides and spins; 8:
characters), with human scores for the first evaluation still to come. Next up: flipbooks and
glow, voxel effects, then a motion evaluation
([the plan](docs/superpowers/specs/2026-10-24-motion-and-effects-design.md)).

Development: `uv run pytest` (Godot on PATH enables the GLB import test) and
`uv run ruff check . && uv run ruff format .`.
