# Cubist — Agent-Native Voxel Asset Authoring System

Cubist is a headless voxel modeling toolkit that lets AI agents build game-ready voxel assets
by writing code, looking at renders of the result, and revising. It outputs MagicaVoxel `.vox`
files for hand touch-up and `.glb` files for game engines. It is not an editor: humans touch up
in existing editors (Goxel, MagicaVoxel, vengi).

This file is the design of record. It started as the "Voxel Forge" design doc (2026-10-23);
the repo is now the source of truth, and decisions made while building are folded in below.

## Goals

- An agent writes a model script and gets back a `.vox`, a rendered contact sheet, and a lint
  report in one command.
- Assets stay consistent across a whole game through a per-project style bible (palette, scale,
  proportions).
- Builds are deterministic: the same script and seed always produce the same voxels.
- Works from Claude Code through a CLI first, with a thin MCP server over the same functions.
- Exports to `.vox` (multi-part) and glTF/GLB with parts preserved as nodes.

## Non-goals for v1

- A GUI editor.
- ML text-to-3D or image-to-3D generation.
- Skeletal or frame-by-frame animation. v1 supports rigid parts with pivots only.
- An in-engine voxel runtime (destructible terrain, chunk streaming).

## Target asset sizes

| Category | Typical size (voxels) |
| --- | --- |
| Small props (potion, crate, sword) | 8³ to 32³ |
| Characters and creatures | 24 to 64 tall |
| Buildings and vehicles | up to 128³ |
| Hard cap per part | 256³ (the `.vox` per-model limit) |

## Design principles

The central bet: LLMs are strong at writing programs and weak at placing thousands of
coordinates by hand, so the agent writes code that generates voxels and never edits voxels one
at a time.

1. **Code is the model.** Each asset is a deterministic Python script. The script is the source
   of truth; `.vox` and `.glb` are build artifacts. There is deliberately no `place_voxel` tool.
2. **See what you build.** Every build returns renders. The agent critiques them against the
   brief before calling an asset done.
3. **Constraints over freedom.** Palette, grid scale, and budgets are fixed per project, and the
   linter enforces them.
4. **Parts, not blobs.** A model is a tree of named parts with pivots and sockets, so parts can
   be reused, animated, and revised independently.
5. **Boring formats.** `.vox`, glTF, PNG, JSON, TOML. Nothing proprietary.
6. **Humans can step in.** Output opens in standard editors. How hand edits flow back into
   scripts is an open question (see Risks).

## Architecture

One Python package does everything; the CLI and the MCP server are thin wrappers over the same
functions. A build runs the asset script in a subprocess, samples its shapes into a voxel grid,
then renders, lints, and exports the result:

```text
script.py ──> build (subprocess, timeout, seed, style bible)
                └─> sample SDF shapes into voxel grids, per part
                      ├─> render  -> contact sheet PNG, ASCII slices
                      ├─> lint    -> lint.json + summary
                      └─> export  -> .vox, .glb (other formats via vengi)
```

The agent only ever edits the script; renders and the lint report come back to it after every
build until the asset passes.

| Component | Responsibility |
| --- | --- |
| `core` | Voxel grids, parts, models, palettes, coordinate conventions |
| `dsl` | Primitives, CSG, transforms, painting, selections (what asset scripts import) |
| `build` | Runs a script in a subprocess with a timeout, a seed, and the project style bible |
| `render` | Headless CPU renderer: contact sheet of orthographic + isometric views, ASCII slices |
| `lint` | Palette, bounds, connectivity, symmetry, and budget checks; JSON report |
| `io` | `.vox` read/write, glTF/GLB export, optional vengi bridge for other formats |
| `cli` / `mcp_server` | Agent-facing entry points |

**Coordinates.** The DSL is Y-up, right-handed (+X right, +Y up, +Z toward the front), integer
voxel coordinates with the origin at the asset's bottom-left-back corner. The `.vox` writer
converts to MagicaVoxel's Z-up on write: DSL cell (x, y, z) maps to MagicaVoxel cell
(x, −1 − z, y), so the DSL front faces MagicaVoxel's −Y. One voxel's real-world size comes from
the style bible (for example 1/16 m) and is applied only at glTF export.

**Internal shape model.** Primitives are signed distance functions (SDFs) sampled into the grid
at build time. That makes smooth unions, rounding, and arbitrary transforms cheap, and keeps
scripts resolution-independent. Painting happens after sampling, on the grid.

**Sampling rule (decided in M1).** A voxel is filled when its center lies inside the shape or
exactly on its surface. Closed sampling keeps exact ties symmetric, so a shape mirrored about a
plane samples to a mirrored set of voxels. Consequence: a shape symmetric about an integer
plane (between two voxels) must have an even width; about a half-integer plane (through a voxel
center), an odd width.

## The DSL

Asset scripts are plain Python using a small, chainable API; Python is chosen because it is the
language agents write most reliably.

### Primitives

Each starts at its local origin and is placed with `.at(x, y, z)`.

| Primitive | Local origin and parameters |
| --- | --- |
| `box(size)` | min corner at the origin |
| `sphere(r)`, `ellipsoid(radii)` | centered |
| `cylinder(r, h, axis)`, `cone(r, h, axis, r_top=0)` | base centered on the origin, extending `h` along `axis` |
| `capsule(p0, p1, r)` | absolute endpoints, round ends |
| `line(p0, p1, r=0.5)` | absolute endpoints; max-norm distance, so the default samples to a connected one-voxel line (ropes, wires) |
| `torus(R, r, axis)` | centered, ring around `axis` |
| `wedge(size, rise="+x")` | like `box`, height 0 at the low end of `rise` (`±x`, `±z`) rising to full height |
| `lathe(profile, axis)` | `(height, radius)` points, heights increasing, revolved around `axis` (barrels, bottles, columns) |
| `extrude(polygon, depth, axis="z")` | 2D polygon in the other two axes (`z`: x,y; `y`: x,z; `x`: z,y) pushed `depth` from the origin |
| `heightmap(heights)` | columns from y = 0 up to `heights[x][z]` |

### Two stages: distance fields, then voxels

Shapes start as signed distance fields. Anything that works on voxels (painting, hollowing,
morphology, heightmaps) turns a shape into a grid-stage shape. CSG, transforms, and `mirror`
work in both stages with the same color rules; `smooth_union`, `taper`, and `displace` need a
distance field and refuse grid-stage input with a message to apply them before painting.

### Operations

- CSG: `union` (`|`), `subtract` (`-`), `intersect` (`&`), `smooth_union(k)`. Union gives
  overlaps the later shape's color; subtract and intersect keep the left shape's colors.
- Transforms: `at`/`translate`, `rotate` (any angle; default pivot is the bounds center),
  `scale`, `reflect(axis, about)` (reflection only), `mirror(axis, about)` (keeps the shape and
  adds its reflection, exactly symmetric in form and color), `array(n, step)`,
  `radial(n, axis, about)`. Grid-stage shapes are resampled exactly for whole-voxel moves and
  90° turns, by nearest voxel otherwise; mirroring them needs a plane on the voxel grid.
- Field modifiers: `taper(top, axis)`, `displace(amp, scale, seed, noise="perlin"|"value",
  octaves)`.
- Voxel morphology (exact Euclidean distances between voxel centers): `round(r)` (morphological
  opening: never grows the shape; `round(1)` takes one voxel off every edge), `chamfer(c)`
  (removes `c` layers of convex-edge voxels), `hollow(t)` (a shell `t` thick measured in all 26
  directions, so curved shells stay watertight), `erode(r)`, `dilate(r, color=None)` (new voxels
  take the nearest voxel's color).
- Paint: `paint(color, where)`, `paint_faces(direction, color, depth=1)`,
  `gradient(axis, ramp, reverse=False)`, `dither(ramp, seed, amount=0.5)`,
  `darken_crevices(steps=1)`, `highlight_edges(steps=1)`. Ramp-stepping ops move along list
  ramps; table and single colors have no order and stay unchanged. A crevice is a surface voxel
  with solid material beside its open face one and two layers out (a floor meeting a wall), so
  bevel steps don't darken; an edge is a voxel exposed on two different axes.
- Selections for `where=`: `band(axis, at, height)`, `region(lo, hi)`, `colored(*colors)`,
  `faces(direction)`, `surface()`, `inside(shape)`, or any shape directly; combine with `&`,
  `|`, `-`, `~`. Selecting by part is done by painting that part's shape.

### Seeds

Seeded ops (`displace`, `dither`) mix their own seed with the build seed (`cubist build
--seed N`, default 0) through a pinned integer hash, so one flag reshuffles every seeded op and
the same script and seed always give the same voxels on any platform.

### Parts, pivots, sockets

A `Model` holds named `Part`s. Each part has a shape, a pivot (default: the bottom center of its
voxels), an optional parent, an optional `symmetric` axis, and is exported as its own `.vox`
model and glTF node. Sockets are named attachment points (`hand_r`, `muzzle`, `door_hinge`).
`Part.mirrored("arm_r", about_x=…)` returns the part and a reflected twin; `Model` flattens such
pairs. An `@asset` function may return a bare `Shape`, which becomes a single part named after
the asset. `@asset` also takes `grounded`, `symmetric`, and `allow_floating` for lint.

### Palette

Scripts reference colors by name, never by index. A palette entry in `style.toml` is a list
(a ramp, dark to light: shades `dark` = first, `mid` = middle, `light` = last, any position as
`p.wood[i]`), a table of named shades (`cloth = { brown = "#6b4a2a" }` → `p.cloth.brown`), or a
single color (`p.cork`). With `source = "palette.png"` (also `.gif`, `.gpl`, `.hex`), entries
may give source positions instead of hex (`warm = [1, 2, 3]`) and every source color is
`p.source[i]`. An unknown name is a build error that suggests close matches.

### Examples

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

```python
@asset(name="goblin_scout", size=(16, 24, 12), tags=["character"], grounded=True)
def goblin(p):
    torso = box((6, 7, 4), color=p.tunic.mid).at(5, 9, 4).taper(top=0.7).highlight_edges()
    head = box((8, 6, 6), color=p.skin.mid).at(4, 16, 3).round(1)
    eye = box((2, 1, 1), color=p.eye.yellow).at(5, 18, 8)
    head = head.union(eye).mirror("x", about=8).darken_crevices()
    arm = capsule((0, 0, 0), (0, -6, 0), r=1, color=p.skin.mid).at(4, 15, 6)
    return Model(
        parts=[
            Part("torso", torso, pivot=(8, 9, 6), symmetric="x"),
            Part("head", head, parent="torso", pivot=(8, 16, 6), symmetric="x"),
            Part("arm_l", arm, parent="torso", pivot=(4, 15, 6)).mirrored("arm_r", about_x=8),
        ],
        sockets={"hand_r": (12, 8, 6)},
    )
```

Both run as written; the full versions are in `projects/demo/assets/`.

**Rules for scripts:** no file or network access, no randomness except through the provided
`seed`, and a build timeout (default 30 s). Status: the timeout is enforced and
`--check-determinism` catches unseeded randomness; the file/network restriction is not yet.

## Renderer and feedback loop

Every build returns one contact-sheet PNG, because the render-critique loop is what turns
plausible code into a good asset. One image per build also keeps token cost down.

### Contact sheet

- Five views on one sheet: front, side, back, top (all orthographic) and a 3/4 isometric.
- Same pixel scale across views, with a voxel-coordinate ruler on two edges so the agent can name
  exact positions.
- Flat per-face shading (top light, sides mid, bottom dark) plus simple ambient occlusion, so
  form reads clearly.
- Optional extras: a silhouette view (solid black) for shape critique, and a before/after diff
  sheet for revisions. Status: silhouette is available (`--views ...,silhouette`); the diff
  sheet is deferred to `cubist diff` (M5).
- Target: under 1 s for a 64³ model on a laptop CPU, no GPU required, so it runs in CI and
  sandboxes. Measured (M2): about 0.1 s for a 64³ sphere on an M-series laptop.

Renderer decisions (M2):

- Pixel scale: even pixels per voxel so the largest orthographic view is about 256 px, clamped
  to 2..12. Rulers label world coordinates at voxel boundaries; a faint grid sits behind the
  model; the declared size is outlined in blue only when the model breaks out of it.
- Orthographic views use the first voxel hit per pixel, per-corner ambient occlusion
  interpolated across each face, a depth cue (farthest faces 30% darker), and dark outlines just
  outside any surface that stands at least 2 voxels in front of what is behind it.
- The iso view is a 2:1 pixel-art projection (30° camera) from the front-right, showing top,
  front (+Z), and right (+X) faces. Each face is a pre-rasterized sprite; a z-buffer resolves
  overlaps. With integer sprite dimensions no pixel center lies on a face edge, so faces tile
  with no gaps or overlaps (tested exactly).
- Text uses a built-in 5×7 bitmap font, so sheets are pixel-identical across machines.
- A color legend under the views maps each color to its palette name and voxel count.
- Golden images for the demo assets live in `tests/golden/`; regenerate deliberately with
  `CUBIST_UPDATE_GOLDEN=1 uv run pytest tests/test_golden.py` and review them before committing.

### ASCII slices

`cubist render --slices y=10..14` prints layers as character grids, one character per palette
name. Images are good for overall form; ASCII is better when the agent needs to reason
precisely about a small region such as a face or a hinge.

Orientation matches the contact sheet: y slices read like the TOP view (rows back to front),
z slices like the FRONT view, x slices like the LEFT view. Letters are assigned from every color
in the asset, so the same color keeps its letter across calls; the legend lists only the colors
in the printed layers.

### Agent workflow

This goes in the repo's `CLAUDE.md` (M5) so Claude Code follows it:

1. Read the brief, the style bible, and 2–3 similar approved examples.
2. Write the script and build.
3. Inspect the contact sheet and lint report.
4. Critique against a fixed checklist: Is the silhouette readable at game distance? Do
   proportions match the style bible? Is the palette used as ramps? Is any asymmetry
   intentional? Does it meet every must-have in the brief?
5. Revise and rebuild. Stop after 5 iterations by default and report what still falls short.

Optional: a second reviewer agent that sees only the brief and the renders, not the code, gives
a less biased verdict.

## Linting and validation

The linter catches the mistakes renders hide, and errors block export. It writes `lint.json`
plus a short human-readable summary that the CLI and MCP tools return with every build.

Implementation notes (M3): connectivity is 6-neighbor; symmetry compares form and color across
the asset's center plane (`size / 2`, or the bounds center without a size), for the whole asset
(`@asset(symmetric=...)`) or single parts (`Part(symmetric=...)`), against
`[rules] symmetry_tolerance`; triangles are counted from the greedy mesh; hidden voxels are those
with no face open to air connected to the outside, so enclosed cavities count as hidden; the
palette check also enforces `[rules] no_pure_black` and `dither = "forbidden"`. Determinism runs
only with `cubist build --check-determinism`. `cubist lint` exits 2 when there are errors.

| Check | What it catches | Severity |
| --- | --- | --- |
| Palette | Colors not in the project palette | Error |
| Bounds | Voxels outside the declared asset size, or a part over 256³ | Error |
| Pivots and sockets | Pivots or sockets outside the part's bounds | Error |
| Determinism | Two builds with the same seed hash differently (CI only) | Error |
| Floating voxels | Components not attached to the main body (6-connectivity); allow with a flag | Warning |
| Symmetry | A part declared symmetric differs from its mirror by more than N voxels | Warning |
| Grounding | Asset marked `grounded` has nothing at y = 0 | Warning |
| Budgets | Voxel, color, part, or post-meshing triangle count over the style bible's limits | Warning |
| Thin features | 1-voxel walls or spans that disappear at game distance | Info |
| Hidden interior | Solid voxels nobody can see (auto-hollowed on export) | Info |

## Style bible and example library

Consistency across a hundred assets matters more than any single asset, so every project
carries a style bible that the agent loads before each build.

### `style.toml`

```toml
[project]
name = "goblin-dungeon"
voxel_size_m = 0.0625        # 1/16 m per voxel

[palette]
source = "palette.png"       # optional; also .gif, .gpl, .hex
# Named ramps, dark to light. Scripts use p.<ramp>.<shade>.
wood  = ["#3b2414", "#5e3a1f", "#8a5a32", "#b07d4a"]
metal = ["#2a2d34", "#4b505a", "#7d8491", "#c2c8d0"]

[scale]
humanoid_height = 24
door_height = 32

[budgets]
max_colors = 32
max_voxels = 40000
max_triangles = 6000

[rules]
no_pure_black = true
dither = "allowed"           # or "forbidden"
symmetry_tolerance = 0       # voxels allowed to differ from the mirror
notes = "Chunky silhouettes, oversized heads and hands, light comes from top-left."
```

Every section is optional; unknown sections and keys are errors with a suggestion, so a typo
never silently disables a budget. `[budgets]` also takes `max_parts`. `cubist style` prints the
parsed bible with every palette name.

### Example library

- `examples/` holds approved assets: the script, its contact sheet, tags, and a one-paragraph
  note on why it is good.
- Before writing a script, the agent retrieves the 2–3 closest examples by tag and category and
  uses them as few-shot references.
- `cubist examples promote <asset>` adds a finished asset once the user approves it, so quality
  compounds as the game grows.

### Asset brief

Every asset starts from a short brief: name, category, size budget, description, must-haves,
tags, and optional reference images. `cubist new` scaffolds a brief and a script stub.

## CLI and MCP tool surface

The CLI comes first because Claude Code can run it directly; the MCP server wraps the same
functions for other agents and clients. Tools that produce renders return them as inline MCP
image content so the agent sees them without opening files.

### CLI (`cubist`)

| Command | Does |
| --- | --- |
| `cubist new <name> --category prop` | Scaffold a brief and script stub |
| `cubist build <script> [--seed N] [--check-determinism]` | Build → `.vox`, contact sheet, `lint.json` in `out/<name>/` |
| `cubist render <asset> --views front,side,iso --slices y=10..14` | Re-render or print ASCII slices |
| `cubist lint <asset>` | Lint only (exit 2 on errors) |
| `cubist style` | Parsed `style.toml` with palette names |
| `cubist export <asset> [--glb] [--vox] [--via-vengi FMT] [--ao] [--vertex-colors]` | Export (blocked by lint errors unless `--force`; exit 2) |
| `cubist inspect <asset>` | Bounds, parts, pivots, voxel counts, color histogram as JSON |
| `cubist diff <a> <b>` | Before/after contact sheet |
| `cubist palette import <file>` | Import a palette into `style.toml` |
| `cubist examples search --tags prop,wood` / `promote <asset>` | Example library |

### MCP tools

| Tool | Returns |
| --- | --- |
| `build_asset(script_path or code)` | Output paths, contact sheet image, lint summary |
| `render(asset, views, slices)` | Image and/or ASCII text |
| `lint(asset)` | Lint report |
| `inspect(asset)` | Structure and stats JSON |
| `export(asset, formats)` | Output paths |
| `get_style_bible()` | Parsed `style.toml` with palette names |
| `search_examples(tags, category)` | Example scripts and their contact sheets |

There is no tool for placing or deleting individual voxels. If finer control is ever needed,
add it as a DSL feature.

## Export pipeline

Cubist writes `.vox` and `.glb` natively and hands every other format to vengi's `voxconvert`
CLI when it is installed. `cubist export` re-runs lint first (so a `.vox` edited by hand and
given off-palette colors is caught too) and refuses to export when there are errors unless
`--force`; it writes to `<project>/export/` by default.

### `.vox`

- Native reader and writer following the public MagicaVoxel format spec (`ephtracy/voxel-model`
  on GitHub).
- Multi-part models use the scene graph chunks (`nTRN`, `nGRP`, `nSHP`) so parts keep names and
  transforms; the palette goes in the `RGBA` chunk. A model is placed by its center,
  `floor(size / 2)` from its min corner.
- The part hierarchy is kept: a part with children becomes a named group holding its own shape
  and its child parts; the reader rebuilds parents from that nesting (third-party groups that
  aren't parts are ignored). Pivots and sockets live in `build.json` and the GLB, not the `.vox`.
- Parts over 256 voxels on any axis are rejected by the writer (they are a lint error anyway).
  Splitting them into several models is deferred until an asset needs it.
- The reader rejects rotated scene-graph nodes for now.
- Validation: round-trip test (write → read → identical grid). If `vengi-voxconvert` is on PATH,
  CI also converts each test file with it to confirm third-party tools accept it.

### glTF / GLB

- Written by hand (JSON and BIN chunks, sorted keys), so the same asset always gives the same
  bytes.
- Greedy meshing with interior faces removed and coplanar faces merged; sealed cavities are
  treated as solid, so only faces open to outside air are meshed (the export side of "hidden
  interior").
- Colors via a 256×1 palette texture with UVs pointing at palette cells and a nearest-filtered,
  clamped sampler (one material per asset). `--vertex-colors` puts colors in `COLOR_0` instead,
  converted to linear floats as glTF requires (the palette is sRGB).
- `--ao` bakes per-corner ambient occlusion into `COLOR_0` (multiplied with the texture). Only
  fully open faces merge when AO is on; occluded faces keep their own corners, and each quad
  splits along its brighter diagonal so AO interpolates without creases.
- Each part becomes a glTF node at its pivot, parented as in the model, with mesh vertices in
  the node's space, so engines can rotate parts without extra setup. Sockets become empty nodes
  under the part named in `(x, y, z, "part")`, or else the part with the nearest voxel.
- Scale from `voxel_size_m`. Y-up and +Z front, matching glTF. By default the root node shifts
  the asset so its footprint is centered on the origin with its base at y = 0 (`--no-center`
  keeps the DSL origin).
- Acceptance (met in M4): `tests/test_godot.py` loads GLBs in headless Godot 4 and checks the
  part hierarchy, node positions and mesh sizes in metres, triangle counts, and the palette
  texture with nearest filtering. It skips when `godot` is not on PATH.

### Everything else

`cubist export --via-vengi <format>` shells out to `vengi-voxconvert --input <vox> --output
<file>` for OBJ, Qubicle, Minecraft schematics, and the other formats vengi supports. Cubist
does not reimplement them.

## Build plan

Build in six milestones, each usable on its own; v1 is done when an agent can take a prop brief
to an approved, exported asset without human help.

### Stack

Python 3.11+, `numpy`, `scipy` (`ndimage`: exact distance transforms and connected components
for morphology and lint), `Pillow`, `pygltflib` (or `trimesh`) for glTF, the official MCP Python
SDK, `pytest`, `ruff`, `uv`. Dev only: `pygltflib` to read GLBs back in tests; Godot 4 on PATH
for the import acceptance test. Optional: `vengi-voxconvert` on PATH. Before adding any
third-party `.vox` library, check its license; a native writer is small enough to own.

### Repo layout

```text
cubist/
  core/      grid.py  palette.py  part.py  model.py  style.py  topology.py
  dsl/       sdf.py  shape.py  ops.py  primitives.py  gridops.py  paint.py  select.py
             noise.py  context.py
  build/     runner.py
  render/    raster.py  contact_sheet.py  ascii.py  font.py
  lint/      __init__.py (report)  checks.py
  io/        vox.py  mesh.py  gltf.py  vengi.py
  export.py
  cli.py
  mcp_server.py
projects/
  <game>/
    style.toml  palette.png
    briefs/  assets/  examples/  out/
tests/
docs/design.md
CLAUDE.md
```

### Milestones

1. **Core and `.vox` writer** — done. Grid, palette, five primitives, CSG, mirror, `.vox`
   read/write. Done when 5 test scripts produce files that open in MagicaVoxel and Goxel, and
   round-trips are exact. (Round-trips are tested; opening in the editors is a manual check.)
2. **Renderer** — done. Contact sheet and ASCII slices. Done when a 64³ model renders in under
   1 s and golden-image tests pass. `cubist build` writes `contact.png`; `cubist render`
   re-renders or prints slices.
3. **Full DSL, lint, style bible** — done. Remaining primitives and ops, paint ops, all lint
   checks, `style.toml` loading, named palette ramps. The design doc's barrel example builds as
   written; every demo asset lints with no errors or warnings.
4. **Parts and glTF** — done. Parts, pivots, sockets, multi-part `.vox`, GLB export. Done when a
   multi-part GLB imports into Godot 4 with the right hierarchy and scale (verified headless
   against Godot 4.6).
5. **Agent integration.** CLI polish, MCP server, `CLAUDE.md` workflow, example library with 10
   seed examples (hand-checked).
6. **Evaluation.** 20 briefs across props, buildings, and characters. Track lint pass rate,
   iterations to acceptance, and a 1–5 human quality score per asset.

### v1 acceptance targets

- At least 80% of prop briefs reach a lint-clean asset the user accepts within 5 iterations.
- Every build is deterministic and finishes (build + render + lint) in under 10 s for assets up
  to 64³.
- All tests run headless in CI with no GPU.

## Risks and open questions

The biggest risk is organic shapes: hard-surface props should work well early, while faces and
creatures above about 32 voxels tall will need better primitives or human touch-up.

### Risks

| Risk | Mitigation |
| --- | --- |
| Organic forms and faces look crude at higher resolutions | SDF smooth unions and rounding; head and hand templates in the example library; hand touch-up in Goxel; a later image-to-3D-then-voxelize path |
| Agent declares victory too early | Fixed critique checklist; iteration cap with an honest shortfall report; optional render-only reviewer agent |
| Image tokens get expensive | One contact sheet per build, modest resolution, ASCII slices for detail work |
| Style drifts across assets | Style bible, named palette ramps, palette lint, few-shot examples |
| Untrusted script execution | Subprocess sandbox, no file or network access, timeout |

### Open questions

- [ ] Hand edits: when someone touches up a `.vox` in an editor, store the difference as a patch
  layer applied on top of the script output, or freeze the asset and stop treating the script
  as the source?
- [ ] Animation: is rigid-part animation via glTF nodes enough, or will the game need
  frame-by-frame voxel animation?
- [ ] Target engine: confirm Godot 4 as the primary import test, or swap for Unity or Unreal.
- [x] Project name: Cubist (`cubist`).

### Prior art

| Project | Relevance |
| --- | --- |
| Goxel-MCP | Closest existing agent tool; its preview renders confirm the render-feedback approach. A good human touch-up path. |
| vengi | Open-source editor and `voxconvert` CLI covering many voxel formats, plus Lua scripting. Use as the format bridge. |
| magicavoxel-mcp | Per-voxel tool design; shows the limits of the low-level approach this design avoids. |
| APT (arXiv 2411.17255) | Research showing LLM spatial reasoning can plan voxel structures in Minecraft. |
