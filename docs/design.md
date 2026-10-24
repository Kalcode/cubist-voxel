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
- Skeletal animation (bones, skinning, blend shapes). Motion is rigid parts with joints and
  clips, plus frame-by-frame flipbooks; see Motion and effects below.
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
language agents write most reliably. **The complete reference, with every signature and
convention, is [dsl.md](dsl.md).** This section records the design decisions behind it.

- **Primitives start at their local origin** and are placed with `.at()`: boxes at their min
  corner, spheres centered, cylinders and cones at the center of their base.
- **Two stages.** Shapes are signed distance fields until a voxel-level op (paint, hollow,
  round, ...). CSG, transforms, and mirror work in both stages with the same color rules: union
  gives overlaps the later shape's color; subtract and intersect keep the left shape's colors,
  so cutters need no color (in either stage; cutters are sampled for occupancy only).
  `smooth_union`, `taper`, and `displace` need a distance field and say so.
- **`mirror` is exact**: colors come from a canonical point per mirrored pair, so both halves
  match. `reflect` is the plain reflection.
- **`round` is a morphological opening** (erode then dilate, never growing) so `round(1)` means
  the same thing on every shape; `hollow` measures depth in all 26 directions so curved shells
  stay watertight; erode and dilate use exact Euclidean distances.
- **Lines are voxel walks**, not distance fields: face-connected by default (a 2-wide staircase
  on diagonals, so they never count as floating), or one voxel per step with `connect=26`.
  `panel` plates default to the thinnest watertight thickness at their tilt.
- **Ramp ops step along list ramps only.** A crevice needs solid material one and two layers
  out (so bevel steps don't darken) and skips one-voxel walls (so nothing shows through); an
  edge is open on one side along two different axes (so thin plates don't light up entirely).
- **Left and right are the character's own**: characters face +Z, so their right is at low x.
- **Errors point at the script**: every shape records the script line that created it, so a
  lazily evaluated failure (a primitive with no color, a shape too large to sample) names the
  primitive, its line, and the part it belongs to.

### Seeds

Seeded ops (`displace`, `dither`, `masonry`) mix their own seed with the build seed (`cubist
build --seed N`, default 0) through a pinned integer hash, so one flag reshuffles every seeded op
and the same script and seed always give the same voxels on any platform.

### Palette

Scripts reference colors by name, never by index. A palette entry in `style.toml` is a list
(a ramp, dark to light), a table of named shades, or a single color. With `source =
"palette.png"` (also `.gif`, `.gpl`, `.hex`), entries may give source positions instead of hex
(`warm = [1, 2, 3]`) and every source color is `p.source[i]`. An unknown name is a build error
that suggests close matches.

### Example

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
            # the goblin's own right arm, at low x; mirrored to make the left
            Part("arm_r", arm, parent="torso", pivot=(4, 15, 6)).mirrored("arm_l", about_x=8),
        ],
        sockets={"hand_r": (4, 8, 6)},
    )
```

The full versions of every example are in `projects/demo/assets/`.

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
| Overlap | Voxels claimed by two parts (they double up in exports and poke through when a part moves) | Warning |
| Determinism | Two builds with the same seed hash differently (CI only) | Error |
| Floating voxels | Components not attached to the main body (6-connectivity); allow with a flag | Warning |
| Symmetry | A part declared symmetric differs from its mirror by more than N voxels | Warning |
| Grounding | Asset marked `grounded` has nothing at y = 0 | Warning |
| Budgets | Voxel, color, part, or post-meshing triangle count over the style bible's limits | Warning |
| Thin features | 1-voxel walls or spans with open air on both sides, reported as the largest clusters | Info |
| Hidden interior | Solid voxels nobody can see (auto-hollowed on export) | Info |
| Joint limits | Clip keys past a hinge's limits or a slide's distance | Error |
| Loop closure | A looping clip whose last pose isn't its first | Warning |
| Motion collision | A moving part passing through another part (sampled 24 times a second; voxels within 1.5 of a turning joint's axis are forgiven) | Warning |
| Motion bounds | How far each clip reaches beyond the rest box | Info |
| Hand edit | An animated asset's `.vox` no longer matches its build | Error |

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

[glow]                       # colors that give off light (a ramp, a shade, or a table entry)
fire = 1.5                   # strength: previews lift shading by 1 - e^-strength; the .glb
"eye.yellow" = 0.5           #   gets an emissive texture (KHR_materials_emissive_strength > 1)

[cycles]                     # list ramps whose voxels step one shade along each tick
water = { ramp = "water", fps = 6 }
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
- Search scores 2 points for a category match and 1 per shared tag. Promoting refuses assets
  with lint errors and writes a portable summary (no paths from the author's machine).
- The demo project ships ten seed examples (barrel, chest, crate, goblin, hut, mushroom,
  potion, sword, torch, tree), each authored with the build/critique loop and marked
  `approved_by = "claude (seed, pending review)"` until a person reviews them.

### Asset brief

Every asset starts from a short brief: name, category, size budget, description, must-haves,
tags, and optional reference images. `cubist new` scaffolds a brief and a script stub. Briefs
are `briefs/<name>.toml` (`name`, `category`, `size`, `tags`, `description`, `must_haves`,
`references`); default sizes come from the category (prop 16³, character 16×24×12, creature
24³, building 32³, vehicle 24×16×40, terrain 32×8×32).

## CLI and MCP tool surface

The CLI comes first because Claude Code can run it directly; the MCP server wraps the same
functions for other agents and clients. Tools that produce renders return them as inline MCP
image content so the agent sees them without opening files.

### CLI (`cubist`)

| Command | Does |
| --- | --- |
| `cubist new <name> --category prop [--size x,y,z] [--tags ...] [--must-have ...]` | Scaffold a brief and script stub |
| `cubist build <script> [--seed N] [--check-determinism]` | Build → `.vox`, contact sheet, `lint.json` in `out/<name>/` |
| `cubist render <asset> --views front,side,iso --slices y=10..14` | Re-render or print ASCII slices |
| `cubist lint <asset>` | Lint only (exit 2 on errors) |
| `cubist style` | Parsed `style.toml` with palette names |
| `cubist export <asset> [--glb] [--vox] [--via-vengi FMT] [--ao] [--vertex-colors]` | Export (blocked by lint errors unless `--force`; exit 2) |
| `cubist inspect <asset>` | Bounds, parts, pivots, voxel counts, color histogram as JSON |
| `cubist diff <asset> [<other>]` | Before/after/changes sheet; with one argument, against the previous build (`<name>.prev.vox`) |
| `cubist palette import <file>` | Import a palette into `style.toml` |
| `cubist examples list` / `search --tags prop,wood --category prop` / `promote <asset> --note ...` | Example library |
| `cubist mcp --project <dir>` | Run the MCP server on stdio |

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

| `diff(asset)` | Before/after/changes sheet against the previous build |
| `new_asset(name, category, ...)` | Scaffolds a brief and script stub |

There is no tool for placing or deleting individual voxels. If finer control is ever needed,
add it as a DSL feature.

Implementation notes (M5): built on the official MCP Python SDK 2.x (`MCPServer`). The server
sends the authoring workflow as its instructions; tool failures come back as tool errors with
the underlying message (an unknown palette name includes its suggestion). `render` accepts
`slices` for ASCII output. Assets are named (`crate` → `out/crate/`) or given as paths.

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

### Manifest

Every export writes `<name>.cubist.json` beside the `.glb` and `.vox`: parts with pivots and
joints, sockets with their owning part, clips with their authored keys, easing and events, and
`glb_origin` for mapping to the `.glb`. It is the contract with a game engine, versioned
(`"cubist_manifest": 1`) and documented in [manifest.md](manifest.md) with a JSON Schema in
`cubist/schema/`. Builds write the same content to `out/<name>/manifest.json`.

Clips also become glTF animations: sampled 30 times a second from the same pose evaluator the
previews use (more often where a part turns fast, so no step between keys turns 60° or more),
linear, with `step` tracks kept as their keys. Blender imports them and matches Cubist's poses
(`tests/test_blender.py`).

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
  render/    raster.py  contact_sheet.py  ascii.py  font.py  diff.py
  lint/      __init__.py (report)  checks.py
  io/        vox.py  mesh.py  gltf.py  vengi.py
  export.py  project.py  examples.py  palette_import.py
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
5. **Agent integration** — done. CLI polish, MCP server, `CLAUDE.md` workflow, example library
   with 10 seed examples (checked by Claude against the critique checklist; awaiting a person's
   review).
6. **Evaluation** — run; human scores pending. 20 briefs across props, buildings, and
   characters. Track lint pass rate, iterations to acceptance, and a 1–5 human quality score per
   asset. Four fresh agents, given only `CLAUDE.md` and the docs, built all 20: every final
   build lint-clean (0 errors, 0 warnings), every one within 5 iterations (mean 3.95), agent
   self-scores 3–4 (mean 3.7). Their reports drove a round of fixes (colorless cutters on painted
   shapes, errors naming the script line, voxel-walk lines, thin-geometry shading, failed builds
   logged, an overlap lint) and new DSL features (`carve`, `decal`, `masonry`, `stripes`,
   `checker`, `panel`, `pyramid`, `polyline`, tapered capsules). Details:
   `projects/eval/eval/agent-notes.md`; acceptance waits on the human scores in
   `projects/eval/eval/scores.toml`.

### Motion and effects (M7–M11)

The design is `docs/superpowers/specs/2026-10-24-motion-and-effects-design.md`: Cubist says what
moves and what it looks like; the engine decides when it happens and how it is simulated.

7. **Hinges, slides, spins** — done. Joints (`hinge`, `slide`, `spin`) and clips with smooth,
   linear or step keys; one pose evaluator shared by previews, lint and export; a posed renderer
   that draws moving parts as rotated blocks (rest frames match the static renderer pixel for
   pixel except at part seams); filmstrips (`motion.png`), GIFs, and posed contact sheets
   (`cubist render --clip NAME [--at T]`); motion lint; glTF animations checked in Blender; the
   manifest. Demo: chest, door, window, desk, biplane. Plan:
   `docs/superpowers/plans/2026-10-24-m7-hinges-slides-spins.md`.
8. **Characters** — done. Free `turn`/`move` tracks on any part (composed `R_joint · R_z · R_y ·
   R_x`, same-channel tracks add up); generators (`swing`, `bob`, `shake`, `spin_track`)
   resolved against the clip length into plain keys; hidden parts and `visible` tracks (left out
   of contact sheets, zero-scale GLB nodes with STEP scale channels); `travel`; oriented sockets;
   the `motion_floor` check; filmstrip diffs; examples keep their filmstrips; the
   `voxel-animate` skill. Demo: the goblin walks and idles, the blaster fires, the wand casts.
   An independent review of M7 found 11 edge-case bugs (step-only clips cut short in glTF,
   crashes on renamed parts, fast spins under-sampled, false collisions for rigidly attached
   parts, GIF timing, ...), all fixed with regression tests in `tests/test_motion_edges.py`.
   Plan: `docs/superpowers/plans/2026-10-24-m8-characters.md`.
9. **Flipbooks, color cycles, glow, lights** — done. `Part(frames=[...], fps=)` flipbooks
   (frame 0 is the rest look; later frames ride in the `.vox` as hidden `<part>@<k>` models and
   in the `.glb` as child nodes switched by STEP scale channels), looping on their own as
   `<part>_frames` clips or driven by `frames(...)` tracks; `[glow]` and `[cycles]` in the style
   bible (glow lifts preview shading by 1 - e^-strength and exports as an emissive palette
   texture; cycles are previewed and listed in the manifest); `light(...)` point lights as
   `KHR_lights_punctual` with flicker in the manifest. Every lint check that matters runs on
   every frame. Demo: the torch and a campfire flicker and glow, a pond's water flows, the
   blaster's flash plays two frames. Godot imports the moving demo assets with their animations,
   lights and emission. Plan: `docs/superpowers/plans/2026-10-24-m9-flipbooks-glow.md`.
10. **Effects** — fracture, baked bursts, particle models and emitters.
11. **Evaluation** — twelve motion briefs built by fresh agents, scored from their GIFs.

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
- [x] Animation: rigid parts with joints and clips, plus flipbooks for voxels that change
  shape (fire, water); no skinning. See Motion and effects.
- [x] Target engine: David's own engine, still to be designed around Cubist. Godot and Blender
  stay as `.glb` checkers only; the manifest is the engine contract.
- [x] Project name: Cubist (`cubist`).

### Prior art

| Project | Relevance |
| --- | --- |
| Goxel-MCP | Closest existing agent tool; its preview renders confirm the render-feedback approach. A good human touch-up path. |
| vengi | Open-source editor and `voxconvert` CLI covering many voxel formats, plus Lua scripting. Use as the format bridge. |
| magicavoxel-mcp | Per-voxel tool design; shows the limits of the low-level approach this design avoids. |
| APT (arXiv 2411.17255) | Research showing LLM spatial reasoning can plan voxel structures in Minecraft. |
