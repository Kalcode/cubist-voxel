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
| `build` | Runs a script in a subprocess with a timeout, a seed, and the project style bible; scripts import the project's own modules (13d) |
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
- **Occlusion tags are per part, and inherited**: `roof`, `upper` and `canopy` say what an engine
  may hide to show the player (a roof, an upper storey, a crown); `clutter` says what its follow
  camera passes through instead of pulling in (small set-dressing: a stall's counter and posts,
  crates). Kiln added `clutter` (Oct 28) after a play report: small objects mustn't squeeze the
  camera, since its see-through cutout and x-ray ghost already show the character. A part
  without a tag takes its nearest tagged ancestor's; the manifest writes the resolved tag.
  Kiln hides a building by storeys: each `upper` and `roof` part's lowest voxel is a hide line,
  and the storey lint keeps those on storey boundaries. A `canopy` part isn't a storey, so in a
  building (an awning on a shopfront) it makes no line and hides with the storey it falls in;
  only a prop with no `upper` or `roof` part (a tree) takes its lines from its `canopy` parts. So
  the storey lint needn't check `canopy` (Kiln's `docs/engine/formats.md`, "Hide lines").
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
| Scale | An asset tagged `<tag>` more than 20% under or 25% over `[scale]` `<tag>_height` (`door`, `table`, `humanoid`; not the walker's keys), or a `building` shorter than `door_height` | Warning |
| Occlusion | A `building` with no part tagged `roof` or `upper` (`canopy` and `clutter` don't count) | Info |
| Walk | A `building` or `walkable` asset (when `[scale]` has the walker keys) with a `door`, `floor_N`, `cellar` or `tour_NN` socket the walker can't reach from the street at its door (error), a tour leg (`door` to `tour_01`, then each stop to the next) it can't walk in a straight line (error), or with no `door` or `floor_0` socket to walk (warning: older buildings no one enters) | Error, Warning |
| Storeys | A part tagged `upper` or `roof` (its own tag or a parent's) reaching up past the next such part's lowest voxel; the topmost may reach down over the one under it | Error |
| Light reach | A grounded asset's light whose range can't reach the floor as far out as the light is high (range under sqrt(2) times its height above the ground line; in a cellar, above the asset's floor; an indoor light, above the floor straight under it, which it warns of standing over none) | Warning |
| Joint limits | Clip keys past a hinge's limits or a slide's distance | Error |
| Loop closure | A looping clip whose last pose isn't its first | Warning |
| Motion collision | A moving part passing through another part (sampled 24 times a second; voxels within 1.5 of a turning joint's axis are forgiven) | Warning |
| Motion bounds | How far each clip reaches beyond the rest box | Info |
| Hand edit | An animated asset's `.vox` no longer matches its build | Error |

The walk lint (13b, `cubist/lint/walk.py`) is the gate that keeps a building enterable by the
game's character, Kiln's goblin (`Goblin.cs`: `MoveAxis`, `HalfWidth` 4, `Height` 22, `StepUp` 6):

- **The world:** the asset's voxels at rest (hidden parts left out), plus solid ground below
  `ground_y` out to 16 voxels past the asset's x–z bounds: on every side, or on the sides
  `@asset(street_sides=...)` names (13h). Kiln carves the x–z bounds of the voxels below
  `ground_y` (a cellar's footprint), so there is no ground there; the street under eaves and
  jetties stays.
- **The walker** is the goblin on a grid. Between whole voxels its box covers `w + 1` columns on
  each axis, and that footprint is what it stands on, as `GroundBelow` finds it. For clearance it
  carries a box `(w + 2) × (h + 2) × (w + 2)` (from `[scale]` `walker_width`, `walker_height` and
  `walker_climb`) that holds the footprint with a column to spare: along a move it holds both
  footprints, across it the spare column may be on either side. A move goes a voxel in x or z as
  `MoveAxis` does: where the box fits it moves and drops onto the highest voxel under the new
  footprint; where it doesn't, it steps up if the box raised by the climb fits, onto the highest
  voxel under the new footprint. Where it lands a box must fit.
- **The walk** starts on the street: the places it stands with its footprint wholly outside the
  asset's x–z bounds, within 16 voxels of `door` and its feet within 2 of the door's y. (`door`
  itself, like every socket, stays inside the model's bounds: on the front face, or under the
  overhang.) A socket is reached when a reached footprint holds its x–z (edges included), the feet
  within 2 voxels of its y; `door` is checked like the rest, so a wall between the street and the
  doorway shows.
- **The way back:** going down is a drop, so a flight walked down is never climbed on the way
  in. Each move is kept, and a flood from the street over the moves turned round marks the places
  with a way back out; a socket reached only where there's none (a cellar down a stair of 7-voxel
  risers, or through a hatch) errors.
- **Findings:** an error for each `door`, `floor_N`, `cellar` or `tour_NN` socket out of reach or
  with no way back, naming the socket and the walker, the nearest place reached (or walked back
  out from) and its distance, and saying to check the stair or doorway between them; an error
  when there's no street near `door` to start from. With no `door` or `floor_0` there's nothing to
  walk, and lint warns: older buildings that no one enters still export.
- **The tour's legs:** Kiln's walk arrives within 2 voxels (x-z) of a stop and steers straight
  for the next (`ARRIVE`; Kiln's `WalkSteering.SocketReachRadius`, since the town's first street,
  Kiln's `docs/city.md` §8 step 4). So from `door` to `tour_01` and from each `tour_NN` to the
  next (name order), from the stop and from 8 points 2 voxels round it, turned to
  the leg so the two square to it are among them (the walker may have arrived anywhere within 2:
  these are the furthest off any line it may take), the walker must get within 2 of the next stop
  over the moves the walk kept, keeping its footprint's centre within a voxel of its straight line
  (`LEG_SLACK`, the grid's rounding of a line) and its feet between the stops' heights (2 to
  spare). Short legs are walked too: an arrival 2 from one stop may be 4 from the next. A leg that
  can't is an error on the stop it ends at, naming the one it starts from and how far along its
  line the walker got; a leg whose ends are already errors isn't walked again. A slack of 2 instead
  would not be the same: it lets the lint weave round an obstacle within 2 of the line that a
  straight walk from an arrival off the line meets, so it passes legs Kiln's walk fails.
- **Room to arrive:** a start the walker can't stand at is an error on the stop ("too close to an
  obstacle for a 2-voxel arrival"), saying how many voxels to move it and which way. The goblin is
  two voxels narrower than the walker, so it might arrive where the walker can't and walk a line
  the lint never tried; erring, rather than starting from the nearest place the walker can stand,
  keeps the lint the stricter of the two.
- **Against the goblin:** `tests/goblin.py` ports `Goblin.Tick` and `VoxelCollision`, and
  `test_walk_lint.py` runs the same strips through it and the walker. It climbs a riser of 6 and
  not 7 and stays up over a gap of 8 but falls into 9, as the walker does. It steps up with 28
  voxels of air above the tread it leaves (the walker needs 30), and passes a doorway 9 wide (8
  when centred on a whole voxel) and 22 tall (the walker needs 10 × 24). The walker allows less
  than the goblin everywhere but two places it can't see: the instant the goblin's footprint is
  `w` wide as it crosses a whole voxel's line (it passes within a frame), and falls, which it
  takes straight down where the goblin keeps some of its speed (a player can let go of the stick:
  it stops within a voxel). Support handed across the path from one edge column to the other
  needs one footprint (`w + 1`, the goblin's) holding both; the goblin can also cross with a
  diagonal step that crosses both whole-voxel lines in one frame (`MoveAxis` x, then z, then the
  ground check), so there the walker, a move at a time, is the stricter: the safe way round.

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

[scale]                      # lint: assets tagged <tag> against <tag>_height; buildings vs door
humanoid_height = 24
door_height = 32
walker_width = 8             # the walk lint's walker, in voxels: all three or none
walker_height = 22
walker_climb = 6

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
never silently disables a budget. `[budgets]` also takes `max_parts`. The walker keys come
together, in whole voxels (a climb below the height). `cubist style` prints the parsed bible
with every palette name.

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
| `cubist build <script> [--only NAME] [--seed N] [--check-determinism]` | Build → `.vox`, contact sheet, `lint.json` in `out/<name>/`; `--only` builds just the named assets of a script defining several |
| `cubist render <asset> --views front,side,iso --slices y=10..14` | Re-render or print ASCII slices |
| `cubist lint <asset>` | Lint only (exit 2 on errors) |
| `cubist style` | Parsed `style.toml` with palette names |
| `cubist export <asset> [--glb] [--vox] [--via-vengi FMT] [--ao] [--vertex-colors]` | Export (blocked by lint errors unless `--force`; exit 2) |
| `cubist export --all [--project DIR]` | Export every built asset in `out/`, same options; a blocked asset is reported and the rest still export (exit 2 if any was blocked) |
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
- Glow is in the `.vox` as `MATL` emissive materials (`_type = _emit`), one per glowing palette
  index, used or not, so a color painted on in MagicaVoxel glows too. The material id is the
  color index. Strength follows MagicaVoxel's HDR emission, `_emit * 10 ^ _flux`: the writer
  picks the smallest whole `_flux` that keeps `_emit` at most 1 (1.5 is `_emit 0.15`, `_flux 1`).
  The reader takes glow from `_emit` materials (or `_weight` in older files) and ignores the
  rest; a `.vox` with no `build.json` gets its glow this way. Engines can read glow from the
  `.vox` alone, for Cubist's files and hand-painted ones alike.
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

It also names what the asset is and how its colors group: `tags` (the asset's tags, in declared order) and
`ramps` (each list ramp with a shade in use, as its whole dark-to-light run of `.vox` indices). Kiln
collapses a building's ramps to one shade in its coarse copies, from level 1 out; it needs `ramps` to know
which colors share a ramp and `tags` to know which assets are buildings. Both are additive, so the version
stays 1.

Clips also become glTF animations: sampled 30 times a second from the same pose evaluator the
previews use (more often where a part turns fast, so no step between keys turns 60° or more),
linear, with `step` tracks kept as their keys. Blender imports them and matches Cubist's poses
(`tests/test_blender.py`).

### Everything else

`cubist export --via-vengi <format>` shells out to `vengi-voxconvert --input <vox> --output
<file>` for OBJ, Qubicle, Minecraft schematics, and the other formats vengi supports. Cubist
does not reimplement them.

## Worlds

Kiln's MMO world is premade, not generated at runtime (Kiln's `docs/engine/world-streaming.md`, "Premade
maps"), so terrain joins the asset pipeline. Cubist authors a world the way it authors an asset: a script
is the model, every build comes back as pictures and a lint report, and the output is a boring format an
engine reads. Status: designed (Oct 25); milestone 12 below.

**Scale.** An asset is one dense grid of at most 256³. A 4 × 4 km world at 1/16 m is 64,000 × 64,000
columns, so a world is never one grid: it is a function evaluated wherever it's needed, a block of
columns at a time, with numpy.

**Fields are sampled on a 0.5 m lattice for the bake** (decided in 12a). Noise and painted maps have no
detail at 1/16 m, and evaluating them per voxel column would take billions of samples for 4 × 4 km. The
bake samples heights every 8 voxels and interpolates each column bilinearly, 64× fewer samples; slopes
use the same spacing. The map samples at its own pixel size.

### World scripts

`worlds/<name>.py` in a project, next to `assets/`. A `@world` function returns layers:

```python
@world(name="vale", size_m=(4000, 4000), seed=1)
def vale(p):
    ground = fbm(scale_m=900, octaves=5, amplitude_m=60) + ridge(...)  # heights in metres
    ground = ground.blend(height_map("maps/vale.png", low_m=0, high_m=700), mask=...)
    ground = ground.flatten(area(...), at_m=42).river(path([...]), width_m=8, depth_m=2)
    return World(
        ground=ground,
        surface=[slope_over(38, p.stone.mid), height_over(520, p.white), p.leaf.mid],
        subsoil=[(1, p.wood.dark), p.stone.dark],         # metres under the surface
        caves=[tunnel(path([...]), radius_m=2.5), cavern((x, y, z), radius_m=8), worms(region, ...)],
        props=[place("cottage", at=(x, z), rotate_y=90), scatter("tree", region, per_ha=40, ...)],
        landmarks={"spawn": (x, z), "town": (x, z)},
    )
```

- **Ground** is a height field in metres: noise (`fbm`, `ridge`, `terrace`), painted maps (`height_map`,
  greyscale PNG), arithmetic and `blend` by masks, and edits: `flatten` an area, cut a `river` or a `road`
  along a path.
- **Materials** are rules on the surface (slope, height, painted masks), first match wins, plus subsoil
  layers by depth. Colors are palette names, as in assets.
- **Caves** are 3D carvers: tunnels and caverns along paths and points, and noise worms in a region.
- **Props** place built assets by name (`out/<asset>/`) at a position and rotation, snapped onto the ground,
  or scatter them by density and rules. They stay placements: Kiln stamps them, so moving one doesn't
  re-bake terrain.
- **Landmarks** are named points for the engine (spawn, towns, cave mouths).
- **Determinism:** the same script, seed and painted maps give the same world (noise from `dsl/noise.py`).

### Feedback

- `cubist world build worlds/<name>.py` writes `out/worlds/<name>/`: `map.png` (the whole world top-down,
  hill-shaded, with contour lines, a ruler in metres, landmarks, props and caves drawn on), a lint report,
  and stats (height range, slope histogram, cave volume, prop counts).
- `cubist world render <name> --at x,z --size 64` renders a region in voxels with the asset renderer
  (iso and top views), for close critique.
- **Lint:** slopes too steep to walk (`[world] max_walk_slope_deg`), props sunk or floating (their footprint
  over uneven ground), props overlapping, caves with no way in from the surface, floating rock left by
  carving, and anything outside the world's size.

### Bake

`cubist world bake <name>` writes `export/worlds/<name>/` in Kiln's map format (Kiln `formats.md`, "Baked
maps"):

- **Tiles** of 64 × 64 columns per level of detail (levels 0–6), each level sampled directly at its own cell
  size, and the far field's height grids.
- **`map.json`** carries the material registry (id → palette name, color, glow), so an engine needs no style
  bible.
- **`world.json`**: props as Kiln scene placements (`asset`, `at` with the snapped y, `rotate_y`), landmarks,
  and the voxel size.
- **Incremental:** each tile records a hash of what it was built from; a bake rewrites only tiles whose inputs
  changed. Tiles bake in parallel (processes).
- Bakes are large (Kiln's valley: ~250 MiB per km²), so `export/` stays out of git.

**Edits made in the game** (Kiln's save file) sit on top of the map in Kiln, so they survive a re-bake: an
edited chunk keeps its edited voxels. Later, Kiln can log edits as operations and replay them on a new base.

**The risk is speed.** Kiln's C# baker took ~35 s per km² on 10 cores. Python with numpy is expected to be
several times slower *(estimate)*. Milestone 12b measures 1 km². If a 4 × 4 km bake would take more than
about 30 minutes, the work splits: Cubist writes a compact description (height and material maps at about
1 m, paths, carvers, placements) and Kiln's baker fills in the voxels.

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
  lint/      __init__.py (report)  checks.py  motion.py  walk.py
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

Cubist says what moves and what it looks like; the engine decides when it happens and how it is
simulated.

7. **Hinges, slides, spins** — done. Joints (`hinge`, `slide`, `spin`) and clips with smooth,
   linear or step keys; one pose evaluator shared by previews, lint and export; a posed renderer
   that draws moving parts as rotated blocks (rest frames match the static renderer pixel for
   pixel except at part seams); filmstrips (`motion.png`), GIFs, and posed contact sheets
   (`cubist render --clip NAME [--at T]`); motion lint; glTF animations checked in Blender; the
   manifest. Demo: chest, door, window, desk, biplane.
8. **Characters** — done. Free `turn`/`move` tracks on any part (composed `R_joint · R_z · R_y ·
   R_x`, same-channel tracks add up); generators (`swing`, `bob`, `shake`, `spin_track`)
   resolved against the clip length into plain keys; hidden parts and `visible` tracks (left out
   of contact sheets, zero-scale GLB nodes with STEP scale channels); `travel`; oriented sockets;
   the `motion_floor` check; filmstrip diffs; examples keep their filmstrips; the
   `voxel-animate` skill. Demo: the goblin walks and idles, the blaster fires, the wand casts.
   An independent review of M7 found 11 edge-case bugs (step-only clips cut short in glTF,
   crashes on renamed parts, fast spins under-sampled, false collisions for rigidly attached
   parts, GIF timing, ...), all fixed with regression tests in `tests/test_motion_edges.py`.
9. **Flipbooks, color cycles, glow, lights** — done. `Part(frames=[...], fps=)` flipbooks
   (frame 0 is the rest look; later frames ride in the `.vox` as hidden `<part>@<k>` models and
   in the `.glb` as child nodes switched by STEP scale channels), looping on their own as
   `<part>_frames` clips or driven by `frames(...)` tracks; `[glow]` and `[cycles]` in the style
   bible (glow lifts preview shading by 1 - e^-strength and exports as an emissive palette
   texture; cycles are previewed and listed in the manifest); `light(...)` point lights as
   `KHR_lights_punctual` with flicker in the manifest. Every lint check that matters runs on
   every frame. Demo: the torch and a campfire flicker and glow, a pond's water flows, the
   blaster's flash plays two frames. Godot imports the moving demo assets with their animations,
   lights and emission.
10. **Effects** — done. `fracture`/`chunks` (seeded farthest-point seeds, jittered cuts,
   strays and slivers merged so every chunk is one connected piece); `burst` (a baked ballistic
   clip of plain keys: chunks roll up off the floor while rising and rest where they touch
   down); `clip(collisions=False)` for debris; `emitter(...)` with rate and/or burst on
   `emit:<name>` events, previewed by a seeded simulation that draws particle assets from
   `out/` (stand-in cubes and a lint warning when unbuilt), oriented empty nodes in the `.glb`,
   emitters and particle dependencies in the manifest. Demo: the barrel explodes with smoke,
   the campfire smokes, the blaster and wand throw sparks; new `smoke_puff` and `spark`
   particle assets.
11. **Evaluation** — run; human scores pending. Twelve briefs, one per moment of David's
   (torch, chest, door, window, drawer, walk, explosion, gun, wand, water, propellers, fire with
   smoke), on objects distinct from the demo, built by four fresh agents from the docs alone:
   12/12 built and lint-clean (0 errors, 0 warnings), 11/12 within 5 iterations (mean 4.1),
   agent self-scores 3–4 (mean 3.5). Their reports and an independent review of M8–M10 drove a
   round of fixes (preview clips in `render --clip`, flipbook frames in slices and thumbnails,
   burst floor contact, `mirrored()` parents, carve through thin walls, stepped glTF keys one
   key late, driven flipbooks in other clips, default owners skipping hidden parts, a 30 s
   regression in dense sampling) and additions (`cubist build --check`, iteration numbers in
   build output, `joint()`, `in`/`out` easing, `burst(start=)`). Details:
   `projects/motion/eval/agent-notes.md`; GIFs to score in `projects/motion/eval/gifs/`.

### Worlds (M12)

12. **Worlds** — in progress. Terrain for premade game worlds (see Worlds above), toward a 4 × 4 km world
   for Kiln: a town, a river, mountains on the horizon, a cave system.
   - **12a. Ground and the map** — done. `@world`, height fields (noise, ridges, painted maps, masks,
     blend, flatten, terrace), surface and subsoil rules, `cubist world build` with `map.png` and
     `stats.json`, the reference in [worlds.md](worlds.md), and the demo `islet`. Builds are
     deterministic and its map has a golden image.
   - **12b. Bake, and Kiln plays it** — done. `cubist world bake` writes tiles for levels 0–6, the far
     grids, `map.json` (with the material registry, deflate) and `world.json` (landmarks); Kiln opens it
     with `--world <dir>`.
     - **The speed decision: Python bakes.** A 1 km² world (7 levels, 633k chunks) bakes in 57 s on 9
       processes of an M1 Max: 55 s per km², so about 15 minutes for 4 × 4 km, against the 30-minute line.
       Kiln's C# baker did its valley at ~35 s per km² with caves, which this world doesn't have yet.
     - **Size:** 153 MiB per km² at level 0, 208 MiB for all levels (Kiln's valley: 171 at level 0).
     - **Encoding:** chunks share one palette per chunk column; mixed bricks are bit-packed against it (1–8
       bits), uniform bricks are table entries. Kiln re-encodes compactly on load, so this only sets the
       disk size. Raw bricks were 7× larger and slower to compress.
     - **Coarse levels** keep the surface band at least one cell deep, so distant ground shows grass, not
       the dirt under it (as Kiln's own downsampling does).
     - **In Kiln:** the 1 km² world streams with 0 holes; main-thread p99 0.74 ms at 8 m/s; validation
       clean.
     - **Not yet:** incremental bakes (each tile's input hash); props; caves.
   - **12c. Features** — done. Rivers (water levels that only fall downstream, water as voxels), roads
     (levelled, painted), caves (tunnels, caverns, noise worms), props (`place`, `scatter`) written to
     `world.json` for Kiln to stamp, landmarks.
     - **Caves cost bake time:** a 1 km² world with 5% of it over worms took 183 s, then 88 s once the worm
       noise was sampled every metre and per row of columns (a worm tile: 144 → 66 s), against 55 s without
       caves.
     - **In Kiln:** 2,981 scattered oaks stamp in 98 ms. Distant chunks under props are generated and
       stamped with coarse copies of the prefabs (Kiln's `WorldStore.CoarseStamps`) instead of downsampling
       eight finer chunks recursively: settling a view went from 25.5 s to 0.7 s.
   - **12d. Lint and region renders** — done. World lint (bounds, edges, slopes near landmarks, props on
     uneven ground or overlapping, caves with no way in) prints with every world build; `cubist world
     render` draws an area in voxels, with a cut height for looking into caves. The bake and the region
     render share one per-column builder (`chunk_row`).
   - **12e. The 4 × 4 km world**, authored with the build and critique loop, baked, and walked in Kiln.
     `vale` lives in Kiln's Cubist project (`assets/cubist/worlds/vale.py`), with its brief at the top.
     - **The first critique round** found snow on a third of the map (the snow line rose to 460 m) and a
       pass on a 55° cliff (lint's slope warning). Fixing the pass needed `level()`: flatten a circle to
       the ground's own height, since a script can't know the height.
     - **Scripts can't know the ground's height** (it depends on the seed), so caves and props also take
       depths below the ground (`below_ground`), resolved when the world loads.
     - **Baked and walked (Oct 26):** 1,366 tiles in 38 minutes on an M1 Max (142 s/km²), 3.8 GB, 181
       MiB/km² at level 0. Kiln streams it with no holes; a character walks from the town through the
       cave mouth to the cavern's torch. (Kiln's `world-streaming.md` has its numbers.)
     - **What walking found:** a worm under the tunnel left a 1.3 m pit the character couldn't climb
       out of, so worms now leave 2 m of rock under tunnel and cavern floors. The spawn stood in the
       alchemist's stall. A landmark can now be a named prop (`"cavern": "cavern_torch"`), so a start
       point can be underground.
     - **Iteration speed:** `--region` re-bakes only the tiles over an area (the cave's 11 tiles: 3
       minutes); `distance(step_m=8)` cut the full bake from 38 to 19.6 minutes. `cubist world build`
       (map, stats, lint) went from 68 s to 11 s: scatters are worked out once per build (they were
       three times), scattered points meet only the path segments near them, and a coarse distance
       grid is measured once per 64-point tile and kept. Tiles bake byte-identical.
     - **Later the same night:** `World(step_m)` (taller ground steps, for the look decision below);
       region renders draw props; lint walks tunnel floors for steps no one climbs; roads, rivers and
       tunnels are named lines in `world.json` (Kiln walks `--route path:cave_road`; caravans later);
       `place(..., carve=True)` digs a prop into the rock, and vale gained a crypt off its cavern
       (`assets/crypt.py`). The bake writes tiles aside and renames them, so a running Kiln reloads a
       region re-bake; vale bakes in 17 minutes and `world build` takes 11 s (from 68).
     - **Open:** slopes at 1/16 m are terraces of one-voxel steps, and on the hills they cost ~300k
       quads at Kiln's level 0 (its gate is 350k for a whole view); the two-tone grass reads as blobs
       from above. Both are look decisions for the next round.

### 13. Town kit (in progress)

A street of townhouses with cellars (Kiln's `docs/city.md`). Cubist's part, in steps:

- **13a. Ground lines (Oct 30).** `@asset(..., ground_y=N)` names the asset-space y where the ground meets
  the building. The manifest's `glb_origin` y and the `.glb` origin move to it, so Kiln, `place()` and
  region renders (which already place by `glb_origin`) stand the street level on the ground and the cellar
  below it. The default 0 changes no output. A negative `ground_y`, or one at or above the asset's height,
  is an error; the scale lint measures a building's height from the ground line. `build.json` carries it,
  so an export (which reads the asset back from `out/`) and `cubist lint` keep it.
- **13b. Walk lint (Oct 30).** The walk and storey lints (above, "Linting and validation"): a
  `building` or `walkable` asset walks from the street at `door` to every `floor_N`, `cellar` and
  `tour_NN` socket when `[scale]` names the walker, and `upper` and `roof` parts are y bands. A
  160 × 368 × 192 townhouse (a cellar, five storeys and their stairs) walks in and back out and
  checks its bands in 0.61–0.65 s over three runs, of 1.48 s for its whole lint (M1 Max; the test
  holds both under 5 s). Review rounds: the walk starts on the street, not in the doorway; the
  ground follows Kiln's carve; the walker stands on the goblin's own footprint (its spare column
  either side across a move); every socket needs a way back to the street; a missing `door` or
  `floor_0` warns.
- **13c. Bridges (Oct 30).** `road(..., bridges=True)` (`docs/worlds.md`, "Rivers and roads"): where the road
  crosses a river's channel it neither levels nor paints it, so the water and bed stay, and its `world.json`
  line crosses at deck height, the straight line between the road's heights at the two banks, with points
  at the channel's edges and every 4 m or so over it. The deck is a placed prop; the `bridge` lint warns of
  a crossing no placed prop's footprint spans (middle and 1 m past each bank), or a bridged road that starts
  or ends in the water. The road's level resumes at the channel's edge with no easing (the abutment).
  `bridges=False` is the old ford, pinned by a tile hash. The first bridge asset is Kiln's `stone_bridge_40`,
  placed over Brassford's cross street (Kiln's `docs/city.md`, step 4).
- **13d. Project modules (Oct 30).** An asset script imports the project's own modules (a `kit/` package
  of shared pieces beside `style.toml`) from any working directory: the build worker puts the project
  directory first on `sys.path`. Before, it worked only from the project directory, by way of `python -m`
  putting the working directory there, so a build from elsewhere (or from the MCP server) failed. A
  failure inside such a module reports its frames as well as the script's. Iterations still count the
  script's text only. World scripts don't import project modules yet; nothing needs it.
- **13e. Lights above the street (Oct 30).** The light-reach lint measures a grounded asset's light from
  the ground line, as the scale lint does since 13a: a townhouse's door lamp 40 voxels above the street
  needs a range of 57, where measuring from the cellar's floor asked 125 of it. A light below the line,
  in the cellar, is measured from the asset's floor.
- **13f. Tour legs (Oct 30).** The walk lint walks a building's tour in straight legs, as Kiln's
  walk will once it arrives within 2 voxels of each stop (it arrived within 12 then and cut corners;
  Kiln made the change with the town's first street, since shipped): from anywhere within 2 of a
  stop, the walker must follow the straight line to within 2 of the next (its footprint's centre
  within a voxel of it, its feet between the stops' heights) over the moves the flood kept. A leg
  that turns, runs through a wall or has no room either side errors: a doorway just past a stop must
  be 12 wide where a walk from the stop itself needs 10 (pinned by a test). A stop the walker can't
  stand all round, 2 out, errors too, saying how far to move it: 4.5 off a wall it must move 2.
  Short legs are walked. The test fixture's tour gained corner stops; the 160 × 368 × 192 case still
  lints in under 5 s.
- **13g. One asset of a family (Oct 30).** `cubist build --only NAME` (repeatable; the MCP
  `build_asset` tool's `only`) builds just the named assets of a script that defines several, so
  one variant of a family rebuilds alone, under its own timeout, and only its iteration counts.
- **13h. Street sides (Oct 30).** `@asset(..., street_sides=("-z",))` names the sides of the asset's
  x–z box the street lies on; the walk lint lays its ground only there (default: all four, as
  before). Kiln's quay steps showed why: with ground on the river side too, the walker dropped off
  it into the flight and reached the landing with the way in from the street shut. A quay's street
  is behind it, a bridge's at its ends. A wrong side, or an empty list, is an error naming the four;
  a `door` with no street side near it says so. `build.json` carries the sides, so export and
  `cubist lint` keep them.
- **13i. Region renders beside tall buildings (Oct 30).** A region render's window, at most 256 cells
  tall, takes its top from the props' voxels over the area only. It took it from every prop the
  search found by its reach, which counts a prop's height, so a level-0 render of Brassford's quay
  3 m from a five-storey townhouse framed the air over the street and drew nothing. A prop taller
  than the window over the area still shows only its top 256 cells: a street of townhouses is seen
  whole from 17 m across (level 1).
- **13j. Points along straight lines (Oct 30).** The bake writes a road's or river's line through its
  path's points, so a straight road given by its two ends had only those: Brassford's 208 m quay
  street joined none of the landmarks beside it in Kiln's walks, which join a line's nearest point
  within 64 m. A segment longer than 12 m is now split into parts of at most 4 m. A smoothed path's
  points are 4 m or so apart already (vale's at most 8.1 m), so its line is as before.
- **13k. Tags and ramps in the manifest (Oct 30).** The manifest names the asset's `tags` and the list
  `ramps` it uses (above, "Manifest"), so Kiln can collapse a building's colour ramps in its coarse copies.
- **13l. Indoor lights (Oct 31).** `light(..., indoor=True)` marks a light that lights a room. The
  light-reach lint measures it from the floor straight under it (the top of the first voxel below it
  in its column), not from the ground line. Kiln's townhouses light each storey from its hearth, 9
  voxels over the floor: measured from the street, the top storey's on a five-storey house asked a
  range of 307, where 160 lights its room. One with nothing under it keeps the ground line and warns
  that it stands over no floor, so a misplaced one is still caught. The flag is the lint's alone:
  `build.json` carries it, so export and `cubist lint` keep it, and the manifest and `.glb` don't.
  Façade lights stay outdoor, measured from the street.
- **13m. A light's hold (Oct 31).** `light(..., hold_m=3)` asks the engine to hold the light's level
  past that many metres, out to where its range fades it. Kiln's room lights need it: its lights fall
  as the inverse square and it has no bounce light. With a draft of the room lights (one per room), each
  lit a pool round itself, and the revealed rooms read 2.3–12.4 luma at noon against the mean luma of 40
  the room lights' gate will ask, 2.6–26.1 at 8× the intensity (Kiln's `docs/city.md`, step 7b). Walls
  still shadow a held light. The manifest's light has `"hold_m"` only when it is set, so every manifest
  before is byte-for-byte the same and the version stays 1; `build.json` carries it the same way. A
  hold that isn't a finite number above 0 is an error naming `hold_m`. The `.glb` has none.
  - `light()` now refuses a `range`, `intensity` or `position` that isn't finite as well, and so do
    `socket()` (its position in either form, `forward` and `up`) and `emitter()` (its position,
    `direction`, `rate`, `life`, `speed`, `spread`, `gravity`, `grow` and `seed`). NaN passed their
    checks (`nan <= 0` is false), and the export's `json.dumps` wrote `NaN` or `Infinity` into the
    manifest, which an engine's JSON reader rejects far from the cause. A direction is checked before
    it is parsed (`finite_vector`), so a bad one says so instead of warning from numpy.

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
| Untrusted script execution | Planned: a subprocess sandbox with no file or network access. Today scripts run in a subprocess with a timeout but no isolation, so `SECURITY.md` says to build only scripts you trust |

### Open questions

- [ ] Hand edits: when someone touches up a `.vox` in an editor, store the difference as a patch
  layer applied on top of the script output, or freeze the asset and stop treating the script
  as the source?
- [x] Animation: rigid parts with joints and clips, plus flipbooks for voxels that change
  shape (fire, water); no skinning. See Motion and effects.
- [x] Target engine: Kiln, David's own C# voxel engine, which lives in its own repository.
  Cubist is its asset pipeline: Kiln loads the `.vox` (glow from `MATL`) and the manifest
  (parts, joints, clips, lights, colors, occlusion), and needs no style bible. Kiln's requests
  are tracked as issues labeled `kiln`. Godot and Blender stay as `.glb` checkers only; the
  manifest is the engine contract.
- [x] Project name: Cubist (`cubist`).

### Prior art

| Project | Relevance |
| --- | --- |
| Goxel-MCP | Closest existing agent tool; its preview renders confirm the render-feedback approach. A good human touch-up path. |
| vengi | Open-source editor and `voxconvert` CLI covering many voxel formats, plus Lua scripting. Use as the format bridge. |
| magicavoxel-mcp | Per-voxel tool design; shows the limits of the low-level approach this design avoids. |
| APT (arXiv 2411.17255) | Research showing LLM spatial reasoning can plan voxel structures in Minecraft. |
