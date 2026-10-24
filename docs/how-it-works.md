# How Cubist works

Cubist turns a short Python script into a voxel asset, shows the agent what it built, and
checks it against the project's style. The agent never places voxels by hand: it writes code,
looks at renders, and revises. This page walks through the moving parts.

## The big picture

```mermaid
flowchart LR
    subgraph Agent["Agent (Claude Code or any MCP client)"]
        brief["Brief + style bible<br/>+ similar examples"]
        script["assets/name.py"]
        critique["Critique against<br/>the checklist"]
    end

    subgraph Cubist
        cli["cubist CLI"]
        mcp["MCP server"]
        core["cubist package<br/>(DSL, build, render, lint, export)"]
    end

    subgraph Outputs["out/name/"]
        vox[".vox"]
        sheet["contact.png"]
        lint["lint.json"]
        summary["build.json<br/>history.jsonl"]
    end

    brief --> script
    script -->|build| cli & mcp
    cli & mcp --> core
    core --> vox & sheet & lint & summary
    sheet & lint --> critique
    critique -->|revise| script
    vox -->|export, if lint passes| game["Game engine<br/>.glb / .vox / vengi formats"]
```

The CLI and the MCP server are thin wrappers over the same functions, so anything one can do
the other can too.

## What happens in a build

```mermaid
sequenceDiagram
    participant A as Agent
    participant R as Runner (parent)
    participant W as Worker subprocess
    participant O as out/name/

    A->>R: cubist build assets/crate.py [--seed N]
    R->>W: spawn with a timeout (default 30 s)
    W->>W: load style.toml (palette, budgets, rules)
    W->>W: run the script, collect @asset functions
    W->>W: call each asset with the palette inside a build context (seed, ops used)
    W->>W: voxelize every part (sample SDFs, run voxel ops)
    W->>O: crate.vox (previous build kept as crate.prev.vox)
    W->>O: contact.png (5 views, rulers, legend)
    W->>O: lint.json, build.json, history.jsonl
    W-->>R: result file (never stdout, so prints can't corrupt it)
    R-->>A: JSON summary with the lint report
```

The parent never imports the script; a runaway script times out instead of hanging the agent.
`--check-determinism` runs the worker twice and compares the bytes.

## Shapes: distance fields first, voxels second

Every primitive is a signed distance field (SDF). CSG, transforms, taper and displace combine
fields cheaply and exactly. Anything that has to look at voxels (painting, hollowing, rounding,
heightmaps) samples the field into a grid and continues there.

```mermaid
stateDiagram-v2
    direction LR
    [*] --> SDF: box, sphere, lathe, extrude, ...
    SDF --> SDF: union / subtract / intersect<br/>smooth_union, taper, displace<br/>at, rotate, scale, mirror, array, radial
    SDF --> Grid: paint, gradient, dither,<br/>darken_crevices, highlight_edges,<br/>hollow, round, chamfer, erode, dilate
    Grid --> Grid: paint ops, morphology<br/>union / subtract / intersect<br/>whole-voxel moves, 90° turns, mirror
    Grid --> [*]: voxelize
    SDF --> [*]: voxelize (sample at voxel centers)
```

A voxel is filled when its center is inside a shape or exactly on its surface. Union gives
overlaps the later shape's color; subtract and intersect keep the left shape's colors. Ops
that need a distance field (`smooth_union`, `taper`, `displace`) refuse grid-stage shapes with a
message to apply them before painting.

## Coordinates

```text
        +Y (up)
         |
         |
         +------ +X (right)
        /
       /
     +Z (front, toward the viewer)
```

Y-up, right-handed, integer voxel cells. The origin is the asset's bottom-left-back corner.
The `.vox` writer converts to MagicaVoxel's Z-up; the GLB stays Y-up like glTF and is scaled by
the style bible's `voxel_size_m`.

## What the agent sees

Every build writes one contact sheet: front, right, back and top views at one pixel scale, with
voxel-coordinate rulers, plus a 3/4 iso view and a legend of palette names. The agent can say
"the left ear at x = 2..3, y = 18..19 is too thin" and find it in the script.

![Contact sheet for the goblin](images/contact-goblin.png)

For exact work on small regions, ASCII slices print a layer per block, one letter per palette
color:

```text
$ cubist render out/goblin_scout --slices y=18
legend: . = empty, a = skin.mid, e = eye.yellow

y=18  (rows: z, back to front; columns: x)
    0    5    10   15
 3  ....aaaaaaaa....
 5  ..aaaaaaaaaaaa..
 8  ....aeeaaeea....
```

Revisions come with a before/after/changes sheet (`cubist diff`): added voxels green, removed
red, recolored orange.

![Diff sheet: ears removed, a hat added, the nose recolored](images/diff-goblin.png)

## Lint: what renders hide

```mermaid
flowchart TD
    asset[Built asset] --> palette{Palette}
    asset --> bounds{Bounds}
    asset --> pivots{Pivots & sockets}
    asset --> floating{Floating pieces}
    asset --> symmetry{Symmetry}
    asset --> grounding{Grounding}
    asset --> budgets{Budgets}
    asset --> thin{Thin walls}
    asset --> hidden{Hidden interior}
    palette & bounds & pivots -->|error| block[Blocks export]
    floating & symmetry & grounding & budgets -->|warning| advice[Advice]
    thin & hidden -->|info| advice
```

Errors block `cubist export` unless `--force`. Determinism is checked on demand in CI.

## Export

```mermaid
flowchart LR
    out["out/name/name.vox<br/>+ build.json"] --> relint{Re-lint}
    relint -->|errors| stop["Refuse (exit 2)<br/>unless --force"]
    relint -->|ok| fmt{Format}
    fmt -->|glb| glb["GLB: one node per part at its pivot,<br/>sockets as empty nodes,<br/>palette texture, optional AO"]
    fmt -->|vox| vox["MagicaVoxel .vox<br/>with the part hierarchy"]
    fmt -->|anything else| vengi["vengi-voxconvert<br/>(OBJ, Qubicle, schematics, ...)"]
```

GLB meshes are greedy-meshed with sealed cavities dropped, so only faces you can see are
exported. Parts become nodes at their pivots, parented like the model, so an engine can rotate
an arm or open a lid with no extra setup. The import is tested in headless Godot 4.

## Module map

| Module | Does |
| --- | --- |
| `cubist/core` | Voxel grids, palettes, the style bible, parts and models, topology (what is visible) |
| `cubist/dsl` | Primitives, CSG, transforms, voxel morphology, paint ops, selections, seeded noise |
| `cubist/build` | Runs scripts in a subprocess; writes outputs and history |
| `cubist/render` | Rasterizer, contact sheets, ASCII slices, diff sheets, bitmap font |
| `cubist/lint` | The checks above and the report |
| `cubist/io` | `.vox` reader/writer, greedy meshing, GLB writer, vengi bridge |
| `cubist/export.py` | Lint-gated export |
| `cubist/project.py`, `examples.py` | Briefs, scaffolding, the example library |
| `cubist/evaluation.py` | The brief-by-brief evaluation report |
| `cubist/mcp_server.py` | MCP tools over the same functions |
| `cubist/cli.py` | The `cubist` command |

The design of record, with every decision and the DSL reference, is [design.md](design.md).
