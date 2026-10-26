# World reference

Worlds are terrain for premade game worlds; [design.md](design.md), "Worlds", says why and how they fit
in. A world script lives in `worlds/<name>.py` of a project and starts with `from cubist.world import *`.
Positions are metres: x east, z toward the front (rows go down the map), y up. Status: milestones 12a,
12b, 12c and 12d (ground, surface rules, rivers, roads, caves, props, the map, lint, region renders, the
bake); the 4 × 4 km world is next.

```python
from cubist.world import *


@world(name="islet", size_m=(512, 512), seed=7)
def islet(p):
    land = circle((256, 256), radius_m=150, edge_m=90)
    hills = (
        noise(scale_m=120, amplitude_m=14, octaves=4) + ridges(scale_m=200, amplitude_m=30) * land
    )
    ground = (flat(2) + hills * land + land * 12).flatten(rect((150, 300), (200, 340)), at_m=19)
    return World(
        ground=ground,
        surface=[
            height_under(3, p.cork),
            height_over(44, p.white),
            slope_over(35, p.stone.mid),
            p.leaf.mid,
        ],
        subsoil=[(1, p.wood[1]), p.stone.dark],
        landmarks={"quarry": (175, 320)},
    )
```

## Declaring a world

`@world(name, size_m=(x, z), seed=0)` on a function taking the palette `p` and returning a `World`.
The world covers `[0, size_m)` on both axes. Its seed mixes with the build's (`--seed`).

`World(ground, surface, subsoil=(), surface_depth_m=0.25, landmarks={})`:

- `ground`: a height field (metres).
- `surface`: rules, first match wins, ending in a plain color (the default). The surface material is
  the top `surface_depth_m` of the ground.
- `subsoil`: `(depth_m, color)` bands under the surface, ending in the color that fills the rest.
- `landmarks`: named points `(x, z)` for the engine (spawn, towns, cave mouths); drawn on the map.

## Fields

`path(...).distance(max_m=1000)` is also a field: the distance to a path in metres (shape a valley
around a river with it).

A field is a value per ground column. Height fields are metres; masks are weights in [0, 1].

| Field | Gives |
| --- | --- |
| `flat(m)` | The same value everywhere |
| `noise(scale_m, amplitude_m=1, octaves=1, kind="perlin", seed=0)` | Rolling noise in ±amplitude; `scale_m` is the size of its features; `kind` is `perlin` or `value` |
| `ridges(scale_m, amplitude_m=1, octaves=4, sharpness=2, seed=0)` | Sharp crests in [0, amplitude] |
| `height_map(path, low_m, high_m, origin_m=(0, 0), size_m=None)` | A greyscale PNG (8 or 16 bit), black `low_m`, white `high_m`, over `size_m` (default: the world) |
| `mask_map(path, origin_m=(0, 0), size_m=None)` | A greyscale PNG as a mask |
| `circle(center, radius_m, edge_m=10)` | Mask: 1 inside, easing to 0 over `edge_m` |
| `rect(lo, hi, edge_m=10)` | Mask: 1 between the corners, easing to 0 over `edge_m` |

Combine fields with `+ - *` (with fields or numbers), and:

- `a.blend(b, mask)`: `a` where the mask is 0, `b` where it is 1.
- `a.flatten(mask, at_m)`: level the ground to `at_m` under the mask (a town, a quarry).
- `a.level(center, radius_m, edge_m=20)`: level a circle to the ground's own height at its center (a
  square, a pass), when you don't know that height.
- `a.max(b)`, `a.min(b)`.
- `a.terrace(step_m, softness=0.3)`: steps every `step_m`.

## Rivers and roads

`path(points, smooth=True)` is a line through `(x, z)` points in metres; `smooth` makes it a curve through
them. Rivers and roads follow paths, in `World(rivers=[...], roads=[...])` (rivers are cut first):

- `river(path, width_m, depth_m=1.5, bank_m=None, bed=None, water=None)`: list the path from the source.
  The water level follows the ground along it but never rises downstream. The channel is cut `depth_m`
  under it, the banks ease into the ground over `bank_m` (default: the width), the bed is painted `bed`,
  and the channel fills with `water`. Water is solid voxels of that color for now.
- `road(path, width_m, color, bank_m=None, smooth_m=30)`: the strip is levelled to the road's own height
  profile (averaged over `smooth_m` of road), cut and filled into the ground over `bank_m`, and painted
  `color`. Where a road crosses a river it fills the channel (a ford).

The map draws water in its own color, a little darker where it's deep; the bake fills it as voxels and
shows it in the far field.

## Caves

`World(caves=[...])` hollows the ground. Caves only remove ground, and the bake reaches each column's
chunks down to the caves under it.

- `tunnel(points, radius_m=2.5, smooth=True, below_ground=False)`: a tube along `(x, y, z)` points in
  metres (y is height). With `below_ground`, each y is a depth under the ground at that point (negative:
  above it), so a mouth opens wherever the ground is: start at `-1`.
- `cavern(center, radius_m, roughness_m=0, seed=0, below_ground=False)`: an ellipsoid (one radius, or
  `(x, y, z)` radii) whose walls wobble by up to `roughness_m`; `below_ground` makes the center's y a
  depth.
- `worms(region, depth_m=(4, 30), scale_m=24, thickness=0.08, seed=0)`: winding noise tunnels under
  `region` (`(x0, z0), (x1, z1)`), between `depth_m` metres under the ground. `scale_m` spaces them;
  `thickness` widens them. Their noise is sampled every metre and interpolated.

Tunnels and caverns carve up to level of detail 2, so a mouth shows from a distance; worms only at level
0. The map draws caves over the ground: tunnels as lines, caverns as outlines, worm regions dashed.
Caves cost bake time: a 1 km² world with 5% of it over worms bakes in about 90 s instead of 55.

## Props

`World(props=[...])` places built assets. They stay placements: the bake writes them into `world.json` and
Kiln stamps the exported assets (`export/<asset>.vox` and its manifest), so moving one doesn't re-bake the
terrain. Export an asset before baking a world that uses it (`cubist export out/<asset>`).

- `place(asset, at, rotate_y=0, name=None, sink_m=0, below_ground_m=None)`: `at` is `(x, z)` to stand it
  on the ground (the highest point under its footprint) or `(x, y, z)` for its base; `below_ground_m`
  puts its base that far under the ground (a torch on a cavern floor). `name` is how game code finds it.
- `scatter(asset, region, per_ha, seed=0, spacing_m=None, rotate=True, slope_max=None, height_m=None,
  avoid=None, sink_m=0.25)`: about `per_ha` per hectare over `region` (a mask, or `((x0, z0), (x1, z1))`),
  on a jittered grid at least `spacing_m` apart (default from the density), turned by random quarter turns.
  Points steeper than `slope_max`, outside `height_m` (low, high) or where `avoid` is over 0.5 are skipped.
  Each sinks `sink_m` into the ground so trunks on slopes don't float.

The map draws scattered props as dark dots and placed ones as small squares; the stats count them per
asset. Kiln stamps 3,000 scattered oaks in about 0.1 s, and draws distant ones from coarse copies of each
asset.

## Surface rules

| Rule | Matches |
| --- | --- |
| `height_over(m, color)` | Ground above `m` metres (snow) |
| `height_under(m, color)` | Ground below `m` metres (a shore) |
| `slope_over(degrees, color)` | Steeper than `degrees` (rock on cliffs) |
| `where(mask, color, over=0.5)` | Where a mask is above `over` (a painted biome, a road) |

## Building

`cubist world build worlds/<name>.py [--seed N] [--out DIR]` writes `out/worlds/<name>/map.png` and
`stats.json`:

- **The map:** the whole world top-down, hill-shaded from the top-left, with contour lines, a ruler in
  metres, landmarks, and a legend of surface colors with their share of the ground.
- **The stats:** height range, slope bands (0–15°, 15–30°, 30–45°, 45°+) and surface shares.

## Lint and close looks

`cubist world build` lints the world and prints the findings (also in `lint.json`):

| Check | Finds | Severity |
| --- | --- | --- |
| bounds | Landmarks or placed props outside the world (rivers or roads leaving it: info) | Error |
| edge | Landmarks within 400 m of the edge, where the engine shows the world's end | Warning |
| slope | Ground within 20 m of a landmark steeper than 45°; the share of all ground that steep (info) | Warning |
| uneven | A placed prop whose footprint's ground falls more than 1 m (it floats on the low side) | Warning |
| overlap | Placed props overlapping, and scattered props growing into a placed one | Warning |
| cave | A cavern or tunnel with no way in from the surface | Warning |

Props are checked against their exported manifests; until they're exported, lint says so and skips them.

`cubist world render worlds/<name>.py --at x,z [--size 64] [--cut y]` draws an area in voxels (iso and top
views, `out/worlds/<name>/region-x-z.png`), built as the bake builds it, at the finest level that fits 256
cells. `--cut` removes everything above a height in metres, to look into caves from above. Props aren't
drawn in it yet.

## Baking for Kiln

`cubist world bake worlds/<name>.py [--levels 7] [--workers N] [--out DIR]` writes the world in Kiln's
map format (Kiln's `docs/engine/formats.md`, "Baked maps") to `export/worlds/<name>/`:

- **Tiles** `L{k}/{tx}_{tz}.tile`: 64 × 64 columns of level k (cells of 2^k voxels, 128·2^k m a tile).
  Heights are sampled every 0.5 m and interpolated per column.
- **Far-field grids** `far-L{k}.bin`, reaching past the world (the ground's fields go on beyond it).
- **`map.json`:** the tiles' extent, compression (deflate) and the material registry (palette index →
  name, color, glow).
- **`world.json`:** landmarks in voxels, with their ground height.

Kiln plays it with `demos/Diorama/run.sh --world <that directory>`, starting at the landmark `--at` names
(or `spawn`). Expect about a minute per km² on a 10-core laptop, and 150–200 MiB per km²; bakes stay out
of git.

