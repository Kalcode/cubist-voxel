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

`World(ground, surface, subsoil=(), surface_depth_m=0.25, landmarks={}, step_m=None)`:

- `ground`: a height field (metres).
- `surface`: rules, first match wins, ending in a plain color (the default). The surface material is
  the top `surface_depth_m` of the ground.
- `subsoil`: `(depth_m, color)` bands under the surface, ending in the color that fills the rest.
- `landmarks`: named points for the engine (spawn, towns, cave mouths); drawn on the map. `(x, z)` sits
  on the ground; the name of a `place(..., name=...)` sits at that prop's base, so a landmark can be on a
  cave floor (`"cavern": "cavern_torch"`).
- `step_m`: the baked ground's heights snap to multiples of it (say `0.25`, four voxels). A slope at one voxel
  per step is a field of tiny risers (busy dots up close, and many quads for Kiln); taller steps read as
  terrace lines, and on a test of vale-like hills `0.25` cut the terrain's quads 3.5× at Kiln's finest
  level and its level-0 disk size from 181 to 101 MiB/km². Props still stand on the unsnapped ground.

## Fields

`path(...).distance(max_m=1000, step_m=None)` is also a field: the distance to a path in metres (shape a
valley around a river with it). For a far reach, `step_m=8` samples it on an 8 m grid and interpolates:
much cheaper, and off by a couple of metres at most, which a valley wall doesn't show (leave it unset for
masks right next to a path).

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
- `road(path, width_m, color, bank_m=None, smooth_m=30, bridges=False)`: the strip is levelled to the road's
  own height profile (averaged over `smooth_m` of road), cut and filled into the ground over `bank_m`, and
  painted `color`. Where a road crosses a river it fills the channel (a ford).
  With `bridges=True` it leaves the channel alone: the water and the bed stay, the strip isn't painted over
  them, and the road's `world.json` line gets a point where it enters the channel and one where it leaves
  (at the road's height on each bank) and a point every 4 m or so between, on the straight line between
  the two: the deck. Kiln's walks follow that line in x–z and the character rides what's under it, so the
  deck is a prop: `place()` a bridge asset turned to the road, its footprint spanning the water and
  landing 1 m on each bank. The road's level resumes at the channel's edge, with no easing: the bridge's
  abutment meets it. A road that starts or ends in the water has no bank to carry a deck from: no deck, and
  lint warns.

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
- `worms(region, depth_m=(4, 30), scale_m=24, thickness=0.08, seed=0, floors=True)`: winding noise
  tunnels under `region` (`(x0, z0), (x1, z1)`), between `depth_m` metres under the ground. `scale_m`
  spaces them; `thickness` widens them. Their noise is sampled every metre and interpolated. They open
  into the world's tunnels and caverns through walls and ceilings but leave 2 m of rock under their
  floors, so the way through has no pits (a worm under a tunnel once left a 1.3 m drop the character
  couldn't climb out of); `floors=False` cuts those too.

Tunnels and caverns carve up to level of detail 2, so a mouth shows from a distance; worms only at level
0. The map draws caves over the ground: tunnels as lines, caverns as outlines, worm regions dashed.
Caves cost bake time: a 1 km² world with 5% of it over worms bakes in about 90 s instead of 55.

## Props

`World(props=[...])` places built assets. They stay placements: the bake writes them into `world.json` and
Kiln stamps the exported assets (`export/<asset>.vox` and its manifest), so moving one doesn't re-bake the
terrain. Export an asset before baking a world that uses it (`cubist export out/<asset>`).

- `place(asset, at, rotate_y=0, name=None, sink_m=0, below_ground_m=None, carve=False)`: `at` is `(x, z)`
  to stand it on the ground (the highest point under its footprint) or `(x, y, z)` for its base;
  `below_ground_m` puts its base that far under the ground (a torch on a cavern floor). `name` is how game
  code finds it. `carve` digs it into the rock: Kiln empties the world wherever the asset's box is empty, so
  a crypt or a room placed underground is hollow (vale's crypt, joined to its cavern by a `tunnel`).
  **Cellars:** an asset with `@asset(..., ground_y=48)` has its `glb_origin` at y = 48, and a placement puts
  that row on the ground, so `place(house, at=(x, z), carve=True)` stands its upper floors in the air and
  digs its cellar (the 48 rows below) into the ground. Region renders draw it the same way.
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
| bridge | A bridged road's crossing whose water no placed prop's footprint spans along the road (it must cover the middle and 1 m past the channel on each bank; a bridge turned along the river, a tree or a 1 m prop doesn't; a prop that isn't exported must stand within 1 m of the middle); a bridged road that starts or ends in a channel | Warning |
| cave | A cavern or tunnel with no way in from the surface; a tunnel whose floor rises or drops more than 1 m in a metre (a step no one climbs: vale's worm pit showed as "rises 3.7 m") | Warning |

Props are checked against their exported manifests; until they're exported, lint says so and skips them.

`cubist world render worlds/<name>.py --at x,z [--size 64] [--cut y]` draws an area in voxels (iso and top
views, `out/worlds/<name>/region-x-z.png`), built as the bake builds it, at the finest level that fits 256
cells. `--cut` removes everything above a height in metres, to look into caves from above. Props are
drawn from the project's exports, placed and turned as Kiln stamps them (at coarse levels a cell takes one
of its voxels). A town of 96 m renders in about 30 s, most of it placing the world's scattered props.

## Baking for Kiln

`cubist world bake worlds/<name>.py [--levels 7] [--workers N] [--out DIR] [--region X0 Z0 X1 Z1]` writes
the world in Kiln's
map format (Kiln's `docs/engine/formats.md`, "Baked maps") to `export/worlds/<name>/`:

- **Tiles** `L{k}/{tx}_{tz}.tile`: 64 × 64 columns of level k (cells of 2^k voxels, 128·2^k m a tile).
  Heights are sampled every 0.5 m and interpolated per column; from level 3 (seen from 128 m), every two
  cells.
- **Far-field grids** `far-L{k}.bin`, reaching past the world (the ground's fields go on beyond it).
- **`map.json`:** the tiles' extent, compression (deflate) and the material registry (palette index →
  name, color, glow).
- **`world.json`:** landmarks in voxels, with their ground height (or their prop's base); `paths`, the roads,
  rivers and tunnels as named lines (`road(..., name="cave_road")`; unnamed ones are `road1`, `river1`,
  `tunnel1`, ...), a point every ~4 m on the road or bed or along a tunnel's floor, for walks and caravans; and
  props.

The bake prints its progress every 5% of tiles. `--region` re-bakes only the tiles over that area (metres)
into an existing bake, plus the far grids and the JSON: a minute instead of half an hour while working on
one part of a big world. Tiles elsewhere keep what they had, so bake everything again once a change reaches
past the region. Tiles are written aside and renamed into place, so a running Kiln can reload them (its
panel's "reload map") without reading half a tile. Kiln plays it with `demos/Diorama/run.sh --world <that directory>`, starting at the landmark `--at` names
(or `spawn`). Expect about a minute per km² on a 10-core laptop, and 150–200 MiB per km²; bakes stay out
of git.

