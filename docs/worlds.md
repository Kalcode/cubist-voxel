# World reference

Worlds are terrain for premade game worlds; [design.md](design.md), "Worlds", says why and how they fit
in. A world script lives in `worlds/<name>.py` of a project and starts with `from cubist.world import *`.
Positions are metres: x east, z toward the front (rows go down the map), y up. Status: milestone 12a
(ground, surface rules, the map); caves, props, lint and the bake come next.

```python
from cubist.world import *

@world(name="islet", size_m=(512, 512), seed=7)
def islet(p):
    land = circle((256, 256), radius_m=150, edge_m=90)
    hills = noise(scale_m=120, amplitude_m=14, octaves=4) + ridges(scale_m=200, amplitude_m=30) * land
    ground = (flat(2) + hills * land + land * 12).flatten(rect((150, 300), (200, 340)), at_m=19)
    return World(
        ground=ground,
        surface=[height_under(3, p.cork), height_over(44, p.white), slope_over(35, p.stone.mid), p.leaf.mid],
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
- `a.max(b)`, `a.min(b)`.
- `a.terrace(step_m, softness=0.3)`: steps every `step_m`.

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
