# DSL reference

Everything an asset script can call. Scripts start with `from cubist import *` and define one
or more `@asset` functions that take the palette `p` and return a `Model` or a shape.

```python
from cubist import *


@asset(name="crate", size=(12, 12, 12), tags=["prop", "wood"], grounded=True)
def crate(p):
    return box((12, 12, 12), color=p.wood.mid).highlight_edges()
```

## Conventions

- **Axes.** Y-up, right-handed, integer voxel cells. +X is the viewer's right in the front view,
  +Y is up, +Z is the front (toward the viewer). The origin is the asset's bottom-left-back
  corner.
- **Left and right** are the character's own. A character faces +Z, so its right hand is at low
  x (the viewer's left): name that arm `arm_r`.
- **Sampling.** A voxel is filled when its center (x + 0.5, y + 0.5, z + 0.5) is inside a shape
  or exactly on its surface.
  - A shape symmetric about a whole-number plane needs an even width; about a half-number plane
    (x.5), an odd width.
  - Pointed shapes (cones, pyramids, spikes) need their center on a voxel center (x.5) so the tip
    has a column. Centered between columns, the tip disappears.
  - Thin rings and loops: voxel centers sit 0.5 off any plane through whole numbers, so a torus
    needs `r` of about 0.85 or more for its voxels to join.
- **Ranges are half-open.** `band(axis, at, height)`, `region(lo, hi)` and `gradient(lo, hi)`
  include the low bound and exclude the high one.
- **Colors** are palette names, never indices: `p.wood.dark`, `p.wood.mid`, `p.wood.light`,
  `p.wood[1]`, `p.cloth.white` (a named table), `p.cork` (a single color). On a list ramp
  `dark` is the first shade, `light` the last, and `mid` index `n // 2` (the third of four).
  Anywhere a ramp is expected, `p.wood`, `p.wood.ramp` and a list of colors all work.
- **Two stages.** Shapes stay distance fields until a voxel op (paint, hollow, round, ...).
  CSG, transforms and mirror work in both stages with the same color rules; `smooth_union`,
  `taper` and `displace` must come before voxel ops.

## Assets, parts, sockets

```python
asset(name=None, size=None, tags=(), grounded=False, symmetric=None, allow_floating=False)
```

Decorates `fn(p) -> Model | Shape`. `size` is the declared bounding box (lint errors on voxels
outside it). `grounded`: the asset should touch y = 0. `symmetric`: `"x"` (mirror across the
center plane `size / 2`) or `("x", 15.5)` for another plane. `allow_floating`: don't warn about
pieces detached from the main body. A bare shape becomes one part named after the asset.

```python
Model(parts, sockets=None)
Part(name, shape, pivot=None, parent=None, symmetric=None)
Part(...).mirrored(name, about_x=None, about_y=None, about_z=None)  # -> (part, twin)
```

- `pivot`: where the part hinges or rotates; default the bottom center of its voxels. Lint
  errors if it's outside the part.
- `parent`: the part it hangs from; GLB export nests nodes the same way.
- `mirrored` returns the part and a reflected twin (pivot reflected too); `Model` flattens the
  pair.
- `sockets`: `{"hand_r": (x, y, z)}`, attached to the nearest part, or
  `{"hand_r": (x, y, z, "arm_r")}` to name the part. They become empty nodes in the GLB.
- Each voxel should belong to one part; lint warns when parts overlap (they double up in exports
  and poke through when a part moves). Subtract one part's shape from the other.

## Primitives

Each starts at its local origin; move it with `.at(x, y, z)`.

| Call | Shape and origin |
| --- | --- |
| `box(size, color)` | min corner at the origin |
| `sphere(r, color)`, `ellipsoid(radii, color)` | centered |
| `cylinder(r, h, axis="y", color)` | base centered on the origin, extending `h` along `axis` |
| `cone(r, h, axis="y", r_top=0, color)` | like `cylinder`; `r_top > 0` makes a frustum |
| `pyramid(size, color)` | like `box`, narrowing to its top center (hip roofs, spires) |
| `wedge(size, rise="+x", color)` | like `box`, height 0 at the low end of `rise` (`+x`, `-x`, `+z`, `-z`) |
| `torus(R, r, axis="y", color)` | centered; ring of radius `R` around `axis`, tube radius `r` |
| `capsule(p0, p1, r, color, r_end=None)` | round-ended tube between absolute points, tapering to `r_end` (tails, horns, limbs) |
| `line(p0, p1, color, connect=6, r=None)` | the voxels a segment passes through: `connect=6` face-connected (a 2-wide staircase on diagonals), `connect=26` one voxel per step; with `r`, a thick line |
| `polyline(points, color, connect=6)` | `line` through a list of points (bows, sagging ropes) |
| `panel(points, color, thickness=None)` | a flat plate through 3+ coplanar points at any angle (wings, sails); the default thickness stays watertight at any tilt |
| `lathe(profile, axis="y", color)` | `[(height, radius), ...]`, heights increasing, revolved around `axis` (barrels, bottles, columns) |
| `extrude(polygon, depth, axis="z", color)` | 2D polygon pushed `depth` from the origin; polygon axes: `z` → (x, y), `y` → (x, z), `x` → (z, y) |
| `heightmap(heights, color)` | columns from y = 0 up to `heights[x][z]` (terrain) |

Give lines their endpoints directly instead of rotating them afterwards; rotation resamples
them into a staircase.

## CSG

| Call | Does |
| --- | --- |
| `a.union(b, ...)`, `a \| b` | Both. Where they overlap the later shape's color wins |
| `a.subtract(b, ...)`, `a - b` | `a` without `b`; keeps `a`'s colors, so cutters need no color |
| `a.intersect(b)`, `a & b` | Only the overlap; keeps `a`'s colors |
| `a.smooth_union(b, k)` | Union with a fillet of radius `k` voxels (distance fields only) |
| `a.carve(cutter, color=None)` | `a - cutter`, coloring the cut's walls and back (windows, arrow slits, eye sockets) |

Selections (`band`, `faces`, ...) only go in `where=`; use a shape to cut.

## Transforms

| Call | Does |
| --- | --- |
| `.at(x, y, z)`, `.translate(...)` | Move by (x, y, z) |
| `.rotate(degrees, axis="y", about=None)` | Right-hand rule about `axis` (or a vector) through `about` (default: the bounds center). `rotate(90, "y")` turns +X toward −Z; `rotate(90, "z")` turns +X toward +Y; `rotate(90, "x")` turns +Y toward +Z |
| `.scale(factor, about=None)` | Uniform or per-axis `(sx, sy, sz)`, about the bounds center by default |
| `.reflect(axis, about=None)` | Replace with the reflection across the plane `axis = about` |
| `.mirror(axis, about=None)` | Keep the shape and add its reflection; exactly symmetric in form and color |
| `.array(n, step)` | `n` copies, each moved `step` further |
| `.radial(n, axis="y", about=(0, 0, 0))` | `n` copies evenly around `axis` through `about` |

Voxel-stage shapes move exactly by whole voxels and 90° turns, and resample by nearest voxel
otherwise. Mirroring them needs a plane on the grid (a whole or half number).

## Shape modifiers

| Call | Does |
| --- | --- |
| `.taper(top=1.0, axis="y")` | Cross-section scale 1 at the bottom, `top` at the top (distance fields only) |
| `.displace(amp, scale=4.0, seed=0, noise="perlin", octaves=1)` | Push the surface in and out by `amp` voxels of noise with feature size `scale` (distance fields only) |
| `.round(r=1)` | Round convex edges with radius `r` without growing; `round(1)` takes one voxel off every edge. Strong on small shapes |
| `.chamfer(c=1)` | Cut `c` layers off every convex edge at 45° |
| `.hollow(t=1)` | Keep a shell `t` thick (measured in all 26 directions, so curved shells stay watertight) |
| `.erode(r=1)`, `.dilate(r=1, color=None)` | Shrink or grow by `r`; grown voxels take the nearest color or `color` |

## Paint

Every paint op takes `where=`: a selection or a shape (the voxels whose centers it contains).

| Call | Does |
| --- | --- |
| `.paint(color, where=None)` | Recolor |
| `.paint_faces(direction, color, depth=1, where=None)` | Voxels whose face toward `direction` is open, `depth` deep (grass on top) |
| `.gradient(axis, ramp, where=None, reverse=False, lo=None, hi=None)` | Equal bands through `ramp`, dark to light along `axis`, over the selected voxels' extent (hidden ones included) or `[lo, hi)` |
| `.dither(ramp, seed=0, amount=0.5, where=None, mirror=None)` | Nudge a share of voxels one shade along the ramp; `mirror=("x", 8)` keeps the pattern symmetric |
| `.darken_crevices(steps=1, where=None)` | Darken surface voxels in concave corners (solid one and two layers out, like a floor meeting a wall); skips one-voxel walls so nothing shows through |
| `.highlight_edges(steps=1, where=None, facing=None)` | Lighten convex edges; thin plates don't count as edges. `facing=("top", "left")` lightens only edges open toward the light |
| `.masonry(ramp, block=(4, 2, 4), mortar=None, seed=0, stagger=True, where=None)` | Blocks with a seeded shade each, staggered courses, optional mortar joints (stone, brick, shingles) |
| `.decal(rows, colors, face="front", at=(u, v), where=None)` | Paint a character grid onto the first surface seen from `face`, as that view shows it (eyes, emblems, signs); see below |

Ramp-stepping ops (`dither`, `darken_crevices`, `highlight_edges`) only move along list ramps;
named-table and single colors stay as they are. Dithering breaks exact color symmetry unless you
pass `mirror=`. Paint per material before a union when one material shouldn't affect another's
shading.

`decal` rows run top to bottom; `colors` maps characters to palette colors, and any other
character is transparent. `at` is the view's ruler coordinates of the top-left character, in the
orientation the contact sheet shows that view:

```python
eyes = ["w.w", "k.k"]
head = head.decal(eyes, {"w": p.eye.white, "k": p.eye.dark}, face="front", at=(6, 20))
```

## Selections

| Call | Selects |
| --- | --- |
| `band(axis, at, height=1)` | `at <= coordinate < at + height` |
| `region(lo, hi)` | the box `[lo, hi)` in world coordinates |
| `stripes(axis, period, width=1, offset=0)` | repeating bands: `width` on, `period - width` off |
| `checker(size=1, axes="xyz")` | a checkerboard of `size`-voxel cells |
| `colored(*colors)` | voxels currently painted any of `colors` |
| `faces(direction)` | voxels whose face toward `top`, `bottom`, `left`, `right`, `front` or `back` (or `+y`, `-x`, ...) is open |
| `surface()` | voxels with any open face |
| `inside(shape)` or a shape | voxels whose centers the shape contains |

Combine with `&`, `|`, `-` and `~`: `faces("top") & ~band("x", 0)`.

## Seeds

`displace`, `dither` and `masonry` take a `seed`, mixed with the build seed
(`cubist build --seed N`), so the same script and seed always give the same voxels. There is no
other randomness: don't use Python's `random`.

## Patterns that work

- **Big masses first**, details second, shading last.
- **Shade half, then mirror** for symmetric assets with noisy paint, or pass
  `dither(..., mirror=...)`.
- **Recesses**: `carve(cutter, color=dark)` gives windows and sockets depth.
- **Profiles**: intersect extrusions from two or three sides
  (`extrude(side, ...) & extrude(top, ..., axis="y")`) for anvils, furniture, weapons.
- **Families**: copy a script and change only what makes the variant different.
