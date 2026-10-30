# DSL reference

Everything an asset script can call. Scripts start with `from cubist import *` and define one
or more `@asset` functions that take the palette `p` and return a `Model` or a shape.

```python
from cubist import *


@asset(name="crate", size=(12, 12, 12), tags=["prop", "wood"], grounded=True)
def crate(p):
    return box((12, 12, 12), color=p.wood.mid).highlight_edges()
```

A family of assets can share pieces through the project's own modules: a package beside
`style.toml` (say `kit/`) that scripts `import`, whatever directory the build runs from. Its
functions take `p` and return shapes like any script's. An error inside one shows its lines too.
A build counts an iteration by the script's text alone, so a change to a shared module alone
isn't counted: say so when you report iterations. The project's directory goes first on the
import path, so give its modules names no installed package uses (`kit`, not `random` or `cubist`).
`cubist build <script> --only <name>` builds one asset of a script that defines several (a family's
variants), and only that asset's iteration counts.

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
asset(
    name=None,
    size=None,
    tags=(),
    grounded=False,
    symmetric=None,
    allow_floating=False,
    ground_y=0,
    street_sides=None,
)
```

Decorates `fn(p) -> Model | Shape`. `size` is the declared bounding box (lint errors on voxels
outside it). `grounded`: the asset should touch y = 0. `symmetric`: `"x"` (mirror across the
center plane `size / 2`) or `("x", 15.5)` for another plane. `allow_floating`: don't warn about
pieces detached from the main body. `ground_y`: the asset-space y (a whole number of voxels, below
the asset's height) where the ground meets the asset: a townhouse with a cellar sets it to the
cellar's depth, so the export's `glb_origin` sits at street level and a placement puts the street
there. The scale lint measures a building's height from it. `street_sides`: the sides of the asset's
x–z box the street lies on, some of `"-x"`, `"+x"`, `"-z"`, `"+z"` (default: all four), where the
walk lint lays its ground: a quay's street is behind it (`("-z",)`), a bridge's at its ends
(`("-x", "+x")`), and the river is everywhere else. A bare shape becomes one part named after the
asset.

```python
Model(parts, sockets=None)
Part(name, shape, pivot=None, parent=None, symmetric=None, occlusion=None)
Part(...).mirrored(name, about_x=None, about_y=None, about_z=None)  # -> (part, twin)
```

- `pivot`: where the part hinges or rotates; default the bottom center of its voxels. Lint
  errors if it's outside the part.
- `parent`: the part it hangs from; GLB export nests nodes the same way.
- `mirrored` returns the part and a reflected twin (pivot reflected too); `Model` flattens the
  pair.
- `sockets`: `{"hand_r": (x, y, z)}`, attached to the nearest part, or
  `{"hand_r": (x, y, z, "arm_r")}` to name the part. They become empty nodes in the GLB.
- `occlusion`: `"roof"`, `"upper"` (an upper storey) or `"canopy"` (a tree's crown), for
  engines that hide what's between the camera and the player; or `"clutter"`, small
  set-dressing (a market stall's counter and posts, crates, a cart) that an engine's follow
  camera passes through instead of pulling in. Parts hanging from a tagged part share its tag
  unless they have their own, so parent the chimney to the roof. Give a building's roof its own
  part; lint notes a building with no `roof` or `upper` part (clutter isn't a roof).
- Each voxel should belong to one part; lint warns when parts overlap (they double up in exports
  and poke through when a part moves). Subtract one part's shape from the other.

Buildings (tagged `building`, or `walkable` for a bridge) name the points a walk needs, and when
the style's `[scale]` gives `walker_width`, `walker_height` and `walker_climb` (the game
character's box and the ledge it climbs, in voxels), lint walks them:

- `door`: at the main entrance, on the ground (y = `ground_y`): on the front face, or under the
  overhang, within 16 voxels of the street (sockets stay inside the model's bounds). The walk
  starts on the street there, outside the asset's x–z bounds, and must reach it.
- `floor_0` … `floor_n`: a standing point on each storey, `floor_0` the ground floor. A bridge puts
  `door` and `floor_0` at its deck ends.
- `cellar`: a standing point in the cellar.
- `tour_01` … `tour_NN`: the stops of a walk through the building, in order (the door, the foot and
  head of each flight, each floor, the cellar). Kiln's walk will arrive within 2 voxels of a stop
  and steer straight for the next (it arrives within 12 today, and changes with the town's first
  street), so from `door` to `tour_01` and from each stop to the next the walker must follow a
  straight line from anywhere within 2 voxels of the stop it leaves: lint errors on a leg that
  turns, runs through a wall or has no room either side of the line (a stop where the way turns
  mends it), and on a stop too close to an obstacle for the walker to stand all round it, 2 out.

Without `door` or `floor_0` lint warns that it can't walk the building. The walker, two voxels wider
and taller than the character, must reach every one of them from the street (on every side of the
asset, or on its `street_sides`, and on under eaves and jetties) and walk back out, its feet within
2 voxels of the socket's y; a stair it can only drop down, or a hatch, doesn't count. So doorways at
least `walker_width + 2` wide and `walker_height + 2` tall, risers of `walker_climb` at most, and
stairwells open `walker_climb + walker_height + 2` above each tread. Give each storey its own part
tagged `upper` and the roof its own `roof` part, split by y bands (stair steps above a line belong
to the storey above): lint errors when one reaches up past the next one's lowest voxel, save the
roof coming down over the top storey.

## Joints and clips

```python
Part(name, shape, ..., joint=None)
hinge(axis, limits)  # turn about axis through the pivot; limits = (lo, hi) degrees
slide(direction, distance)  # move 0..distance voxels along direction
spin(axis, rps=None)  # turn without limits; rps = a default speed, turns per second
clip(name, length, tracks, loop=False, ease="smooth", events=())
Model(parts, sockets=None, clips=None)
```

A joint says how a part moves about its pivot; a clip is named, timed motion of joints.

- **Axes** are `"x"`, `"-x"`, `"+z"` (and so on) or a vector, in the parent's frame: a part
  rides along when its parent moves.
- **Units:** seconds, degrees for hinges, voxels for slides, turns for spins. The pose as
  modeled is joint value 0, and clip values are relative to it.
- **Tracks** map part names to keys: plain values spread evenly from 0 to `length` (one value
  holds still), or `(time, value)` pairs. In list-form clips (below), `joint(part, keys)` is
  the same thing next to `turn`, `visible` and the rest.
- **`ease`**: `smooth` (default) eases in and out and never overshoots its keys, so to
  overshoot, add a key past the target: `[0, 115, 105]`. It stops only at keys where the motion
  turns around (a peak, a valley, the ends of a one-shot clip) and keeps moving through keys
  in between. `linear` keeps a constant speed (spins, conveyor belts); `step` jumps from key to
  key; `in` accelerates into each key (falls, slams); `out` decelerates into each key.
- **`loop=True`** clips should end in the pose they start in; spins end on whole turns. Lint
  warns otherwise.
- **`events`** are `(time, name)` markers for the engine (`"creak"`, `"footstep"`); Cubist
  gives them no meaning.
- `Part.mirrored(...)` reflects joints too: mirrored legs swing forward together. The twin
  hangs from the mirrored parent (`foot_r` on `leg_r` gives `foot_l` on `leg_l`; an unsided
  parent like `torso` is kept), or from `parent=`.
- Clip names become file names (`out/<name>/<clip>.gif`): letters, digits, `_` and `-`.

Which way a hinge turns follows the right-hand rule, like `rotate`:

| Axis | A positive angle turns | So |
| --- | --- | --- |
| `hinge("x")` | +Y toward +Z | a lid hinged at its **back** edge opens with `hinge("-x")` |
| `hinge("y")` | +X toward -Z | a door hinged at its low-x edge swings toward the viewer with negative angles |
| `hinge("z")` | +X toward +Y | a part hinged at its low-x end lifts with positive angles |

Put a hinge's pivot where the part really hinges: the edge it turns on, at the face it swings
away from. A pivot inside the neighboring part makes the part sweep through it (lint:
`motion_collision`).

```python
lid = Part("lid", lid_shape, parent="body", pivot=(7, 7, 0), joint=hinge("-x", (0, 120)))
drawer = Part("drawer", drawer_shape, parent="desk", joint=slide("+z", 6))
prop = Part("prop", blades, parent="plane", pivot=(10.5, 5.5, 20.5), joint=spin("z", rps=2))
clips = [
    clip("open", 0.6, {"lid": [0, 115, 105]}),  # overshoot, then settle
    clip("pull", 0.7, {"drawer": [0, 6]}),
    clip("spin", 1.0, {"prop": [0, 2]}, loop=True, ease="linear"),
]
```

### Free tracks and generators

Any part can move without a joint. Free tracks key one channel each, about the part's pivot
along the parent's axes; tracks on the same channel add up (a swing plus a shake):

```python
turn(part, axis, keys, ease=None)  # degrees about axis ("x", "-y", ...)
move(part, axis, keys, ease=None)  # voxels along axis
visible(part, keys)  # 1 shown, 0 hidden; always steps
```

A part's local motion is the joint's slide plus the moves, turned by `R_joint @ R_z @ R_y @
R_x`: free rotations x first, then y, then z, then the joint. Generators write the keys for
common motion:

| Call | Makes |
| --- | --- |
| `swing(part, axis, amount, phase=0, cycles=1, center=0)` | `center + amount·sin(2π(cycles·u + phase))` degrees; `phase=0.5` is the opposite limb |
| `bob(part, height, cycles=2, phase=0, axis="y")` | from rest to `height` voxels and back, `cycles` times (two steps per walk cycle); negative dips |
| `shake(part, amount, channel="rotate_z", rate=12, seed=0)` | seeded jitter, starting and ending at rest (recoil, impacts); follows `--seed` |
| `spin_track(part, turns, axis=None)` | constant-speed turns on a spin joint, or about `axis` on any part |

Curves are sampled 8 times per cycle; whole cycles close loops exactly. `keys` can also be a
function of `u` in [0, 1]: `TrackSpec(part, "rotate_z", lambda u: 30 * u, samples=9)`.

```python
walk = clip(
    "walk",
    0.8,
    [
        swing("leg_r", "x", 25),
        swing("leg_l", "x", 25, phase=0.5),
        swing("arm_r", "x", 20, phase=0.5),
        swing("arm_l", "x", 20),
        bob("torso", -0.4),
    ],  # dip as the legs spread so the feet stay planted
    loop=True,
    events=[(0.2, "footstep"), (0.6, "footstep")],
    travel=(0, 0, 19),  # in place; the engine moves it 19 voxels a second
)
flash = Part("flash", star_shape, parent="frame", hidden=True)
fire = clip(
    "fire",
    0.35,
    [visible("flash", [(0, 1), (0.06, 0)]), turn("frame", "x", [(0, 0), (0.05, -10), (0.35, 0)])],
)
```

- **Hidden parts** (`Part(..., hidden=True)`) start hidden, and so do their children. Contact
  sheets leave them out and name them (`HIDDEN: FLASH`); filmstrips show them while visible.
- **`travel`** is `(x, y, z)` voxels per second, for clips that walk in place.
- A clip needs at least one track, and at most one `joint`, `visible` or `frames` track per
  part (free `turn`/`move` tracks on one channel add up).
- Hidden parts and flipbook frames count for the declared size, palette and overlap like any
  part, and every frame is checked for floating pieces. A light on a hidden part keeps shining
  (glTF lights ignore scale); tie a muzzle-flash light to an event for the engine instead.
- Part names can't contain `@`: `.vox` files use it for flipbook frames.

### Flipbooks

```python
Part(name, frames=[shape, shape, ...], fps=8, ...)   # instead of a shape
frames(part, keys)                                   # a clip track picking frames, stepped
```

A flipbook part swaps whole shapes over time: flames, splashes, a muzzle flash. Frames share
the part's pivot, parent, joint and visibility; frame 0 is the rest look the contact sheet
shows. Without a `frames` track a flipbook loops on its own at `fps` (the build previews it as a
`<part>_frames` clip and exports that as a glTF animation); a clip's `frames(...)` track plays
chosen frames instead, like a two-frame flash played once:

```python
flame = Part("flame", frames=[flame_shape(p, k) for k in range(6)], fps=10, parent="torch")
flash = Part("flash", frames=[small_star, big_star], parent="frame", hidden=True)
fire = clip(
    "fire", 0.35, [visible("flash", [(0, 1), (0.06, 0)]), frames("flash", [(0, 0), (0.03, 1)])]
)
```

Make frames from one function with a seed or a parameter per frame, so they stay the same
flame. Every frame needs voxels; to blink, hide the part with `visible(...)` instead of an empty
frame. Once any clip plays a flipbook with `frames(...)`, it no longer loops on its own: other
clips show it at frame 0 (lint notes this as `flipbook_driven`). `cubist render --clip
<part>_frames` renders the loop like any clip. Lint checks palette, bounds and floating pieces on every frame, and notes frames that
change most of their voxels (`frame_pop`): small changes per frame read as motion.

### Glow, color cycles and lights

Glow and color cycles are set per project in `style.toml` (`[glow]`, `[cycles]`; see
`cubist style`), so every asset using those colors gets them. Glowing colors render nearly
unshaded in previews and export as emissive (in the default palette-texture export;
`--vertex-colors` has no emissive) and as emissive materials in the `.vox`, which MagicaVoxel
shows and engines can read. A cycling ramp's voxels step one shade along it each tick
(water, lava): a voxel painted shade k shows shade k+1, k+2, ... wrapping around, so paint
bands dark-to-light in the direction the water should flow. Every voxel of that ramp cycles,
in every part, so paint spray, foam and flashes from another ramp.

```python
light(position, color, range=8, intensity=1.0, flicker=0.0, part=None)
Model(parts, ..., lights={"flame": light((3.5, 14, 3.5), p.fire.light, range=96, flicker=0.35)})
```

A point light in a palette color, moving with `part` (default: the part with the nearest voxel).
It becomes a `KHR_lights_punctual` light in the `.glb`; `flicker` (0 steady to 1 wild) goes in
the manifest for the engine. Posed previews mark it with a dot.

`range` is where the light fades to nothing, in voxels, and is easy to set far too short:
think in metres and divide by `voxel_size_m`. Typical ranges, with voxels at 1/16 m:

| Light | Range | Voxels at 1/16 m |
| --- | --- | --- |
| Candle, glowing potion, wand tip | 2-3 m | 32-48 |
| Torch, sconce | 5-6 m | 80-96 |
| Lantern, lamp post, lit window | 6-10 m | 96-160 |
| Campfire, hearth, forge | 8-12 m | 128-192 |

Lint warns (`light_reach`) when a light reaches less than 2 m, a candle's reach, and, on a
grounded asset, when it can't reach the floor as far out as it is high (`range` under 1.4 times
its height). The floor is the ground line (`ground_y`): a building's door lamp is measured from the
street, not from its cellar's floor. A light below the ground line (in the cellar) is measured from
the asset's floor.

### Fracture and bursts

```python
fracture(shape, pieces=8, seed=0, min_voxels=None)  # -> list of voxel shapes
chunks(name, shape, pieces=8, seed=0, parent=None, min_voxels=None)  # -> parts name_0, ...
burst(parts, origin=None, speed=6, up=6, spin=540, gravity=40, rate=20, seed=0, start=0)
```

`fracture` splits a shape into about `pieces` face-connected chunks that keep their colors (no
slivers: tiny pieces merge into a neighbor). `chunks` makes them parts pivoted at their centers
of mass, so the intact asset is just the chunks together. `burst` throws them out from `origin`
(default their center at the floor) at about `speed` plus `up` voxels/s, tumbling at up to
`spin` degrees/s under `gravity`; rising chunks roll up off the floor and each comes to rest
where its lowest corner touches down. It's a baked clip of plain keys, so it plays anywhere;
the manifest has each chunk for engines that want real physics.

```python
pieces = chunks("stave", body, pieces=8, seed=2)
boom = clip(
    "explode",
    1.6,
    burst(pieces, speed=11, up=10, seed=1),
    events=[(0.0, "emit:smoke")],
    collisions=False,
)
return Model(parts=pieces, clips=[boom], emitters={"smoke": smoke})
```

`collisions=False` turns off the collision check for a clip whose parts are meant to pass
through each other, like debris. `start=` delays the blast for a wind-up; `rate` is keys per
second (the keys are lifted wherever the clip would play a chunk through the floor).

### Emitters and particles

```python
emitter(
    position,
    particle,
    direction="+y",
    rate=None,
    burst=None,
    life=1.0,
    speed=2.0,
    spread=20,
    gravity=0.0,
    grow=(1, 1),
    fade=True,
    seed=0,
    part=None,
)
Model(parts, ..., emitters={"smoke": emitter(...)})
```

An emitter sprays copies of a particle asset (an ordinary asset, tagged `particle`, like
`smoke_puff` or `spark`): `rate` per second always, and/or `burst` at once on each
`emit:<name>` clip event. `spread` is degrees around `direction`; `gravity` pulls down in
voxels/s² (negative rises, for smoke); `grow` is the size at birth and at death; `fade` shrinks
particles away over the last 30% of their life. It moves with `part` (default the nearest).

Cubist only describes and previews emitters: filmstrips and GIFs run a simple seeded
simulation and draw each particle's model from `out/<particle>/` (build the particle first;
until then previews draw plain cubes and lint warns). The engine plays them for real from the
manifest, which also lists the particle assets the asset needs. The `.glb` gets an empty node
per emitter, its +Z along `direction`. Lint warns when an emitter sprays straight into the
model (always-on emitters only: a burst may start inside, like an explosion) or a clip fires
an emitter that doesn't exist or has no `burst`.

### Oriented sockets

```python
socket(position, part=None, forward=None, up="+y")
```

A socket with a `forward` direction (muzzles, wand tips, grips) becomes a GLB node whose local
+Z points forward and +Y up; `up` is made perpendicular to `forward`. `(x, y, z)` and
`(x, y, z, part)` tuples still work for sockets without a direction.

A build of an asset with clips also writes `motion.png` (a filmstrip of every clip: iso frames
above an orthographic view chosen to show the motion, each labeled with its time) and a GIF per
clip. `cubist render out/<name> --clip open --at 0.3` draws a full contact sheet of one moment;
`--frames 12 --view right` gives a denser filmstrip, and `cubist diff` also writes
`motion-diff.png` (last build's filmstrip above this one's). Motion lint flags keys past a
joint's limits (error), loops that don't close, parts moving through each other, grounded
assets pushed through the floor (warnings), and how far each clip reaches beyond the rest box
(info).

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
| `line(p0, p1, color, connect=6, r=None)` | the voxels a segment passes through: `connect=6` face-connected (a 2-wide staircase on diagonals), `connect=26` one voxel per step (its voxels touch only at edges, which the floating check counts as detached); with `r`, a thick line |
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
| `a.carve(cutter, color=None)` | `a - cutter`, coloring the cut's walls and back (windows, arrow slits, eye sockets); one-voxel walls with outside air behind stay uncolored, so the color never shows through |

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
