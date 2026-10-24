# The manifest

`cubist export` writes `<name>.cubist.json` next to the `.glb` and `.vox`. It is the contract
between Cubist and a game engine: it holds everything needed to place an asset's parts, move
them, and play its clips. Some of that the `.glb` can't express, such as joint limits, the clip
keys as authored, and events. The rest the `.glb` repeats in its own terms. The JSON Schema is
[`cubist/schema/manifest-1.schema.json`](../cubist/schema/manifest-1.schema.json).

A build also writes `out/<name>/manifest.json` with the same content, minus `files`, so agents
can read it before exporting.

## Versioning

`cubist_manifest` is the version, currently `1`.

- New optional fields can be added without changing the version, so readers should ignore keys
  they don't know.
- A change to what an existing field means bumps the version.

## Units and axes

```json
"units": {"length": "voxel", "angle": "degree", "time": "second", "spin": "turn"}
```

- **Positions** are in voxels in the asset's own space. The origin is the bottom-left-back
  corner, Y is up, +Z is the front, and the frame is right-handed.
- **Metres:** multiply voxels by `voxel_size_m`.
- **Rotations** follow the right-hand rule: a positive angle about +X turns +Y toward +Z.
- **`glb_origin`** is the asset-space point that sits at the `.glb` origin. By default that is
  the center of the footprint at y = 0. To go from a manifest position to `.glb` scene metres,
  use `(p - glb_origin) * voxel_size_m`.

## Top level

| Field | Meaning |
| --- | --- |
| `name` | Asset name |
| `size` | Declared bounding box `[x, y, z]` in voxels, or `null` |
| `voxel_size_m` | Metres per voxel |
| `glb_origin` | See above |
| `parts` | Parts, parents before children |
| `sockets` | Named attachment points |
| `clips` | Named motion |
| `files` | The files written alongside, e.g. `{"glb": "chest.glb", "vox": "chest.vox"}` |

## Parts

```json
{"name": "lid", "parent": "body", "pivot": [7, 7, 0],
 "joint": {"type": "hinge", "axis": [-1, 0, 0], "limits": [0, 120]},
 "voxels": 372, "vox_model": "lid"}
```

| Field | Meaning |
| --- | --- |
| `pivot` | Where the part turns, in asset space. In the `.glb` this is the part node's origin. |
| `parent` | The part it hangs from, or `null` |
| `joint` | How it moves (below), or `null` for a part that doesn't move on its own |
| `frames` | A flipbook's `{count, fps, vox_models}`, or `null`. In the `.glb` the part node has a child node `<part>@<k>` per frame, frames other than 0 at scale 0; `frame` tracks become `STEP` scale channels on those nodes. Without a `frame` track it loops at `fps` (exported as a `<part>_frames` animation) |
| `hidden` | Starts hidden (a clip's `visible` track shows it); children of hidden parts are hidden too. In the `.glb` its node has scale 0, and `visible` tracks become `STEP` scale channels |
| `voxels` | Voxel count |
| `vox_model` | The model's name in the `.vox` scene graph |

In the `.glb`, a part node's rest translation is `pivot - parent.pivot` (metres), with no rest
rotation. Mesh vertices are relative to the pivot.

## Joints

The axis is a unit vector in the parent's frame. The rest pose (as modeled) is joint value 0.

| `type` | Value means | Range |
| --- | --- | --- |
| `hinge` | Degrees about `axis` through the pivot | `limits: [lo, hi]` |
| `slide` | Voxels along `axis` | `limits: [0, distance]` |
| `spin` | Turns about `axis` through the pivot | unlimited; `rps` is a default speed in turns per second, or `null` |

A part at joint value `v` has the local transform
`T(pivot - parent.pivot) · T(slide_offset) · R(axis, angle)`:

- a hinge uses `angle = v` degrees;
- a spin uses `angle = 360·v` degrees;
- a slide uses `slide_offset = axis · v` voxels.

An engine can drive joints directly, such as a door the player pushes, using the limits.

## Glow, color cycles, lights

```json
"glow": {"fire.light": 1.5},
"cycles": [{"name": "water", "ramp": "water", "fps": 6,
            "colors": ["water.dark", "water.1", "water.mid", "water.light"]}],
"lights": {"flame": {"position": [3.5, 14, 3.5], "part": "torch", "color": "fire.light",
                     "rgb": "#ffe066", "range": 10, "intensity": 1, "flicker": 0.35}}
```

- **`glow`**: the glowing colors this asset uses, with strengths. The `.glb` material has an
  emissive palette texture (glowing cells colored, others black), `emissiveFactor` 1 and, when a
  strength is above 1, `KHR_materials_emissive_strength`.
- **`cycles`**: each tick (`1/fps` s) every voxel of a listed color shows the next color in the
  list, wrapping around. Not in the `.glb`: with a 256×1 palette texture an engine can rotate
  those texels.
- **`lights`**: point lights in asset space; `range` in voxels; `flicker` 0..1 for the engine.
  The `.glb` has them as `KHR_lights_punctual` point lights on nodes under `part`.

## Emitters and particles

```json
"emitters": {"sparks": {"position": [1.5, 6.5, 17.5], "part": "frame", "particle": "spark",
                        "direction": [0, 0, 1], "rate": null, "burst": 10, "life": 0.3,
                        "speed": 24, "spread": 40, "gravity": 30, "grow": [1, 1],
                        "fade": true, "seed": 0}},
"particles": ["spark"]
```

- An emitter sprays the `particle` asset (its own `.glb`/manifest; `particles` lists every one
  this asset needs) from `position` along `direction`, within `spread` degrees: `rate` per
  second always, and `burst` at once whenever a clip event `emit:<name>` fires.
- Particles live `life` seconds, start at `speed` voxels/s (Cubist's previews vary it by
  ±25%), fall under `gravity` voxels/s² (negative rises), scale from `grow[0]` to `grow[1]`,
  and with `fade` shrink to nothing over the last 30% of their life.
- The `.glb` has an empty node per emitter under `part`, its local +Z along `direction`.
- Clips with `"collisions": false` are effects whose parts pass through each other (debris).

## Sockets

```json
"sockets": {"muzzle": {"position": [1.5, 6.5, 17], "part": "frame",
                       "forward": [0, 0, 1], "up": [0, 1, 0]}}
```

`forward` and `up` (or `null`) give an oriented socket's direction: in the `.glb` the socket
node's local +Z points `forward` and +Y `up`. `position` is in asset space at rest. `part` is the part the socket moves with. When the script
didn't name one, it is the part with the nearest voxel. In the `.glb` each socket is an empty
node under that part.

## Clips

```json
{"name": "open", "length": 0.6, "loop": false, "ease": "smooth",
 "tracks": [{"part": "lid", "channel": "joint",
             "times": [0, 0.3, 0.6], "values": [0, 115, 105], "ease": null}],
 "events": [[0.0, "creak"]]}
```

- A **track** keys one channel of one part:
  - `joint`: the part's joint value, at most one per part;
  - `rotate_x|y|z`: degrees about the pivot along the parent's axes;
  - `move_x|y|z`: voxels along the parent's axes;
  - `visible`: 1 shown, 0 hidden, always stepped;
  - `frame`: which frame of a flipbook part shows, always stepped.

  Free tracks on the same channel add up. A part's local transform is
  `T(pivot - parent.pivot) · T(slide + moves) · R_joint · R_z · R_y · R_x`.
- A track's `ease` of `null` means the clip's `ease`.
- **Events** are `[time, name]` markers for the engine (sounds, effects, gameplay). Cubist gives
  them no meaning.
- `travel` (or `null`) is `[x, y, z]` voxels per second: how fast an in-place clip such as a
  walk moves the asset through the world. The `.glb` animation carries it in metres per second
  as `extras.travel`.
- `loop` means the clip repeats. Its last pose should equal its first; Cubist lints for this.
- Parts without tracks stay at rest. A part also moves when one of its ancestors moves.

### Easing

Between keys `k` and `k+1`, with `s = (t - t_k) / (t_{k+1} - t_k)`:

- **`step`** holds `v_k` until `t_{k+1}`.
- **`linear`** gives `v_k + s·(v_{k+1} - v_k)`.
- **`smooth`** is a monotone cubic Hermite (PCHIP):

  ```text
  v(s) = h00·v_k + h10·h·m_k + h01·v_{k+1} + h11·h·m_{k+1}
  h00 = 2s³ - 3s² + 1   h10 = s³ - 2s² + s   h01 = -2s³ + 3s²   h11 = s³ - s²
  ```

  `h` is the segment length in seconds. The tangent `m_k` at an interior key is 0 when the
  neighboring slopes `d_{k-1}`, `d_k` differ in sign or either is 0. Otherwise it is
  `(w1 + w2) / (w1/d_{k-1} + w2/d_k)`, with `w1 = 2h_k + h_{k-1}` and `w2 = h_k + 2h_{k-1}`.
  End tangents are 0, so motion eases in and out. In a looping clip that has keys at 0 and at
  `length`, the end tangents use the same formula across the wrap (last segment, then first).
  Smooth motion never overshoots its keys.

Before the first key a track holds the first value; after the last key, the last value.

### In the `.glb`

Each clip is a glTF animation with the same name, and `extras: {"loop", "length"}`. Tracks are
sampled 30 times a second (more where a part turns fast, so no step between keys turns 60° or
more) with `LINEAR` interpolation. A part whose tracks are all `step` keeps only its keys, plus
keys at 0 and `length` so every animation lasts the whole clip, with `STEP`. A step track mixed
with smooth ones gets an extra key 1 ms before each jump. Node translations are full local
translations: rest offset plus slide. An engine can play the `.glb` animations as they are, or
rebuild motion from the authored keys above.

## Hand edits

For animated assets the script is the only source. If the `.vox` in `out/` no longer matches its
build, lint reports a `hand_edit` error and export refuses. `--force` exports the edited voxels
with the build's motion. Static assets can be touched up by hand as before.
