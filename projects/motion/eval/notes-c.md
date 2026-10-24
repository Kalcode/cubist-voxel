# Motion eval, agent C: dresser, flintlock, fountain

All three assets (and the `flintlock_smoke` particle) build with 0 lint errors and 0 warnings.
Iterations are distinct script versions in `out/<name>/history.jsonl`.

## dresser: 2 iterations, self-score 4/5

Solid carcass on four feet. The fixed top and bottom drawer fronts stand one voxel proud, with
gold knobs. The middle drawer is an open tray that exactly fills a pocket cut through the carcass,
on `slide("+z", 5)`. `open` (0.55 s, eases out) and `close` (0.35 s, slow then fast into the stop,
`thud` event). No `motion_collision`. Iteration 2 fixed a dark band showing through the sides and
back (see bugs) and toned down blotchy dither.

Still short:
- The sides and back are plain slabs: no panels or trim.
- `darken_crevices` darkens the top drawer's knobs under the overhang (gold.1 against gold.mid), so
  one pair of knobs differs from the others.
- The drawer stops dead at full travel with no bump or settle. Overshoot can't go past the slide's
  limit, and I didn't author a bounce inside the range.

## flintlock: 3 iterations, self-score 3/5

Faces +Z, 3 wide:
- Wood stock running to a brass forend cap at z=13.
- Grip slanting down and back to a brass butt cap; brass trigger guard and a metal trigger.
- Half-round barrel with a light spine, a dark bored muzzle band and a brass front sight.
- Hammer on `hinge("x", (0, 60))` and frizzen on `hinge("x", (0, 70))`.
- Hidden two-frame flash flipbook, a `flintlock_smoke` burst emitter on `emit:smoke`, and oriented
  `grip` and `muzzle` sockets.

`fire` (0.4 s): the hammer snaps forward in 0.03 s with a small settle, and the frizzen flips off
the pan. The flash shows from 0.04 to 0.1 s, big star then small. The muzzle kicks -12° and
recovers. Events are `hammer` and `fire`, plus `emit:smoke`. `cock` returns the hammer and the
frizzen. I worked out the hammer's arc by hand so its jaws stop one voxel short of the frizzen; no
collisions. Iteration 1 had the barrel dominating and a stubby stock. Iteration 2 made the barrel
half-round and ran the wood to the muzzle. Iteration 3 cut barrel glare and added the sight.

Still short:
- The grip is short (3 rows below the stock line). In 9 rows of height, a longer grip would cost
  hammer height.
- The hammer is a 1-voxel-wide "T" and the frizzen a plain post. At game distance they read as
  bits on top, not as a lock.
- The flash is one flat layer (z=15). It has to fit inside the declared 16-deep box, so the barrel
  ends at z=14.
- In the 0.4 s preview the smoke mostly piles up at the muzzle. You can't see it drift and fade,
  because the filmstrip ends with the clip.

## fountain: 5 iterations (the cap), self-score 3/5

The basin is a masonry wall under a wider, lighter rim. The water is painted in one-voxel rings by
radius, darkest first going outward, so the style's water cycle sends ripples out from the spout.
The spout is a plinth, shaft, flared capital and brass nozzle. The splash is a 4-frame flipbook at
8 fps, with no clip:
- a jet column shaded by height, so the cycle climbs it
- a white crown whose side bits swirl round
- eight arcs of four drops, each a quarter step further along a parabola every frame, so the loop
  is seamless; drops turn from white spray to water shades as they fall
- landing foam that pops on alternate arcs
History:
- Iteration 1 had a tazza bowl with four overflow streams, which read as blue corner pillars, and
  a blob splash.
- Iteration 2 had arcs in water.light. The cycle turned them all navy every third tick (dark bars).
- Iteration 3 used blue.light, which is off-hue against the teal water.
- Iteration 4 had all-white drops, which looked like popcorn.
- Iteration 5 used white spray turning into staggered water shades.

Still short:
- The crown is a flat white plus-shaped plate.
- The drops are floating cubes, and the water.dark ones read as holes.
- `frame_pop` notes that most voxels change every frame, so the splash flickers more than it
  flows.
- The jet is mostly hidden under the crown.
- The water is visible mainly in the outer ring.

## Bugs

- **`render --clip` can't see preview clips.** `uv run cubist render out/fountain --clip
  splash_frames --at 0` gives `error: no clip 'splash_frames' in fountain; its clips: none (add
  clips=[...] to the Model)`. `--clip splash_frames --frames 6 --view top` and `--clip colors` fail
  the same way. motion.png, the GIF name (`splash_frames.gif`) and docs/dsl.md all name the
  flipbook's `<part>_frames` preview clip, but render only knows authored clips. The error's advice
  ("add clips") is misleading. As a result, I couldn't inspect a color-cycle or flipbook moment at
  full size.
- **`carve(color=...)` shows through one-voxel walls.** `box(...).carve(pocket,
  color=p.wood.dark)` on a carcass with 1-thick sides and back painted the outside faces dark (a
  band on the right and back views). `darken_crevices` is documented to skip one-voxel walls "so
  nothing shows through", but `carve` doesn't. I worked around it with a plain subtract and a
  region paint of the pocket's floor and ceiling.

## Confusing errors or docs

- **Color cycle direction.** The docs say "step one shade along it each tick" but not which way. I
  read `cubist/core/style.py` (`cycle_table`): a voxel painted shade k shows shade k+tick. So you
  paint rings `p.water[-k % 3]` to flow outward, and `p.water[y % 3]` to flow down.
- **Cycles hit every voxel of the ramp, in every part,** including flipbook spray and flashes. The
  docs don't warn that something meant to stay light (a drop, a spray) will go dark every n-th
  tick. There is also no non-cycling water-blue in this palette.
- **Multi-key `smooth` easing.** The docs don't say whether a middle key stops the motion or passes
  through it. I read `motion/pose.py`: it's monotone cubic, so keys in the same direction don't
  stop.
- **Joint tracks in list-form clips need `TrackSpec(part, "joint", keys)`.** That form only
  appears in the blaster example. docs/dsl.md shows joints only in the dict form, and free tracks
  only through `turn` / `move` / `visible` / `frames`. The "joint" channel name isn't documented,
  and there is no `joint(part, keys)` helper. You need this as soon as a clip mixes a hinge with a
  free `turn`.
- **The color-cycle preview's placement is undocumented.** With a flipbook present, it is folded
  into the `splash_frames` clip; the pond example has a separate `COLORS` clip only because it has
  nothing else.
- **`cubist examples search` returns `[]` in projects/motion,** which has no example library, with
  no hint to pass `--project ../demo`.

## Missing docs

- Whether hidden parts and flipbook frames count against `size`. I assumed they do and ended the
  barrel at z=14 so the flash fits.
- How a `light(...)` on a hidden part behaves: is it off while the part is hidden? I wanted a
  muzzle-flash light and skipped it.
- That the filmstrip and GIF scale change between builds when emitter particles widen the bounds
  (6 px to 4 px per voxel). This makes `cubist diff` harder to compare by eye.

## Features I wished for

- `render --clip` for preview clips (flipbook loops, color cycles), with `--at` and `--frames`.
- A per-part opt-out from color cycles, e.g. `Part(..., cycles=False)`, for spray, foam and
  flashes.
- A quick motion check that reports collisions and a part's posed voxel positions at a time
  without counting as an iteration. I computed the hammer and frizzen arcs by hand.
- An emitter preview that runs past the clip's end (`--tail 1.0`), so smoke can be seen drifting
  and fading.
- A `joint(part, keys)` helper next to `turn` and `move`.
- A "hit the stop" generator for slides and hinges (a bump and settle within the limit).
- A way to mark `frame_pop` as intended for droplet flipbooks.
