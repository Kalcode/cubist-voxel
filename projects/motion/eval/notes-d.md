# Motion eval notes: agent D

Assets: `assets/mage_staff.py` (+ `mage_staff_spark.py`), `assets/twin_prop_plane.py`,
`assets/forge.py` (+ `forge_smoke.py`). All build with 0 errors, 0 warnings (infos only: thin, hidden,
motion_bounds).

## mage_staff: 6 iterations (one over the cap), self-score 3/5

Gnarled shaft (capsules through wobbling points, knots, a spiral bark seam, iron ferrule), gold collar
and cup, four root prongs with bone talons gripping a glow-ramp orb. The orb is a 4-frame flipbook.
Clips: `idle` (1.6 s loop, orb pulse) and `cast` (1.0 s: rise and wind back to -30 deg, thrust to
+38 deg and 3 voxels forward in 0.12 s, overshoot, settle). `cast` and `emit:sparks` fire at 0.5 s, an
18-spark burst of `mage_staff_spark`, and there is a light on the orb.

Falls short:
- 6 iterations. I didn't know every distinct script hash counts until I read `history.jsonl`. Two of
  the six were lint fix-ups (capsule end below y=0, floating prongs).
- The orb is small (15 voxels in a 5-wide box), and the claw hides part of it in iso. The idle pulse
  barely shows: glow colors render almost unshaded, so dim and bright shades look alike.
- The thrust is mostly a rotation about the grip, so the butt swings back as the head goes forward.
  A real jab would move forward more.
- One comment ("Pulses on its own") is stale. I left it as built so the script hash matches the last
  build.

## twin_prop_plane: 2 iterations, self-score 4/5

Fuselage from intersected side and top extrusions, low wing (2 voxels thick inboard), lathed nacelles,
tail, tricycle gear, cheatline and windows. Each prop is its own part pivoted on its hub with
`spin("z")`. `spin` is a 0.5 s linear loop: `spin_track` +1 on `prop_r` and -1 on `prop_l` (tops sweep
outboard). The front-view filmstrip shows mirror-image blades in every frame, and the loop closes.

Falls short: the windshield is a flat patch on top. Blades and stabilizer are 1 voxel thick (thin
info). The props are plain 2-blade, with no blur disc for real speed.

## forge: 4 iterations, self-score 4/5

Masonry hearth with a recessed bed of dithered glowing coals, an ash pit with embers, a back wall, a
stepped hood with an iron band, and a chimney with a cap and sooty flue. The fire is a 6-frame
flipbook at 10 fps (4 displaced cone tongues, one seed per frame). Light flicker is 0.5, and an
always-on emitter (rate 3) puffs `forge_smoke`. Also an iron bar heating in the coals with a glowing
end, and a hammer on the rim.

Falls short: the hood reads as a stepped "wedding cake". From high angles the hood hides part of the
fire. The hammer is a small dark block.

## Bugs

1. **`render --slices` shows the wrong flipbook frame.** For mage_staff, the contact legend listed
   `GLOW.DARK 5` (frame 0), but `cubist render out/mage_staff --slices y=17..22` had no glow.dark in its
   legend, only glow.mid and glow.light (a later frame's colors). It looks like all frame models in
   the .vox (`orb`, `orb@1`..`orb@3`) are layered into the slice and the last one wins.
2. **Can't render the auto flipbook loop.** `cubist render out/forge --clip fire_frames --at 0.15` fails
   with `Error: no clip 'fire_frames' in forge; its clips: none (add clips=[...] to the Model)`. Yet
   motion.png labels that loop `FIRE_FRAMES` and the build writes `fire_frames.gif`. There is no way to
   pose it or get more frames.
3. **The auto `<part>_frames` loop vanishes silently.** Once any clip has a `frames(...)` track for a
   part, its auto loop is gone: no preview, no manifest entry, no glTF animation. The docs say "Without
   a frames track a flipbook loops on its own", which I read as per clip. Workaround: an explicit
   looping `idle` clip.
4. **Some failed builds aren't logged.** A build that fails before the asset is known (`error: forge.py
   defines no @asset functions`; the cause was my own bad edit) isn't added to `history.jsonl`. The
   `_append_history` docstring says "failed attempts included". Harmless.

## Confusing errors or docs

- **Iteration counting.** CLAUDE.md says "stop after 5 iterations", but only `cubist/evaluation.py`
  says an iteration is a distinct script hash. So a lint fix or a comment-only edit counts, and the
  build output never says "iteration N".
- **`region` as a cutter.** `region` is listed under Selections, and the docs say "use a shape to cut".
  I swapped `wing - region(...)` for a box before building, without knowing whether it would have
  failed.
- **`connect=26` and floating.** `line`/`polyline` with `connect=26` diagonals touch only at edges, so a
  claw built that way got `floating: 8 piece(s) not attached`. The floating check wants face contact,
  and the docs don't say so. `connect=6` fixes it but makes 2-wide staircases.
- **`paint_faces("bottom")` on plates.** On a model with 1-voxel plates (wings, stabilizer) it paints
  the whole plate. Obvious afterwards, but it cost an iteration.
- **`carve(..., color=)` on a thin lip.** A recess with a 1-voxel front lip colors the whole lip (every
  lip voxel counts as a cut wall), which made a black bar across the hearth front.
- **`masonry` overwrites earlier paint.** With no `where=`, it wipes `carve` colors. I needed
  `where=colored(p.stone.mid)`.

## Missing docs

- Which flipbook frame `--slices`, `inspect` and lint show.
- How many frames the filmstrip samples and at what spacing. This decides whether a 2-blade prop's
  direction reads: 1 turn per clip gives 45 deg steps, while 2 turns per clip (like the biplane
  example) gives ambiguous 90 deg steps.
- Whether `Part.mirrored` flips a z spin's direction. The docs only cover mirrored legs.

## Features wished for

- "iteration N of 5" in the build output.
- `render --clip <part>_frames` for auto flipbook loops.
- A separate idle frame list for a flipbook (e.g. `loop_frames=[0, 1, 2, 1]`), so a frame can be kept
  for clips only (a "blazing" orb just for the cast).
- A glow-strength key in clips (or a per-frame emissive boost), so a glow pulse reads even though glow
  colors are nearly unshaded.
- `build --frames N` to set motion.png density.
