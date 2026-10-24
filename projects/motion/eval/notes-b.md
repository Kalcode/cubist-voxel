# Motion eval, agent B

Assets: `assets/jewelry_box.py`, `assets/shop_window.py`, `assets/powder_keg.py` (+
`assets/powder_keg_smoke.py`). All four build with 0 errors, 0 warnings. Iterations are distinct
script hashes in `out/<name>/history.jsonl`, the same count `cubist eval` uses.

## jewelry_box: 4 iterations, self-score 4

Gold feet, gold corner trim, a clasp and hasp, and a lid hinged at the back edge
(`hinge("-x", (0, 120))`, pivot `(5, 5, 0)`). Inside: a red velvet cushion, a blue gem standing
proud of the rim with an emissive `glow` highlight, and a gold ring. `open` (0.6 s) pops to 118,
swings back to 98, then 108, and settles at 105. A hidden four-point `glint` part in `glow.light`
flashes twice beside the gem. `close` (0.3 s, linear) is hand-keyed to accelerate, slams shut
at 0.23 s, rebounds 6° and settles. Events: click, sparkle, snap.

Still short:
- The lid top is one voxel thick, so the gold emblem also shows on the lid's underside when it's
  open, and the lid can't have a velvet lining.
- The sparkle is a 3x3 plus of cubes, so in iso it reads as a small cream blob rather than a
  star. The 10x8x8 size leaves no room for a bigger one without overlapping the closed lid.
- The bounce is visible in the side view but small in iso.

## shop_window: 4 iterations, self-score 4

The frame has a sill on an apron and a stepped cornice. There are two sashes of dark glass, each
with two streaks of shine. `pane_r` is fixed in the front track (z = 2). `pane_l` rides in the
back track (z = 1) on `slide("+x", 6)` and ends exactly behind `pane_r`. A flush metal pull marks
the pane that moves. `open` takes 0.8 s. `close` takes 0.6 s: it hits the jamb at 0.4 s and
rebounds one voxel. No collisions, and motion stays inside the rest box.

Still short:
- It reads as a generic window, not a shop window. There's no sign, display or awning; the 4-deep
  box leaves little room.
- The panes are one voxel thick (INFO `thin`, 144 voxels). That's unavoidable with two tracks in
  4 voxels of depth.
- There's no glass color in the palette, so the glass uses the `blue` ramp.

## powder_keg: 4 iterations, self-score 4 (powder_keg_smoke: 2 iterations)

The keg is a hollow lathe with stave seams, two iron hoops, a recessed head inside a chime rim,
and a gray fuse cord with a burning end. It's fractured with `chunks("stave", ..., pieces=10)`.
`explode` (1.6 s) is
`burst(speed=11, up=20, spin=720, rate=60)` with `collisions=False`. The staves fly about 5
voxels up and scatter, and all land by about 1.0 s. A hidden two-frame fireball flipbook inside
the hollow shows through the gaps for 0.2 s. The `emit:smoke` event bursts 16 `powder_keg_smoke`
puffs from the open top. The puffs are 4x4x4 balls; at 3x3x3 the ball becomes a spiky plus.

Still short:
- No anticipation. `burst` always starts at t = 0, so there's no wind-up (fuse sputter, keg
  tremble) in the same clip.
- In the preview the smoke is a cluster of gray cubes that reads like boulders for about 0.3 s
  before it rises and thins.
- The fireball is only visible in a posed render (`--at 0.08`); the 8-frame filmstrip never
  catches it.

## Bugs

1. **`burst` output trips `motion_floor`.** Raising `up` 11→20 and `spin` 600→720 gave
   `WARNING motion_floor: clip 'explode' pushes 'stave_0' 0.3 voxels through the floor at t=0.02 s;
   lift it (bob the body less, swing less, or move the pivot)`. Cause, from
   `cubist/motion/generators.py`: `burst` bakes keys at `rate=20`/s and lifts chunks off the floor
   only at those keys. The lint samples more densely, so a corner turning about 36° between keys
   sags below the straight-line interpolation. Workaround: `rate=60`. Either `burst` should
   sample densely enough for its `spin`, or it should check the floor between keys.
2. **Following the barrel example leads to `emitter_blocked`.** The approved demo barrel puts its
   emitter inside the barrel (`(6, 8, 6)`, `part="stave_0"`), and its stored `build.json` shows no
   warning. My keg did the same at `(5, 6, 5)` and got
   `WARNING emitter_blocked: emitter 'smoke' sprays into the model ('stave_0') right away; move it
   to the surface or turn its direction outward`. For an `emit:` burst fired at t = 0 in a clip
   that throws the chunks away, spraying from inside is the point. I moved the emitter above the
   rim. Either the example is stale or the check should allow this case.

## Confusing errors or docs

- **The `motion_floor` hint doesn't fit `burst` clips.** "bob the body less, swing less, or move
  the pivot" names no knob a burst has.
- **Failed builds count as iterations.** An uncolored `box` in a union (a typo-level mistake) cost
  one of the five, because history records failed builds and `eval` counts their hashes.
- **The uncolored-box error.** I unioned uncolored boxes and then called `.paint(color)` on the
  result: `part 'body': box (jewelry_box.py:9), box (jewelry_box.py:9) has no color: pass
  color=p.<ramp>.<shade>. ...`. The message is clear, but nothing in the docs says paint can't
  color shapes that were uncolored when unioned.
- **`cubist examples search` from `projects/motion` prints `[]`.** There's no hint that the
  library is per-project and empty here. CLAUDE.md step 1 sends you there.
- **`cubist build` dumps the whole manifest as JSON.** I had to `grep '"summary"'` to find the
  lint.
- **`motion_bounds` reports `3 at -y`** for `explode`, while `motion_floor` passes. It's unclear
  whether chunks really go under the floor.
- **The `water` ramp is a trap for glass.** It's the obvious "glass-ish" color, but it's in
  `[cycles]`, so it would animate.

## Missing docs

- **How to mix a joint track with `visible`/`frames`/`turn` in one clip.** The dict form only does
  joints, and the list form has no joint helper. `TrackSpec(part, "joint", keys)` works and is
  exported, but the docs only show `TrackSpec` with a function and `rotate_z`.
- **What `burst(rate=)` means.** It's keys per second, from the source; the docs list it but don't
  say.
- **Whether hidden parts and non-zero flipbook frames count for overlap and floating checks.** I
  designed around both: a glint in the lid's hollow, a fireball touching the keg floor.
- **Why the frame you see when an emitter fires might not be the one it moves with.** The
  `part=` default is "nearest". For an explosion, which chunk is nearest decides where a
  `rate` emitter would go.

## Features wished for

- `burst(..., start=0.3)`, or a way to offset any track list in time, for a wind-up before the
  throw.
- An accelerating ease (`ease="in"`) for slams and falls. I hand-keyed piecewise-linear
  segments instead.
- A shorter default `cubist build` output: lint summary plus output paths.
- A glass ramp in the motion palette, or a doc note on what to use.
- A smoother particle preview, so a smoke burst doesn't look like rocks in filmstrips.
