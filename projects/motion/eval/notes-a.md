# Motion eval: agent A

Assets: `wall_sconce`, `cellar_hatch` (and its particle `cellar_hatch_dust`), `kobold`. All
three end at 0 lint errors and 0 warnings. Infos only: `thin`, `hidden`, `motion_bounds`.

## wall_sconce: 5 iterations (1 was a failed build), self-score 3/5

Iron shield plate with a raised rim and rivets, an arm and brace out to a collar, a wooden torch
with a leather wrap and an iron cup. The flame is a 6-frame flipbook at 10 fps, made from one
function: a teardrop plus a side tongue that switches sides, displaced with a new seed each
frame. The core is pale yellow, the rim is orange (two gradients: one for the core, one masked
to the outside), and the tips are red-orange. There is a flickering light at the flame
(flicker 0.4) and `fire`/`glow` glow from the style.

Still falls short:
- The flame is small and blocky: about 5x6 voxels at 16 tall. On some frames the tip is a
  single-voxel column that reads as a stick.
- Most of the plate is hidden behind the torch in the front view. The brace is barely visible.
- I spent one of the five iterations on a Python error (bug 1), so the final flame pass never
  got a follow-up.

## cellar_hatch: 5 iterations (1 was a failed build), self-score 4/5

Stone masonry frame flush with the floor at y = 3, around a dark shaft. The hatch is two voxels
thick: five planks, two iron strap hinges that continue onto the frame, a flat ring pull, and a
Z-batten underside that shows when it is open. It uses `hinge("-x", (0, 120))` with the pivot
at the top-back edge.
- `open` (1.1 s): a tug that rattles it, a heave past vertical to 118°, then it settles at 106°.
- `close` (0.65 s, linear): pushed back over vertical, falls faster and faster, slams at 0.5 s
  with a 4° bounce. A `slam` event and `emit:dust` fire a dust burst at the front gap.

Still falls short:
- Nothing visibly holds the hatch at 106° (no chain or stop). "Resting past vertical" is
  implied, not shown.
- The underside brace is a jaggy staircase.
- The ring pull is a solid 4x4 square ring and does not flop with gravity.
- The 0.08 s tug falls between the 8 filmstrip frames, so you only see it in the GIF.

## kobold: 5 iterations, self-score 3/5

Rust-red kobold:
- Head: rounded skull, a 4-long snout, a pale hinged lower jaw with fangs, gold eyes,
  swept-back bone horns.
- Body: leather vest open over a straw belly, a belt, clawed hands and feet.
- Tail: two segments (`tail`, `tail_tip`) sweeping down, back and out to +x, with a
  face-connected curl at the end.
- Feet are separate parts that counter-rotate, so the soles stay flat.

Clips:
- `walk` (0.7 s loop): legs ±28°, arms ±24° in the opposite phase, torso bob −0.45, a head sway,
  and a tail lash with the tip lagging (phase −0.12). Footsteps at 0.175 and 0.525 s; travel
  (0, 0, 10.7), worked out from how far the ankles sweep.
- `idle` (2.4 s loop): looks left and right, the tail sways, and the jaw drops 25° to hiss at
  1.45 s.

Still falls short:
- From the front the silhouette is a box: the 6-wide torso plus 2-wide arms fill x 1–11, with
  no gaps or taper.
- With a 10-deep bounding box the tail can't trail far behind. It reads as long only in the iso
  view, because it sweeps sideways, and its tip is a 1-voxel staircase.
- No knees and no foot lift. Both feet are flat at mid-stance, so the walk reads as a stiff
  scurry. Lifting a leg with `move` pushed it up into the torso (`motion_collision`).
- The horns are still staircased. The face reads in iso but is tiny from the front.

## Bugs hit

1. **`~shape` in a selection crashes with an unhelpful error.**
   `where=~cylinder(r=1.45, h=8).at(C, 9, TZ)` gave
   `TypeError: bad operand type for unary ~: 'Transform'`. The DSL docs say a shape works
   anywhere a selection does and that selections combine with `~`, but a shape needs
   `~inside(shape)`. The error should say so.
2. **Selection passed to `inside()`: error with no script location.**
   `inside(band("z", 3, 2) | band("z", 9, 2) | line(...))` gave
   `AttributeError: '_Combine' object has no attribute 'is_sdf'`. The traceback had no script
   line, because selections are evaluated lazily and `_script_traceback` keeps only frames from
   the script. I had to guess which line it was. The fix was
   `band(...) | band(...) | inside(line(...))`.
3. **`Part.mirrored()` keeps the original's `parent`.**
   `Part("foot_r", foot, parent="leg_r", ...).mirrored("foot_l", about_x=6)` gives a `foot_l`
   that is a child of `leg_r`. Its world angle came out at 56° instead of 0°, the foot pierced
   the floor, and lint said only
   `motion_floor: clip 'walk' pushes 'foot_l' 2.0 voxels through the floor at t=0.17 s`. This is
   not documented, there is no `parent=` argument, and nothing warns that a `_l` part hangs off
   an `_r` part. The workaround was to build `foot_l` by hand with `foot.reflect("x", 6)` and
   `parent="leg_l"`. I found the cause only by reading `cubist/core/part.py` and evaluating
   `world_matrices` in a scratch script.
4. **Failed builds count as iterations.** This is by design (the `runner.py` docstring says
   "failed attempts included"), but CLAUDE.md and the skill don't say so. Two of my 15
   iterations went on Python exceptions that rendered nothing.

## Confusing errors or docs

- `cubist build` prints about 200 lines of JSON (sha256s, palette indices), with the lint
  summary at the very end. I piped every build through Python to pull out voxels, parts and
  lint. A short human or agent summary by default, with `--json` on request, would help.
- `motion_bounds` reported "walk reaches ... 1 at -y" while `motion_floor` was clean. Is
  something below the floor or not? It is probably rotated voxel corners, but the two checks
  disagree on the page.
- `chamfer(1)` on a 1-thick plate ate its edge columns: a 5-wide plate came out 3 wide.
  `pyramid((5, 3, 1))` also narrows in z, so a 1-deep pyramid is almost empty. Both match the
  docs if you read them closely, but they are easy traps for flat parts.
- `motion_collision` on the leg lift said "torso through leg_l". Correct, but no hint that
  translating a child limb up is what caused it.

## Missing docs

- How to set `travel` from a walk: stride per step = 2·L·sin(amplitude), with L from the hip to
  where the foot sits. I first used L = 5 (hip to sole) and got 13.4; the ankle radius (L = 4)
  gives 10.7. A worked example in "Free tracks and generators" would help.
- How to keep feet flat or planted when feet stick out in front of the leg: counter-rotating
  child feet. The goblin example has no feet, so it never hits this.
- That `mirrored()` keeps `parent` (bug 3), and that shapes need `inside()` under `~`.
- Whether hidden parts must stay inside the declared `size`. I dropped a tongue idea because I
  couldn't tell.

## Features wished for

- `cubist check <script>`: evaluate the script without rendering or counting an iteration.
  Or don't count builds that raise.
- `mirrored(..., parent=...)`, or automatic `_r`/`_l` remapping of `parent`.
- A swing-phase-only `lift` generator (a half-rectified sine), plus some two-bone or knee help,
  so a foot can lift without translating the whole leg into the torso.
- A tail or chain generator: swing N chained parts with an increasing phase lag.
- Emitters along a line or edge (dust from a slot, steam from a seam), not only a point.
- `--slices` on a posed frame (`--clip walk --at 0.17 --slices y=0..2`) to check floor contact
  exactly.
