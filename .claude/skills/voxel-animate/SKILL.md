---
name: voxel-animate
description: Use when the user wants a Cubist voxel asset to move or change over time - open, close, slide, spin, walk, idle, recoil, flourish, wave, flicker - or gives feedback on an existing clip's timing or motion.
---

# Animate a voxel asset

Motion is code in the asset script, next to the shapes: joints on parts, clips on the model.
Never key motion by hand frame by frame; use generators. Work from the project directory with
`uv run`. The DSL is in `docs/dsl.md` ("Joints and clips", "Free tracks and generators").

1. **Read the asset and the request.** Read `assets/<name>.py`, its brief, and
   `out/<name>/contact.png`. Decide which parts move and how:
   - opens, closes or swings on an edge: `hinge(axis, limits)`
   - slides: `slide(direction, distance)`
   - spins: `spin(axis)` with `spin_track` or linear keys
   - limbs and anything else: free tracks, no joint needed (`turn`, `move`, `swing`, `bob`,
     `shake`)
   - appears for a moment (flash, spark): `Part(..., hidden=True)` and `visible(...)`
   - changes shape (flame, splash): a flipbook, `Part(..., frames=[...], fps=)`, built from one
     function with a different seed per frame; `frames(...)` plays chosen frames in a clip
   - gives off light: a `[glow]` color in `style.toml` and a `light(...)` with some flicker
2. **Split the model into parts that move separately**, each with its pivot where it really
   turns: the hinge edge at the face it swings away from, the hip, the shoulder, the grip.
   Moving parts must be separate parts. Children ride along with their parents.
3. **Write the clips.**
   - Prefer generators: `swing("leg_l", "x", 25, phase=0.5)` beats nine keys.
   - Overshoot by adding a key past the target, then settle: `[0, 115, 105]`.
   - Loops end where they start; spins end on whole turns.
   - Add `events` for sounds and effects, and `travel` for walks (in place, voxels per second).
4. **Build and look.** Run `cubist build assets/<name>.py`, then Read `out/<name>/motion.png`:
   iso frames above a side view picked to show the motion, each with its time.
   - For a moment that matters, run `cubist render out/<name> --clip NAME --at T`: a full
     contact sheet with rulers.
   - For more frames, add `--frames 12 --view right`.
   - Then read the motion lint in the build output.
5. **Critique against this checklist:**
   - Does the motion read at game distance? Big, clear arcs beat small wiggles.
   - Is the timing right? Fast actions are short (0.2–0.4 s); a hinge settles after it
     overshoots; anticipation (a wind-up) where it matters.
   - Do loops hitch (`loop_closure`)?
   - Does anything pass through anything (`motion_collision`)? Feet through the floor
     (`motion_floor`)?
   - Do joint limits hold (`joint_limits`)?
   - Does it match the brief and the user's words?
6. **Revise and rebuild.** `cubist diff out/<name>` also writes `motion-diff.png` (the last
   filmstrip above this one). Stop after 5 iterations and say honestly what still falls short.
7. **Report** with the filmstrip and the GIF paths (`out/<name>/<clip>.gif`), so the user can
   watch it.

Gotchas:
- Hinges follow the right-hand rule. A lid hinged at its back edge opens with `hinge("-x")`, and
  a door hinged at its low-x edge swings toward the viewer with negative angles.
- If a hinge collides, flip its axis before anything else.
- Free rotations compose x, then y, then z about the parent's axes, and a joint goes on top.
- A walking body that bobs *down* as the legs spread keeps the feet planted. Bobbing up makes
  the asset float, and too far down sinks the feet (`motion_floor`).
