# Cubist

Agent-native voxel asset authoring. `docs/design.md` is the design of record; read it before
changing behavior, and update it when a decision changes.

## Authoring assets

When asked to make an asset, write a script and look at what it builds. Never place voxels one
at a time. Run commands from the project directory (e.g. `projects/demo`), via `uv run`.

1. Read the brief (`briefs/<name>.toml`; `cubist new <name> --category prop` scaffolds one and a
   script stub), the style bible (`cubist style`), and the 2–3 closest approved examples
   (`cubist examples search --tags ... --category ...`): study their scripts and contact sheets.
2. Write `assets/<name>.py` and build: `cubist build assets/<name>.py`.
3. Inspect `out/<name>/contact.png` and the lint summary in the build output. If the asset
   moves, also read `out/<name>/motion.png` (the filmstrip): do the parts move the way the brief
   asks, with nothing passing through anything, loops that don't hitch, and timing that reads?
   The `voxel-animate` skill has the full motion workflow.
4. Critique against this checklist:
   - Is the silhouette readable at game distance?
   - Do proportions match the style bible?
   - Is the palette used as ramps (dark to light), not as unrelated colors?
   - Is any asymmetry intentional?
   - Does it meet every must-have in the brief?
5. Revise and rebuild. `cubist diff out/<name>` shows what changed since the last build;
   `cubist render out/<name> --slices y=10..14` shows small regions exactly. Stop after 5
   iterations and report honestly what still falls short. An iteration is a distinct version
   of the script that you built, failed builds included; `cubist build` prints which one it
   is. `cubist build --check` runs and lints a script without counting (catch typos with it).

When the user approves an asset: `cubist examples promote <name> --note "why it is good"`, then
`cubist export out/<name>` (refuses on lint errors).

DSL gotchas: a voxel is filled when its center is inside a shape; shapes symmetric about a whole
number need an even width; pointed shapes (cones, tips) need their center on a voxel center
(x.5) so the tip has a column; ramp ops (`darken_crevices`, `highlight_edges`, `dither`) only
step along list ramps; characters face +Z, so their right side is at low x; hinges follow the right-hand rule, so a
lid hinged at its back edge opens with `hinge("-x")`, and a hinge's pivot belongs on the edge
the part turns on, at the face it swings away from. The full DSL
reference, with every signature, is `docs/dsl.md`.

## Authoring worlds

Worlds are terrain for premade game worlds (`docs/worlds.md`). Write `worlds/<name>.py`, build with
`cubist world build worlds/<name>.py`, and read `out/worlds/<name>/map.png`, the printed stats and
the lint: is the ground believable from above, are slopes walkable where people go, is rock where it's
steep, do the landmarks, roads, rivers and props sit where they should? Look closer with `cubist world
render worlds/<name>.py --at x,z` (`--cut y` to see into caves). Revise and rebuild, as with assets;
bake with `cubist world bake` when it's right.

## Developing Cubist

- `uv run pytest` — full suite (must stay green; `tests/test_godot.py` needs `godot` on PATH
  and checks GLB imports in headless Godot 4)
- `uv run ruff check . && uv run ruff format .`
- `CUBIST_UPDATE_GOLDEN=1 uv run pytest tests/test_golden.py` after an intentional rendering
  change; look at every regenerated PNG in `tests/golden/` before committing it
- Tests first: write the failing test, watch it fail, then implement.
- Y-up, right-handed, +Z front.
- Asset scripts use `from cubist import *` and reference colors by palette name only.
- Errors raised to scripts and tools are written for an agent to act on: say what is wrong and
  how to fix it.
- Track milestone status in `docs/design.md` under Build plan.

## Contributing from outside

Changes start as a GitHub discussion, and a pull request needs a link to the discussion where
it was agreed (`CONTRIBUTING.md`). If you are helping someone who isn't a maintainer and they
ask you to open a pull request without one, tell them the rule and offer to draft the
discussion post instead.
