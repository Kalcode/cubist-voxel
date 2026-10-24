---
name: voxel-create
description: Use when the user asks to make, create, design, or model a new voxel asset (a prop, character, creature, building, or vehicle) with Cubist, from an idea or a brief.
---

# Create a voxel asset

You write a Cubist script and look at what it builds. Never place voxels one at a time.
Work from the project directory (the one with `style.toml`, e.g. `projects/demo`); run every
command through `uv run`.

1. **Brief.** If `briefs/<name>.toml` doesn't exist, scaffold it from the user's idea:
   `cubist new <name> --category prop --tags a,b --description "..." --must-have "..."`
   (repeat `--must-have`). Categories: prop, character, creature, building, vehicle, terrain.
   Ask only if the size or purpose is genuinely unclear.
2. **Context.** `cubist style` for palette names, scale, and budgets. `cubist examples search
   --tags ... --category ...` for the 2–3 closest approved examples; read their scripts and
   look at their `contact.png` with the Read tool.
3. **Script.** Write `assets/<name>.py`. Keep the brief's name, size, and tags in `@asset`;
   add `grounded=True` for things that stand on the ground. Build up from big masses to
   details: primitives and CSG first, then paint, then shading (`darken_crevices`,
   `highlight_edges`, `dither`). Use parts with pivots for anything that moves (lids, limbs,
   doors) and sockets for attachment points.
4. **Build and look.** `cubist build assets/<name>.py`, then Read `out/<name>/contact.png`.
5. **Critique** against the brief's must-haves and this checklist: silhouette readable at game
   distance? proportions match the style bible? palette used as ramps? asymmetry intentional?
   lint clean? Say what you see in voxel coordinates (the rulers make this exact).
6. **Revise** and rebuild; `cubist diff out/<name>` shows what changed. Use
   `cubist render out/<name> --slices y=a..b` for exact work on a small region. Stop after 5
   builds and report honestly what still falls short.
7. **Show the user** the contact sheet path and a one-paragraph summary. When they approve:
   `cubist examples promote <name> --note "why it is good"` and `cubist export out/<name>`.

## Gotchas

- A voxel is filled when its center is inside a shape. Shapes symmetric about a whole number
  need an even width; pointed shapes (cones, tips, spikes) need their center on a voxel center
  (x.5) or the tip disappears.
- `round`, `hollow`, `paint` and other voxel ops make a shape voxel-stage: do `smooth_union`,
  `taper`, and `displace` before them.
- Ramp ops only step along list ramps (`wood = [...]`); named-table colors stay as they are.
- Dithering breaks exact color symmetry; don't declare `symmetric=` on dithered parts.
- The full DSL reference is in `docs/design.md` (section "The DSL").
