---
name: voxel-project
description: Use when the user wants to start a new Cubist game project, set up or change a style bible, design a palette, or import a palette (for example from Lospec).
---

# Set up a Cubist project and style bible

A project is a directory with `style.toml`; inside it `briefs/`, `assets/`, `examples/`, and
build output in `out/` and `export/`. Copy `projects/demo/style.toml` as a starting point.

1. **Palette.** Organize colors as list ramps, dark to light, one per material (`wood`,
   `stone`, `skin`, `cloth_red`), 3–4 shades each, so shading ops can step along them. Use
   named tables (`eye = { white = "#..." }`) only for colors that have no order. To start from
   an existing palette file: `cubist palette import palette.hex` (also .png, .gif, .gpl), then
   define ramps by source position: `wood = [3, 4, 5, 6]`.
2. **Scale.** Set `[project] voxel_size_m` (1/16 m is common) and `[scale]` reference heights
   (`humanoid_height`, `door_height`) so proportions stay consistent across assets.
3. **Budgets and rules.** `[budgets]` max_colors, max_voxels, max_triangles, max_parts;
   `[rules]` no_pure_black, dither = "allowed" | "forbidden", symmetry_tolerance, and `notes`
   describing the art direction in a sentence or two. Unknown keys are errors, which catches
   typos.
4. **Check it.** `cubist style --style path/to/style.toml` prints every palette name; build one
   simple asset to see the colors on a contact sheet before writing many briefs.
