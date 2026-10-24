---
name: voxel-refine
description: Use when the user gives feedback on an existing Cubist voxel asset or asks to edit, fix, tweak, refine, recolor, resize, or add to it.
---

# Refine an existing voxel asset

Edits go in the script, never in voxels. Work from the project directory with `uv run`.

1. **Find the asset.** Read `assets/<name>.py`, `briefs/<name>.toml`, and the current
   `out/<name>/contact.png` (rebuild first if there is no build).
2. **Locate the feedback.** Turn the user's words into coordinates: use the contact sheet's
   rulers, `cubist render out/<name> --slices y=a..b` for exact layers, and `cubist inspect
   out/<name>/<name>.vox` for part bounds. Find the shapes in the script that produce that
   region.
3. **Change as little as possible.** Adjust the shapes, positions, or paint that cause the
   problem; keep everything else byte-for-byte so the diff is easy to read. If the request
   changes the design (a new must-have), update the brief too.
4. **Rebuild and compare.** `cubist build assets/<name>.py`, then `cubist diff out/<name>`
   and Read `out/<name>/diff.png`: green is added, red removed, orange recolored. Check that
   only the intended region changed and lint is still clean.
5. **Report** what changed in one or two sentences with the diff sheet path. If the user's
   request conflicts with the style bible (palette, scale, budgets), say so and offer the
   closest compliant option.

Small, frequent rebuilds beat one big rewrite: the diff makes every step reviewable.
