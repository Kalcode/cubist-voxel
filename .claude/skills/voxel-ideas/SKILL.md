---
name: voxel-ideas
description: Use when the user wants ideas for voxel assets, an asset list for a game area or level, variations on an existing asset, or a batch of briefs to build later with Cubist.
---

# Brainstorm assets and write briefs

Ideas become briefs; briefs become assets. Don't build until the user picks what to make.

1. **Ground the ideas.** Read `style.toml` (`cubist style`: palette, scale, rules, notes) and
   list what exists: `ls briefs assets` and `cubist examples list`. Ideas should fit the
   style bible's palette and scale and fill gaps, not repeat what's there.
2. **Propose a set**, grouped by purpose (e.g. for a village: buildings, street props, NPCs,
   nature). For each: name, category, size, one-line description, and 2–4 must-haves that a
   contact sheet can verify. Prefer variety in silhouette over variety in color. Offer
   variations as families (`crate`, `crate_broken`, `crate_stacked`) that can share code.
3. **Write the briefs the user picks** with `cubist new <name> --category ... --size x,y,z
   --tags ... --description "..." --must-have "..."` (one `--must-have` per item).
4. **Suggest a build order**: start with assets closest to existing examples, then promote good
   results so later, harder assets have better references.

For variations of an existing asset, copy its script to the new name and change only what makes
the variant different; note the shared origin in the brief's description.
