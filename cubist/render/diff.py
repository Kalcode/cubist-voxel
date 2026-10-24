"""Before/after diff sheets: the old model, the new one, and what changed between them."""

from __future__ import annotations

import numpy as np
from PIL import Image

from cubist.core.grid import VoxelGrid
from cubist.render.contact_sheet import SHEET_BG, contact_sheet

DIFF_VIEWS = ("front", "right", "top", "iso")
CHANGE_COLORS = {  # index: (name, rgb)
    1: ("same", (200, 198, 192)),
    2: ("added", (63, 174, 73)),
    3: ("removed", (214, 69, 69)),
    4: ("recolored", (240, 160, 32)),
}


def change_grid(before: VoxelGrid, after: VoxelGrid) -> tuple[VoxelGrid, dict[str, int]]:
    """A grid colored by change (indices of CHANGE_COLORS) and a count per kind."""
    frame = VoxelGrid.merge([before, after])
    if frame.count() == 0:
        return frame, {name: 0 for name, _ in CHANGE_COLORS.values()}
    cells = np.indices(frame.shape).reshape(3, -1).T + np.array(frame.origin)
    b, a = before.lookup(cells), after.lookup(cells)
    kind = np.zeros(len(cells), dtype=np.uint8)
    kind[(b > 0) & (a == b)] = 1
    kind[(b == 0) & (a > 0)] = 2
    kind[(b > 0) & (a == 0)] = 3
    kind[(b > 0) & (a > 0) & (a != b)] = 4
    counts = {name: int(np.sum(kind == i)) for i, (name, _) in CHANGE_COLORS.items()}
    return VoxelGrid(kind.reshape(frame.shape), frame.origin), counts


def diff_sheet(
    before: VoxelGrid,
    after: VoxelGrid,
    palette: np.ndarray,
    name: str,
    size=None,
    names: dict[int, str] | None = None,
    before_palette: np.ndarray | None = None,
) -> Image.Image:
    changes, counts = change_grid(before, after)
    table = np.zeros((256, 4), dtype=np.uint8)
    for i, (_, rgb) in CHANGE_COLORS.items():
        table[i] = (*rgb, 255)
    change_names = {i: label for i, (label, _) in CHANGE_COLORS.items()}
    summary = (
        f"{name} CHANGES: +{counts['added']} ADDED, -{counts['removed']} REMOVED, "
        f"~{counts['recolored']} RECOLORED"
    )
    # One scale for all three so they line up.
    region_grid = VoxelGrid.merge([before, after])
    from cubist.render.contact_sheet import asset_region, choose_scale

    region = asset_region(region_grid, size)
    scale = choose_scale([h - lo for lo, h in zip(*region, strict=True)])
    sheets = [
        contact_sheet(
            before,
            before_palette if before_palette is not None else palette,
            name=f"{name} BEFORE",
            size=size,
            views=DIFF_VIEWS,
            names=names,
            scale=scale,
        ),
        contact_sheet(
            after,
            palette,
            name=f"{name} AFTER",
            size=size,
            views=DIFF_VIEWS,
            names=names,
            scale=scale,
        ),
        contact_sheet(
            changes,
            table,
            name=summary,
            size=size,
            views=DIFF_VIEWS,
            names=change_names,
            scale=scale,
        ),
    ]
    width = max(s.width for s in sheets)
    out = Image.new("RGB", (width, sum(s.height for s in sheets)), SHEET_BG)
    y = 0
    for s in sheets:
        out.paste(s, (0, y))
        y += s.height
    return out
