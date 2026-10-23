"""Contact sheets: every view of an asset on one image, at one pixel scale.

Orthographic panels carry voxel-coordinate rulers on their bottom and left edges (world
coordinates at voxel boundaries) and a faint grid behind the model, so an agent can name
exact positions. The declared asset size is outlined in blue when the model leaves it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image

from cubist.core.grid import VoxelGrid
from cubist.core.palette import Palette
from cubist.render.font import draw_text, text_height, text_width
from cubist.render.raster import BACKGROUND, VIEWS, ViewRender, iso, ortho

DEFAULT_VIEWS = ("front", "right", "back", "top", "iso")
TITLES = {
    "front": "FRONT (+Z)",
    "back": "BACK (-Z)",
    "right": "RIGHT (+X)",
    "left": "LEFT (-X)",
    "top": "TOP (+Y)",
    "bottom": "BOTTOM (-Y)",
    "iso": "ISO (FRONT-RIGHT)",
    "silhouette": "SILHOUETTE (FRONT)",
}
TARGET_VIEW_PX = 256
MAX_ROW_PX = 1100
MARGIN, GAP = 16, 24
SHEET_BG = (250, 249, 246)
INK = (40, 40, 44)
MUTED = (110, 108, 112)
GRID_MINOR = (224, 221, 213)
GRID_MAJOR = (206, 202, 193)
SIZE_BOX = (70, 120, 210)
_STEPS = (1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 50, 64, 100, 128, 256)


def choose_scale(dims: Sequence[int]) -> int:
    """Even pixels-per-voxel so the largest orthographic view is about 256 px."""
    s = max(2, min(12, TARGET_VIEW_PX // max(1, max(dims))))
    return s - s % 2


def asset_region(grid: VoxelGrid, size: Sequence[int] | None):
    los, his = [], []
    b = grid.bounds()
    if b is not None:
        los.append(b[0])
        his.append(b[1])
    if size is not None:
        los.append((0, 0, 0))
        his.append(tuple(int(s) for s in size))
    if not los:
        return (0, 0, 0), (1, 1, 1)
    return tuple(np.min(los, axis=0).tolist()), tuple(np.max(his, axis=0).tolist())


def _major_step(scale: int, max_label: int) -> int:
    need = text_width(str(max_label)) + 6
    return next((s for s in _STEPS if s * scale >= need), _STEPS[-1])


def _px(ticks: list[tuple[int, int]], world: int, limit: int) -> int | None:
    for px, w in ticks:
        if w == world:
            return min(px, limit - 1)
    return None


def _decorate(view: ViewRender, step: int, size_box) -> np.ndarray:
    """Background grid and declared-size box, drawn only over empty background."""
    rgb = view.rgb.copy()
    h, w = rgb.shape[:2]
    empty = np.all(rgb == BACKGROUND.astype(np.uint8), axis=-1)
    for ticks, horizontal in ((view.u_ticks, False), (view.v_ticks, True)):
        for px, world in ticks:
            if px >= (h if horizontal else w):
                continue
            color = GRID_MAJOR if world % step == 0 else GRID_MINOR
            line = (
                (slice(px, px + 1), slice(None)) if horizontal else (slice(None), slice(px, px + 1))
            )
            rgb[line][empty[line]] = color
    if size_box is not None:
        (u_lo, u_hi), (v_lo, v_hi) = size_box
        cols = [_px(view.u_ticks, u, w) for u in (u_lo, u_hi)]
        rows = [_px(view.v_ticks, v, h) for v in (v_lo, v_hi)]
        if None not in cols and None not in rows:
            c0, c1 = sorted(cols)
            r0, r1 = sorted(rows)
            for line in (
                (slice(r0, r1 + 1), c0),
                (slice(r0, r1 + 1), c1),
                (r0, slice(c0, c1 + 1)),
                (r1, slice(c0, c1 + 1)),
            ):
                rgb[line][empty[line]] = SIZE_BOX
    return rgb


def _titled(title: str, body: np.ndarray, body_x: int = 0) -> np.ndarray:
    th = text_height(2) + 6
    w = max(body.shape[1], body_x + text_width(title, 2))
    panel = np.full((th + body.shape[0], w, 3), SHEET_BG, dtype=np.uint8)
    draw_text(panel, body_x, 0, title, INK, scale=2)
    panel[th:, : body.shape[1]] = body
    return panel


def _ruled(view: ViewRender, rgb: np.ndarray, step: int) -> tuple[np.ndarray, int]:
    """The view with tick rulers below and to the left; returns (image, view x offset)."""
    labels = [str(w) for _, w in view.v_ticks if w % step == 0]
    left = max((text_width(s) for s in labels), default=0) + 8
    bottom = text_height() + 8
    h, w = rgb.shape[:2]
    out = np.full((h + bottom, left + w + 4, 3), SHEET_BG, dtype=np.uint8)
    out[:h, left : left + w] = rgb
    for px, world in view.u_ticks:
        major = world % step == 0
        x = min(left + px, left + w - 1)
        out[h : h + (5 if major else 2), x] = INK if major else MUTED
        if major:
            s = str(world)
            draw_text(out, x - text_width(s) // 2, h + 7, s, INK)
    for py, world in view.v_ticks:
        major = world % step == 0
        y = min(py, h - 1)
        out[y, left - (5 if major else 2) : left] = INK if major else MUTED
        if major:
            s = str(world)
            draw_text(out, left - 7 - text_width(s), y - text_height() // 2, s, INK)
    return out, left


def _flow(panels: list[np.ndarray]) -> list[list[np.ndarray]]:
    rows, row, width = [], [], 0
    limit = max([MAX_ROW_PX] + [p.shape[1] for p in panels])
    for p in panels:
        if row and width + GAP + p.shape[1] > limit:
            rows.append(row)
            row, width = [], 0
        width += (GAP if row else 0) + p.shape[1]
        row.append(p)
    if row:
        rows.append(row)
    return rows


def _legend(counts: dict[int, int], table: np.ndarray, names: dict[int, str], width: int):
    items = []
    for index, n in sorted(counts.items()):
        r, g, b = (int(v) for v in table[index, :3])
        label = f"{names.get(index, f'#{r:02x}{g:02x}{b:02x}')} {n}"
        items.append(((r, g, b), label))
    lines, line, x = [], [], 0
    for item in items:
        w = 14 + text_width(item[1]) + 18
        if line and x + w > width:
            lines.append(line)
            line, x = [], 0
        line.append((x, item))
        x += w
    if line:
        lines.append(line)
    row_h = 16
    out = np.full((max(1, len(lines)) * row_h, width, 3), SHEET_BG, dtype=np.uint8)
    for i, line in enumerate(lines):
        y = i * row_h
        for x, (color, label) in line:
            out[y : y + 11, x : x + 11] = INK
            out[y + 1 : y + 10, x + 1 : x + 10] = color
            draw_text(out, x + 16, y + 2, label, INK)
    return out


def contact_sheet(
    grid: VoxelGrid,
    palette: Palette | np.ndarray,
    name: str,
    size: Sequence[int] | None = None,
    views: Sequence[str] = DEFAULT_VIEWS,
    names: dict[int, str] | None = None,
    scale: int | None = None,
) -> Image.Image:
    unknown = [v for v in views if v not in TITLES]
    if unknown:
        raise ValueError(f"unknown view {unknown[0]!r}; choose from {', '.join(TITLES)}")
    if isinstance(palette, Palette):
        table = palette.rgba_table()
        names = {c.index: c.name for c in palette.colors()} if names is None else names
    else:
        table = np.asarray(palette, dtype=np.uint8)
    names = names or {}

    region = asset_region(grid, size)
    dims = [b - a for a, b in zip(*region, strict=True)]
    scale = scale or choose_scale(dims)
    step = _major_step(scale, max(abs(v) for v in (*region[0], *region[1])))
    size_box = None
    # Outline the declared size only when the model breaks out of it.
    if size is not None and region != ((0, 0, 0), tuple(int(s) for s in size)):
        lo, hi = (0, 0, 0), tuple(size)
        size_box = {
            "front": ((lo[0], hi[0]), (hi[1], lo[1])),
            "back": ((hi[0], lo[0]), (hi[1], lo[1])),
            "right": ((hi[2], lo[2]), (hi[1], lo[1])),
            "left": ((lo[2], hi[2]), (hi[1], lo[1])),
            "top": ((lo[0], hi[0]), (lo[2], hi[2])),
            "bottom": ((lo[0], hi[0]), (hi[2], lo[2])),
        }

    panels = []
    for v in views:
        if v == "iso":
            panels.append(_titled(TITLES[v], iso(grid, table, scale, region).rgb))
            continue
        base_view = "front" if v == "silhouette" else v
        assert base_view in VIEWS
        render = ortho(grid, table, base_view, scale, region)
        if v == "silhouette":
            render.rgb = np.where(render.mask[..., None], 0, BACKGROUND).astype(np.uint8)
        rgb = _decorate(render, step, size_box[base_view] if size_box else None)
        ruled, offset = _ruled(render, rgb, step)
        panels.append(_titled(TITLES[v], ruled, body_x=offset))

    rows = _flow(panels)
    body_w = max(sum(p.shape[1] for p in row) + GAP * (len(row) - 1) for row in rows)
    width = body_w + 2 * MARGIN

    header = np.full((text_height(2) + text_height() + 14, body_w, 3), SHEET_BG, dtype=np.uint8)
    draw_text(header, 0, 0, name, INK, scale=2)
    b = grid.bounds()
    info = [f"{grid.count()} VOXELS"]
    if size is not None:
        info.insert(0, "SIZE " + "X".join(str(s) for s in size))
    if b is not None:
        info.append(f"BOUNDS ({','.join(map(str, b[0]))})-({','.join(map(str, b[1]))})")
    info.append(f"1 VOXEL = {scale} PX")
    draw_text(header, 0, text_height(2) + 7, "   ".join(info), MUTED)

    legend = _legend(grid.colors_used(), table, names, body_w)
    blocks = [header]
    for row in rows:
        row_h = max(p.shape[0] for p in row)
        strip = np.full((row_h, body_w, 3), SHEET_BG, dtype=np.uint8)
        x = 0
        for p in row:
            strip[: p.shape[0], x : x + p.shape[1]] = p
            x += p.shape[1] + GAP
        blocks.append(strip)
    blocks.append(legend)

    height = sum(blk.shape[0] for blk in blocks) + GAP * (len(blocks) - 1) + 2 * MARGIN
    sheet = np.full((height, width, 3), SHEET_BG, dtype=np.uint8)
    y = MARGIN
    for blk in blocks:
        sheet[y : y + blk.shape[0], MARGIN : MARGIN + blk.shape[1]] = blk
        y += blk.shape[0] + GAP
    return Image.fromarray(sheet)
