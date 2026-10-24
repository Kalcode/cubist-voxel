"""Seeing motion: filmstrips for the agent, GIFs for people, and posed contact sheets.

A filmstrip shows each clip as a row of frames: the iso view above one orthographic view chosen
to show the motion, each labeled with its time. Looping clips repeat their first pose at the end
so a hitch shows. Every frame of a clip shares one canvas, sized to everything the clip reaches.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from cubist.core.grid import VoxelGrid
from cubist.core.model import BuiltAsset
from cubist.motion.clips import Clip
from cubist.motion.pose import hidden_at, sample_times, world_matrices
from cubist.render.contact_sheet import (
    DEFAULT_VIEWS,
    GAP,
    INK,
    MUTED,
    SHEET_BG,
    TITLES,
    _decorate,
    _major_step,
    _ruled,
    choose_scale,
    iso_framed,
    sheet,
    stack,
)
from cubist.render.font import draw_text, text_height, text_width
from cubist.render.posed import PosedScene
from cubist.render.raster import BACKGROUND, Region

GIF_HOLD_MS = 600  # one-shot clips rest on their last frame this long before restarting
GIF_MAX_ISO_PX = 420
_ROTATE_VIEW = {0: "right", 1: "top", 2: "front"}  # looking along the rotation axis
_SLIDE_VIEW = {0: "front", 1: "front", 2: "right"}  # looking across the slide


def motion_view(asset: BuiltAsset, clip: Clip) -> str:
    """The orthographic view that shows a clip's motion best."""
    votes: list[str] = []
    for track in clip.tracks:
        joint = asset.part(track.part).joint
        if joint is None:
            continue
        axis = int(np.argmax(np.abs(joint.axis)))
        votes.append((_SLIDE_VIEW if joint.kind == "slide" else _ROTATE_VIEW)[axis])
    if not votes:
        return "front"
    counts = Counter(votes)
    return max(votes, key=lambda v: (counts[v], -votes.index(v)))


def _union(regions: Sequence[Region]) -> Region:
    lo = tuple(int(min(r[0][i] for r in regions)) for i in range(3))
    hi = tuple(int(max(r[1][i] for r in regions)) for i in range(3))
    return lo, hi  # type: ignore[return-value]


def motion_region(asset: BuiltAsset, clips: Sequence[Clip], scene: PosedScene) -> Region:
    """Whole-voxel box around the rest pose, the declared size, and every sampled pose."""
    regions = [scene.bounds({})]
    if asset.size is not None:
        regions.append(((0, 0, 0), tuple(int(s) for s in asset.size)))  # type: ignore[arg-type]
    for c in clips:
        for t in sample_times(c, rate=24):
            regions.append(scene.bounds(world_matrices(asset, c, float(t))))
    return _union(regions)


def frame_times(clip: Clip, frames: int) -> list[float]:
    """Times to show: evenly across a one-shot clip, ends included; a loop's frames plus its
    first pose again at the end."""
    frames = max(2, int(frames))
    if clip.loop:
        return [clip.length * k / frames for k in range(frames)] + [clip.length]
    return [clip.length * k / (frames - 1) for k in range(frames)]


def _strip_scale(region: Region) -> int:
    dims = [h - lo for lo, h in zip(*region, strict=True)]
    s = max(2, choose_scale(dims) // 2)
    return s - s % 2


def _cell(iso_rgb: np.ndarray, ortho_rgb: np.ndarray, label: str) -> np.ndarray:
    w = max(iso_rgb.shape[1], ortho_rgb.shape[1], text_width(label))
    h = iso_rgb.shape[0] + 6 + ortho_rgb.shape[0] + 6 + text_height()
    out = np.full((h, w, 3), SHEET_BG, dtype=np.uint8)
    out[: iso_rgb.shape[0], : iso_rgb.shape[1]] = iso_rgb
    y = iso_rgb.shape[0] + 6
    out[y : y + ortho_rgb.shape[0], : ortho_rgb.shape[1]] = ortho_rgb
    draw_text(out, 0, h - text_height(), label, MUTED)
    return out


def _row(cells: list[np.ndarray]) -> np.ndarray:
    h = max(c.shape[0] for c in cells)
    w = sum(c.shape[1] for c in cells) + GAP * (len(cells) - 1)
    out = np.full((h, w, 3), SHEET_BG, dtype=np.uint8)
    x = 0
    for c in cells:
        out[: c.shape[0], x : x + c.shape[1]] = c
        x += c.shape[1] + GAP
    return out


def filmstrip(
    asset: BuiltAsset,
    table: np.ndarray,
    clips: Sequence[Clip],
    frames: int = 8,
    view: str | None = None,
    scale: int | None = None,
) -> Image.Image:
    """One block per clip: a title line, then frames (iso above ``view``, default the view that
    shows the motion best) labeled with their times."""
    scene = PosedScene(asset, table)
    region = motion_region(asset, clips, scene)
    scale = scale or _strip_scale(region)
    step = _major_step(scale, max(abs(v) for v in (*region[0], *region[1])))
    size_box = _size_boxes(asset, region)
    blocks = []
    header = f"{asset.name} MOTION"
    info = f"{len(clips)} CLIP{'S' if len(clips) != 1 else ''}   1 VOXEL = {scale} PX"
    head = np.full((text_height(2) + text_height() + 14, 1, 3), SHEET_BG, dtype=np.uint8)
    blocks.append(head)  # filled in once the width is known
    for c in clips:
        v = view or motion_view(asset, c)
        cells = []
        for t in frame_times(c, frames):
            m, hide = world_matrices(asset, c, t), hidden_at(asset, c, t)
            iso_rgb = iso_framed(scene.render(m, "iso", scale, region, hide).rgb)
            ortho = scene.render(m, v, scale, region, hide)
            rgb = _decorate(ortho, step, size_box.get(v) if size_box else None)
            ruled, _ = _ruled(ortho, rgb, step)
            cells.append(_cell(iso_rgb, ruled, f"T={t:.2f}S"))
        title = (
            f"{c.name.upper()}   {c.length:g} S   {'LOOP' if c.loop else 'ONCE'}   "
            f"ISO + {TITLES[v]}"
        )
        row = _row(cells)
        th = text_height(2) + 8
        block = np.full((th + row.shape[0], max(row.shape[1], text_width(title, 2)), 3), SHEET_BG,
                        dtype=np.uint8)  # fmt: skip
        draw_text(block, 0, 0, title, INK, scale=2)
        block[th:, : row.shape[1]] = row
        blocks.append(block)
    body_w = max([b.shape[1] for b in blocks[1:]] + [text_width(header, 2), text_width(info)])
    head = np.full((blocks[0].shape[0], body_w, 3), SHEET_BG, dtype=np.uint8)
    draw_text(head, 0, 0, header, INK, scale=2)
    draw_text(head, 0, text_height(2) + 7, info, MUTED)
    blocks[0] = head
    padded = []
    for b in blocks:
        full = np.full((b.shape[0], body_w, 3), SHEET_BG, dtype=np.uint8)
        full[:, : b.shape[1]] = b
        padded.append(full)
    return stack(padded, body_w + 2 * 16)


def _size_boxes(asset: BuiltAsset, region: Region) -> dict | None:
    if asset.size is None or region == ((0, 0, 0), tuple(int(s) for s in asset.size)):
        return None
    lo, hi = (0, 0, 0), tuple(asset.size)
    return {
        "front": ((lo[0], hi[0]), (hi[1], lo[1])),
        "back": ((hi[0], lo[0]), (hi[1], lo[1])),
        "right": ((hi[2], lo[2]), (hi[1], lo[1])),
        "left": ((lo[2], hi[2]), (hi[1], lo[1])),
        "top": ((lo[0], hi[0]), (lo[2], hi[2])),
    }


def clip_frames(
    asset: BuiltAsset, table: np.ndarray, clip: Clip, fps: float = 15, scale: int | None = None
) -> tuple[list[Image.Image], list[int]]:
    """Iso frames of ``clip`` at ``fps`` and each frame's duration in milliseconds."""
    scene = PosedScene(asset, table)
    region = motion_region(asset, [clip], scene)
    if scale is None:
        dims = [h - lo for lo, h in zip(*region, strict=True)]
        cap = max(2, GIF_MAX_ISO_PX // max(1, dims[0] + dims[2]))
        scale = min(choose_scale(dims), cap)
        scale = max(2, scale - scale % 2)
    n = max(2, round(clip.length * fps))
    if clip.loop:
        times = [clip.length * k / n for k in range(n)]
    else:
        times = [clip.length * k / n for k in range(n + 1)]
    frame_ms = round(1000 / fps)
    images = []
    for t in times:
        m, hide = world_matrices(asset, clip, t), hidden_at(asset, clip, t)
        images.append(Image.fromarray(iso_framed(scene.render(m, "iso", scale, region, hide).rgb)))
    durations = [frame_ms] * len(images)
    if not clip.loop:
        durations[-1] += GIF_HOLD_MS
    return images, durations


def save_gif(frames: Sequence[Image.Image], durations: Sequence[int], path: str | Path) -> None:
    """A looping GIF with one palette shared by every frame, so colors don't flicker."""
    w, h = frames[0].size
    tall = Image.new("RGB", (w, h * len(frames)), tuple(int(c) for c in BACKGROUND))
    for i, f in enumerate(frames):
        tall.paste(f.convert("RGB"), (0, i * h))
    shared = tall.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    images = [f.convert("RGB").quantize(palette=shared, dither=Image.Dither.NONE) for f in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=list(durations),
        loop=0,
        optimize=False,
        disposal=1,
    )


def posed_sheet(
    asset: BuiltAsset,
    table: np.ndarray,
    clip: Clip,
    t: float,
    names: dict[int, str] | None = None,
    views: Sequence[str] = DEFAULT_VIEWS,
) -> Image.Image:
    """Today's contact sheet of the asset posed at time ``t`` of ``clip``."""
    scene = PosedScene(asset, table)
    m, hide = world_matrices(asset, clip, t), hidden_at(asset, clip, t)
    bounds = scene.bounds(m)
    regions = [bounds]
    if asset.size is not None:
        regions.append(((0, 0, 0), tuple(int(s) for s in asset.size)))  # type: ignore[arg-type]
    region = _union(regions)
    grid = VoxelGrid.merge([p.grid for p in asset.parts if p.name not in hide])
    return sheet(
        lambda view, s, r: scene.render(m, view, s, r, hide),
        region,
        np.asarray(table),
        names or {},
        asset.name,
        grid.colors_used(),
        grid.count(),
        bounds,
        size=asset.size,
        views=views,
        note=f"CLIP {clip.name.upper()} T={t:.2f}S",
    )


STRIP_VIEWS = ("front", "back", "left", "right", "top")


def clip_named(asset: BuiltAsset, name: str) -> Clip:
    for c in asset.clips:
        if c.name == name:
            return c
    known = ", ".join(c.name for c in asset.clips) or "none (add clips=[...] to the Model)"
    raise ValueError(f"no clip {name!r} in {asset.name}; its clips: {known}")


def render_clip(
    asset: BuiltAsset,
    table: np.ndarray,
    name: str,
    at: float | None = None,
    frames: int = 8,
    view: str | None = None,
    names: dict[int, str] | None = None,
) -> tuple[str, Image.Image]:
    """("filmstrip", image) for a clip, or ("contact_sheet", image) posed at time ``at``."""
    c = clip_named(asset, name)
    if at is not None:
        if not 0 <= at <= c.length:
            raise ValueError(f"--at {at:g} is outside clip {name!r} (0 to {c.length:g} s)")
        return "contact_sheet", posed_sheet(asset, table, c, at, names=names)
    if view is not None and view not in STRIP_VIEWS:
        raise ValueError(f"unknown view {view!r}; choose from {', '.join(STRIP_VIEWS)}")
    if frames < 2:
        raise ValueError(f"frames must be at least 2, got {frames}")
    return "filmstrip", filmstrip(asset, table, [c], frames=frames, view=view)
