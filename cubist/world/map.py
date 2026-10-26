"""The world map: the whole world top-down, for the agent to critique (design.md, "Worlds").

Surface colors hill-shaded with light from the top-left (north-west), contour lines, a ruler in
metres on two edges, landmarks, and a legend of surface colors with their share of the ground. x
runs right, z down, like the asset TOP view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from cubist.render.font import draw_text, text_height, text_width
from cubist.world.model import LoadedWorld
from cubist.world.sample import columns, heights

MAX_PIXELS = 768  # the longer side
_BG = np.array([24, 24, 28], dtype=np.uint8)
_INK = np.array([228, 228, 232], dtype=np.uint8)
_DIM = np.array([140, 140, 150], dtype=np.uint8)
MARGIN_LEFT, MARGIN_TOP, PAD = 40, 34, 8


@dataclass
class WorldMap:
    image: Image.Image
    metres_per_pixel: float
    heights: np.ndarray  # metres, rows along z
    slopes: np.ndarray  # degrees
    surface: np.ndarray  # palette indices
    contour_m: float


def _nice(value: float, steps=(1, 2, 2.5, 5)) -> float:
    """The smallest 'nice' number (1, 2, 2.5, 5 × 10^k) at least ``value``."""
    exp = 10 ** math.floor(math.log10(max(value, 1e-9)))
    for s in (*steps, 10):
        if s * exp >= value:
            return s * exp
    return 10 * exp


def render_map(loaded: LoadedWorld) -> WorldMap:
    sx, sz = loaded.size_m
    mpp = _nice(max(sx, sz) / MAX_PIXELS)
    w, h = max(1, math.ceil(sx / mpp)), max(1, math.ceil(sz / mpp))
    x, z = np.meshgrid((np.arange(w) + 0.5) * mpp, (np.arange(h) + 0.5) * mpp)
    hm = heights(loaded, x, z)
    gz, gx = np.gradient(hm, mpp)
    slope = np.degrees(np.arctan(np.hypot(gx, gz)))
    _, _, surf = columns(loaded, x, z, slope=slope)

    # Hill shading: the light comes from the top-left and above.
    normal = np.stack([-gx, np.ones_like(hm), -gz], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    light = np.array([-1.0, 1.5, -1.0]) / np.linalg.norm([-1.0, 1.5, -1.0])
    shade = np.clip(normal @ light, 0.0, 1.0)
    rgb = loaded.palette.rgba_table()[surf][..., :3].astype(np.float64) * (
        0.45 + 0.6 * shade[..., None]
    )

    # Contour lines where the height crosses a multiple of the interval (aim for ~12 bands).
    span = float(hm.max() - hm.min())
    contour = _nice(span / 12) if span > 0 else 1.0
    band = np.floor(hm / contour)
    edge = np.zeros_like(band, dtype=bool)
    edge[:, :-1] |= band[:, :-1] != band[:, 1:]
    edge[:-1, :] |= band[:-1, :] != band[1:, :]
    rgb[edge] *= 0.72
    body = np.clip(rgb, 0, 255).astype(np.uint8)

    canvas = _frame(loaded, body, mpp, contour, hm, surf)
    _landmarks(canvas, loaded, mpp)
    return WorldMap(Image.fromarray(canvas), mpp, hm, slope, surf, contour)


def _frame(
    loaded: LoadedWorld,
    body: np.ndarray,
    mpp: float,
    contour: float,
    hm: np.ndarray,
    surf: np.ndarray,
) -> np.ndarray:
    h, w = body.shape[:2]
    legend = _legend(loaded, surf, w)
    canvas = np.empty(
        (MARGIN_TOP + h + PAD + legend.shape[0] + PAD, MARGIN_LEFT + w + PAD, 3), dtype=np.uint8
    )
    canvas[:] = _BG
    canvas[MARGIN_TOP : MARGIN_TOP + h, MARGIN_LEFT : MARGIN_LEFT + w] = body
    sx, sz = loaded.size_m
    title = (
        f"{loaded.name}  {sx:g} x {sz:g} m  height {hm.min():.0f} to {hm.max():.0f} m"
        f"  contours {contour:g} m"
    )
    draw_text(canvas, PAD, PAD, title, _INK)
    # Rulers: labeled ticks in metres along the top and left edges.
    step = _nice(text_width("0000") * 2.5 * mpp)
    for axis, size in ((0, sx), (1, sz)):
        for i in range(int(size // step) + 1):
            metres = i * step
            px = int(round(metres / mpp))
            label = f"{metres:g}"
            if axis == 0:
                xx = MARGIN_LEFT + min(px, w - 1)
                canvas[MARGIN_TOP - 4 : MARGIN_TOP, xx] = _DIM
                draw_text(
                    canvas, xx - text_width(label) // 2, MARGIN_TOP - 6 - text_height(), label, _DIM
                )
            else:
                yy = MARGIN_TOP + min(px, h - 1)
                canvas[yy, MARGIN_LEFT - 4 : MARGIN_LEFT] = _DIM
                draw_text(
                    canvas,
                    MARGIN_LEFT - 6 - text_width(label),
                    yy - text_height() // 2,
                    label,
                    _DIM,
                )
    canvas[
        MARGIN_TOP + h + PAD : MARGIN_TOP + h + PAD + legend.shape[0],
        MARGIN_LEFT : MARGIN_LEFT + legend.shape[1],
    ] = legend
    return canvas


def surface_shares(loaded: LoadedWorld, surf: np.ndarray) -> dict[str, float]:
    """Each surface color's share of the ground in percent, largest first."""
    indices, counts = np.unique(surf, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    return {loaded.palette.name_of(int(indices[i])): 100.0 * counts[i] / surf.size for i in order}


def _legend(loaded: LoadedWorld, surf: np.ndarray, width: int) -> np.ndarray:
    """A row per surface color: a swatch, its palette name and its share of the ground."""
    shares = surface_shares(loaded, surf)
    table = loaded.palette.rgba_table()
    row = text_height() + 4
    legend = np.empty((row * len(shares), width, 3), dtype=np.uint8)
    legend[:] = _BG
    by_name = {c.name: c.index for c in loaded.palette.colors()}
    for i, (name, pct) in enumerate(shares.items()):
        y = i * row
        legend[y : y + text_height(), 0:10] = table[by_name[name]][:3]
        draw_text(legend, 14, y, f"{name}  {pct:.1f}%", _INK)
    return legend


def _landmarks(canvas: np.ndarray, loaded: LoadedWorld, mpp: float) -> None:
    for name, (lx, lz) in loaded.world.landmarks.items():
        px, py = MARGIN_LEFT + int(lx / mpp), MARGIN_TOP + int(lz / mpp)
        for d in range(-3, 4):
            for yy, xx in ((py, px + d), (py + d, px)):
                if 0 <= yy < canvas.shape[0] and 0 <= xx < canvas.shape[1]:
                    canvas[yy, xx] = _INK
        draw_text(canvas, px + 5, py - text_height() // 2, name, _INK)
