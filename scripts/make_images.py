"""Regenerate the README and docs images (icon, banner, gallery) with Cubist itself.

uv run python scripts/make_images.py
"""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
from PIL import Image

from cubist import Model, Part, box
from cubist.build.runner import load_project_style
from cubist.core.grid import VoxelGrid
from cubist.core.model import AssetDef, asset
from cubist.core.palette import Palette
from cubist.render.font import draw_text, text_width
from cubist.render.raster import iso

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "docs" / "images"

BRAND = Palette(
    {
        "deep": ["#1d1b3a", "#2c2a5a", "#3d3b7c", "#5552a3"],
        "teal": ["#0f4c5c", "#16697a", "#2a8fa0", "#5fc1c9"],
        "gold": ["#8a5a12", "#c8891c", "#f2b33d", "#ffe08a"],
    }
)


@asset(name="logo", size=(13, 16, 13))
def logo(p):
    """A 2x2x2 cube of blocks with one lifted out and lit gold: building by code."""
    blocks = []
    for bx in (0, 1):
        for by in (0, 1):
            for bz in (0, 1):
                lifted = (bx, by, bz) == (1, 1, 1)
                lift = 3 if lifted else 0
                color = p.gold.mid if lifted else (p.teal.mid if (bx + by + bz) % 2 else p.deep.mid)
                block = box((6, 6, 6), color=color).at(bx * 7, by * 7 + lift, bz * 7)
                blocks.append(block.highlight_edges())
    return Model(parts=[Part("logo", blocks[0].union(*blocks[1:]))])


def rgba(view, alpha_background=True) -> Image.Image:
    img = np.dstack([view.rgb, np.full(view.rgb.shape[:2], 255, dtype=np.uint8)])
    if alpha_background:
        # keep the dark outline pixels that ring the model; drop the flat background
        bg = np.all(view.rgb == view.rgb[0, 0], axis=-1)
        img[bg, 3] = 0
    return Image.fromarray(img, "RGBA")


def trim(img: Image.Image, pad: int) -> Image.Image:
    box_ = img.getbbox()
    img = img.crop(box_)
    out = Image.new("RGBA", (img.width + 2 * pad, img.height + 2 * pad), (0, 0, 0, 0))
    out.paste(img, (pad, pad))
    return out


def square(img: Image.Image, size: int) -> Image.Image:
    side = max(img.size)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return canvas.resize((size, size), Image.NEAREST)


def make_icon() -> None:
    built = logo.build(BRAND)
    view = iso(built.merged_grid(), BRAND.rgba_table(), scale=24)
    icon = trim(rgba(view), pad=8)
    for size in (512, 256, 64, 32):
        square(icon, size).save(IMAGES / f"icon-{size}.png")


def demo_iso(name: str, scale: int) -> Image.Image:
    style = load_project_style(ROOT / "projects" / "demo" / "style.toml")
    script = ROOT / "projects" / "demo" / "assets" / f"{name}.py"
    [a] = [v for v in runpy.run_path(str(script)).values() if isinstance(v, AssetDef)]
    grid = a.build(style.palette).merged_grid()
    return trim(rgba(iso(grid, style.palette.rgba_table(), scale)), pad=4)


def demo_built(name: str):
    style = load_project_style(ROOT / "projects" / "demo" / "style.toml")
    script = ROOT / "projects" / "demo" / "assets" / f"{name}.py"
    [a] = [v for v in runpy.run_path(str(script)).values() if isinstance(v, AssetDef)]
    return a, a.build(style.palette), style.palette


def make_sheets() -> None:
    """The goblin's contact sheet, and a diff of a revision: ears off, a hat on."""
    from cubist.render.contact_sheet import contact_sheet
    from cubist.render.diff import diff_sheet

    a, built, palette = demo_built("goblin")
    before = built.merged_grid()
    contact_sheet(before, palette, name=a.name, size=a.size).save(IMAGES / "contact-goblin.png")
    coords, colors = before.voxels()
    keep = ((coords[:, 0] >= 4) & (coords[:, 0] < 12)) | (coords[:, 1] < 18)
    coords, colors = coords[keep], colors[keep].copy()
    nose = (coords[:, 1] == 16) & (coords[:, 2] == 9)
    colors[nose] = palette.eye.yellow.index
    hat = np.array([[x, 22, z] for x in range(5, 11) for z in range(4, 9)])
    after = VoxelGrid.from_voxels(
        np.vstack([coords, hat]),
        np.concatenate([colors, np.full(len(hat), palette.cloth.brown.index, dtype=np.uint8)]),
    )
    names = {c.index: c.name for c in palette.colors()}
    diff_sheet(before, after, palette.rgba_table(), name=a.name, size=a.size, names=names).save(
        IMAGES / "diff-goblin.png"
    )


def make_banner() -> None:
    names = ["goblin", "chest", "barrel", "torch", "hut", "tree", "mushroom", "potion", "sword"]
    tiles = [demo_iso(n, 8 if n != "hut" else 6) for n in names]
    height = max(t.height for t in tiles)
    width = sum(t.width for t in tiles) + 24 * (len(tiles) + 1)
    title_scale = 10
    title_h = 7 * title_scale + 40
    canvas = np.zeros((height + title_h + 32, width, 4), dtype=np.uint8)
    canvas[..., :3] = (24, 23, 33)
    canvas[..., 3] = 255
    title = "CUBIST"
    tw = text_width(title, title_scale)
    draw_text(canvas, (width - tw) // 2, 20, title, (242, 179, 61, 255), scale=title_scale)
    sub = "VOXEL ASSETS, WRITTEN AS CODE BY AGENTS"
    draw_text(
        canvas,
        (width - text_width(sub, 2)) // 2,
        20 + 7 * title_scale + 10,
        sub,
        (170, 168, 190, 255),
        scale=2,
    )
    img = Image.fromarray(canvas, "RGBA")
    x = 24
    for t in tiles:
        img.alpha_composite(t, (x, title_h + 16 + height - t.height))
        x += t.width + 24
    img.convert("RGB").save(IMAGES / "banner.png")


if __name__ == "__main__":
    IMAGES.mkdir(parents=True, exist_ok=True)
    make_icon()
    make_banner()
    make_sheets()
    print("wrote", sorted(p.name for p in IMAGES.glob("*.png")))
