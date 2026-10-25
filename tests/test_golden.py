"""Golden-image tests: demo contact sheets must match the reviewed PNGs in tests/golden/.

After an intentional rendering change, regenerate with ``CUBIST_UPDATE_GOLDEN=1 uv run pytest
tests/test_golden.py``, look at the new images, and commit them.
"""

import os
import runpy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cubist.build.runner import load_project_style
from cubist.core.model import AssetDef
from cubist.motion.pose import preview_clips
from cubist.render.contact_sheet import contact_sheet, hidden_note
from cubist.render.filmstrip import filmstrip

ROOT = Path(__file__).parent.parent
DEMO = ROOT / "projects" / "demo"
GOLDEN = Path(__file__).parent / "golden"
SCRIPTS = sorted((DEMO / "assets").glob("*.py"))
UPDATE = os.environ.get("CUBIST_UPDATE_GOLDEN") == "1"
# Math libraries differ in the last bit of sin and cos, so on another platform an edge that lands
# exactly on a pixel center can flip (3 pixels on Linux x86 vs macOS arm64). Recoloring a single
# voxel changes at least 33 pixels in these images, so a real change still fails.
EDGE_PIXELS = 16


def _compare(image, path: Path) -> None:
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return
    assert path.exists(), f"missing {path}; run with CUBIST_UPDATE_GOLDEN=1 to create it"
    expected = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    actual = np.asarray(image, dtype=np.int16)
    assert actual.shape == expected.shape
    differ = (np.abs(actual - expected) > 2).any(axis=-1).sum()
    assert differ <= EDGE_PIXELS, f"{differ} pixels differ from {path.name}"


def _built(script):
    style = load_project_style(DEMO / "style.toml")
    [asset] = [v for v in runpy.run_path(str(script)).values() if isinstance(v, AssetDef)]
    built = asset.build(style.palette, glow=style.glow, cycles=style.cycles)
    for e in built.emitters.values():  # particle models come from their own demo scripts
        [part_def] = [v for v in runpy.run_path(str(DEMO / "assets" / f"{e.particle}.py"))
                      .values() if isinstance(v, AssetDef)]  # fmt: skip
        built.particle_models[e.particle] = part_def.build(style.palette).merged_grid()
    return built, style.palette


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda s: s.stem)
def test_filmstrip_matches_golden(script):
    built, palette = _built(script)
    clips = preview_clips(built, with_previews=True)
    if not clips:
        pytest.skip("nothing moves")
    _compare(filmstrip(built, palette.rgba_table(), clips), GOLDEN / "motion" /
             f"{built.name}.png")  # fmt: skip


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda s: s.stem)
def test_contact_sheet_matches_golden(script):
    built, palette = _built(script)
    asset = built
    sheet = contact_sheet(
        built.visible_grid(),
        palette,
        name=asset.name,
        size=asset.size,
        note=hidden_note(built.hidden_parts()),
        glow=built.glow,
    )
    path = GOLDEN / f"{asset.name}.png"
    if UPDATE:
        GOLDEN.mkdir(exist_ok=True)
        sheet.save(path)
        return
    assert path.exists(), f"missing {path}; run with CUBIST_UPDATE_GOLDEN=1 to create it"
    expected = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    actual = np.asarray(sheet, dtype=np.int16)
    assert actual.shape == expected.shape
    differ = (np.abs(actual - expected) > 2).any(axis=-1).sum()
    assert differ <= EDGE_PIXELS, f"{differ} pixels differ from {path.name}"
