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

from cubist.build.runner import load_palette
from cubist.core.model import AssetDef
from cubist.render.contact_sheet import contact_sheet
from cubist.render.filmstrip import filmstrip

ROOT = Path(__file__).parent.parent
DEMO = ROOT / "projects" / "demo"
GOLDEN = Path(__file__).parent / "golden"
SCRIPTS = sorted((DEMO / "assets").glob("*.py"))
UPDATE = os.environ.get("CUBIST_UPDATE_GOLDEN") == "1"


def _compare(image, path: Path) -> None:
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return
    assert path.exists(), f"missing {path}; run with CUBIST_UPDATE_GOLDEN=1 to create it"
    expected = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    actual = np.asarray(image, dtype=np.int16)
    assert actual.shape == expected.shape
    diff = np.abs(actual - expected)
    assert diff.max() <= 2, f"{(diff > 2).any(axis=-1).sum()} pixels differ from {path.name}"


def _built(script):
    palette = load_palette(DEMO / "style.toml")
    [asset] = [v for v in runpy.run_path(str(script)).values() if isinstance(v, AssetDef)]
    return asset.build(palette), palette


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda s: s.stem)
def test_filmstrip_matches_golden(script):
    built, palette = _built(script)
    if not built.clips:
        pytest.skip("no clips")
    _compare(filmstrip(built, palette.rgba_table(), built.clips), GOLDEN / "motion" /
             f"{built.name}.png")  # fmt: skip


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda s: s.stem)
def test_contact_sheet_matches_golden(script):
    palette = load_palette(DEMO / "style.toml")
    [asset] = [v for v in runpy.run_path(str(script)).values() if isinstance(v, AssetDef)]
    built = asset.build(palette)
    sheet = contact_sheet(built.merged_grid(), palette, name=asset.name, size=asset.size)
    path = GOLDEN / f"{asset.name}.png"
    if UPDATE:
        GOLDEN.mkdir(exist_ok=True)
        sheet.save(path)
        return
    assert path.exists(), f"missing {path}; run with CUBIST_UPDATE_GOLDEN=1 to create it"
    expected = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    actual = np.asarray(sheet, dtype=np.int16)
    assert actual.shape == expected.shape
    diff = np.abs(actual - expected)
    assert diff.max() <= 2, f"{(diff > 2).any(axis=-1).sum()} pixels differ from {path.name}"
