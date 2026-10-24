"""The demo scripts build, round-trip through .vox exactly, fit their size, and lint clean."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from cubist.build.runner import build_script, load_palette, load_project_style
from cubist.core.model import AssetDef, BuiltAsset
from cubist.io.vox import read_vox
from cubist.lint import lint

DEMO = Path(__file__).parent.parent / "projects" / "demo"
SCRIPTS = sorted((DEMO / "assets").glob("*.py"))


def test_there_are_eighteen_demo_scripts():
    assert len(SCRIPTS) == 18


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda s: s.stem)
def test_demo_builds_and_round_trips_exactly(script, tmp_path):
    [result] = build_script(script, out=tmp_path)
    palette = load_palette(DEMO / "style.toml")
    [asset] = [v for v in runpy.run_path(str(script)).values() if isinstance(v, AssetDef)]
    built = asset.build(palette)

    scene = read_vox(result.vox_path)
    again = BuiltAsset.from_vox(scene, result.summary, palette)  # folds flipbook frames back in
    assert [p.name for p in again.parts] == [p.name for p in built.parts]
    for read_part, built_part in zip(again.parts, built.parts, strict=True):
        assert read_part.grid == built_part.grid, read_part.name
        assert read_part.frames == built_part.frames, read_part.name
    assert np.array_equal(scene.palette, palette.rgba_table())


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda s: s.stem)
def test_demo_fits_its_declared_size(script):
    palette = load_palette(DEMO / "style.toml")
    [asset] = [v for v in runpy.run_path(str(script)).values() if isinstance(v, AssetDef)]
    for part in asset.build(palette).parts:
        lo, hi = part.grid.bounds()
        assert min(lo) >= 0, part.name
        assert all(h <= s for h, s in zip(hi, asset.size, strict=True)), (part.name, hi)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda s: s.stem)
def test_demo_lints_without_errors_or_warnings(script):
    style = load_project_style(DEMO / "style.toml")
    [asset] = [v for v in runpy.run_path(str(script)).values() if isinstance(v, AssetDef)]
    report = lint(asset.build(style.palette), style)
    problems = [f for f in report.findings if f.severity in ("error", "warning")]
    assert problems == [], report.summary()
