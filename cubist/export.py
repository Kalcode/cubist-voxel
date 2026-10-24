"""Exporting a built asset: lint gates it, then .glb and .vox are written natively and any
other format goes through vengi."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from cubist.build.runner import find_style, load_project_style
from cubist.core.grid import VoxelGrid
from cubist.core.model import BuiltAsset
from cubist.core.style import StyleBible
from cubist.io import vengi
from cubist.io.gltf import write_glb
from cubist.io.vox import VoxError, VoxScene, read_vox
from cubist.lint import LintReport, lint

NATIVE = ("glb", "vox")


class ExportBlocked(RuntimeError):
    """Lint found errors; the message lists them."""

    def __init__(self, report: LintReport):
        self.report = report
        errors = [line for line in report.summary().splitlines() if line.startswith("ERROR")]
        super().__init__(
            f"export blocked by {len(errors)} lint error(s); fix them or pass --force\n"
            + "\n".join(errors)
        )


@dataclass
class LoadedAsset:
    vox: Path
    scene: VoxScene
    summary: dict  # build.json, or {} when the .vox has none beside it

    @property
    def grid(self) -> VoxelGrid:
        return VoxelGrid.merge([p.grid for p in self.scene.parts])

    @property
    def names(self) -> dict[int, str]:
        return {int(k): v for k, v in self.summary.get("palette_names", {}).items()}

    @property
    def name(self) -> str:
        return self.summary.get("name", self.vox.stem)


def load_asset(path: str | Path) -> LoadedAsset:
    """A built asset from its .vox file or its out/<name>/ directory."""
    path = Path(path)
    vox = path
    if path.is_dir():
        vox = path / f"{path.name}.vox"
        if not vox.exists():
            found = sorted(path.glob("*.vox"))
            if len(found) != 1:
                raise VoxError(f"expected one .vox file in {path}, found {len(found)}")
            vox = found[0]
    summary_path = vox.parent / "build.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return LoadedAsset(vox, read_vox(vox), summary)


def lint_loaded(asset: LoadedAsset, style: StyleBible) -> tuple[BuiltAsset, LintReport]:
    built = BuiltAsset.from_vox(asset.scene, asset.summary, style.palette)
    return built, lint(built, style, rgba=asset.scene.palette)


def export_asset(
    path: str | Path,
    formats: Sequence[str] = NATIVE,
    out_dir: str | Path | None = None,
    force: bool = False,
    style: str | Path | None = None,
    vertex_colors: bool = False,
    ao: bool = False,
    center: bool = True,
) -> dict[str, str]:
    """Export to ``out_dir`` (default ``<project>/export``); returns {format: path}."""
    asset = load_asset(path)
    style_path = Path(style) if style else find_style(asset.vox)
    bible = load_project_style(style_path)
    built, report = lint_loaded(asset, bible)
    if not report.ok and not force:
        raise ExportBlocked(report)
    out = Path(out_dir) if out_dir else style_path.parent / "export"
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        target = out / f"{asset.name}.{fmt}"
        if fmt == "glb":
            write_glb(
                target,
                built,
                asset.scene.palette,
                voxel_size_m=bible.voxel_size_m,
                vertex_colors=vertex_colors,
                ao=ao,
                center=center,
            )
        elif fmt == "vox":
            if target.resolve() != asset.vox.resolve():
                shutil.copyfile(asset.vox, target)
        else:
            vengi.convert(asset.vox, target)
        written[fmt] = str(target)
    return written
