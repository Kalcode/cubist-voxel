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
from cubist.lint.motion import check_hand_edits
from cubist.manifest import build_manifest
from cubist.motion.particles import load_particles

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
    def parts(self) -> list:
        """The scene's parts, without flipbook frames 1.. (models named ``<part>@<k>``)."""
        names = {p.name for p in self.scene.parts}
        return [
            p for p in self.scene.parts
            if not ("@" in p.name and p.name.rsplit("@", 1)[0] in names)
        ]  # fmt: skip

    @property
    def grid(self) -> VoxelGrid:
        """Every part at rest (flipbooks at frame 0) in one grid."""
        return VoxelGrid.merge([p.grid for p in self.parts])

    @property
    def glow(self) -> dict[int, float]:
        return {int(k): float(v) for k, v in self.summary.get("glow", {}).items()}

    @property
    def hidden(self) -> list[str]:
        """Parts the build marked hidden at rest, and parts hanging from them."""
        meta = {p["name"]: p for p in self.summary.get("parts", [])}
        parents = {p.name: p.parent for p in self.scene.parts}

        def hidden(name: str | None) -> bool:
            while name is not None:
                if meta.get(name, {}).get("hidden"):
                    return True
                name = parents.get(name)
            return False

        return [p.name for p in self.parts if hidden(p.name)]

    @property
    def visible_grid(self) -> VoxelGrid:
        hidden = set(self.hidden)
        return VoxelGrid.merge([p.grid for p in self.parts if p.name not in hidden])

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
    load_particles(built, asset.vox.parent.parent)
    extra = check_hand_edits(built, asset.summary)
    return built, lint(built, style, rgba=asset.scene.palette, extra=extra)


def export_asset(
    path: str | Path,
    formats: Sequence[str] = NATIVE,
    out_dir: str | Path | None = None,
    force: bool = False,
    style: str | Path | None = None,
    vertex_colors: bool = False,
    ao: bool = False,
    center: bool = True,
    notes: list[str] | None = None,
) -> dict[str, str]:
    """Export to ``out_dir`` (default ``<project>/export``); returns {format: path}. A
    manifest (``<name>.cubist.json``) describing the parts and motion goes beside any .glb or
    .vox, under the key ``manifest``. ``notes`` collects warnings, such as the lint errors a
    forced export went past."""
    asset = load_asset(path)
    style_path = Path(style) if style else find_style(asset.vox)
    bible = load_project_style(style_path)
    built, report = lint_loaded(asset, bible)
    if not report.ok and not force:
        raise ExportBlocked(report)
    if not report.ok and notes is not None:
        errors = [ln for ln in report.summary().splitlines() if ln.startswith("ERROR")]
        notes.append(f"exported despite {len(errors)} lint error(s) (--force)")
        notes.extend(errors)
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
    native = {f: Path(p).name for f, p in written.items() if f in NATIVE}
    if native:
        target = out / f"{asset.name}.cubist.json"
        manifest = build_manifest(built, bible.voxel_size_m, files=native, center=center)
        target.write_text(json.dumps(manifest, indent=2) + "\n")
        written["manifest"] = str(target)
    return written
