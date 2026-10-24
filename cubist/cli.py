"""The ``cubist`` command line. Commands print JSON on stdout (ASCII slices print as text);
errors go to stderr. Exit codes: 0 ok, 1 the command failed, 2 lint found errors."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from cubist.build.runner import (
    DEFAULT_TIMEOUT_S,
    BuildError,
    build_script,
    find_style,
    load_project_style,
)
from cubist.core.grid import VoxelGrid
from cubist.core.model import BuiltAsset
from cubist.io.vox import VoxError, VoxScene, read_vox
from cubist.lint import lint
from cubist.render.ascii import parse_slices, slices
from cubist.render.contact_sheet import DEFAULT_VIEWS, asset_region, contact_sheet


def _hex(rgba) -> str:
    r, g, b, a = (int(v) for v in rgba)
    return f"#{r:02x}{g:02x}{b:02x}" + ("" if a == 255 else f"{a:02x}")


def _print(data) -> None:
    print(json.dumps(data, indent=2))


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


def load_asset(path: Path) -> LoadedAsset:
    """A built asset from its .vox file or its out/<name>/ directory."""
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


def cmd_build(args: argparse.Namespace) -> int:
    results = build_script(
        args.script,
        style=args.style,
        out=args.out,
        timeout=args.timeout,
        seed=args.seed,
        check_determinism=args.check_determinism,
    )
    _print([r.summary for r in results])
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    scene = read_vox(args.asset)
    parts, totals = [], {}
    for part in scene.parts:
        b = part.grid.bounds()
        parts.append(
            {
                "name": part.name,
                "voxels": part.grid.count(),
                "bounds": [list(b[0]), list(b[1])] if b else None,
            }
        )
        for index, n in part.grid.colors_used().items():
            totals[index] = totals.get(index, 0) + n
    _print(
        {
            "path": str(args.asset),
            "voxels": sum(p["voxels"] for p in parts),
            "parts": parts,
            "colors": {_hex(scene.palette[i]): n for i, n in sorted(totals.items())},
        }
    )
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    asset = load_asset(args.asset)
    grid, size = asset.grid, asset.summary.get("size")
    name = asset.summary.get("name", asset.vox.stem)
    if args.slices:
        axis, start, stop = parse_slices(args.slices)
        print(slices(grid, axis, start, stop, asset.names, asset_region(grid, size)), end="")
    if args.views or not args.slices:
        views = args.views.split(",") if args.views else DEFAULT_VIEWS
        out = args.out or asset.vox.parent / "contact.png"
        sheet = contact_sheet(
            grid, asset.scene.palette, name=name, size=size, views=views, names=asset.names
        )
        sheet.save(out)
        _print({"contact_sheet": str(out)})
    return 0


def cmd_lint(args: argparse.Namespace) -> int:
    asset = load_asset(args.asset)
    style = load_project_style(args.style or find_style(asset.vox))
    built = BuiltAsset.from_vox(asset.scene, asset.summary, style.palette)
    report = lint(built, style, rgba=asset.scene.palette)
    _print(report.to_dict())
    return 0 if report.ok else 2


def cmd_style(args: argparse.Namespace) -> int:
    style_path = args.style or find_style(Path.cwd() / "_")
    _print(load_project_style(style_path).to_dict())
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cubist", description="Build voxel assets from scripts.")
    sub = ap.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="build a script: .vox, contact sheet, lint report")
    b.add_argument("script", type=Path)
    b.add_argument("--style", type=Path, help="style.toml (default: nearest above the script)")
    b.add_argument("--out", type=Path, help="output root (default: <project>/out)")
    b.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds")
    b.add_argument("--seed", type=int, default=0, help="build seed mixed into every seeded op")
    b.add_argument(
        "--check-determinism",
        action="store_true",
        help="build twice in separate processes and compare (for CI)",
    )
    b.set_defaults(fn=cmd_build)

    r = sub.add_parser("render", help="re-render the contact sheet or print ASCII slices")
    r.add_argument("asset", type=Path, help="a .vox file or its out/<name>/ directory")
    r.add_argument("--views", help=f"comma-separated (default: {','.join(DEFAULT_VIEWS)})")
    r.add_argument("--slices", help="layers to print, like y=10..14 or x=3")
    r.add_argument("--out", type=Path, help="contact sheet path (default: beside the .vox)")
    r.set_defaults(fn=cmd_render)

    li = sub.add_parser("lint", help="lint a built asset (exit 2 when there are errors)")
    li.add_argument("asset", type=Path, help="a .vox file or its out/<name>/ directory")
    li.add_argument("--style", type=Path, help="style.toml (default: nearest above the asset)")
    li.set_defaults(fn=cmd_lint)

    i = sub.add_parser("inspect", help="parts, bounds, and color counts of a .vox file")
    i.add_argument("asset", type=Path)
    i.set_defaults(fn=cmd_inspect)

    s = sub.add_parser("style", help="print the parsed style bible with palette names")
    s.add_argument("--style", type=Path, help="style.toml (default: nearest above the cwd)")
    s.set_defaults(fn=cmd_style)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.fn(args)
    except (BuildError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
