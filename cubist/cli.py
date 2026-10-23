"""The ``cubist`` command line. Commands print JSON on stdout (ASCII slices print as text);
errors go to stderr."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cubist.build.runner import DEFAULT_TIMEOUT_S, BuildError, build_script
from cubist.core.grid import VoxelGrid
from cubist.io.vox import VoxError, read_vox
from cubist.render.ascii import parse_slices, slices
from cubist.render.contact_sheet import DEFAULT_VIEWS, asset_region, contact_sheet


def _hex(rgba) -> str:
    r, g, b, a = (int(v) for v in rgba)
    return f"#{r:02x}{g:02x}{b:02x}" + ("" if a == 255 else f"{a:02x}")


def cmd_build(args: argparse.Namespace) -> int:
    results = build_script(args.script, style=args.style, out=args.out, timeout=args.timeout)
    print(json.dumps([r.summary for r in results], indent=2))
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
    info = {
        "path": str(args.asset),
        "voxels": sum(p["voxels"] for p in parts),
        "parts": parts,
        "colors": {_hex(scene.palette[i]): n for i, n in sorted(totals.items())},
    }
    print(json.dumps(info, indent=2))
    return 0


def _load_asset(path: Path):
    """A built asset from its .vox file or its out/<name>/ directory, plus build.json if any."""
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
    scene = read_vox(vox)
    grid = VoxelGrid.merge([p.grid for p in scene.parts])
    names = {int(k): v for k, v in summary.get("palette_names", {}).items()}
    return vox, grid, scene.palette, names, summary.get("size"), summary.get("name", vox.stem)


def cmd_render(args: argparse.Namespace) -> int:
    vox, grid, table, names, size, name = _load_asset(args.asset)
    if args.slices:
        axis, start, stop = parse_slices(args.slices)
        print(slices(grid, axis, start, stop, names, asset_region(grid, size)), end="")
    if args.views or not args.slices:
        views = args.views.split(",") if args.views else DEFAULT_VIEWS
        out = args.out or vox.parent / "contact.png"
        contact_sheet(grid, table, name=name, size=size, views=views, names=names).save(out)
        print(json.dumps({"contact_sheet": str(out)}))
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cubist", description="Build voxel assets from scripts.")
    sub = ap.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="build a script to out/<name>/<name>.vox")
    b.add_argument("script", type=Path)
    b.add_argument("--style", type=Path, help="style.toml (default: nearest above the script)")
    b.add_argument("--out", type=Path, help="output root (default: <project>/out)")
    b.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds")
    b.set_defaults(fn=cmd_build)

    r = sub.add_parser("render", help="re-render the contact sheet or print ASCII slices")
    r.add_argument("asset", type=Path, help="a .vox file or its out/<name>/ directory")
    r.add_argument("--views", help=f"comma-separated (default: {','.join(DEFAULT_VIEWS)})")
    r.add_argument("--slices", help="layers to print, like y=10..14 or x=3")
    r.add_argument("--out", type=Path, help="contact sheet path (default: beside the .vox)")
    r.set_defaults(fn=cmd_render)

    i = sub.add_parser("inspect", help="parts, bounds, and color counts of a .vox file")
    i.add_argument("asset", type=Path)
    i.set_defaults(fn=cmd_inspect)
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
