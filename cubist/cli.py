"""The ``cubist`` command line. Every command prints JSON on stdout; errors go to stderr."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cubist.build.runner import DEFAULT_TIMEOUT_S, BuildError, build_script
from cubist.io.vox import VoxError, read_vox


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


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cubist", description="Build voxel assets from scripts.")
    sub = ap.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="build a script to out/<name>/<name>.vox")
    b.add_argument("script", type=Path)
    b.add_argument("--style", type=Path, help="style.toml (default: nearest above the script)")
    b.add_argument("--out", type=Path, help="output root (default: <project>/out)")
    b.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds")
    b.set_defaults(fn=cmd_build)

    i = sub.add_parser("inspect", help="parts, bounds, and color counts of a .vox file")
    i.add_argument("asset", type=Path)
    i.set_defaults(fn=cmd_inspect)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.fn(args)
    except (BuildError, VoxError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
