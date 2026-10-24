"""The ``cubist`` command line. Commands print JSON on stdout (ASCII slices print as text);
errors go to stderr. Exit codes: 0 ok, 1 the command failed, 2 lint found errors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cubist.build.runner import (
    DEFAULT_TIMEOUT_S,
    BuildError,
    build_script,
    find_style,
    load_project_style,
)
from cubist.examples import ExampleError, list_examples, promote, search_examples
from cubist.export import NATIVE, ExportBlocked, export_asset, lint_loaded, load_asset
from cubist.io.vengi import VengiError
from cubist.io.vox import read_vox
from cubist.palette_import import import_palette
from cubist.project import ProjectError, find_project, new_asset
from cubist.render.ascii import parse_slices, slices
from cubist.render.contact_sheet import DEFAULT_VIEWS, asset_region, contact_sheet
from cubist.render.diff import diff_sheet


def _hex(rgba) -> str:
    r, g, b, a = (int(v) for v in rgba)
    return f"#{r:02x}{g:02x}{b:02x}" + ("" if a == 255 else f"{a:02x}")


def _print(data) -> None:
    print(json.dumps(data, indent=2))


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
    asset = load_asset(args.asset)
    names = asset.names
    meta = {p["name"]: p for p in asset.summary.get("parts", [])}
    parts, totals = [], {}
    for part in asset.scene.parts:
        b = part.grid.bounds()
        info = meta.get(part.name, {})
        parts.append(
            {
                "name": part.name,
                "parent": part.parent,
                "voxels": part.grid.count(),
                "bounds": [list(b[0]), list(b[1])] if b else None,
                "pivot": info.get("pivot"),
            }
        )
        for index, n in part.grid.colors_used().items():
            totals[index] = totals.get(index, 0) + n
    _print(
        {
            "path": str(asset.vox),
            "voxels": asset.grid.count(),
            "parts": parts,
            "sockets": asset.summary.get("sockets", {}),
            # palette names when the build summary is beside the file, else hex
            "colors": {
                names.get(i, _hex(asset.scene.palette[i])): n for i, n in sorted(totals.items())
            },
        }
    )
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    asset = load_asset(args.asset)
    grid, size, name = asset.grid, asset.summary.get("size"), asset.name
    if args.slices:
        axis, start, stop = parse_slices(args.slices)
        print(slices(grid, axis, start, stop, asset.names, asset_region(grid, size)), end="")
    if args.views or not args.slices:
        views = args.views.split(",") if args.views else DEFAULT_VIEWS
        out = args.out or asset.vox.parent / "render.png"  # contact.png belongs to the build
        sheet = contact_sheet(
            grid, asset.scene.palette, name=name, size=size, views=views, names=asset.names
        )
        sheet.save(out)
        _print({"contact_sheet": str(out)})
    return 0


def cmd_lint(args: argparse.Namespace) -> int:
    asset = load_asset(args.asset)
    style = load_project_style(args.style or find_style(asset.vox))
    _, report = lint_loaded(asset, style)
    _print(report.to_dict())
    return 0 if report.ok else 2


def cmd_export(args: argparse.Namespace) -> int:
    formats = [f for f in NATIVE if getattr(args, f)] + (args.via_vengi or [])
    try:
        written = export_asset(
            args.asset,
            formats=formats or NATIVE,
            out_dir=args.out,
            force=args.force,
            style=args.style,
            vertex_colors=args.vertex_colors,
            ao=args.ao,
            center=not args.no_center,
        )
    except ExportBlocked as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    _print({"exported": written})
    return 0


def _project(args: argparse.Namespace) -> Path:
    return find_project(args.project or Path.cwd())


def _csv(value: str | None) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()] if value else []


def cmd_new(args: argparse.Namespace) -> int:
    size = tuple(int(v) for v in _csv(args.size)) if args.size else None
    if size is not None and len(size) != 3:
        raise ProjectError("--size takes three numbers, like 12,16,12")
    brief, script = new_asset(
        _project(args),
        args.name,
        category=args.category,
        size=size,
        tags=_csv(args.tags),
        description=args.description or "",
        must_haves=args.must_have or [],
    )
    _print({"brief": str(brief), "script": str(script)})
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    after = load_asset(args.asset)
    if args.other:
        before, after = after, load_asset(args.other)
    else:
        prev = after.vox.with_name(f"{after.vox.stem}.prev.vox")
        if not prev.exists():
            raise ProjectError(f"no previous build of {after.name} to compare ({prev.name})")
        from cubist.export import LoadedAsset

        before = LoadedAsset(prev, read_vox(prev), after.summary)
    sheet = diff_sheet(
        before.grid,
        after.grid,
        after.scene.palette,
        name=after.name,
        size=after.summary.get("size"),
        names=after.names,
        before_palette=before.scene.palette,
    )
    out = args.out or after.vox.parent / "diff.png"
    sheet.save(out)
    from cubist.render.diff import change_grid

    _, counts = change_grid(before.grid, after.grid)
    _print({"diff_sheet": str(out), **counts})
    return 0


def cmd_palette_import(args: argparse.Namespace) -> int:
    style = args.style or find_style(Path.cwd() / "_")
    colors = import_palette(style, args.file)
    _print({"style": str(style), "source": Path(args.file).name, "colors": colors})
    return 0


def cmd_examples(args: argparse.Namespace) -> int:
    project = _project(args)
    if args.action == "list":
        _print(list_examples(project))
    elif args.action == "search":
        _print(
            search_examples(project, tags=_csv(args.tags), category=args.category, limit=args.limit)
        )
    else:
        if not args.name or not args.note:
            raise ExampleError("promote needs an asset name and --note on why it is good")
        dest = promote(
            project, args.name, note=args.note, approved_by=args.approved_by, force=args.force
        )
        _print({"example": str(dest)})
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from cubist.evaluation import evaluate, write_report

    project = _project(args)
    report = evaluate(project)
    paths = write_report(project, report)
    _print({"summary": report["summary"], **{k: str(v) for k, v in paths.items()}})
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from cubist.mcp_server import create_server

    create_server(_project(args)).run("stdio")
    return 0


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
    r.add_argument("--out", type=Path, help="image path (default: render.png beside the .vox)")
    r.set_defaults(fn=cmd_render)

    li = sub.add_parser("lint", help="lint a built asset (exit 2 when there are errors)")
    li.add_argument("asset", type=Path, help="a .vox file or its out/<name>/ directory")
    li.add_argument("--style", type=Path, help="style.toml (default: nearest above the asset)")
    li.set_defaults(fn=cmd_lint)

    e = sub.add_parser("export", help="export .glb/.vox (blocked by lint errors unless --force)")
    e.add_argument("asset", type=Path, help="a .vox file or its out/<name>/ directory")
    e.add_argument("--glb", action="store_true", help="glTF binary (default: --glb --vox)")
    e.add_argument("--vox", action="store_true", help="MagicaVoxel .vox")
    e.add_argument(
        "--via-vengi",
        action="append",
        metavar="FORMAT",
        help="any format vengi-voxconvert writes (obj, qb, schematic, ...)",
    )
    e.add_argument("--out", type=Path, help="output directory (default: <project>/export)")
    e.add_argument("--force", action="store_true", help="export despite lint errors")
    e.add_argument(
        "--vertex-colors",
        action="store_true",
        help="colors in vertices instead of a palette texture",
    )
    e.add_argument("--ao", action="store_true", help="bake ambient occlusion into vertex colors")
    e.add_argument(
        "--no-center",
        action="store_true",
        help="keep the asset's origin instead of centering its footprint",
    )
    e.add_argument("--style", type=Path, help="style.toml (default: nearest above the asset)")
    e.set_defaults(fn=cmd_export)

    i = sub.add_parser("inspect", help="parts, bounds, pivots, and color counts as JSON")
    i.add_argument("asset", type=Path, help="a .vox file or its out/<name>/ directory")
    i.set_defaults(fn=cmd_inspect)

    n = sub.add_parser("new", help="scaffold a brief and a script stub for a new asset")
    n.add_argument("name")
    n.add_argument("--category", default="prop", help="prop, character, creature, building, ...")
    n.add_argument("--size", help="x,y,z in voxels (default by category)")
    n.add_argument("--tags", help="comma-separated")
    n.add_argument("--description")
    n.add_argument("--must-have", action="append", help="repeat for each must-have")
    n.add_argument("--project", type=Path, help="project directory (default: from cwd)")
    n.set_defaults(fn=cmd_new)

    d = sub.add_parser("diff", help="before/after/changes sheet (default: vs. the last build)")
    d.add_argument("asset", type=Path, help="the newer asset (or the older one with OTHER)")
    d.add_argument("other", type=Path, nargs="?", help="compare ASSET (before) to OTHER (after)")
    d.add_argument("--out", type=Path, help="sheet path (default: diff.png beside the asset)")
    d.set_defaults(fn=cmd_diff)

    pal = sub.add_parser("palette", help="palette tools")
    pal_sub = pal.add_subparsers(dest="action", required=True)
    pi = pal_sub.add_parser("import", help="set a palette file as the style's palette source")
    pi.add_argument("file", type=Path, help=".png, .gif, .gpl, or .hex")
    pi.add_argument("--style", type=Path, help="style.toml (default: nearest above the cwd)")
    pi.set_defaults(fn=cmd_palette_import)

    ex = sub.add_parser("examples", help="the example library")
    ex.add_argument("action", choices=("list", "search", "promote"))
    ex.add_argument("name", nargs="?", help="asset to promote")
    ex.add_argument("--tags", help="comma-separated, for search")
    ex.add_argument("--category", help="for search")
    ex.add_argument("--limit", type=int, default=3)
    ex.add_argument("--note", help="for promote: one paragraph on why it is good")
    ex.add_argument("--approved-by", default="user")
    ex.add_argument("--force", action="store_true", help="promote despite lint errors")
    ex.add_argument("--project", type=Path, help="project directory (default: from cwd)")
    ex.set_defaults(fn=cmd_examples)

    ev = sub.add_parser("eval", help="evaluate every brief: lint, iterations, scores, gallery")
    ev.add_argument("--project", type=Path, help="project directory (default: from cwd)")
    ev.set_defaults(fn=cmd_eval)

    m = sub.add_parser("mcp", help="run the MCP server on stdio")
    m.add_argument("--project", type=Path, help="project directory (default: from cwd)")
    m.set_defaults(fn=cmd_mcp)

    s = sub.add_parser("style", help="print the parsed style bible with palette names")
    s.add_argument("--style", type=Path, help="style.toml (default: nearest above the cwd)")
    s.set_defaults(fn=cmd_style)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.fn(args)
    except (BuildError, VengiError, ValueError, OSError) as e:  # incl. Project/ExampleError
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
