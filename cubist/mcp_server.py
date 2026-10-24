"""MCP server: the same functions as the CLI, for agents and clients that speak MCP.

Run with ``cubist mcp --project <dir>``. Tools that produce renders return them as inline
images so the agent sees them without opening files. There is deliberately no tool for
placing or deleting single voxels.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from cubist.build.runner import BuildError, build_script, load_project_style
from cubist.core.model import BuiltAsset
from cubist.examples import search_examples as find_examples
from cubist.export import ExportBlocked, LoadedAsset, export_asset, lint_loaded, load_asset
from cubist.io.vengi import VengiError
from cubist.io.vox import read_vox
from cubist.project import new_asset as scaffold
from cubist.render.ascii import parse_slices, slices
from cubist.render.contact_sheet import DEFAULT_VIEWS, asset_region, contact_sheet, hidden_note
from cubist.render.diff import change_grid, diff_sheet
from cubist.render.filmstrip import motion_diff, render_clip

INSTRUCTIONS = """Cubist builds game-ready voxel assets from Python scripts. Never place voxels one
at a time: write a script with the DSL (primitives, CSG, transforms, paint ops) and build it.

Workflow for each asset:
1. Read the brief (briefs/<name>.toml), call get_style_bible, and call search_examples for the
   2-3 closest approved examples; study their scripts and contact sheets.
2. Write the script (new_asset scaffolds a brief and stub) and call build_asset.
3. Inspect the contact sheet and the lint report.
4. Critique against this checklist: Is the silhouette readable at game distance? Do proportions
   match the style bible? Is the palette used as ramps? Is any asymmetry intentional? Does it
   meet every must-have in the brief?
5. Revise and rebuild (diff shows what changed). Stop after 5 iterations and report honestly what
   still falls short. Use render with slices for precise work on small regions.
Export only when lint has no errors."""


def _tool_errors(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except (BuildError, ExportBlocked, VengiError, ValueError, OSError) as e:
            raise ToolError(str(e)) from e

    return wrapper


def create_server(project: str | Path) -> MCPServer:
    project = Path(project).resolve()
    server = MCPServer("cubist", instructions=INSTRUCTIONS)

    def resolve(asset: str) -> Path:
        path = Path(asset)
        for candidate in (path, project / path, project / "out" / asset):
            if candidate.exists():
                return candidate
        raise ToolError(f"no built asset {asset!r}; build it first (looked in {project / 'out'})")

    @server.tool()
    @_tool_errors
    def build_asset(
        script_path: str | None = None,
        code: str | None = None,
        name: str | None = None,
        seed: int = 0,
    ) -> list:
        """Build an asset script and return its lint summary, output paths, and contact sheet.

        Pass script_path (relative to the project), or code plus name to save the code as
        assets/<name>.py first. seed reshuffles every seeded op."""
        if code is not None:
            if not name:
                raise ToolError("pass name with code; the script is saved as assets/<name>.py")
            script = project / "assets" / f"{name}.py"
            script.parent.mkdir(parents=True, exist_ok=True)
            script.write_text(code)
        elif script_path:
            script = Path(script_path) if Path(script_path).is_absolute() else project / script_path
        else:
            raise ToolError("pass script_path, or code and name")
        out = []
        for r in build_script(script, style=project / "style.toml", seed=seed):
            s = r.summary
            motion = s["outputs"].get("motion")
            gifs = s["outputs"].get("gifs", {})
            out.append(
                f"{s['name']}: {s['voxels']} voxels, {len(s['parts'])} part(s)\n"
                f"{s['lint']['summary']}\nvox: {r.vox_path}\ncontact sheet: {r.contact_path}"
                + (f"\nmotion: {motion}" if motion else "")
                + "".join(f"\n{name}.gif: {path}" for name, path in gifs.items())
            )
            out.append(Image(path=r.contact_path))
            if motion:
                out.append(Image(path=motion))
        return out

    @server.tool()
    @_tool_errors
    def render(
        asset: str,
        views: list[str] | None = None,
        slices: str | None = None,
        clip: str | None = None,
        at: float | None = None,
        frames: int = 8,
        view: str | None = None,
    ) -> list:
        """Re-render a built asset's contact sheet (views like front,right,back,top,iso,
        silhouette) and/or print ASCII slices (like y=10..14) for precise work. With clip: a
        filmstrip of that clip (frames, view), or with at (seconds) a contact sheet posed then."""
        loaded = load_asset(resolve(asset))
        if clip:
            built = BuiltAsset.from_vox(loaded.scene, loaded.summary, None)
            _, image = render_clip(
                built, loaded.scene.palette, clip, at=at, frames=frames, view=view,
                names=loaded.names,
            )  # fmt: skip
            path = loaded.vox.parent / "render.png"
            image.save(path)
            return [Image(path=path)]
        spec = slices  # the parameter shadows the module function; _slices calls that
        out: list = []
        if spec:
            axis, start, stop = parse_slices(spec)
            out.append(_slices(loaded, axis, start, stop))
        if views or not spec:
            sheet = contact_sheet(
                loaded.visible_grid,
                loaded.scene.palette,
                name=loaded.name,
                size=loaded.summary.get("size"),
                views=views or DEFAULT_VIEWS,
                names=loaded.names,
                note=hidden_note(loaded.hidden),
                glow=loaded.glow,
            )
            path = loaded.vox.parent / "render.png"
            sheet.save(path)
            out.append(Image(path=path))
        return out

    def _slices(loaded: LoadedAsset, axis: str, start: int, stop: int) -> str:
        size = loaded.summary.get("size")
        return slices(loaded.grid, axis, start, stop, loaded.names, asset_region(loaded.grid, size))

    @server.tool()
    @_tool_errors
    def lint(asset: str) -> str:
        """Lint a built asset against the style bible; returns the report as JSON."""
        loaded = load_asset(resolve(asset))
        _, report = lint_loaded(loaded, load_project_style(project / "style.toml"))
        return json.dumps(report.to_dict(), indent=2)

    @server.tool()
    @_tool_errors
    def inspect(asset: str) -> str:
        """Structure and stats of a built asset as JSON: parts, bounds, pivots, sockets,
        voxel counts, and colors by palette name."""
        loaded = load_asset(resolve(asset))
        parts = []
        for part in loaded.scene.parts:
            b = part.grid.bounds()
            meta = next((p for p in loaded.summary.get("parts", []) if p["name"] == part.name), {})
            parts.append(
                {
                    "name": part.name,
                    "parent": part.parent,
                    "voxels": part.grid.count(),
                    "bounds": [list(b[0]), list(b[1])] if b else None,
                    "pivot": meta.get("pivot"),
                }
            )
        return json.dumps(
            {
                "name": loaded.name,
                "size": loaded.summary.get("size"),
                "voxels": sum(p["voxels"] for p in parts),
                "parts": parts,
                "sockets": loaded.summary.get("sockets", {}),
                "colors": loaded.summary.get("colors", {}),
            },
            indent=2,
        )

    @server.tool()
    @_tool_errors
    def export(
        asset: str,
        formats: list[str] | None = None,
        force: bool = False,
        ao: bool = False,
        vertex_colors: bool = False,
    ) -> str:
        """Export a built asset (default glb and vox) to <project>/export. Blocked by lint
        errors unless force. Other formats go through vengi-voxconvert."""
        notes: list[str] = []
        written = export_asset(
            resolve(asset),
            formats=formats or ("glb", "vox"),
            force=force,
            style=project / "style.toml",
            ao=ao,
            vertex_colors=vertex_colors,
            notes=notes,
        )
        return json.dumps({**written, **({"warnings": notes} if notes else {})}, indent=2)

    @server.tool()
    @_tool_errors
    def get_style_bible() -> str:
        """The project's parsed style.toml: palette names and hex, scale, budgets, rules."""
        return json.dumps(load_project_style(project / "style.toml").to_dict(), indent=2)

    @server.tool()
    @_tool_errors
    def search_examples(
        tags: list[str] | None = None, category: str | None = None, limit: int = 3
    ) -> list:
        """The closest approved examples by category and tags: each one's note, script, and
        contact sheet."""
        found = find_examples(project, tags=tags, category=category, limit=limit)
        if not found:
            return ["no matching examples in the library yet"]
        out: list = []
        for ex in found:
            script = Path(ex["script"]).read_text()
            out.append(
                f"# example: {ex['name']} ({ex['category']}; tags: "
                f"{', '.join(ex['tags'])})\n# why it is good: {ex['note']}\n\n{script}"
            )
            out.append(Image(path=ex["contact_sheet"]))
            if ex.get("filmstrip"):
                out.append(Image(path=ex["filmstrip"]))
        return out

    @server.tool()
    @_tool_errors
    def diff(asset: str) -> list:
        """Compare a built asset with its previous build: before, after, and what changed."""
        after = load_asset(resolve(asset))
        prev = after.vox.with_name(f"{after.vox.stem}.prev.vox")
        if not prev.exists():
            raise ToolError(f"{after.name} has no previous build to compare")
        before_scene = read_vox(prev)
        before = LoadedAsset(prev, before_scene, after.summary)
        _, counts = change_grid(before.grid, after.grid)
        sheet = diff_sheet(
            before.grid,
            after.grid,
            after.scene.palette,
            name=after.name,
            size=after.summary.get("size"),
            names=after.names,
            before_palette=before_scene.palette,
        )
        path = after.vox.parent / "diff.png"
        sheet.save(path)
        summary = (
            f"{counts['added']} added, {counts['removed']} removed, "
            f"{counts['recolored']} recolored, {counts['same']} unchanged"
        )
        out = [summary, Image(path=path)]
        folder = after.vox.parent
        strips = motion_diff(folder / "motion.prev.png", folder / "motion.png")
        if strips is not None:
            strips.save(folder / "motion-diff.png")
            out.append(Image(path=folder / "motion-diff.png"))
        return out

    @server.tool()
    @_tool_errors
    def new_asset(
        name: str,
        category: str = "prop",
        tags: list[str] | None = None,
        description: str = "",
        must_haves: list[str] | None = None,
        size: list[int] | None = None,
    ) -> str:
        """Scaffold briefs/<name>.toml and a buildable assets/<name>.py stub."""
        brief, script = scaffold(
            project,
            name,
            category=category,
            size=tuple(size) if size else None,
            tags=tags,
            description=description,
            must_haves=must_haves,
        )
        return (
            f"brief: {brief.relative_to(project)}\nscript: {script.relative_to(project)}\n"
            "Fill in the brief's must-haves, then replace the stub and build_asset."
        )

    return server
