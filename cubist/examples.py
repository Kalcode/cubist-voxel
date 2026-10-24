"""The example library: approved assets the agent studies before writing a new script.

``examples/<name>/`` holds the script, its contact sheet, its build summary, and
``meta.toml`` (category, tags, and a note on why it is good).
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import tomllib
from pathlib import Path

from cubist.build.runner import load_project_style
from cubist.export import lint_loaded, load_asset
from cubist.project import brief_path, dump_toml, load_brief


class ExampleError(ValueError):
    """An asset can't be promoted to the library."""


def promote(
    project: str | Path, name: str, note: str, approved_by: str = "user", force: bool = False
) -> Path:
    """Copy a built asset into ``examples/<name>/`` once it is approved."""
    project = Path(project)
    out = project / "out" / name
    if not (out / f"{name}.vox").exists():
        raise ExampleError(f"no build for {name!r} in {out}; run `cubist build` first")
    asset = load_asset(out)
    _, report = lint_loaded(asset, load_project_style(project / "style.toml"))
    if not report.ok and not force:
        raise ExampleError(
            f"{name} has lint errors; examples must be lint-clean\n" + report.summary()
        )
    summary = asset.summary
    script = Path(summary.get("script", ""))
    if not script.is_file():
        raise ExampleError(f"the script for {name!r} ({script}) is missing")
    brief = load_brief(brief_path(project, name)) if brief_path(project, name).exists() else {}
    tags = list(brief.get("tags") or summary.get("tags") or [])
    category = brief.get("category") or (tags[0] if tags else "prop")
    dest = project / "examples" / name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(script, dest / f"{name}.py")
    shutil.copyfile(out / "contact.png", dest / "contact.png")
    # The library is shared: keep the summary, but no paths from this machine.
    portable = {k: v for k, v in summary.items() if k not in ("outputs",)}
    portable["script"] = f"{name}.py"
    (dest / "build.json").write_text(json.dumps(portable, indent=2) + "\n")
    (dest / "meta.toml").write_text(
        dump_toml(
            {
                "name": name,
                "category": category,
                "tags": tags,
                "note": note,
                "approved_by": approved_by,
                "date": dt.date.today().isoformat(),
            }
        )
    )
    return dest


def list_examples(project: str | Path) -> list[dict]:
    root = Path(project) / "examples"
    out = []
    for meta_path in sorted(root.glob("*/meta.toml")):
        with open(meta_path, "rb") as f:
            meta = tomllib.load(f)
        folder = meta_path.parent
        out.append(
            {
                **meta,
                "script": str(folder / f"{meta['name']}.py"),
                "contact_sheet": str(folder / "contact.png"),
                "voxels": json.loads((folder / "build.json").read_text()).get("voxels")
                if (folder / "build.json").exists()
                else None,
            }
        )
    return out


def search_examples(
    project: str | Path, tags: list[str] | None = None, category: str | None = None, limit: int = 3
) -> list[dict]:
    """Closest examples: 2 points for a category match, 1 per shared tag; best first."""
    wanted = set(tags or [])
    scored = []
    for ex in list_examples(project):
        score = len(wanted & set(ex.get("tags", [])))
        if category and ex.get("category") == category:
            score += 2
        if score > 0:
            scored.append((-score, ex["name"], ex))
    return [ex for _, _, ex in sorted(scored)[:limit]]
