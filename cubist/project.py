"""Projects and briefs.

A project is a directory with a ``style.toml``; inside it ``briefs/``, ``assets/``,
``examples/``, ``out/`` and ``export/``. Every asset starts from a brief:
``briefs/<name>.toml`` with name, category, size, tags, description, must-haves, and
optional reference images.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from cubist.build.runner import load_project_style

CATEGORY_SIZES = {
    "prop": (16, 16, 16),
    "character": (16, 24, 12),
    "creature": (24, 24, 24),
    "building": (32, 32, 32),
    "vehicle": (24, 16, 40),
    "terrain": (32, 8, 32),
}
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class ProjectError(ValueError):
    """A project or brief operation can't proceed."""


def find_project(start: str | Path) -> Path:
    start = Path(start).resolve()
    for directory in [start, *start.parents]:
        if (directory / "style.toml").is_file():
            return directory
    raise ProjectError(f"no style.toml in {start} or any directory above it")


def _toml_value(value) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise ProjectError(f"can't write {value!r} to TOML")


def dump_toml(table: dict) -> str:
    """Flat tables of strings, numbers, booleans, and lists only."""
    return "".join(f"{k} = {_toml_value(v)}\n" for k, v in table.items())


def load_brief(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def brief_path(project: Path, name: str) -> Path:
    return project / "briefs" / f"{name}.toml"


_STUB = """from cubist import *

# Brief: briefs/{name}.toml
# {description}


@asset(name="{name}", size={size}, tags={tags}, grounded=True)
def {name}(p):
    # Replace this placeholder block with the asset. See examples/ for approved scripts.
    return box({size}, color=p.{color})
"""


def new_asset(
    project: str | Path,
    name: str,
    category: str = "prop",
    size: tuple[int, int, int] | None = None,
    tags: list[str] | None = None,
    description: str = "",
    must_haves: list[str] | None = None,
) -> tuple[Path, Path]:
    """Scaffold ``briefs/<name>.toml`` and ``assets/<name>.py``; returns both paths."""
    project = Path(project)
    if not _NAME.match(name):
        raise ProjectError(f"asset name {name!r} must be lowercase letters, digits, underscores")
    if category not in CATEGORY_SIZES:
        raise ProjectError(f"category must be one of {', '.join(CATEGORY_SIZES)}")
    brief = brief_path(project, name)
    script = project / "assets" / f"{name}.py"
    for path in (brief, script):
        if path.exists():
            raise ProjectError(f"{path} already exists; pick another name for {name!r}")
    size = tuple(size or CATEGORY_SIZES[category])
    tags = [category] + [t for t in (tags or []) if t != category]
    palette = load_project_style(project / "style.toml").palette
    colors = palette.colors()
    color = "source[0]"
    for ramp in palette.ramp_names():
        entry = getattr(palette, ramp)
        if hasattr(entry, "ramp") and ramp != "source":
            color = f"{ramp}.mid"
            break
        if not hasattr(entry, "ramp"):
            color = ramp
            break
    if not colors:
        raise ProjectError("the style palette is empty; add colors to style.toml first")
    brief.parent.mkdir(parents=True, exist_ok=True)
    script.parent.mkdir(parents=True, exist_ok=True)
    brief.write_text(
        dump_toml(
            {
                "name": name,
                "category": category,
                "size": list(size),
                "tags": tags,
                "description": description,
                "must_haves": list(must_haves or []),
                "references": [],
            }
        )
    )
    script.write_text(
        _STUB.format(
            name=name,
            size=size,
            tags=json.dumps(tags),
            description=description or "(describe the asset in the brief)",
            color=color,
        )
    )
    return brief, script
