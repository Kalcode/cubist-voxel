"""The per-project style bible, ``style.toml``: palette, scale, budgets, and rules.

Every section is optional. Unknown sections and keys are errors, so a typo never silently
turns a budget off.
"""

from __future__ import annotations

import difflib
import os
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from cubist.core.palette import Palette, PaletteError

DEFAULT_VOXEL_SIZE_M = 0.0625
DITHER_RULES = ("allowed", "forbidden")


class StyleError(ValueError):
    """style.toml is missing, malformed, or has invalid values."""


@dataclass
class Budgets:
    max_colors: int | None = None
    max_voxels: int | None = None
    max_triangles: int | None = None
    max_parts: int | None = None


@dataclass
class Rules:
    no_pure_black: bool = False
    dither: str = "allowed"
    symmetry_tolerance: int = 0
    notes: str = ""


@dataclass
class StyleBible:
    path: Path
    name: str
    voxel_size_m: float
    palette: Palette
    scale: dict[str, float] = field(default_factory=dict)
    budgets: Budgets = field(default_factory=Budgets)
    rules: Rules = field(default_factory=Rules)

    def to_dict(self) -> dict:
        def hex_of(rgba):
            return "#" + "".join(f"{v:02x}" for v in rgba[:3])

        return {
            "path": str(self.path),
            "name": self.name,
            "voxel_size_m": self.voxel_size_m,
            "palette": [
                {"index": c.index, "name": c.name, "hex": hex_of(c.rgba)}
                for c in self.palette.colors()
            ],
            "ramps": self.palette.list_ramps(),
            "named": [n for n in self.palette.ramp_names() if n not in self.palette.list_ramps()],
            "scale": self.scale,
            "budgets": asdict(self.budgets),
            "rules": asdict(self.rules),
        }


_SECTIONS = {
    "project": {"name": str, "voxel_size_m": (int, float)},
    "palette": None,  # ramps are free-form; "source" is the one reserved key
    "scale": None,  # free-form numbers
    "budgets": {k: int for k in Budgets.__dataclass_fields__},
    "rules": {"no_pure_black": bool, "dither": str, "symmetry_tolerance": int, "notes": str},
}


def _unknown(kind: str, name: str, known) -> StyleError:
    close = difflib.get_close_matches(name, list(known), n=1)
    hint = f" Did you mean {close[0]!r}?" if close else ""
    return StyleError(f"unknown {kind} {name!r}.{hint} Known: {', '.join(known)}.")


def _checked(section: str, table: dict) -> dict:
    schema = _SECTIONS[section]
    for key, value in table.items():
        if key not in schema:
            raise _unknown(f"key in [{section}]", key, schema)
        expected = schema[key]
        if isinstance(value, bool) and expected is not bool:
            ok = False
        else:
            ok = isinstance(value, expected)
        if not ok:
            raise StyleError(f"[{section}] {key} has the wrong type: {value!r}")
    return table


def read_palette_file(path: str | os.PathLike) -> list[str]:
    """Hex colors from a palette file: .png/.gif (distinct pixels in reading order),
    .gpl (GIMP), or .hex/.txt (one hex color per line)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".png", ".gif"):
        with Image.open(path) as img:
            pixels = np.asarray(img.convert("RGBA")).reshape(-1, 4)
        seen: dict[str, None] = {}
        for r, g, b, a in pixels.tolist():
            if a > 0:
                seen.setdefault(f"#{r:02x}{g:02x}{b:02x}")
        return list(seen)
    lines = path.read_text().splitlines()
    if suffix == ".gpl":
        colors = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
                r, g, b = (int(p) for p in parts[:3])
                colors.append(f"#{r:02x}{g:02x}{b:02x}")
        return colors
    if suffix in (".hex", ".txt"):
        out = []
        for line in lines:
            value = line.strip().lstrip("#")
            if value and not value.startswith(";"):
                out.append("#" + value.lower())
        return out
    raise StyleError(f"unsupported palette file {path.name}; use .png, .gif, .gpl, or .hex")


def load_style(path: str | os.PathLike) -> StyleBible:
    path = Path(path)
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError as e:
        raise StyleError(f"style bible not found: {path}") from e
    except tomllib.TOMLDecodeError as e:
        raise StyleError(f"{path}: {e}") from e

    for section in data:
        if section not in _SECTIONS:
            raise _unknown("section", section, _SECTIONS)
    project = _checked("project", data.get("project", {}))
    budgets = Budgets(**_checked("budgets", data.get("budgets", {})))
    rules = Rules(**_checked("rules", data.get("rules", {})))
    if rules.dither not in DITHER_RULES:
        raise StyleError(f"[rules] dither must be one of {', '.join(DITHER_RULES)}")
    scale = data.get("scale", {})
    for key, value in scale.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StyleError(f"[scale] {key} must be a number, got {value!r}")

    entries = dict(data.get("palette", {}))
    source = None
    if "source" in entries:
        source_path = path.parent / str(entries.pop("source"))
        try:
            source = read_palette_file(source_path)
        except FileNotFoundError as e:
            raise StyleError(f"palette source not found: {source_path}") from e
    try:
        palette = Palette(entries, source=source)
    except PaletteError as e:
        raise StyleError(f"{path}: {e}") from e

    return StyleBible(
        path=path,
        name=project.get("name", path.parent.name),
        voxel_size_m=float(project.get("voxel_size_m", DEFAULT_VOXEL_SIZE_M)),
        palette=palette,
        scale=dict(scale),
        budgets=budgets,
        rules=rules,
    )
