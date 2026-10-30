"""The per-project style bible, ``style.toml``: palette, scale, budgets, rules, which colors
glow, and which ramps cycle.

Every section is optional. Unknown sections and keys are errors, so a typo never silently
turns a budget off.
"""

from __future__ import annotations

import difflib
import math
import os
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from cubist.core.palette import Palette, PaletteError

DEFAULT_VOXEL_SIZE_M = 0.0625
DITHER_RULES = ("allowed", "forbidden")
# The walk lint's walker, in voxels: its box's width (x and z) and height, and the ledge it climbs.
WALKER_KEYS = ("walker_width", "walker_height", "walker_climb")


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


@dataclass(frozen=True)
class Cycle:
    """A list ramp whose voxels step one shade along it ``fps`` times a second (water, lava)."""

    name: str
    ramp: str
    indices: tuple[int, ...]
    fps: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ramp": self.ramp,
            "indices": list(self.indices),
            "fps": self.fps,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Cycle:
        return cls(d["name"], d["ramp"], tuple(int(i) for i in d["indices"]), float(d["fps"]))


def cycle_table(table: np.ndarray, cycles, t: float) -> np.ndarray:
    """The palette table at time ``t``: each cycle's colors shifted ``floor(t * fps)`` shades
    along their ramp, wrapping around."""
    out = np.array(table, copy=True)
    for c in cycles:
        n = len(c.indices)
        ticks = int(np.floor(t * c.fps + 1e-9)) % n
        if ticks:
            for k, index in enumerate(c.indices):
                out[index] = table[c.indices[(k + ticks) % n]]
    return out


def glow_array(glow: dict[int, float] | None) -> np.ndarray:
    """Glow strength per palette index (0: not glowing), for the renderers."""
    out = np.zeros(256)
    for index, strength in (glow or {}).items():
        out[int(index)] = strength
    return out


@dataclass
class StyleBible:
    path: Path
    name: str
    voxel_size_m: float
    palette: Palette
    scale: dict[str, float] = field(default_factory=dict)
    budgets: Budgets = field(default_factory=Budgets)
    rules: Rules = field(default_factory=Rules)
    glow: dict[int, float] = field(default_factory=dict)
    cycles: list[Cycle] = field(default_factory=list)

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
            "glow": {self.palette.name_of(i): s for i, s in sorted(self.glow.items())},
            "cycles": [{"name": c.name, "ramp": c.ramp, "fps": c.fps} for c in self.cycles],
        }


_SECTIONS = {
    "project": {"name": str, "voxel_size_m": (int, float)},
    "palette": None,  # ramps are free-form; "source" is the one reserved key
    "scale": None,  # free-form numbers
    "budgets": {k: int for k in Budgets.__dataclass_fields__},
    "rules": {"no_pure_black": bool, "dither": str, "symmetry_tolerance": int, "notes": str},
    "glow": None,  # palette name -> strength
    "cycles": None,  # name -> {ramp, fps}
}


def _colors_named(palette: Palette, name: str, section: str) -> list[int]:
    """Indices for a palette name: a ramp or table (all its colors) or one shade."""
    obj = palette
    try:
        for piece in str(name).split("."):
            obj = obj[int(piece)] if piece.isdigit() else getattr(obj, piece)
    except (PaletteError, IndexError, TypeError, AttributeError) as e:
        raise StyleError(f"[{section}] {name!r}: not a palette color ({e})") from None
    if hasattr(obj, "index"):
        return [obj.index]
    return [c.index for c in obj]


def _glow(palette: Palette, table: dict) -> dict[int, float]:
    out: dict[int, float] = {}
    for name, strength in table.items():
        if isinstance(strength, dict):
            raise StyleError(
                f'[glow] {name} = {strength!r}: quote dotted names, like "{name}.'
                f'{next(iter(strength), "light")}" = 1.5 (TOML reads an unquoted dot as a table)'
            )
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or strength <= 0:
            raise StyleError(f"[glow] {name} must be a strength > 0, got {strength!r}")
        for index in _colors_named(palette, name, "glow"):
            out[index] = float(strength)
    return out


def _cycles(palette: Palette, table: dict) -> list[Cycle]:
    out = []
    for name, spec in table.items():
        if not isinstance(spec, dict) or set(spec) != {"ramp", "fps"}:
            raise StyleError(f'[cycles] {name} must be {{ ramp = "<ramp>", fps = <number> }}')
        ramp, fps = spec["ramp"], spec["fps"]
        if ramp not in palette.list_ramps():
            raise StyleError(
                f"[cycles] {name}: cycles step along a list ramp; {ramp!r} is not one (list "
                f"ramps: {', '.join(palette.list_ramps())})"
            )
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
            raise StyleError(f"[cycles] {name}: fps must be a number > 0, got {fps!r}")
        out.append(Cycle(name, ramp, tuple(c.index for c in getattr(palette, ramp)), float(fps)))
    return out


def _walker(scale: dict) -> None:
    """The walker keys come together, as whole voxels: a width and height of 1 or more, and a climb
    below the height. Whole floats (8.0) become ints."""
    given = [k for k in WALKER_KEYS if k in scale]
    if not given:
        return
    missing = [k for k in WALKER_KEYS if k not in scale]
    if missing:
        raise StyleError(
            f"[scale] has {', '.join(given)} but not {', '.join(missing)}: the walk lint needs all "
            "three (the walker's width, its height and the ledge it climbs, in voxels), or none"
        )
    for key in WALKER_KEYS:
        value = scale[key]
        low = 0 if key == "walker_climb" else 1
        high = scale["walker_height"] if key == "walker_climb" else math.inf
        if not float(value).is_integer() or not low <= value < high:
            limits = "0 or more, below walker_height" if key == "walker_climb" else "1 or more"
            raise StyleError(
                f"[scale] {key} must be a whole number of voxels, {limits}; got {value!r}"
            )
        scale[key] = int(value)


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
    _walker(scale)

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
        glow=_glow(palette, data.get("glow", {})),
        cycles=_cycles(palette, data.get("cycles", {})),
    )
