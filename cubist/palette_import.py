"""``cubist palette import``: point a style bible at a palette file, keeping its comments."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from cubist.core.palette import PaletteError, parse_hex
from cubist.core.style import StyleError, load_style, read_palette_file

_SECTION = re.compile(r"^\s*\[([^\]]+)\]\s*$")
_SOURCE = re.compile(r"^\s*source\s*=")


def import_palette(style_path: str | Path, palette_file: str | Path) -> list[dict]:
    """Copy ``palette_file`` next to ``style.toml`` (if it isn't there) and set it as the
    ``[palette] source``. Returns the imported colors with their ``source.i`` names."""
    style_path, palette_file = Path(style_path), Path(palette_file)
    try:
        colors = read_palette_file(palette_file)
    except (OSError, UnicodeDecodeError) as e:
        raise StyleError(f"can't read palette file {palette_file}: {e}") from e
    if not colors:
        raise StyleError(f"{palette_file} has no colors")
    for hx in colors:
        try:
            parse_hex(hx)
        except PaletteError as e:
            raise StyleError(f"{palette_file}: {e}") from e

    target = style_path.parent / palette_file.name
    if palette_file.resolve() != target.resolve():
        shutil.copyfile(palette_file, target)

    line = f"source = {json.dumps(target.name)}\n"
    lines = style_path.read_text().splitlines(keepends=True)
    start = next(
        (
            i
            for i, ln in enumerate(lines)
            if (m := _SECTION.match(ln)) and m.group(1).strip() == "palette"
        ),
        None,
    )
    if start is None:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines += ["\n", "[palette]\n", line]
    else:
        end = next(
            (i for i in range(start + 1, len(lines)) if _SECTION.match(lines[i])), len(lines)
        )
        existing = next((i for i in range(start + 1, end) if _SOURCE.match(lines[i])), None)
        if existing is not None:
            lines[existing] = line
        else:
            lines.insert(start + 1, line)
    original = style_path.read_text()
    style_path.write_text("".join(lines))
    try:
        load_style(style_path)
    except StyleError:
        style_path.write_text(original)
        raise
    return [{"name": f"source.{i}", "hex": hx} for i, hx in enumerate(colors)]
