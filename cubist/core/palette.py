"""Named palettes. Scripts refer to colors as ``p.<ramp>.<shade>``, never by index.

A palette entry is one of:

* a list of hex colors, dark to light: shades are ``dark`` (first), ``mid`` (middle),
  ``light`` (last), and any position by index (``p.wood[1]``);
* a table of named hex colors: ``{"green": "#3f6b2a"}`` -> ``p.cloth.green``;
* a single hex color: ``p.eye``.

Color indices are assigned from 1 in definition order (0 means empty), matching the
``.vox`` color index space.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

MAX_COLORS = 255
_HEX = re.compile(r"^#([0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")


class PaletteError(AttributeError):
    """Unknown color name or invalid palette definition."""


@dataclass(frozen=True)
class Color:
    name: str
    index: int
    rgba: tuple[int, int, int, int]

    def __repr__(self) -> str:
        r, g, b, a = self.rgba
        return f"Color({self.name}, #{r:02x}{g:02x}{b:02x}{a:02x}, index={self.index})"


def parse_hex(value: str) -> tuple[int, int, int, int]:
    if not isinstance(value, str) or not _HEX.match(value):
        raise PaletteError(f"invalid color {value!r}: expected '#rrggbb' or '#rrggbbaa'")
    digits = value[1:]
    if len(digits) == 6:
        digits += "ff"
    return tuple(int(digits[i : i + 2], 16) for i in range(0, 8, 2))  # type: ignore[return-value]


def _suggest(name: str, options: Sequence[str]) -> str:
    close = difflib.get_close_matches(name, options, n=3)
    hint = f" Did you mean {', '.join(close)}?" if close else ""
    return f"{hint} Known: {', '.join(options)}."


class Ramp:
    """A named group of colors, accessed by shade name or position."""

    def __init__(self, name: str, colors: list[Color], shades: dict[str, int]):
        self._name = name
        self._colors = colors
        self._shades = shades

    @property
    def ramp(self) -> tuple[Color, ...]:
        return tuple(self._colors)

    def __getattr__(self, shade: str) -> Color:
        if shade.startswith("_"):
            raise AttributeError(shade)
        if shade not in self._shades:
            raise PaletteError(
                f"unknown color {self._name}.{shade}." + _suggest(shade, list(self._shades))
            )
        return self._colors[self._shades[shade]]

    def __getitem__(self, i: int) -> Color:
        return self._colors[i]

    def __iter__(self) -> Iterator[Color]:
        return iter(self._colors)

    def __len__(self) -> int:
        return len(self._colors)

    def __repr__(self) -> str:
        return f"Ramp({self._name}, {list(self._shades)})"


def _list_shades(n: int) -> dict[str, int]:
    # Later keys win when positions coincide in short ramps, so name canonicalization
    # (see Palette.__init__) prefers dark, then light, then mid.
    return {"mid": n // 2, "light": n - 1, "dark": 0}


class Palette:
    def __init__(self, entries: Mapping[str, Sequence[str] | Mapping[str, str] | str]):
        self._entries: dict[str, Ramp | Color] = {}
        self._by_index: list[Color] = []

        def add(name: str, hex_value: str) -> Color:
            if len(self._by_index) >= MAX_COLORS:
                raise PaletteError(f"palette has more than {MAX_COLORS} colors")
            color = Color(name, len(self._by_index) + 1, parse_hex(hex_value))
            self._by_index.append(color)
            return color

        for ramp_name, value in entries.items():
            if isinstance(value, str):
                self._entries[ramp_name] = add(ramp_name, value)
            elif isinstance(value, Mapping):
                shades = {shade: i for i, shade in enumerate(value)}
                colors = [add(f"{ramp_name}.{shade}", hx) for shade, hx in value.items()]
                self._entries[ramp_name] = Ramp(ramp_name, colors, shades)
            else:
                if not value:
                    raise PaletteError(f"ramp {ramp_name!r} is empty")
                shades = _list_shades(len(value))
                names = {i: shade for shade, i in shades.items()}
                colors = [add(f"{ramp_name}.{names.get(i, i)}", hx) for i, hx in enumerate(value)]
                self._entries[ramp_name] = Ramp(ramp_name, colors, shades)

    def __getattr__(self, name: str) -> Ramp | Color:
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._entries:
            raise PaletteError(f"unknown color {name}." + _suggest(name, list(self._entries)))
        return self._entries[name]

    def __len__(self) -> int:
        return len(self._by_index)

    def colors(self) -> list[Color]:
        return list(self._by_index)

    def name_of(self, index: int) -> str:
        return self._by_index[index - 1].name

    def rgba_table(self) -> np.ndarray:
        """(256, 4) uint8 table indexed by color index; row 0 is empty."""
        table = np.zeros((256, 4), dtype=np.uint8)
        for color in self._by_index:
            table[color.index] = color.rgba
        return table
