"""Named palettes. Scripts refer to colors as ``p.<ramp>.<shade>``, never by index.

A palette entry is one of:

* a list of hex colors, dark to light: shades are ``dark`` (first), ``mid`` (middle),
  ``light`` (last), and any position by index (``p.wood[1]``);
* a table of named hex colors: ``{"green": "#3f6b2a"}`` -> ``p.cloth.green``;
* a single hex color: ``p.eye``.

With a palette source file, any of these may give a source color's position instead of a
hex value, and the file's colors are also reachable as ``p.source[i]``.

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
    """Named colors. With ``source`` (hex colors from a palette file), entries may also refer
    to source colors by position, and every source color is reachable as ``p.source[i]``."""

    def __init__(
        self,
        entries: Mapping[str, Sequence[str | int] | Mapping[str, str | int] | str | int],
        source: Sequence[str] | None = None,
    ):
        slots: list[list] = []  # [hex, canonical name]; slot i becomes color index i + 1
        by_hex: dict[str, int] = {}
        source_slots: list[int] = []
        for hx in source or []:
            parse_hex(hx)
            key = hx.lower()
            if key not in by_hex:
                by_hex[key] = len(slots)
                slots.append([hx, None])
            source_slots.append(by_hex[key])

        def slot_for(value: str | int, name: str) -> int:
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                raise PaletteError(f"{name}: expected a hex color or a source index, got {value!r}")
            if isinstance(value, int):
                if not source:
                    raise PaletteError(
                        f"{name} refers to source color {value}, but the palette has no source"
                    )
                if not 0 <= value < len(source_slots):
                    raise PaletteError(
                        f"{name} refers to source color {value}; the source has "
                        f"{len(source_slots)} colors (0..{len(source_slots) - 1})"
                    )
                slot = source_slots[value]
            else:
                parse_hex(value)
                slot = by_hex.get(value.lower(), len(slots))
                if slot == len(slots):
                    slots.append([value, None])
            if slots[slot][1] is None:
                slots[slot][1] = name
            return slot

        layout: dict[str, tuple] = {}
        for ramp_name, value in entries.items():
            if ramp_name == "source":
                raise PaletteError("'source' is reserved for the palette file's colors")
            if isinstance(value, (str, int)):
                layout[ramp_name] = ("single", slot_for(value, ramp_name))
            elif isinstance(value, Mapping):
                shades = {shade: i for i, shade in enumerate(value)}
                ids = [slot_for(v, f"{ramp_name}.{shade}") for shade, v in value.items()]
                layout[ramp_name] = ("table", ids, shades)
            else:
                if not value:
                    raise PaletteError(f"ramp {ramp_name!r} is empty")
                shades = _list_shades(len(value))
                names = {i: shade for shade, i in shades.items()}
                ids = [slot_for(v, f"{ramp_name}.{names.get(i, i)}") for i, v in enumerate(value)]
                layout[ramp_name] = ("list", ids, shades)
        for i, slot in enumerate(source_slots):
            if slots[slot][1] is None:
                slots[slot][1] = f"source.{i}"
        if len(slots) > MAX_COLORS:
            raise PaletteError(f"palette has {len(slots)} colors; at most {MAX_COLORS} fit")

        self._by_index = [Color(name, i + 1, parse_hex(hx)) for i, (hx, name) in enumerate(slots)]
        self._entries: dict[str, Ramp | Color] = {}
        self._ramp_of: dict[int, tuple[list[int], int]] = {}
        self._list_ramps = [name for name, spec in layout.items() if spec[0] == "list"]
        for ramp_name, spec in layout.items():
            if spec[0] == "single":
                self._entries[ramp_name] = self._by_index[spec[1]]
                continue
            kind, ids, shades = spec
            colors = [self._by_index[s] for s in ids]
            self._entries[ramp_name] = Ramp(ramp_name, colors, shades)
            if kind == "list":
                order = [c.index for c in colors]
                for pos, c in enumerate(colors):
                    self._ramp_of.setdefault(c.index, (order, pos))
        if source_slots:
            colors = [self._by_index[s] for s in source_slots]
            self._entries["source"] = Ramp("source", colors, _list_shades(len(colors)))

    def shift(self, index: int, steps: int) -> int:
        """The color ``steps`` shades lighter (negative: darker) in the color's list ramp,
        clamped to the ramp's ends. Colors outside list ramps are returned unchanged."""
        if index not in self._ramp_of:
            return index
        order, pos = self._ramp_of[index]
        return order[max(0, min(len(order) - 1, pos + steps))]

    def ramp_names(self) -> list[str]:
        """Every top-level palette name: list ramps, named tables, and single colors."""
        return list(self._entries)

    def list_ramps(self) -> list[str]:
        """Names of the ordered dark-to-light ramps (what ramp ops step along)."""
        return [n for n in self._list_ramps if n in self._entries]

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
