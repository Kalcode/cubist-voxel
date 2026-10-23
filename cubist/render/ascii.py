"""ASCII slices: one layer of the grid per block, one character per palette color.

Orientation matches the contact sheet: y slices read like the TOP view (rows run back to
front), z slices like the FRONT view, x slices like the LEFT view.
"""

from __future__ import annotations

import re
import string

import numpy as np

from cubist.core.grid import VoxelGrid

_LETTERS = string.ascii_lowercase + string.ascii_uppercase + string.digits
_SPEC = re.compile(r"^([xyz])=(-?\d+)(?:\.\.(-?\d+))?$")

# slice axis -> (row axis, rows descending, row label, column axis, column label)
_LAYOUT = {
    "y": (2, False, "z, back to front", 0, "x"),
    "z": (1, True, "y, top to bottom", 0, "x"),
    "x": (1, True, "y, top to bottom", 2, "z, back to front"),
}


def parse_slices(spec: str) -> tuple[str, int, int]:
    m = _SPEC.match(spec.strip())
    if not m:
        raise ValueError(f"bad slice spec {spec!r}; expected like 'y=10..14' or 'x=3'")
    axis, start = m.group(1), int(m.group(2))
    stop = int(m.group(3)) if m.group(3) is not None else start
    if stop < start:
        raise ValueError(f"slice range {spec!r} runs backwards")
    return axis, start, stop


def _header(values: list[int]) -> str:
    chars = [" "] * len(values)
    free = 0
    for i, v in enumerate(values):
        label = str(v)
        if v % 5 == 0 and i >= free and i + len(label) <= len(values) + 2:
            chars.extend(" " * max(0, i + len(label) - len(chars)))
            chars[i : i + len(label)] = label
            free = i + len(label) + 1
    return "".join(chars).rstrip()


def slices(
    grid: VoxelGrid,
    axis: str,
    start: int,
    stop: int,
    names: dict[int, str] | None = None,
    region=None,
) -> str:
    if axis not in _LAYOUT:
        raise ValueError(f"axis must be x, y or z, got {axis!r}")
    names = names or {}
    a = "xyz".index(axis)
    row_axis, rows_desc, row_label, col_axis, col_label = _LAYOUT[axis]
    b = grid.bounds()
    lo, hi = region if region is not None else (b if b is not None else ((0, 0, 0), (1, 1, 1)))
    rows = list(range(lo[row_axis], hi[row_axis]))
    if rows_desc:
        rows.reverse()
    cols = list(range(lo[col_axis], hi[col_axis]))

    layers = []
    for k in range(start, stop + 1):
        layer = np.zeros((len(rows), len(cols)), dtype=np.uint8)
        for r, rv in enumerate(rows):
            for c, cv in enumerate(cols):
                pos = [0, 0, 0]
                pos[a], pos[row_axis], pos[col_axis] = k, rv, cv
                layer[r, c] = grid.get(*pos)
        layers.append((k, layer))

    # Letters come from every color in the grid, so they stay the same across calls.
    everything = sorted(grid.colors_used())
    chars = {index: _LETTERS[i] if i < len(_LETTERS) else "?" for i, index in enumerate(everything)}
    used = sorted({int(v) for _, layer in layers for v in np.unique(layer) if v})
    legend = ["legend: . = empty"] + [f"{chars[i]} = {names.get(i, f'color {i}')}" for i in used]

    width = max((len(str(v)) for v in rows), default=1)
    out = [", ".join(legend)]
    for k, layer in layers:
        out.append("")
        out.append(f"{axis}={k}  (rows: {row_label}; columns: {col_label})")
        out.append(" " * (width + 2) + _header(cols))
        for rv, line in zip(rows, layer, strict=True):
            body = "".join(chars[v] if v else "." for v in line.tolist())
            out.append(f"{str(rv).rjust(width)}  {body}")
    return "\n".join(out) + "\n"
