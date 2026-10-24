"""Lint checks: the mistakes renders hide. Each check returns a list of findings."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from cubist.core.grid import VoxelGrid
from cubist.core.model import BuiltAsset
from cubist.core.style import StyleBible
from cubist.core.topology import SIX, hidden_mask, outside_air
from cubist.io.mesh import greedy_mesh

MAX_PART_AXIS = 256
_FACE_DIRS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


@dataclass
class Finding:
    check: str
    severity: str  # "error" | "warning" | "info"
    message: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
        }


def _hex(rgb) -> str:
    return "#" + "".join(f"{int(v):02x}" for v in rgb[:3])


def _box(lo, hi) -> str:
    return f"({','.join(map(str, lo))})-({','.join(map(str, hi))})"


def check_palette(asset: BuiltAsset, style: StyleBible, rgba: np.ndarray) -> list[Finding]:
    out = []
    allowed = {tuple(c.rgba[:3]) for c in style.palette.colors()}
    used: dict[int, int] = {}
    for part in asset.parts:
        for i, n in part.grid.colors_used().items():
            used[i] = used.get(i, 0) + n
    foreign = {i: n for i, n in used.items() if tuple(int(v) for v in rgba[i, :3]) not in allowed}
    if foreign:
        colors = ", ".join(f"{_hex(rgba[i])} ({n})" for i, n in sorted(foreign.items()))
        out.append(
            Finding(
                "palette",
                "error",
                f"colors not in the project palette: {colors}",
                {"colors": {_hex(rgba[i]): n for i, n in foreign.items()}},
            )
        )
    if style.rules.no_pure_black:
        black = sum(n for i, n in used.items() if tuple(rgba[i, :3]) == (0, 0, 0))
        if black:
            out.append(
                Finding(
                    "palette",
                    "error",
                    f"{black} voxels are pure black (#000000); the style forbids it",
                    {"voxels": black},
                )
            )
    if style.rules.dither == "forbidden" and "dither" in asset.ops_used:
        out.append(Finding("palette", "error", "dither is used but the style forbids it"))
    return out


def check_bounds(asset: BuiltAsset) -> list[Finding]:
    out = []
    for part in asset.parts:
        b = part.grid.bounds()
        if b and any(h - lo > MAX_PART_AXIS for lo, h in zip(*b, strict=True)):
            dims = "x".join(str(h - lo) for lo, h in zip(*b, strict=True))
            out.append(
                Finding(
                    "bounds",
                    "error",
                    f"part {part.name!r} is {dims}; parts are limited to "
                    f"{MAX_PART_AXIS} voxels per axis",
                    {"part": part.name},
                )
            )
    if asset.size is None:
        return out
    size = np.array(asset.size)
    outside, lo_all, hi_all = 0, None, None
    for part in asset.parts:
        coords, _ = part.grid.voxels()
        if len(coords) == 0:
            continue
        bad = np.any((coords < 0) | (coords >= size), axis=1)
        outside += int(bad.sum())
        lo = coords.min(axis=0)
        hi = coords.max(axis=0) + 1
        lo_all = lo if lo_all is None else np.minimum(lo_all, lo)
        hi_all = hi if hi_all is None else np.maximum(hi_all, hi)
    if outside:
        declared = "x".join(str(int(s)) for s in size)
        out.append(
            Finding(
                "bounds",
                "error",
                f"{outside} voxels lie outside the declared size {declared}; "
                f"the model spans {_box(lo_all.tolist(), hi_all.tolist())}",
                {"outside": outside, "lo": lo_all.tolist(), "hi": hi_all.tolist()},
            )
        )
    return out


def check_pivots(asset: BuiltAsset) -> list[Finding]:
    out = []
    for part in asset.parts:
        b = part.grid.bounds()
        if b is None:
            continue
        lo, hi = np.array(b[0]), np.array(b[1])
        pivot = np.array(part.pivot, dtype=float)
        if np.any(pivot < lo) or np.any(pivot > hi):
            out.append(
                Finding(
                    "pivots",
                    "error",
                    f"part {part.name!r} pivot {tuple(part.pivot)} is outside its "
                    f"bounds {_box(b[0], b[1])}",
                    {"part": part.name},
                )
            )
    merged = VoxelGrid.merge([p.grid for p in asset.parts]).bounds()
    if merged is not None:
        lo, hi = np.array(merged[0]), np.array(merged[1])
        for name, value in asset.sockets.items():
            pos = tuple(value[:3])
            p = np.array(pos, dtype=float)
            if np.any(p < lo) or np.any(p > hi):
                out.append(
                    Finding(
                        "pivots",
                        "error",
                        f"socket {name!r} at {tuple(pos)} is outside the model "
                        f"{_box(merged[0], merged[1])}",
                        {"socket": name},
                    )
                )
    return out


def check_floating(asset: BuiltAsset, merged: VoxelGrid) -> list[Finding]:
    if asset.allow_floating or merged.count() == 0:
        return []
    labels, n = ndimage.label(merged.data > 0, structure=SIX)
    if n <= 1:
        return []
    sizes = np.bincount(labels.ravel())[1:]
    main = int(np.argmax(sizes)) + 1
    pieces = []
    for label in range(1, n + 1):
        if label == main:
            continue
        cells = np.argwhere(labels == label) + np.array(merged.origin)
        pieces.append(
            {
                "voxels": int(len(cells)),
                "lo": cells.min(axis=0).tolist(),
                "hi": (cells.max(axis=0) + 1).tolist(),
            }
        )
    pieces.sort(key=lambda p: -p["voxels"])
    total = sum(p["voxels"] for p in pieces)
    where = "; ".join(f"{p['voxels']} at {_box(p['lo'], p['hi'])}" for p in pieces[:3])
    more = f" (+{len(pieces) - 3} more)" if len(pieces) > 3 else ""
    return [
        Finding(
            "floating",
            "warning",
            f"{len(pieces)} piece(s) not attached to the main body ({total} voxels): "
            f"{where}{more}. Pass allow_floating=True if intended.",
            {"pieces": len(pieces), "voxels": total, "list": pieces[:10]},
        )
    ]


def _mirror_mismatch(grid: VoxelGrid, axis: int, plane: float) -> int:
    flip = int(round(2 * plane)) - 1
    coords, colors = grid.voxels()
    if len(coords) == 0:
        return 0
    mirrored = coords.copy()
    mirrored[:, axis] = flip - coords[:, axis]
    return int(np.sum(grid.lookup(mirrored) != colors))


def check_symmetry(asset: BuiltAsset, merged: VoxelGrid, style: StyleBible) -> list[Finding]:
    out = []
    tolerance = style.rules.symmetry_tolerance

    def plane(axis: int) -> float:
        if asset.size is not None:
            return asset.size[axis] / 2.0
        b = merged.bounds()
        return (b[0][axis] + b[1][axis]) / 2.0 if b else 0.0

    targets = []
    if asset.symmetric:
        targets.append(("asset", merged, asset.symmetric))
    targets += [(f"part {p.name!r}", p.grid, p.symmetric) for p in asset.parts if p.symmetric]
    for label, grid, axis_name in targets:
        axis = "xyz".index(axis_name)
        at = plane(axis)
        mismatched = _mirror_mismatch(grid, axis, at)
        if mismatched > tolerance:
            out.append(
                Finding(
                    "symmetry",
                    "warning",
                    f"{label} is declared symmetric in {axis_name} but {mismatched} "
                    f"voxels differ from their mirror across {axis_name} = {at:g}",
                    {"target": label, "axis": axis_name, "plane": at, "mismatched": mismatched},
                )
            )
    return out


def check_grounding(asset: BuiltAsset, merged: VoxelGrid) -> list[Finding]:
    b = merged.bounds()
    if not asset.grounded or b is None or b[0][1] <= 0:
        return []
    return [
        Finding(
            "grounding",
            "warning",
            f"asset is marked grounded but its lowest voxel is at y = {b[0][1]}",
            {"lowest_y": b[0][1]},
        )
    ]


def check_budgets(asset: BuiltAsset, merged: VoxelGrid, style: StyleBible) -> list[Finding]:
    budgets = style.budgets
    triangles = sum(2 * len(greedy_mesh(p.grid)) for p in asset.parts)
    actual = {
        "max_voxels": asset.voxel_count(),
        "max_colors": len(merged.colors_used()),
        "max_parts": len(asset.parts),
        "max_triangles": triangles,
    }
    out = []
    for budget, value in actual.items():
        limit = getattr(budgets, budget)
        if limit is not None and value > limit:
            what = budget.removeprefix("max_")
            out.append(
                Finding(
                    "budgets",
                    "warning",
                    f"{value} {what} is over the budget of {limit}",
                    {"budget": budget, "value": value, "limit": limit},
                )
            )
    return out


def check_thin(merged: VoxelGrid) -> list[Finding]:
    """Voxels with outside air on both sides along some axis: walls you can see past."""
    if merged.count() == 0:
        return []
    filled = np.pad(merged.data > 0, 1)
    air = outside_air(filled)
    thin = np.zeros_like(filled)
    for axis in range(3):
        thin |= filled & np.roll(air, 1, axis) & np.roll(air, -1, axis)
    n = int(thin.sum())
    if n == 0:
        return []
    cells = np.argwhere(thin) - 1 + np.array(merged.origin)
    lo, hi = cells.min(axis=0).tolist(), (cells.max(axis=0) + 1).tolist()
    return [
        Finding(
            "thin",
            "info",
            f"{n} voxels are one voxel thick with open air on both sides and may vanish at "
            f"game distance, within {_box(lo, hi)}",
            {"voxels": n, "lo": lo, "hi": hi},
        )
    ]


def check_hidden(merged: VoxelGrid) -> list[Finding]:
    if merged.count() == 0:
        return []
    n = int(hidden_mask(merged).sum())
    if n == 0:
        return []
    return [
        Finding(
            "hidden",
            "info",
            f"{n} solid voxels can't be seen from outside; export hollows them",
            {"voxels": n},
        )
    ]
