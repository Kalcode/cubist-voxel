"""Models, the ``@asset`` decorator, and built (voxelized) assets."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from cubist.core.errors import ModelError
from cubist.core.grid import VoxelGrid
from cubist.core.palette import Palette
from cubist.core.part import Part, Vec3
from cubist.dsl.context import building
from cubist.dsl.shape import Shape, ShapeError
from cubist.motion.joints import Joint


def check_symmetric(value, what: str) -> None:
    """``symmetric`` is an axis ('x', 'y', 'z') or (axis, plane) for a plane off the
    asset's center (e.g. ("x", 15.5) for an odd-width model in an even-width asset)."""
    if value is None or value in ("x", "y", "z"):
        return
    if (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and value[0] in ("x", "y", "z")
        and isinstance(value[1], (int, float))
    ):
        return
    raise ModelError(f"{what} must be 'x', 'y', 'z', (axis, plane) or None, got {value!r}")


def _flatten(parts: Iterable) -> list[Part]:
    out = []
    for item in parts:
        if isinstance(item, Part):
            out.append(item)
        elif isinstance(item, (list, tuple)):
            out.extend(_flatten(item))
        else:
            raise ModelError(f"Model parts must be Part objects, got {item!r}")
    return out


class Model:
    def __init__(self, parts: Sequence, sockets: dict[str, Vec3] | None = None):
        self.parts = _flatten(parts)
        self.sockets = dict(sockets or {})
        names = [p.name for p in self.parts]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ModelError(f"duplicate part names: {', '.join(dupes)}")
        for part in self.parts:
            if part.parent is not None and part.parent not in names:
                raise ModelError(f"part {part.name!r} has unknown parent {part.parent!r}")
            if part.joint is not None and not isinstance(part.joint, Joint):
                raise ModelError(
                    f"part {part.name!r} joint must be hinge(...), slide(...) or spin(...), "
                    f"got {part.joint!r}"
                )
        for socket, value in self.sockets.items():
            if len(value) not in (3, 4):
                raise ModelError(f"socket {socket!r} must be (x, y, z) or (x, y, z, part)")
            if len(value) == 4 and value[3] not in names:
                raise ModelError(f"socket {socket!r} names unknown part {value[3]!r}")
        parents = {p.name: p.parent for p in self.parts}
        for name in names:
            seen = set()
            while name is not None:
                if name in seen:
                    raise ModelError(f"parent cycle through part {name!r}")
                seen.add(name)
                name = parents[name]


def default_pivot(grid: VoxelGrid) -> Vec3:
    """The bottom center of a part's voxels: where a part usually hinges or stands."""
    b = grid.bounds()
    if b is None:
        return (0, 0, 0)
    (x0, y0, z0), (x1, _, z1) = b

    def num(v: float):
        return int(v) if float(v).is_integer() else v

    return (num((x0 + x1) / 2), y0, num((z0 + z1) / 2))


@dataclass
class BuiltPart:
    name: str
    grid: VoxelGrid
    pivot: Vec3
    parent: str | None
    symmetric: str | None = None


@dataclass
class BuiltAsset:
    name: str
    size: Vec3 | None
    tags: tuple[str, ...]
    parts: list[BuiltPart]
    palette: Palette
    sockets: dict[str, Vec3] = field(default_factory=dict)
    grounded: bool = False
    symmetric: str | None = None
    allow_floating: bool = False
    seed: int = 0
    ops_used: set[str] = field(default_factory=set)

    def voxel_count(self) -> int:
        return sum(p.grid.count() for p in self.parts)

    def color_counts(self) -> dict[str, int]:
        totals: dict[int, int] = {}
        for part in self.parts:
            for index, n in part.grid.colors_used().items():
                totals[index] = totals.get(index, 0) + n
        return {self.palette.name_of(i): n for i, n in sorted(totals.items())}

    def merged_grid(self) -> VoxelGrid:
        """Every part in one world-space grid (later parts win overlaps)."""
        return VoxelGrid.merge([p.grid for p in self.parts])

    def palette_names(self) -> dict[int, str]:
        """Names of the color indices this asset uses."""
        used = set()
        for part in self.parts:
            used.update(part.grid.colors_used())
        return {i: self.palette.name_of(i) for i in sorted(used)}

    def to_vox_scene(self):
        from cubist.io.vox import VoxPart, VoxScene

        return VoxScene(
            [VoxPart(p.name, p.grid, p.parent) for p in self.parts], self.palette.rgba_table()
        )

    def summary(self) -> dict:
        def part_info(p: BuiltPart) -> dict:
            b = p.grid.bounds()
            return {
                "name": p.name,
                "voxels": p.grid.count(),
                "bounds": [list(b[0]), list(b[1])] if b else None,
                "pivot": list(p.pivot),
                "parent": p.parent,
                "symmetric": p.symmetric,
            }

        return {
            "name": self.name,
            "size": list(self.size) if self.size else None,
            "tags": list(self.tags),
            "seed": self.seed,
            "grounded": self.grounded,
            "symmetric": self.symmetric,
            "allow_floating": self.allow_floating,
            "voxels": self.merged_grid().count(),
            "part_voxels": self.voxel_count(),
            "parts": [part_info(p) for p in self.parts],
            "sockets": {k: list(v) for k, v in self.sockets.items()},
            "colors": self.color_counts(),
            "palette_names": {str(i): n for i, n in self.palette_names().items()},
            "ops_used": sorted(self.ops_used),
        }

    @classmethod
    def from_vox(cls, scene, summary: dict, palette: Palette) -> BuiltAsset:
        """Rebuild an asset from its .vox file and build.json (empty dict if none) for lint.

        Color indices stay as they are in the file, which may differ from ``palette``."""
        info = {p["name"]: p for p in summary.get("parts", [])}
        parts = []
        for part in scene.parts:
            meta = info.get(part.name, {})
            parts.append(
                BuiltPart(
                    part.name,
                    part.grid,
                    tuple(meta.get("pivot", (0, 0, 0))),
                    meta.get("parent"),
                    meta.get("symmetric"),
                )
            )
        size = summary.get("size")
        return cls(
            name=summary.get("name", parts[0].name if parts else "asset"),
            size=tuple(size) if size else None,
            tags=tuple(summary.get("tags", ())),
            parts=parts,
            palette=palette,
            sockets={k: tuple(v) for k, v in summary.get("sockets", {}).items()},
            grounded=bool(summary.get("grounded", False)),
            symmetric=summary.get("symmetric"),
            allow_floating=bool(summary.get("allow_floating", False)),
            seed=int(summary.get("seed", 0)),
            ops_used=set(summary.get("ops_used", ())),
        )


class AssetDef:
    """An asset script function plus its metadata, produced by ``@asset``."""

    def __init__(
        self,
        fn: Callable,
        name: str,
        size: Vec3 | None,
        tags: Sequence[str],
        grounded: bool = False,
        symmetric: str | None = None,
        allow_floating: bool = False,
    ):
        check_symmetric(symmetric, "symmetric")
        self.fn, self.name, self.size, self.tags = fn, name, size, tuple(tags)
        self.grounded, self.symmetric, self.allow_floating = grounded, symmetric, allow_floating

    def build(self, palette: Palette, seed: int = 0) -> BuiltAsset:
        with building(palette, seed) as ctx:
            result = self.fn(palette)
            if isinstance(result, Shape):
                result = Model(parts=[Part(self.name, result)])
            if not isinstance(result, Model):
                raise ModelError(
                    f"asset {self.name!r} must return a Model or a Shape, "
                    f"got {type(result).__name__}"
                )
            parts = []
            for p in result.parts:
                check_symmetric(p.symmetric, f"part {p.name!r} symmetric")
                try:
                    grid = p.shape.voxelize()
                except ShapeError as e:
                    raise ShapeError(f"part {p.name!r}: {e}") from None
                pivot = tuple(p.pivot) if p.pivot is not None else default_pivot(grid)
                parts.append(BuiltPart(p.name, grid, pivot, p.parent, p.symmetric))
        return BuiltAsset(
            self.name,
            self.size,
            self.tags,
            parts,
            palette,
            result.sockets,
            grounded=self.grounded,
            symmetric=self.symmetric,
            allow_floating=self.allow_floating,
            seed=int(seed),
            ops_used=set(ctx.ops_used),
        )

    def __repr__(self) -> str:
        return f"AssetDef({self.name!r}, size={self.size})"


def asset(
    name: str | None = None,
    size: Vec3 | None = None,
    tags: Sequence[str] = (),
    grounded: bool = False,
    symmetric: str | None = None,
    allow_floating: bool = False,
):
    """Mark a function ``fn(p) -> Model | Shape`` as a buildable asset.

    ``grounded``: the asset should touch y = 0. ``symmetric``: the whole asset mirrors across
    the center plane of that axis. ``allow_floating``: pieces may float free of the main body.
    Lint warns when these don't hold.
    """

    def wrap(fn: Callable) -> AssetDef:
        return AssetDef(
            fn,
            name or fn.__name__,
            tuple(size) if size else None,
            tags,
            grounded=grounded,
            symmetric=symmetric,
            allow_floating=allow_floating,
        )

    return wrap
