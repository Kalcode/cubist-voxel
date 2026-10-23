"""Models, the ``@asset`` decorator, and built (voxelized) assets."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from cubist.core.grid import VoxelGrid
from cubist.core.palette import Palette
from cubist.core.part import Part, Vec3
from cubist.dsl.shape import Shape


class ModelError(ValueError):
    """A model's structure is invalid."""


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
        parents = {p.name: p.parent for p in self.parts}
        for name in names:
            seen = set()
            while name is not None:
                if name in seen:
                    raise ModelError(f"parent cycle through part {name!r}")
                seen.add(name)
                name = parents[name]


@dataclass
class BuiltPart:
    name: str
    grid: VoxelGrid
    pivot: Vec3
    parent: str | None


@dataclass
class BuiltAsset:
    name: str
    size: Vec3 | None
    tags: tuple[str, ...]
    parts: list[BuiltPart]
    palette: Palette
    sockets: dict[str, Vec3] = field(default_factory=dict)

    def voxel_count(self) -> int:
        return sum(p.grid.count() for p in self.parts)

    def color_counts(self) -> dict[str, int]:
        totals: dict[int, int] = {}
        for part in self.parts:
            for index, n in part.grid.colors_used().items():
                totals[index] = totals.get(index, 0) + n
        return {self.palette.name_of(i): n for i, n in sorted(totals.items())}

    def to_vox_scene(self):
        from cubist.io.vox import VoxPart, VoxScene

        return VoxScene([VoxPart(p.name, p.grid) for p in self.parts], self.palette.rgba_table())

    def summary(self) -> dict:
        def part_info(p: BuiltPart) -> dict:
            b = p.grid.bounds()
            return {
                "name": p.name,
                "voxels": p.grid.count(),
                "bounds": [list(b[0]), list(b[1])] if b else None,
                "pivot": list(p.pivot),
                "parent": p.parent,
            }

        return {
            "name": self.name,
            "size": list(self.size) if self.size else None,
            "tags": list(self.tags),
            "voxels": self.voxel_count(),
            "parts": [part_info(p) for p in self.parts],
            "sockets": {k: list(v) for k, v in self.sockets.items()},
            "colors": self.color_counts(),
        }


class AssetDef:
    """An asset script function plus its metadata, produced by ``@asset``."""

    def __init__(self, fn: Callable, name: str, size: Vec3 | None, tags: Sequence[str]):
        self.fn, self.name, self.size, self.tags = fn, name, size, tuple(tags)

    def build(self, palette: Palette) -> BuiltAsset:
        result = self.fn(palette)
        if isinstance(result, Shape):
            result = Model(parts=[Part(self.name, result)])
        if not isinstance(result, Model):
            raise ModelError(
                f"asset {self.name!r} must return a Model or a Shape, got {type(result).__name__}"
            )
        parts = [
            BuiltPart(p.name, p.shape.voxelize(), tuple(p.pivot), p.parent) for p in result.parts
        ]
        return BuiltAsset(self.name, self.size, self.tags, parts, palette, result.sockets)

    def __repr__(self) -> str:
        return f"AssetDef({self.name!r}, size={self.size})"


def asset(name: str | None = None, size: Vec3 | None = None, tags: Sequence[str] = ()):
    """Mark a function ``fn(p) -> Model | Shape`` as a buildable asset."""

    def wrap(fn: Callable) -> AssetDef:
        return AssetDef(fn, name or fn.__name__, tuple(size) if size else None, tags)

    return wrap
