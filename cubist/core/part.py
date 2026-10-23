"""Named parts: a shape plus a pivot and an optional parent part."""

from __future__ import annotations

from dataclasses import dataclass, replace

from cubist.dsl.shape import Shape, ShapeError

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Part:
    name: str
    shape: Shape
    pivot: Vec3 = (0, 0, 0)
    parent: str | None = None

    def mirrored(
        self,
        name: str,
        about_x: float | None = None,
        about_y: float | None = None,
        about_z: float | None = None,
    ) -> tuple[Part, Part]:
        """This part and a reflected twin called ``name`` (e.g. arm_l -> arm_r)."""
        planes = {"x": about_x, "y": about_y, "z": about_z}
        given = [(axis, about) for axis, about in planes.items() if about is not None]
        if len(given) != 1:
            raise ShapeError("mirrored() takes exactly one of about_x, about_y, about_z")
        axis, about = given[0]
        i = "xyz".index(axis)
        pivot = list(self.pivot)
        pivot[i] = 2 * about - pivot[i]
        twin = replace(self, name=name, shape=self.shape.reflect(axis, about), pivot=tuple(pivot))
        return self, twin
