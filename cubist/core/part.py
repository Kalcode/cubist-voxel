"""Named parts: a shape plus a pivot, an optional parent part, an optional symmetry axis that
lint checks against the asset's center plane, an optional joint saying how it moves, and
whether it starts hidden (shown by clips)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from cubist.dsl.shape import Shape, ShapeError
from cubist.motion.joints import Joint

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Part:
    name: str
    shape: Shape | None = None  # or frames=[...] for a flipbook
    pivot: Vec3 | None = None  # None: the bottom center of the part's voxels
    parent: str | None = None
    symmetric: str | None = None
    joint: Joint | None = None
    hidden: bool = False  # hidden at rest; a clip's visible(...) track shows it
    frames: tuple[Shape, ...] | list[Shape] | None = None  # a flipbook: one shape per frame
    fps: float = 8.0  # flipbook frames per second when it plays on its own

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
        pivot = None
        if self.pivot is not None:
            pivot = list(self.pivot)
            pivot[i] = 2 * about - pivot[i]
            pivot = tuple(pivot)
        joint = self.joint.reflected(i) if isinstance(self.joint, Joint) else self.joint
        shape = self.shape.reflect(axis, about) if self.shape is not None else None
        frames = [f.reflect(axis, about) for f in self.frames] if self.frames else self.frames
        twin = replace(self, name=name, shape=shape, pivot=pivot, joint=joint, frames=frames)
        return self, twin
