"""Named parts: a shape plus a pivot, an optional parent part, an optional symmetry axis that
lint checks against the asset's center plane, an optional joint saying how it moves, and
whether it starts hidden (shown by clips), and an occlusion tag for engines that hide parts
between the camera and the player, or let their follow camera pass through them."""

from __future__ import annotations

from dataclasses import dataclass, replace

from cubist.core.errors import ModelError
from cubist.dsl.shape import Shape, ShapeError
from cubist.motion.joints import Joint

Vec3 = tuple[float, float, float]
# Roofs, upper storeys and tree crowns (an engine hides them to show the player), and clutter: small
# set-dressing (a stall's counter and posts, crates) that a follow camera passes through.
OCCLUSION = ("roof", "upper", "canopy", "clutter")


def _swap_side(parent: str | None, old: str, new: str) -> str | None:
    """``parent`` with the side ``old`` -> ``new`` swaps (``_r``/``_l``, ``r_``/``l_``)."""
    if parent is None or old == new:
        return parent
    for a, b in (("_r", "_l"), ("_l", "_r")):
        if old.endswith(a) and new.endswith(b) and parent.endswith(a):
            return parent[: -len(a)] + b
    for a, b in (("r_", "l_"), ("l_", "r_")):
        if old.startswith(a) and new.startswith(b) and parent.startswith(a):
            return b + parent[len(a) :]
    return parent


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
    occlusion: str | None = None  # one of OCCLUSION: what an engine hides or its camera passes

    def __post_init__(self):
        if self.occlusion is not None and self.occlusion not in OCCLUSION:
            raise ModelError(
                f"part {self.name!r}: occlusion must be one of {', '.join(OCCLUSION)} "
                f"(got {self.occlusion!r})"
            )
        if self.frames is not None and not isinstance(self.frames, (list, tuple)):
            object.__setattr__(self, "frames", tuple(self.frames))  # generators too

    def mirrored(
        self,
        name: str,
        about_x: float | None = None,
        about_y: float | None = None,
        about_z: float | None = None,
        parent: str | None = None,
    ) -> tuple[Part, Part]:
        """This part and a reflected twin called ``name`` (e.g. arm_l -> arm_r). The twin hangs
        from ``parent``, or else from this part's parent with the same side swapped (foot_r on
        leg_r mirrors to foot_l on leg_l); an unsided parent (torso) is kept."""
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
        twin = replace(
            self,
            name=name,
            shape=shape,
            pivot=pivot,
            joint=joint,
            frames=frames,
            parent=parent if parent is not None else _swap_side(self.parent, self.name, name),
        )
        return self, twin
