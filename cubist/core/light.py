"""Point lights: a position, a palette color, a range, and how much they flicker.

They become ``KHR_lights_punctual`` point lights in the GLB, on a node that moves with their
part. Flicker is for the engine (glTF can't animate it portably); previews draw a marker. An
indoor light lights a room: the reach lint measures it from the floor under it, and the engine
lights nothing outside its part's storey. A hold (``hold_m``) is for the engine too: past that many
metres the light stops falling off. So is a fitting: the box of the glowing voxels (a lantern's
glass, a grate's embers) that glow with the light rather than with the whole asset. And an indoor
light's room: the box its bounce stays in, the room's air and its walls short of their faces
outside.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from cubist.core.errors import ModelError
from cubist.dsl.shape import Shape

Vec3 = tuple[float, float, float]
Vec3i = tuple[int, int, int]


@dataclass(frozen=True)
class Light:
    position: Vec3
    color: int  # palette index
    rgb: tuple[int, int, int]
    range: float = 8.0  # voxels
    intensity: float = 1.0
    flicker: float = 0.0  # 0 steady .. 1 wild
    part: str | None = None
    indoor: bool = False  # lights a room: measured from the floor under it, kept to its storey
    hold_m: float | None = None  # metres past which the engine holds its level; None: no hold
    fitting: tuple[Vec3i, Vec3i] | None = None  # (lo, hi) voxels, half-open: what glows with it
    room: tuple[Vec3i, Vec3i] | None = None  # (lo, hi) voxels, half-open: where its bounce stays

    def to_summary(self) -> dict:
        summary = {
            "position": list(self.position),
            "color": self.color,
            "rgb": list(self.rgb),
            "range": self.range,
            "intensity": self.intensity,
            "flicker": self.flicker,
            "part": self.part,
            "indoor": self.indoor,
        }
        if self.hold_m is not None:
            summary["hold_m"] = self.hold_m
        if self.fitting is not None:
            summary["fitting"] = [list(self.fitting[0]), list(self.fitting[1])]
        if self.room is not None:
            summary["room"] = [list(self.room[0]), list(self.room[1])]
        return summary

    def reflect(self, axis: str, about: float) -> Light:
        """The light reflected across the plane ``axis = about``, as ``Shape.reflect`` reflects a
        shape: its position, and its fitting's and its room's boxes with it. For a mirrored
        building's lights, say.
        The plane must lie on a whole or half voxel: there voxels land on voxels, so the box holds
        the reflected shape's; anywhere else a shape is resampled and a box can't follow it."""
        if axis not in ("x", "y", "z"):
            raise ModelError(f"light reflect axis must be 'x', 'y' or 'z', got {axis!r}")
        number = isinstance(about, (int, float)) and not isinstance(about, bool)
        if not number or not math.isfinite(about):
            raise ModelError(f"light reflect about must be a finite number, got {about!r}")
        if 2 * about != round(2 * about):
            raise ModelError(
                f"light reflect about must be on a whole or half voxel (4 or 4.5), got {about!r}: "
                "elsewhere a reflected shape is resampled and the fitting's box can't follow it"
            )
        i = "xyz".index(axis)
        position = list(self.position)
        position[i] = 2 * about - position[i]

        def mirrored(bounds):
            # Voxel v's centre v + 0.5 lands on 2a - v - 0.5, voxel 2a - v - 1: [lo, hi) becomes
            # [2a - hi, 2a - lo).
            if bounds is None:
                return None
            lo, hi = list(bounds[0]), list(bounds[1])
            lo[i], hi[i] = round(2 * about) - bounds[1][i], round(2 * about) - bounds[0][i]
            return tuple(lo), tuple(hi)

        moved = tuple(float(v) for v in position)
        return replace(  # type: ignore[arg-type]
            self, position=moved, fitting=mirrored(self.fitting), room=mirrored(self.room)
        )

    @classmethod
    def from_summary(cls, d: dict) -> Light:
        return cls(
            tuple(float(v) for v in d["position"]),  # type: ignore[arg-type]
            int(d["color"]),
            tuple(int(v) for v in d["rgb"]),  # type: ignore[arg-type]
            float(d["range"]),
            float(d["intensity"]),
            float(d["flicker"]),
            d.get("part"),
            bool(d.get("indoor", False)),
            None if d.get("hold_m") is None else float(d["hold_m"]),
            None if d.get("fitting") is None else _box(d["fitting"]),
            None if d.get("room") is None else _box(d["room"]),
        )


def _box(pair) -> tuple[Vec3i, Vec3i]:
    lo, hi = pair
    return tuple(int(v) for v in lo), tuple(int(v) for v in hi)  # type: ignore[return-value]


def _fitting(shape) -> tuple[Vec3i, Vec3i]:
    """The voxel bounds of ``shape``, half-open: what glows with the light."""
    return _bounds(
        shape,
        "fitting",
        "the shape of the light's glowing voxels (its lantern's glass, its grate's embers)",
    )


def _room(shape) -> tuple[Vec3i, Vec3i]:
    """The voxel bounds of ``shape``, half-open: the air the light's bounce stays in."""
    return _bounds(
        shape,
        "room",
        "the shape of the room: its air, and its walls, floor and ceiling short of their faces "
        "outside",
    )


def _bounds(shape, what: str, should_be: str) -> tuple[Vec3i, Vec3i]:
    if not isinstance(shape, Shape):
        raise ModelError(f"light {what} must be {should_be}, or None, got {shape!r}")
    bounds = shape.occupancy().bounds()
    if bounds is None:
        raise ModelError(
            f"light {what} has no voxels: no voxel's centre is inside {shape!r}; pass {should_be}"
        )
    return bounds


def _positive(value, what: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ModelError(f"light {what} must be a finite number > 0, got {value!r}")
    return float(value)


def light(
    position,
    color,
    range=8,
    intensity=1.0,
    flicker=0.0,
    part: str | None = None,
    indoor=False,
    hold_m=None,
    fitting=None,
    room=None,
) -> Light:
    """A point light at ``position`` in palette ``color`` (e.g. ``p.fire.light``), reaching
    ``range`` voxels, moving with ``part`` (default: the part with the nearest voxel).
    ``flicker`` from 0 (steady) to 1 tells the engine how much it wavers. ``indoor`` marks a
    light that lights a room (a hearth's on an upper floor): the reach lint measures its height
    from the floor straight under it, not from the ground line, and the manifest marks it so the
    engine lights nothing outside the storey ``part`` is in. In the manifest only when set.
    ``hold_m`` (metres, or None) tells the engine to stop the light falling off past that
    distance: it holds its level out to where its range fades it, as a room's walls would
    pass its light on. Walls still shadow it. In the manifest only when set. ``fitting`` (a
    shape, or None) is the light's fixture, the shape of its glowing voxels (its lantern's
    glass, its grate's embers): the engine glows the glowing voxels in its bounds with the light,
    so they go dark when it is put out, while the rest of the asset's glow keeps its own switch.
    In the manifest, as those bounds, only when set. ``room`` (a shape, or None; an indoor light
    only) is the room it lights: the shape of its air, and of its walls, floor and ceiling short of
    their faces outside, and short of a pane's outer face where panes are set into a wall, so the
    panes' inner faces and the reveals inside them are in. The engine passes a share of an indoor
    light round the surfaces in its room, unshadowed, and only those, so the faces of the walls
    outside get none. In the manifest, as its bounds, only when set."""
    try:
        pos = tuple(float(v) for v in position)
    except (TypeError, ValueError):
        pos = ()
    if len(pos) != 3 or not all(math.isfinite(v) for v in pos):
        raise ModelError(f"light position must be (x, y, z), finite numbers, got {position!r}")
    if not (hasattr(color, "index") and hasattr(color, "rgba")):
        raise ModelError(f"light color must be a palette color like p.fire.light, got {color!r}")
    if isinstance(flicker, bool) or not isinstance(flicker, (int, float)) or not 0 <= flicker <= 1:
        raise ModelError(f"light flicker must be between 0 and 1, got {flicker!r}")
    if not isinstance(indoor, bool):
        raise ModelError(f"light indoor must be True or False, got {indoor!r}")
    if hold_m is not None and (
        isinstance(hold_m, bool)
        or not isinstance(hold_m, (int, float))
        or not math.isfinite(hold_m)
        or hold_m <= 0
    ):
        raise ModelError(
            f"light hold_m must be a number of metres > 0, or None for a light that falls off "
            f"all the way, got {hold_m!r}"
        )
    if room is not None and not indoor:
        raise ModelError(
            "light room is the room an indoor light lights; pass indoor=True with it, or no room"
        )
    return Light(
        pos,  # type: ignore[arg-type]
        int(color.index),
        tuple(int(v) for v in color.rgba[:3]),  # type: ignore[arg-type]
        _positive(range, "range"),
        _positive(intensity, "intensity"),
        float(flicker),
        part,
        indoor,
        None if hold_m is None else float(hold_m),
        None if fitting is None else _fitting(fitting),
        None if room is None else _room(room),
    )
