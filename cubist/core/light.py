"""Point lights: a position, a palette color, a range, and how much they flicker.

They become ``KHR_lights_punctual`` point lights in the GLB, on a node that moves with their
part. Flicker is for the engine (glTF can't animate it portably); previews draw a marker. An
indoor light lights a room: the reach lint measures it from the floor under it, and the engine
lights nothing outside its part's storey. A hold (``hold_m``) is for the engine too: past that many
metres the light stops falling off. So is a fitting: the box of the glowing voxels (a lantern's
glass, a grate's embers) that glow with the light rather than with the whole asset.
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
        return summary

    def reflect(self, axis: str, about: float) -> Light:
        """The light reflected across the plane ``axis = about``, as ``Shape.reflect`` reflects a
        shape: its position, and its fitting's box with it (the voxels whose centres the reflected
        box holds). For a mirrored building's lights, say."""
        if axis not in ("x", "y", "z"):
            raise ModelError(f"light reflect axis must be 'x', 'y' or 'z', got {axis!r}")
        number = isinstance(about, (int, float)) and not isinstance(about, bool)
        if not number or not math.isfinite(about):
            raise ModelError(f"light reflect about must be a finite number, got {about!r}")
        i = "xyz".index(axis)
        position = list(self.position)
        position[i] = 2 * about - position[i]
        fitting = self.fitting
        if fitting is not None:
            # The voxels whose centres the reflected box [2a - hi, 2a - lo) holds.
            lo, hi = list(fitting[0]), list(fitting[1])
            lo[i] = math.ceil(2 * about - fitting[1][i] - 0.5)
            hi[i] = math.ceil(2 * about - fitting[0][i] - 0.5)
            fitting = (tuple(lo), tuple(hi))  # type: ignore[assignment]
        moved = tuple(float(v) for v in position)
        return replace(self, position=moved, fitting=fitting)  # type: ignore[arg-type]

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
        )


def _box(pair) -> tuple[Vec3i, Vec3i]:
    lo, hi = pair
    return tuple(int(v) for v in lo), tuple(int(v) for v in hi)  # type: ignore[return-value]


def _fitting(shape) -> tuple[Vec3i, Vec3i]:
    """The voxel bounds of ``shape``, half-open: what glows with the light."""
    if not isinstance(shape, Shape):
        raise ModelError(
            f"light fitting must be the shape of the light's glowing voxels (its lantern's glass, "
            f"its grate's embers), or None, got {shape!r}"
        )
    bounds = shape.occupancy().bounds()
    if bounds is None:
        raise ModelError(
            f"light fitting has no voxels: no voxel's centre is inside {shape!r}; pass the shape "
            "of the light's glowing voxels"
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
    In the manifest, as those bounds, only when set."""
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
    )
