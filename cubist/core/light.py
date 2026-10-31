"""Point lights: a position, a palette color, a range, and how much they flicker.

They become ``KHR_lights_punctual`` point lights in the GLB, on a node that moves with their
part. Flicker is for the engine (glTF can't animate it portably); previews draw a marker. An
indoor light lights a room: the reach lint measures it from the floor under it.
"""

from __future__ import annotations

from dataclasses import dataclass

from cubist.core.errors import ModelError

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Light:
    position: Vec3
    color: int  # palette index
    rgb: tuple[int, int, int]
    range: float = 8.0  # voxels
    intensity: float = 1.0
    flicker: float = 0.0  # 0 steady .. 1 wild
    part: str | None = None
    indoor: bool = False  # lights a room: the lint measures it from the floor under it

    def to_summary(self) -> dict:
        return {
            "position": list(self.position),
            "color": self.color,
            "rgb": list(self.rgb),
            "range": self.range,
            "intensity": self.intensity,
            "flicker": self.flicker,
            "part": self.part,
            "indoor": self.indoor,
        }

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
        )


def _positive(value, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ModelError(f"light {what} must be a number > 0, got {value!r}")
    return float(value)


def light(
    position, color, range=8, intensity=1.0, flicker=0.0, part: str | None = None, indoor=False
) -> Light:
    """A point light at ``position`` in palette ``color`` (e.g. ``p.fire.light``), reaching
    ``range`` voxels, moving with ``part`` (default: the part with the nearest voxel).
    ``flicker`` from 0 (steady) to 1 tells the engine how much it wavers. ``indoor`` marks a
    light that lights a room (a hearth's on an upper floor): the reach lint measures its height
    from the floor straight under it, not from the ground line. Only the lint reads it."""
    try:
        pos = tuple(float(v) for v in position)
    except (TypeError, ValueError):
        pos = ()
    if len(pos) != 3:
        raise ModelError(f"light position must be (x, y, z), got {position!r}")
    if not (hasattr(color, "index") and hasattr(color, "rgba")):
        raise ModelError(f"light color must be a palette color like p.fire.light, got {color!r}")
    if isinstance(flicker, bool) or not isinstance(flicker, (int, float)) or not 0 <= flicker <= 1:
        raise ModelError(f"light flicker must be between 0 and 1, got {flicker!r}")
    if not isinstance(indoor, bool):
        raise ModelError(f"light indoor must be True or False, got {indoor!r}")
    return Light(
        pos,  # type: ignore[arg-type]
        int(color.index),
        tuple(int(v) for v in color.rgba[:3]),  # type: ignore[arg-type]
        _positive(range, "range"),
        _positive(intensity, "intensity"),
        float(flicker),
        part,
        indoor,
    )
