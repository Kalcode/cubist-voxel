"""Emitters: where particles (smoke, sparks, dust) come from, described for the engine.

An emitter names a particle model (an ordinary asset, usually tagged ``particle``) and how it
sprays: continuously at ``rate`` per second, and/or ``burst`` at a time when a clip has an
``emit:<name>`` event. Cubist previews it with a simple seeded simulation; the engine's own
particle system plays it for real from the manifest.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from cubist.core.errors import ModelError
from cubist.motion.joints import finite_vector, parse_axis

Vec3 = tuple[float, float, float]
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class Emitter:
    position: Vec3
    particle: str
    direction: Vec3 = (0.0, 1.0, 0.0)
    rate: float | None = None  # particles per second, always on
    burst: int | None = None  # particles per emit:<name> clip event
    life: float = 1.0  # seconds
    speed: float = 2.0  # voxels per second
    spread: float = 20.0  # degrees around direction
    gravity: float = 0.0  # voxels/s² downward; negative rises (smoke)
    grow: tuple[float, float] = (1.0, 1.0)  # size at birth, at death
    fade: bool = True  # shrink away over the last 30% of life
    seed: int = 0
    part: str | None = None

    def to_summary(self) -> dict:
        return {
            "position": list(self.position),
            "particle": self.particle,
            "direction": list(self.direction),
            "rate": self.rate,
            "burst": self.burst,
            "life": self.life,
            "speed": self.speed,
            "spread": self.spread,
            "gravity": self.gravity,
            "grow": list(self.grow),
            "fade": self.fade,
            "seed": self.seed,
            "part": self.part,
        }

    @classmethod
    def from_summary(cls, d: dict) -> Emitter:
        return cls(
            tuple(float(v) for v in d["position"]),  # type: ignore[arg-type]
            d["particle"],
            tuple(float(v) for v in d["direction"]),  # type: ignore[arg-type]
            None if d.get("rate") is None else float(d["rate"]),
            None if d.get("burst") is None else int(d["burst"]),
            float(d["life"]),
            float(d["speed"]),
            float(d["spread"]),
            float(d["gravity"]),
            tuple(float(v) for v in d["grow"]),  # type: ignore[arg-type]
            bool(d["fade"]),
            int(d["seed"]),
            d.get("part"),
        )


def _number(value, what: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ModelError(f"emitter {what} must be a finite number, got {value!r}")
    if positive and value <= 0:
        raise ModelError(f"emitter {what} must be > 0, got {value!r}")
    return float(value)


def emitter(
    position,
    particle: str,
    direction="+y",
    rate: float | None = None,
    burst: int | None = None,
    life: float = 1.0,
    speed: float = 2.0,
    spread: float = 20.0,
    gravity: float = 0.0,
    grow=(1.0, 1.0),
    fade: bool = True,
    seed: int = 0,
    part: str | None = None,
) -> Emitter:
    """Particles of asset ``particle`` leaving ``position`` along ``direction``: ``rate`` per
    second continuously, and/or ``burst`` at once on each ``emit:<name>`` clip event."""
    try:
        pos = tuple(float(v) for v in position)
    except (TypeError, ValueError):
        pos = ()
    if len(pos) != 3 or not all(math.isfinite(v) for v in pos):
        raise ModelError(f"emitter position must be (x, y, z), finite numbers, got {position!r}")
    if not isinstance(particle, str) or not _NAME.match(particle):
        raise ModelError(
            f"emitter particle must be an asset name (lowercase letters, digits, _ and -), got "
            f"{particle!r}"
        )
    if rate is None and burst is None:
        raise ModelError("an emitter needs rate=... (always on) or burst=... (on emit events)")
    if burst is not None and (isinstance(burst, bool) or not isinstance(burst, int) or burst < 1):
        raise ModelError(f"emitter burst must be a whole number >= 1, got {burst!r}")
    try:
        g0, g1 = (_number(v, "grow", positive=True) for v in grow)
    except TypeError:
        raise ModelError(f"emitter grow must be (start, end) sizes, got {grow!r}") from None
    spread = _number(spread, "spread")
    if not 0 <= spread <= 180:
        raise ModelError(f"emitter spread must be 0 to 180 degrees, got {spread!r}")
    if not finite_vector(direction):
        raise ModelError(
            f"emitter direction must be finite, like '+y' or (0, 1, 0), got {direction!r}"
        )
    try:
        seed = int(seed)
    except (TypeError, ValueError, OverflowError):
        raise ModelError(f"emitter seed must be a whole number, got {seed!r}") from None
    return Emitter(
        pos,  # type: ignore[arg-type]
        particle,
        parse_axis(direction),
        None if rate is None else _number(rate, "rate", positive=True),
        burst,
        _number(life, "life", positive=True),
        _number(speed, "speed"),
        spread,
        _number(gravity, "gravity"),
        (g0, g1),
        bool(fade),
        seed,
        part,
    )
