"""Sockets: named attachment points on a model, optionally with a direction.

A socket is a position (asset space), the part it moves with (by default the part with the
nearest voxel), and for muzzles, wand tips and grips a ``forward`` and ``up`` direction. In the
GLB an oriented socket's local +Z points forward and +Y up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cubist.core.errors import ModelError
from cubist.motion.joints import parse_axis

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Socket:
    position: Vec3
    part: str | None = None
    forward: Vec3 | None = None
    up: Vec3 | None = None

    def to_summary(self) -> list | dict:
        """The list form ``[x, y, z]`` / ``[x, y, z, part]`` unless oriented."""
        if self.forward is None:
            pos = [_num(v) for v in self.position]
            return pos + [self.part] if self.part else pos
        return {
            "position": list(self.position),
            "part": self.part,
            "forward": list(self.forward),
            "up": list(self.up) if self.up else None,
        }

    @classmethod
    def from_value(cls, value, name: str = "socket") -> Socket:
        """A Socket from a Socket, (x, y, z), (x, y, z, part) or a summary dict."""
        if isinstance(value, Socket):
            return value
        if isinstance(value, dict):
            fwd, up = value.get("forward"), value.get("up")
            return cls(
                tuple(float(v) for v in value["position"]),  # type: ignore[arg-type]
                value.get("part"),
                tuple(float(v) for v in fwd) if fwd else None,  # type: ignore[arg-type]
                tuple(float(v) for v in up) if up else None,  # type: ignore[arg-type]
            )
        if isinstance(value, (tuple, list)) and len(value) in (3, 4):
            try:
                pos = tuple(float(v) for v in value[:3])
            except (TypeError, ValueError):
                pos = None
            if pos is not None:
                return cls(pos, value[3] if len(value) == 4 else None)  # type: ignore[arg-type]
        raise ModelError(
            f"socket {name!r} must be (x, y, z), (x, y, z, part) or socket(...), got {value!r}"
        )


def _num(v: float):
    return int(v) if float(v).is_integer() else float(v)


def socket(position, part: str | None = None, forward=None, up="+y") -> Socket:
    """An attachment point at ``position``, moving with ``part`` (default: the part with the
    nearest voxel). With ``forward`` ('+z', '-x', a vector) it has a direction too; ``up`` is
    made perpendicular to it."""
    try:
        pos = tuple(float(v) for v in position)
    except (TypeError, ValueError):
        pos = ()
    if len(pos) != 3:
        raise ModelError(f"socket position must be (x, y, z), got {position!r}")
    if forward is None:
        return Socket(pos, part)  # type: ignore[arg-type]
    f = np.array(parse_axis(forward))
    u = np.array(parse_axis(up))
    u = u - (u @ f) * f
    n = float(np.linalg.norm(u))
    if n < 1e-6:
        raise ModelError(
            f"socket up {up!r} is parallel to forward {forward!r}; pick an up that crosses it, "
            "like up='+y' for forward='+z'"
        )
    u = u / n
    return Socket(
        pos,  # type: ignore[arg-type]
        part,
        tuple(float(c) + 0.0 for c in f),  # type: ignore[arg-type]
        tuple(float(np.round(c, 12)) + 0.0 for c in u),  # type: ignore[arg-type]
    )
