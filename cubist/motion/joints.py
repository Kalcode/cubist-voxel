"""Joints: how a part moves about its pivot.

A hinge turns about an axis through the pivot within limits (degrees), a slide moves along a
direction from 0 to a distance (voxels), and a spin turns without limits. Axes are in the
parent's frame, and the rest pose (as modeled) is joint value 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cubist.core.errors import ModelError

Vec3 = tuple[float, float, float]
KINDS = ("hinge", "slide", "spin")
_NAMED = {"x": 0, "y": 1, "z": 2}


def parse_axis(value) -> Vec3:
    """'x', '+x', '-x' (and y, z) or any non-zero 3-vector, as a unit vector."""
    if isinstance(value, str):
        name = value.strip().lower()
        sign = -1.0 if name.startswith("-") else 1.0
        name = name.lstrip("+-")
        if name in _NAMED:
            out = [0.0, 0.0, 0.0]
            out[_NAMED[name]] = sign
            return tuple(out)  # type: ignore[return-value]
    else:
        try:
            v = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            v = None
        if v is not None and v.shape == (3,):
            n = float(np.linalg.norm(v))
            if n == 0:
                raise ModelError("joint axis must be non-zero")
            return tuple(float(c) for c in v / n)  # type: ignore[return-value]
    raise ModelError(
        f"joint axis must be 'x', '-y', '+z' or a vector like (0, 1, 0), got {value!r}"
    )


def _number(value, what: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ModelError(f"{what} must be a number, got {value!r}") from None


@dataclass(frozen=True)
class Joint:
    kind: str
    axis: Vec3
    limits: tuple[float, float] | None = None
    rps: float | None = None

    def to_dict(self) -> dict:
        out: dict = {"type": self.kind, "axis": list(self.axis)}
        if self.kind == "spin":
            out["rps"] = self.rps
        else:
            out["limits"] = list(self.limits) if self.limits else None
        return out

    @classmethod
    def from_dict(cls, d: dict) -> Joint:
        limits = d.get("limits")
        return cls(
            d["type"],
            tuple(float(c) for c in d["axis"]),  # type: ignore[arg-type]
            tuple(float(v) for v in limits) if limits else None,  # type: ignore[arg-type]
            None if d.get("rps") is None else float(d["rps"]),
        )

    def reflected(self, axis_index: int) -> Joint:
        """This joint mirrored across a plane normal to ``axis_index``. A slide direction
        flips that component; a rotation axis (a pseudovector) flips the other two."""
        a = list(self.axis)
        if self.kind == "slide":
            a[axis_index] = -a[axis_index]
        else:
            a = [c if i == axis_index else -c for i, c in enumerate(a)]
        return Joint(self.kind, tuple(c + 0.0 for c in a), self.limits, self.rps)  # type: ignore[arg-type]


def hinge(axis, limits) -> Joint:
    """Turn about ``axis`` through the part's pivot, between ``limits`` = (lo, hi) degrees
    (right-hand rule: hinge("x") turns +Y toward +Z)."""
    try:
        lo, hi = (_number(v, "hinge limits") for v in limits)
    except TypeError:
        raise ModelError(f"hinge limits must be (lo, hi) in degrees, got {limits!r}") from None
    if not lo < hi:
        raise ModelError(f"hinge limits must be (lo, hi) with lo < hi, got {limits!r}")
    return Joint("hinge", parse_axis(axis), (lo, hi))


def slide(direction, distance) -> Joint:
    """Move along ``direction`` from 0 to ``distance`` voxels."""
    d = _number(distance, "slide distance")
    if d <= 0:
        raise ModelError(
            f"slide distance must be > 0, got {distance!r}; flip the direction to slide the "
            "other way"
        )
    return Joint("slide", parse_axis(direction), (0.0, d))


def spin(axis, rps=None) -> Joint:
    """Turn about ``axis`` without limits; ``rps`` is a default speed in turns per second."""
    return Joint("spin", parse_axis(axis), None, None if rps is None else _number(rps, "rps"))
