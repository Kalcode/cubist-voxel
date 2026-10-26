"""Fields: values per ground column of a world, in metres, evaluated with numpy.

A field is a lazy expression. ``sample(x, z, seed)`` evaluates it at world positions (metres;
x east, z toward the front, as in the DSL) for arrays of any shape. Height fields give heights in
metres; masks give weights in [0, 1]. Noise mixes its own seed with the world's, so one seed
reshuffles a world and the same seed always gives the same ground.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path

import numpy as np
from PIL import Image

from cubist.dsl.noise import fbm, mix_seed

# Where relative map paths resolve, and the default size of a painted map: set while a world builds.
script_dir: ContextVar[Path | None] = ContextVar("cubist_world_dir", default=None)
world_size: ContextVar[tuple[float, float] | None] = ContextVar("cubist_world_size", default=None)


class WorldError(ValueError):
    """A world script or world operation can't proceed; the message says how to fix it."""


def _field(value) -> Field:
    if isinstance(value, Field):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return flat(float(value))
    raise WorldError(f"expected a field or a number, got {value!r}")


class Field:
    """A value per column.

    Combine with + - * (fields or numbers), ``blend``, ``flatten``, ``max`` and ``min``.
    """

    name = "field"

    def sample(self, x, z, seed: int) -> np.ndarray:
        raise NotImplementedError

    def __add__(self, other) -> Field:
        return _Op(lambda a, b: a + b, self, _field(other), "+")

    __radd__ = __add__

    def __sub__(self, other) -> Field:
        return _Op(lambda a, b: a - b, self, _field(other), "-")

    def __rsub__(self, other) -> Field:
        return _Op(lambda a, b: a - b, _field(other), self, "-")

    def __mul__(self, other) -> Field:
        return _Op(lambda a, b: a * b, self, _field(other), "*")

    __rmul__ = __mul__

    def __neg__(self) -> Field:
        return self * -1.0

    def max(self, other) -> Field:
        """The larger of the two per column."""
        return _Op(np.maximum, self, _field(other), "max")

    def min(self, other) -> Field:
        """The smaller of the two per column."""
        return _Op(np.minimum, self, _field(other), "min")

    def blend(self, other, mask: Field) -> Field:
        """This field where ``mask`` is 0, ``other`` where it is 1, mixed in between."""
        return _Blend(self, _field(other), _field(mask))

    def flatten(self, mask: Field, at_m: float) -> Field:
        """Level the ground to ``at_m`` where ``mask`` is 1 (a town), easing out at its edge."""
        return self.blend(flat(at_m), mask)

    def level(self, center, radius_m: float, edge_m: float = 20.0) -> Field:
        """Flatten a circle to this ground's own height at its center (a square, a pass), easing out
        over ``edge_m``: like ``flatten`` without knowing the height."""
        return _Level(self, (float(center[0]), float(center[1])), float(radius_m), float(edge_m))

    def terrace(self, step_m: float, softness: float = 0.3) -> Field:
        """Steps every ``step_m`` metres; ``softness`` (0–1) rounds their edges."""
        if step_m <= 0:
            raise WorldError(f"terrace step_m must be positive, got {step_m}")
        return _Terrace(self, float(step_m), float(np.clip(softness, 0.0, 1.0)))


class _Op(Field):
    def __init__(self, fn, a: Field, b: Field, name: str):
        self.fn, self.a, self.b, self.name = fn, a, b, name

    def sample(self, x, z, seed):
        return self.fn(self.a.sample(x, z, seed), self.b.sample(x, z, seed))


class _Blend(Field):
    name = "blend"

    def __init__(self, a: Field, b: Field, mask: Field):
        self.a, self.b, self.mask = a, b, mask

    def sample(self, x, z, seed):
        w = np.clip(self.mask.sample(x, z, seed), 0.0, 1.0)
        a = self.a.sample(x, z, seed)
        return a + (self.b.sample(x, z, seed) - a) * w


class _Level(Field):
    name = "level"

    def __init__(self, base: Field, center, radius: float, edge: float):
        self.base, self.center, self.radius, self.edge = base, center, radius, edge
        self._heights: dict[int, float] = {}

    def sample(self, x, z, seed):
        if seed not in self._heights:
            c = self.base.sample(np.array([self.center[0]]), np.array([self.center[1]]), seed)
            self._heights[seed] = float(c[0])
        h = self.base.sample(x, z, seed)
        d = np.hypot(np.asarray(x) - self.center[0], np.asarray(z) - self.center[1]) - self.radius
        w = _falloff(d, self.edge)
        return h + (self._heights[seed] - h) * w


class _Terrace(Field):
    name = "terrace"

    def __init__(self, base: Field, step: float, softness: float):
        self.base, self.step, self.softness = base, step, softness

    def sample(self, x, z, seed):
        h = self.base.sample(x, z, seed) / self.step
        f = h - np.floor(h)
        # The riser takes the last `softness` of each step; the tread is level.
        t = np.clip((f - (1 - self.softness)) / max(self.softness, 1e-9), 0.0, 1.0)
        return (np.floor(h) + t * t * (3 - 2 * t)) * self.step


class Custom(Field):
    """A field from a function ``fn(x, z, seed) -> array`` (for tests and one-off shapes)."""

    def __init__(self, fn: Callable, name: str = "custom"):
        self.fn, self.name = fn, name

    def sample(self, x, z, seed):
        return np.asarray(self.fn(x, z, seed), dtype=np.float64)


class _Flat(Field):
    name = "flat"

    def __init__(self, value: float):
        self.value = float(value)

    def sample(self, x, z, seed):
        return np.full(np.shape(x), self.value)


def flat(height_m: float) -> Field:
    """The same value everywhere."""
    return _Flat(height_m)


def _noise01(x, z, scale_m: float, octaves: int, kind: str, seed: int, op_seed: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    pts = np.stack(
        [
            x.ravel() / scale_m,
            np.full(x.size, 0.5),
            np.asarray(z, dtype=np.float64).ravel() / scale_m,
        ],
        axis=-1,
    )
    n = fbm(pts, mix_seed(op_seed, seed), octaves=octaves, kind=kind)
    return np.clip(n, -1.0, 1.0).reshape(x.shape)


class _Noise(Field):
    name = "noise"

    def __init__(self, scale_m, amplitude_m, octaves, kind, seed, ridged, sharpness):
        if scale_m <= 0:
            raise WorldError(
                f"noise scale_m must be positive (its features' size in metres), got {scale_m}"
            )
        if octaves < 1:
            raise WorldError(f"noise octaves must be at least 1, got {octaves}")
        self.scale, self.amplitude, self.octaves, self.kind = (
            float(scale_m),
            float(amplitude_m),
            int(octaves),
            kind,
        )
        self.seed, self.ridged, self.sharpness = int(seed), ridged, float(sharpness)

    def sample(self, x, z, seed):
        n = _noise01(x, z, self.scale, self.octaves, self.kind, seed, self.seed)
        if self.ridged:
            return (1.0 - np.abs(n)) ** self.sharpness * self.amplitude
        return n * self.amplitude


def noise(
    scale_m: float, amplitude_m: float = 1.0, octaves: int = 1, kind: str = "perlin", seed: int = 0
) -> Field:
    """Rolling noise in [-amplitude_m, amplitude_m]; ``scale_m`` is the size of its features."""
    return _Noise(scale_m, amplitude_m, octaves, kind, seed, ridged=False, sharpness=1.0)


def ridges(
    scale_m: float,
    amplitude_m: float = 1.0,
    octaves: int = 4,
    sharpness: float = 2.0,
    seed: int = 0,
) -> Field:
    """Sharp crests in [0, amplitude_m] (mountain ridges): folded noise; ``sharpness`` sharpens."""
    return _Noise(scale_m, amplitude_m, octaves, "perlin", seed, ridged=True, sharpness=sharpness)


class _HeightMap(Field):
    name = "height_map"

    def __init__(self, path, low_m, high_m, origin_m, size_m):
        self.path, self.low, self.high = Path(path), float(low_m), float(high_m)
        self.origin = (float(origin_m[0]), float(origin_m[1]))
        self.size = size_m
        self._pixels: np.ndarray | None = None

    def _load(self) -> np.ndarray:
        if self._pixels is None:
            path = (
                self.path
                if self.path.is_absolute() or script_dir.get() is None
                else script_dir.get() / self.path
            )
            if not path.is_file():
                raise WorldError(
                    f"no such height map: {path} (paths are relative to the world script)"
                )
            image = Image.open(path)
            if image.mode in ("I;16", "I;16B", "I"):
                self._pixels = np.asarray(image, dtype=np.float64) / 65535.0
            else:
                self._pixels = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
        return self._pixels

    def sample(self, x, z, seed):
        pixels = self._load()
        size = self.size or world_size.get()
        if size is None:
            raise WorldError(
                "height_map needs size_m outside a world (it defaults to the world's size)"
            )
        h, w = pixels.shape
        # Pixel centers sit at (i + 0.5) / w of the map's width; bilinear, clamped at the edges.
        u = np.clip((np.asarray(x) - self.origin[0]) / size[0] * w - 0.5, 0, w - 1)
        v = np.clip((np.asarray(z) - self.origin[1]) / size[1] * h - 0.5, 0, h - 1)
        i0, j0 = np.floor(u).astype(int), np.floor(v).astype(int)
        i1, j1 = np.minimum(i0 + 1, w - 1), np.minimum(j0 + 1, h - 1)
        fu, fv = u - i0, v - j0
        top = pixels[j0, i0] * (1 - fu) + pixels[j0, i1] * fu
        bottom = pixels[j1, i0] * (1 - fu) + pixels[j1, i1] * fu
        return self.low + (top * (1 - fv) + bottom * fv) * (self.high - self.low)


def height_map(path, low_m: float, high_m: float, origin_m=(0.0, 0.0), size_m=None) -> Field:
    """A painted greyscale PNG (8 or 16 bit): black is ``low_m``, white ``high_m``.

    It covers ``size_m`` (default: the whole world) from ``origin_m``, sampled bilinearly. Paths are
    relative to the world script.
    """
    return _HeightMap(
        path,
        low_m,
        high_m,
        origin_m,
        None if size_m is None else (float(size_m[0]), float(size_m[1])),
    )


def mask_map(path, origin_m=(0.0, 0.0), size_m=None) -> Field:
    """A painted greyscale PNG as a mask: black 0, white 1."""
    return _HeightMap(
        path, 0.0, 1.0, origin_m, None if size_m is None else (float(size_m[0]), float(size_m[1]))
    )


def _falloff(distance: np.ndarray, edge_m: float) -> np.ndarray:
    """1 at distance ≤ 0 (inside), easing to 0 at ``edge_m`` outside."""
    if edge_m <= 0:
        return (distance <= 0).astype(np.float64)
    t = np.clip(distance / edge_m, 0.0, 1.0)
    return 1.0 - t * t * (3 - 2 * t)


class _Circle(Field):
    name = "circle"

    def __init__(self, center, radius_m, edge_m):
        self.cx, self.cz, self.r, self.edge = (
            float(center[0]),
            float(center[1]),
            float(radius_m),
            float(edge_m),
        )

    def sample(self, x, z, seed):
        return _falloff(
            np.hypot(np.asarray(x) - self.cx, np.asarray(z) - self.cz) - self.r, self.edge
        )


def circle(center, radius_m: float, edge_m: float = 10.0) -> Field:
    """A round mask: 1 within ``radius_m`` of ``center`` (x, z), easing to 0 over ``edge_m``."""
    return _Circle(center, radius_m, edge_m)


class _Rect(Field):
    name = "rect"

    def __init__(self, lo, hi, edge_m):
        self.x0, self.z0 = min(lo[0], hi[0]), min(lo[1], hi[1])
        self.x1, self.z1 = max(lo[0], hi[0]), max(lo[1], hi[1])
        self.edge = float(edge_m)

    def sample(self, x, z, seed):
        dx = np.maximum(np.maximum(self.x0 - np.asarray(x), np.asarray(x) - self.x1), 0)
        dz = np.maximum(np.maximum(self.z0 - np.asarray(z), np.asarray(z) - self.z1), 0)
        return _falloff(np.hypot(dx, dz), self.edge)


def rect(lo, hi, edge_m: float = 10.0) -> Field:
    """A rectangular mask between corners ``lo`` and ``hi`` (x, z), easing to 0 over ``edge_m``."""
    return _Rect(lo, hi, edge_m)
