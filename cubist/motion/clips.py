"""Clips: named, timed motion made of tracks of keys.

A track drives one channel of one part, relative to rest:

* ``joint``: the part's joint value (degrees for a hinge, voxels for a slide, turns for a spin);
* ``rotate_x|y|z``: degrees about the part's pivot along the parent's axes, on any part;
* ``move_x|y|z``: voxels along the parent's axes, on any part;
* ``visible``: 1 shown, 0 hidden (always stepped);
* ``frame``: which frame of a flipbook part shows (always stepped).

Track specs (``turn``, ``move``, the generators) describe keys before the clip's length is
known; ``clip()`` resolves them into plain tracks, so a clip only ever holds keys.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from cubist.core.errors import ModelError

EASES = ("smooth", "linear", "step", "in", "out")
ROTATE = ("rotate_x", "rotate_y", "rotate_z")
MOVE = ("move_x", "move_y", "move_z")
CHANNELS = ("joint", *ROTATE, *MOVE, "visible", "frame")
STEPPED = ("visible", "frame")
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
_EPS = 1e-9


@dataclass(frozen=True)
class Track:
    part: str
    channel: str
    times: tuple[float, ...]
    values: tuple[float, ...]
    ease: str | None = None  # None: the clip's ease

    def to_dict(self) -> dict:
        return {
            "part": self.part,
            "channel": self.channel,
            "times": list(self.times),
            "values": list(self.values),
            "ease": self.ease,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Track:
        return cls(
            d["part"],
            d["channel"],
            tuple(float(t) for t in d["times"]),
            tuple(float(v) for v in d["values"]),
            d.get("ease"),
        )


@dataclass(frozen=True)
class Clip:
    name: str
    length: float
    tracks: tuple[Track, ...]
    loop: bool = False
    ease: str = "smooth"
    events: tuple[tuple[float, str], ...] = ()
    travel: tuple[float, float, float] | None = None  # voxels per second, for in-place walks
    collisions: bool = True  # False: parts may pass through each other (debris, effects)

    def track_ease(self, track: Track) -> str:
        if track.channel in STEPPED:
            return "step"
        return track.ease or self.ease

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "length": self.length,
            "loop": self.loop,
            "ease": self.ease,
            "tracks": [t.to_dict() for t in self.tracks],
            "events": [[t, name] for t, name in self.events],
            "travel": list(self.travel) if self.travel else None,
            "collisions": self.collisions,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Clip:
        return cls(
            d["name"],
            float(d["length"]),
            tuple(Track.from_dict(t) for t in d["tracks"]),
            bool(d.get("loop", False)),
            d.get("ease", "smooth"),
            tuple((float(t), str(name)) for t, name in d.get("events", ())),
            tuple(float(v) for v in d["travel"]) if d.get("travel") else None,  # type: ignore[arg-type]
            bool(d.get("collisions", True)),
        )


@dataclass(frozen=True)
class TrackSpec:
    """Keys for one channel of one part, resolved against the clip's length by ``clip()``.

    ``keys``: values spread evenly from 0 to the length, ``(time, value)`` pairs, or a function
    of ``u`` in [0, 1] sampled at ``samples`` evenly spaced points (ends included). ``make``,
    when given, takes the length and returns ``(times, values)`` itself."""

    part: str
    channel: str
    keys: Sequence | Callable[[float], float] | None = None
    ease: str | None = None
    samples: int | None = None
    make: Callable[[float], tuple[Sequence[float], Sequence[float]]] | None = None

    def resolve(self, clip_name: str, length: float) -> Track:
        if self.make is not None:
            times, values = self.make(length)
            return Track(self.part, self.channel, tuple(map(float, times)),
                         tuple(map(float, values)), self.ease)  # fmt: skip
        keys = self.keys
        if callable(keys):
            n = max(2, int(self.samples or 9))
            us = [i / (n - 1) for i in range(n)]
            times = tuple(length * u for u in us)
            values = tuple(_float(keys(u), f"clip {clip_name!r} track {self.part!r}") for u in us)
            return Track(self.part, self.channel, times, values, self.ease)
        times, values = _keys(clip_name, self.part, keys, length)
        return Track(self.part, self.channel, times, values, self.ease)


def _free(kind: str, part: str, axis: str, keys, ease: str | None) -> TrackSpec:
    name = str(axis).strip().lower()
    sign = -1.0 if name.startswith("-") else 1.0
    name = name.lstrip("+-")
    if name not in ("x", "y", "z"):
        raise ModelError(f"{kind} axis must be 'x', 'y', 'z' (or '-x', ...), got {axis!r}")
    if sign < 0 and not callable(keys):
        keys = [(k[0], -k[1]) if isinstance(k, (tuple, list)) else -k for k in keys]
    elif sign < 0:
        f = keys
        keys = lambda u: -f(u)  # noqa: E731
    return TrackSpec(part, f"{kind}_{name}", keys, ease)


def joint(part: str, keys, ease: str | None = None) -> TrackSpec:
    """Keys for a part's joint (degrees for a hinge, voxels for a slide, turns for a spin), for
    list-form clips that mix it with ``turn``, ``visible`` and the rest."""
    return TrackSpec(part, "joint", keys, ease)


def visible(part: str, keys) -> TrackSpec:
    """Show (1) and hide (0) a part over the clip, stepping between keys. Pair it with
    ``Part(..., hidden=True)`` for flashes and sparks that appear for a moment."""
    return TrackSpec(part, "visible", keys, "step")


def frames(part: str, keys) -> TrackSpec:
    """Pick which frame of a flipbook part shows, stepping between keys: ``[(0, 0), (0.05, 1)]``
    plays a two-frame muzzle flash once. Without a frames track a flipbook loops at its fps."""
    return TrackSpec(part, "frame", keys, "step")


def turn(part: str, axis: str, keys, ease: str | None = None) -> TrackSpec:
    """Rotate any part about its pivot: degrees along the parent's ``axis`` ('x', '-y', ...)."""
    return _free("rotate", part, axis, keys, ease)


def move(part: str, axis: str, keys, ease: str | None = None) -> TrackSpec:
    """Move any part: voxels along the parent's ``axis`` ('y', '-z', ...)."""
    return _free("move", part, axis, keys, ease)


def _float(value, what: str) -> float:
    if isinstance(value, bool):
        raise ModelError(f"{what} must be a number, got {value!r}")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ModelError(f"{what} must be a number, got {value!r}") from None
    if not math.isfinite(out):
        raise ModelError(f"{what} must be a finite number, got {value!r}")
    return out


def _keys(name: str, part: str, keys, length: float) -> tuple[tuple[float, ...], tuple]:
    what = f"clip {name!r} track {part!r}"
    if isinstance(keys, (str, bytes)) or not isinstance(keys, Iterable):
        raise ModelError(f"{what}: keys must be a list of values or (time, value) pairs")
    keys = list(keys)
    if not keys:
        raise ModelError(f"{what} has no keys; give at least one value")
    if all(isinstance(k, (tuple, list)) for k in keys):
        if any(len(k) != 2 for k in keys):
            raise ModelError(f"{what}: each key must be (time, value)")
        times = [_float(t, f"{what} key time") for t, _ in keys]
        values = [_float(v, f"{what} key value") for _, v in keys]
    elif any(isinstance(k, (tuple, list)) for k in keys):
        raise ModelError(f"{what}: use all plain values or all (time, value) pairs, not a mix")
    else:
        values = [_float(v, f"{what} key value") for v in keys]
        n = len(values)
        times = [0.0] if n == 1 else [length * i / (n - 1) for i in range(n)]
    return tuple(times), tuple(values)


def _check_track(name: str, length: float, track: Track) -> None:
    what = f"clip {name!r} track {track.part!r}"
    if track.channel not in CHANNELS:
        raise ModelError(f"{what}: unknown channel {track.channel!r}; use {', '.join(CHANNELS)}")
    if track.ease is not None and track.ease not in EASES:
        raise ModelError(f"{what}: ease must be one of {', '.join(EASES)}, got {track.ease!r}")
    if not track.times or len(track.times) != len(track.values):
        raise ModelError(f"{what} needs one value per key time")
    if any(b < a for a, b in zip(track.times, track.times[1:], strict=False)):
        raise ModelError(f"{what}: key times must be in order, got {list(track.times)}")
    if len(set(track.times)) != len(track.times):
        raise ModelError(f"{what}: two keys share a time; give each key its own time")
    if track.channel == "visible" and any(v not in (0.0, 1.0) for v in track.values):
        raise ModelError(f"{what}: visible keys must be 0 or 1 (hidden or shown)")
    if track.channel == "frame" and any(v < 0 or v != int(v) for v in track.values):
        raise ModelError(f"{what}: frame keys must be frame numbers 0, 1, 2, ...")
    if track.times[0] < -_EPS or track.times[-1] > length + _EPS:
        raise ModelError(
            f"{what}: key times must be inside the clip (0 to {length:g} s), "
            f"got {list(track.times)}"
        )


def clip(
    name: str,
    length: float,
    tracks: Mapping[str, Sequence] | Sequence[Track] | None = None,
    *,
    loop: bool = False,
    ease: str = "smooth",
    events: Iterable[tuple[float, str]] = (),
    travel=None,
    collisions: bool = True,
) -> Clip:
    """A named clip ``length`` seconds long.

    ``tracks`` maps part names to keys for their joints: plain values spread evenly from 0 to
    ``length`` (one value holds at t = 0), or ``(time, value)`` pairs. Or it is a list of
    tracks: ``turn(...)``, ``move(...)``, ``visible(...)``, generators like ``swing(...)``
    (nested lists are fine). ``ease`` is ``smooth`` (default), ``linear``, ``step``, ``in``
    (accelerating into each key: falls, slams) or ``out`` (decelerating).
    ``travel`` is how fast an in-place clip (a walk) moves the asset through the world, in
    voxels per second, for the engine to match so feet don't slide. ``collisions=False`` turns
    off the check for parts passing through each other (debris tumbling past debris).
    """
    if not isinstance(name, str) or not _NAME.match(name):
        raise ModelError(
            f"clip names use letters, digits, '_' and '-', starting with a letter (they become "
            f"file names), got {name!r}"
        )
    length = _float(length, f"clip {name!r} length")
    if not length > 0:
        raise ModelError(f"clip {name!r} length must be > 0 seconds, got {length:g}")
    if ease not in EASES:
        raise ModelError(f"clip {name!r} ease must be one of {', '.join(EASES)}, got {ease!r}")
    built: list[Track] = []
    if isinstance(tracks, Mapping):
        for part, keys in tracks.items():
            times, values = _keys(name, part, keys, length)
            built.append(Track(part, "joint", times, values))
    elif tracks is not None:

        def flatten(items):
            for t in items:
                if isinstance(t, (list, tuple)):
                    yield from flatten(t)
                else:
                    yield t

        for t in flatten(tracks):
            if isinstance(t, TrackSpec):
                t = t.resolve(name, length)
            if not isinstance(t, Track):
                raise ModelError(
                    f"clip {name!r} tracks must be a dict of part -> keys, or a list of tracks "
                    f"(turn(...), move(...), swing(...), ...), got {t!r}"
                )
            built.append(t)
    if not built:
        raise ModelError(f"clip {name!r} has no tracks; give it keys for at least one part")
    for t in built:
        _check_track(name, length, t)
    joints = [t.part for t in built if t.channel == "joint"]
    twice = sorted({p for p in joints if joints.count(p) > 1})
    if twice:
        raise ModelError(
            f"clip {name!r} has two joint tracks for {twice[0]!r}; a joint takes one track "
            "(free turn/move tracks on one channel add up instead)"
        )
    for channel, helper in (("visible", "visible"), ("frame", "frames")):
        on = [t.part for t in built if t.channel == channel]
        twice = sorted({p for p in on if on.count(p) > 1})
        if twice:
            raise ModelError(
                f"clip {name!r} has two {helper}(...) tracks for {twice[0]!r}; merge them"
            )
    evs = []
    for ev in events:
        try:
            t, label = ev
        except (TypeError, ValueError):
            raise ModelError(f"clip {name!r} events must be (time, name) pairs") from None
        t = _float(t, f"clip {name!r} event time")
        if not -_EPS <= t <= length + _EPS:
            raise ModelError(f"clip {name!r} event {label!r} at {t:g} s is outside the clip")
        evs.append((t, str(label)))
    move_by = None
    if travel is not None:
        try:
            move_by = tuple(_float(v, f"clip {name!r} travel") for v in travel)
        except TypeError:
            move_by = ()
        if len(move_by) != 3:
            raise ModelError(f"clip {name!r} travel must be (x, y, z) voxels per second")
    return Clip(
        name, length, tuple(built), bool(loop), ease, tuple(sorted(evs)), move_by, bool(collisions)
    )
