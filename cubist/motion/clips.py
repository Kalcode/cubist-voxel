"""Clips: named, timed motion made of tracks of keys.

A track drives one channel of one part. In M7 the only channel is ``"joint"``: the value of the
part's joint (degrees for a hinge, voxels for a slide, turns for a spin), relative to rest.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from cubist.core.errors import ModelError

EASES = ("smooth", "linear", "step")
CHANNELS = ("joint",)
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

    def track_ease(self, track: Track) -> str:
        return track.ease or self.ease

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "length": self.length,
            "loop": self.loop,
            "ease": self.ease,
            "tracks": [t.to_dict() for t in self.tracks],
            "events": [[t, name] for t, name in self.events],
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
        )


def _float(value, what: str) -> float:
    if isinstance(value, bool):
        raise ModelError(f"{what} must be a number, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ModelError(f"{what} must be a number, got {value!r}") from None


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
) -> Clip:
    """A named clip ``length`` seconds long.

    ``tracks`` maps part names to keys for their joints: plain values spread evenly from 0 to
    ``length`` (one value holds at t = 0), or ``(time, value)`` pairs. A list of ``Track``
    objects also works. ``ease`` is ``smooth`` (default), ``linear`` or ``step``.
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
        for t in tracks:
            if not isinstance(t, Track):
                raise ModelError(
                    f"clip {name!r} tracks must be a dict of part -> keys or a list of Track, "
                    f"got {t!r}"
                )
            built.append(t)
    for t in built:
        _check_track(name, length, t)
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
    return Clip(name, length, tuple(built), bool(loop), ease, tuple(sorted(evs)))
