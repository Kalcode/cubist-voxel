"""The build context: the palette, the build seed, and which ops a build used.

``AssetDef.build`` sets it; paint ops that step along ramps read the palette from it, and
seeded ops mix their own seed with the build seed so one ``--seed`` reshuffles everything.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from cubist.core.palette import Palette


@dataclass
class BuildContext:
    palette: Palette | None = None
    seed: int = 0
    ops_used: set[str] = field(default_factory=set)


_current: ContextVar[BuildContext | None] = ContextVar("cubist_build", default=None)


def current() -> BuildContext:
    """The active context, or a throwaway default when nothing is being built."""
    return _current.get() or BuildContext()


@contextmanager
def building(palette: Palette | None = None, seed: int = 0) -> Iterator[BuildContext]:
    ctx = BuildContext(palette, int(seed))
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)


def active() -> BuildContext | None:
    """The active context, or None outside a build."""
    return _current.get()
