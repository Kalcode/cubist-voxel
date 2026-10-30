"""The walk lint and the storey lint, for buildings an engine's character walks through.

The walk lint flood-fills a building from its ``door`` socket with a walker the size of the
game's character (``[scale]`` ``walker_width``, ``walker_height`` and ``walker_climb``, plus
MARGIN) and errors on each storey, cellar or tour socket it never reaches. The storey lint keeps
each hideable part a horizontal band, because an engine hides the storeys above the character by y.
"""

from __future__ import annotations

import math

import numpy as np

from cubist.core.model import BuiltAsset
from cubist.core.socket import Socket
from cubist.core.style import WALKER_KEYS, StyleBible
from cubist.lint.checks import Finding

WALK_TAGS = ("building", "walkable")  # a bridge is walkable: door and floor_0 at its deck ends
MARGIN = 2  # voxels added to the walker's width and height, so what passes has room to spare
GROUND_PAD = 16  # voxels of ground around the asset's footprint
REACH_Y = 2  # a socket is reached by feet within this many voxels of its y
STOREY_TAGS = ("upper", "roof")
_MOVES = ((1, 0), (-1, 0), (0, 1), (0, -1))


def walked(name: str) -> bool:
    """The sockets the walker must reach: the door, each storey, the cellar and a tour's stops."""
    return name in ("door", "cellar") or name.startswith(("floor_", "tour_"))


def _num(v: float):
    return int(v) if float(v).is_integer() else round(float(v), 2)


def _at(pos) -> str:
    return f"({', '.join(str(_num(v)) for v in pos)})"


def _any_in_window(a: np.ndarray, n: int, axis: int) -> np.ndarray:
    """``out[i] = a[i] | ... | a[i + n - 1]`` along ``axis`` (n - 1 shorter, or empty), by
    doubling windows."""
    a = np.moveaxis(a, axis, 0)
    m = a.shape[0] - n + 1
    if m <= 0:
        return np.moveaxis(np.zeros((0, *a.shape[1:]), bool), 0, axis)
    span, size = a, 1
    while size * 2 <= n:
        span = span[:-size] | span[size:]
        size *= 2
    return np.moveaxis(span[:m] | span[n - size : n - size + m], 0, axis)


def _tops(foot: np.ndarray) -> np.ndarray:
    """For each row (axis 1), the top of the highest voxel ``foot`` holds at that row or below
    (0: none)."""
    rows = np.arange(1, foot.shape[1] + 1, dtype=np.min_scalar_type(foot.shape[1]))
    out = np.where(foot, rows[None, :, None], 0).astype(rows.dtype)
    return np.maximum.accumulate(out, axis=1, out=out)


class Walker:
    """Kiln's character on a grid: a box W × H × W (``W = width + MARGIN``, ``H = height +
    MARGIN``) that moves a voxel at a time in x or z as the goblin's ``MoveAxis`` does. Where the
    box fits at the new spot it moves there and drops onto the highest voxel under it; where it
    doesn't, it steps up if the box raised by ``climb`` fits there, onto the highest voxel under
    it; else it stops.

    Halfway between two spots the goblin's box covers width + 1 columns along the move (W - 1,
    one fewer than the walker's), and it falls if nothing is under them. So a move needs
    something under that footprint at the walker's height before a step up, and a drop must land
    where the walker's does: a gap the goblin would fall into stops the walk.

    The world is the asset's voxels at rest (hidden parts left out) and, outside the asset's x-z
    bounds and up to GROUND_PAD voxels past them, solid ground below ``ground_y``. Inside the
    bounds there is no ground: Kiln carves a cellared building's footprint.

    A state is a cell (X, F, Z) in local coordinates (they start at ``origin``): the box spans
    columns X to X + W - 1 and Z to Z + W - 1 and rows F to F + H - 1. ``clear`` says the box
    holds no voxel. ``ground`` is the top of the highest voxel under the footprint at each row or
    below (0: none), so feet dropping from F land at ``ground[X, F - 1, Z]``; ``half_x`` and
    ``half_z`` are the same for the footprint halfway through a move in x or z, indexed by the
    larger of the two cells' X or Z."""

    def __init__(self, asset: BuiltAsset, width: int, height: int, climb: int):
        self.width, self.height, self.climb = width, height, climb
        self.W, self.H = width + MARGIN, height + MARGIN
        grid = asset.visible_grid()
        (x0, y0, z0), (x1, y1, z1) = grid.bounds()
        bottom = min(y0, asset.ground_y) - 1  # at least one row of ground
        # Room to raise the box over the highest place to stand (ground_y is below y1 when built).
        top = max(y1, asset.ground_y) + climb + self.H
        self.origin = np.array((x0 - GROUND_PAD, bottom, z0 - GROUND_PAD))
        pad = 2 * GROUND_PAD
        solid = np.zeros((x1 - x0 + pad, top - bottom, z1 - z0 + pad), bool)
        solid[:, : asset.ground_y - bottom] = True
        solid[GROUND_PAD:-GROUND_PAD, :, GROUND_PAD:-GROUND_PAD] = False
        o = np.subtract(grid.origin, self.origin)
        s = grid.shape
        solid[o[0] : o[0] + s[0], o[1] : o[1] + s[1], o[2] : o[2] + s[2]] |= grid.data > 0

        along_x = _any_in_window(solid, self.W, 0)
        foot = _any_in_window(along_x, self.W, 2)
        self.clear = ~_any_in_window(foot, self.H, 1)
        self.ground = _tops(foot)
        self.half_x = _tops(_any_in_window(_any_in_window(solid, self.W - 1, 0), self.W, 2))
        self.half_z = _tops(_any_in_window(along_x, self.W - 1, 2))
        self.reached = np.zeros(self.clear.shape, bool)

    def _around(self, pos) -> tuple[slice, slice, slice]:
        """The cells whose footprint holds pos's x and z (edges included), with their feet within
        REACH_Y of its y."""
        lx, ly, lz = np.asarray(pos, float) - self.origin

        def span(lo, hi, n):
            lo, hi = max(math.ceil(lo), 0), min(math.floor(hi), n - 1) + 1
            return slice(lo, max(lo, hi))

        nx, nf, nz = self.clear.shape
        return (
            span(lx - self.W, lx, nx),
            span(max(ly - REACH_Y, 1), ly + REACH_Y, nf),
            span(lz - self.W, lz, nz),
        )

    def start(self, pos) -> bool:
        """Flood-fill from the places the walker can stand at ``pos``; False if there are none."""
        sx, sf, sz = self._around(pos)
        feet = np.arange(sf.start, sf.stop)[None, :, None]
        under = self.ground[sx, sf.start - 1 : sf.stop - 1, sz]
        xs, fs, zs = np.nonzero(self.clear[sx, sf, sz] & (under == feet))
        if len(xs) == 0:
            return False
        shape = self.reached.shape
        flat = self.reached.reshape(-1)
        cells = np.ravel_multi_index((xs + sx.start, fs + sf.start, zs + sz.start), shape)
        while len(cells):
            flat[cells] = True
            cells = np.unique(np.concatenate(self._moves(*np.unravel_index(cells, shape))))
            cells = cells[~flat[cells]]
        return True

    def _moves(self, xs, fs, zs) -> list[np.ndarray]:
        """Where each cell's four moves land, as flat indices: Goblin.cs's MoveAxis, then its
        gravity."""
        nx, _, nz = self.clear.shape
        out = []
        for dx, dz in _MOVES:
            x, z = xs + dx, zs + dz
            ok = (x >= 0) & (x < nx) & (z >= 0) & (z < nz)
            x, f, z = x[ok], fs[ok], z[ok]
            if dx:
                half = self.half_x[np.maximum(x, xs[ok]), f - 1, z]
            else:
                half = self.half_z[x, f - 1, np.maximum(z, zs[ok])]
            level = self.clear[x, f, z]
            land = self.ground[x, f - 1, z]
            raised = ~level & (half == f) & self.clear[x, f + self.climb, z]
            to = np.where(level & (land == half), land, 0)
            to = np.where(raised, self.ground[x, f + self.climb - 1, z], to).astype(np.intp)
            keep = to > 0  # 0: it stopped, fell where the walker can't follow, or left the world
            out.append(np.ravel_multi_index((x[keep], to[keep], z[keep]), self.reached.shape))
        return out

    def reaches(self, pos) -> bool:
        return bool(self.reached[self._around(pos)].any())

    def nearest(self, pos) -> tuple[list, float]:
        """The reached place nearest to ``pos`` (the middle of the walker's feet), and how far."""
        cells = np.argwhere(self.reached).astype(float)
        cells[:, [0, 2]] += self.W / 2
        cells += self.origin
        d = np.linalg.norm(cells - np.asarray(pos, float), axis=1)
        k = int(np.argmin(d))
        return [_num(v) for v in cells[k]], float(d[k])

    def needs(self) -> str:
        return (
            f"the walker is {self.width} × {self.height} voxels and climbs {self.climb} ([scale] "
            f"walker_*), checked with {MARGIN} voxels to spare, so it needs doorways {self.W} "
            f"wide and {self.H} tall, risers of {self.climb} at most, and stairwells open "
            f"{self.climb + self.H} voxels above each tread"
        )


def check_walk(asset: BuiltAsset, style: StyleBible) -> list[Finding]:
    """A building (or a walkable asset, like a bridge) must let the walker in at its ``door``
    and up to every ``floor_N``, ``cellar`` and ``tour_NN`` socket."""
    if not set(WALK_TAGS) & set(asset.tags) or not all(k in style.scale for k in WALKER_KEYS):
        return []
    width, height, climb = (int(style.scale[k]) for k in WALKER_KEYS)
    sockets = {n: Socket.from_value(v, n).position for n, v in asset.sockets.items()}
    ground = f"on the ground (y = ground_y, {asset.ground_y} here)"
    if "building" in asset.tags:
        door_at = f"outside the main entrance, {ground}"
        floor_at = "on the ground floor, where someone would stand"
    else:
        door_at = f"where the walk starts, {ground}: one end of a bridge's deck"
        floor_at = "where it ends: the other end of the deck"
    out = []
    for name, where in (("door", door_at), ("floor_0", floor_at)):
        if name not in sockets:
            out.append(
                Finding(
                    "walk",
                    "error",
                    f"{asset.name} has no {name!r} socket; add one {where}: the walk lint walks "
                    "from 'door' to every floor_N, cellar and tour_NN socket",
                    {"socket": name},
                )
            )
    if "door" not in sockets or asset.visible_grid().count() == 0:
        return out
    walker = Walker(asset, width, height, climb)
    details = {"walker": [width, height, climb], "box": [walker.W, walker.H, walker.W]}
    door = sockets["door"]
    if not walker.start(door):
        out.append(
            Finding(
                "walk",
                "error",
                f"the walker can't stand at 'door' {_at(door)}: no spot there fits its "
                f"{walker.W} × {walker.H} × {walker.W} box on something solid with its feet "
                f"within {REACH_Y} voxels of y = {_num(door[1])}. Put 'door' {door_at}, clear of "
                "the walls; nothing else was walked",
                {"socket": "door", **details},
            )
        )
        return out
    for name, pos in sorted(sockets.items()):
        if name == "door" or not walked(name) or walker.reaches(pos):
            continue
        point, distance = walker.nearest(pos)
        out.append(
            Finding(
                "walk",
                "error",
                f"the walker can't reach {name!r} {_at(pos)} from 'door': the nearest it gets is "
                f"{_at(point)}, {distance:.0f} voxels away. Check the stair or doorway between "
                f"them: {walker.needs()}",
                {"socket": name, "nearest": point, "distance": round(distance, 2), **details},
            )
        )
    return out


def check_storeys(asset: BuiltAsset) -> list[Finding]:
    """Parts tagged ``upper`` or ``roof`` (their own tag or a parent's) are horizontal bands: an
    engine hides the storeys above the character from each such part's lowest voxel up, so a part
    reaching up past the next one's lowest voxel loses its top. Parts starting on the same line
    share a band. The topmost band is exempt: the one under it may reach up into it (a roof whose
    eaves come down past the top storey's walls)."""
    hidden = set(asset.hidden_parts())
    bands: dict[int, list] = {}  # lowest y -> [(top, part)]
    for p in asset.parts:
        b = p.grid.bounds()
        if p.name not in hidden and b and asset.occlusion_of(p.name) in STOREY_TAGS:
            bands.setdefault(b[0][1], []).append((b[1][1], p))
    lows = sorted(bands)
    out = []
    for lo, nlo in zip(lows[:-2], lows[1:-1], strict=True):
        n = bands[nlo][0][1]
        via = ""
        if n.occlusion is None:
            tag = asset.occlusion_of(n.name)
            via = f" ({n.name!r} hangs from {n.parent!r} and shares its {tag!r} tag)"
        for hi, p in bands[lo]:
            if hi <= nlo:
                continue
            out.append(
                Finding(
                    "storeys",
                    "error",
                    f"part {p.name!r} reaches {hi - nlo} row(s) into {n.name!r}, which starts at "
                    f"y = {nlo}{via}: an engine hides the storeys above the character from each "
                    f"upper or roof part's lowest voxel up, so {p.name!r} would lose its top. "
                    f"Split the building by y bands, so what's at or above y = {nlo} belongs to "
                    "the part above",
                    {"part": p.name, "next": n.name, "rows": hi - nlo, "next_y": nlo},
                )
            )
    return out
