"""The walk lint and the storey lint, for buildings an engine's character walks through.

The walk lint flood-fills a building from the street at its ``door`` socket with a walker the
size of the game's character (``[scale]`` ``walker_width``, ``walker_height`` and
``walker_climb``, plus MARGIN) and errors on each door, storey, cellar or tour socket it never
reaches. The storey lint keeps each hideable part a horizontal band, because an engine hides the
storeys above the character by y.
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
DOOR_REACH = 16  # voxels from 'door' the walk may start on the street (eaves or a jetty over it)
REACH_Y = 2  # a socket is reached by feet within this many voxels of its y
STOREY_TAGS = ("upper", "roof")
_MOVES = ((0, 1), (0, -1), (2, 1), (2, -1))  # (axis, step)


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
    """Kiln's goblin (``Goblin.cs``) walking a grid, checked with MARGIN voxels to spare.

    **Where it stands.** Between whole voxels the goblin's box covers ``width + 1`` columns on
    each axis (``foot``); on a whole voxel, for an instant, ``width``. A state is that footprint's
    corner (i, k) and the feet's height F, in local coordinates (they start at ``origin``): the
    footprint is columns i to i + foot - 1 and k to k + foot - 1, and it stands on the highest
    voxel under them, as ``GroundBelow`` finds it. ``ground`` holds the top of the highest voxel
    under each footprint at each row or below (0: none), so feet dropping from F land at
    ``ground[i, F - 1, k]``.

    **Its box.** The margin box is W × H × W (``W = width + MARGIN``, ``H = height + MARGIN``), one
    column wider than the footprint. ``clear[X, F, Z]`` says the box with its corner at (X, F, Z)
    holds no voxel. Along a move the box must hold both footprints (X = the smaller i); across it,
    either box holding the footprint will do (Z = k - 1 or k), as the walker may keep its spare
    column on either side.

    **A move** goes one voxel in x or z as ``MoveAxis`` does. Where the box fits it moves and drops
    to the ground under the new footprint; where it doesn't, it steps up if the box raised by
    ``climb`` fits, onto the highest voxel under the new footprint (it must be higher). Where it
    lands, some box holding the footprint must fit. The instant on a whole voxel's line (the
    narrower footprint) doesn't count: the goblin passes it within a frame.

    **The world** is the asset's voxels at rest (hidden parts left out) and, up to GROUND_PAD
    voxels past the asset's x-z bounds, solid ground below ``ground_y``, except over the x-z
    bounds of the voxels below ``ground_y``: Kiln carves that (a cellar's footprint), and leaves
    the street under eaves and jetties. The walk starts out on the street: footprints wholly
    outside the asset's x-z bounds, within DOOR_REACH voxels of 'door'."""

    def __init__(self, asset: BuiltAsset, grid, width: int, height: int, climb: int):
        self.width, self.height, self.climb = width, height, climb
        self.foot, self.W, self.H = width + 1, width + MARGIN, height + MARGIN
        (x0, y0, z0), (x1, y1, z1) = grid.bounds()
        line = asset.ground_y
        bottom = min(y0, line) - 1  # at least one row of ground
        top = max(y1, line) + climb + self.H  # room to raise the box over the highest standing
        self.origin = np.array((x0 - GROUND_PAD, bottom, z0 - GROUND_PAD))
        pad = 2 * GROUND_PAD
        solid = np.zeros((x1 - x0 + pad, top - bottom, z1 - z0 + pad), bool)
        solid[:, : line - bottom] = True
        o = np.subtract(grid.origin, self.origin)
        below = grid.data[:, : max(line - grid.origin[1], 0)] > 0
        if below.any():
            xs = np.flatnonzero(below.any(axis=(1, 2))) + o[0]
            zs = np.flatnonzero(below.any(axis=(0, 1))) + o[2]
            solid[xs[0] : xs[-1] + 1, :, zs[0] : zs[-1] + 1] = False
        s = grid.shape
        solid[o[0] : o[0] + s[0], o[1] : o[1] + s[1], o[2] : o[2] + s[2]] |= grid.data > 0
        self.bounds = (o[0], o[0] + s[0], o[2], o[2] + s[2])  # local x-z bounds, half-open

        box = _any_in_window(_any_in_window(solid, self.W, 0), self.W, 2)
        self.clear = ~_any_in_window(box, self.H, 1)
        self.ground = _tops(_any_in_window(_any_in_window(solid, self.foot, 0), self.foot, 2))
        shape = self.ground.shape
        self.reached = np.zeros((shape[0], self.clear.shape[1], shape[2]), bool)

    def _span(self, lo, hi, axis: int) -> np.ndarray:
        """The whole numbers from lo to hi (rounded inward) that index ``reached`` on axis."""
        floor = 1 if axis == 1 else 0  # feet stand on a row below them
        return np.arange(
            max(math.ceil(lo), floor), min(math.floor(hi), self.reached.shape[axis] - 1) + 1
        )

    def _around(self, pos, reach: float = 0) -> tuple[np.ndarray, ...]:
        """Every state whose footprint is within ``reach`` of pos's x and z (0: holds them, edges
        included), with its feet within REACH_Y of its y."""
        lx, ly, lz = np.asarray(pos, float) - self.origin
        a = self.foot
        i, f, k = np.meshgrid(
            self._span(lx - a - reach, lx + reach, 0),
            self._span(ly - REACH_Y, ly + REACH_Y, 1),
            self._span(lz - a - reach, lz + reach, 2),
            indexing="ij",
        )
        i, f, k = i.ravel(), f.ravel(), k.ravel()
        dx = np.maximum(np.maximum(i - lx, lx - (i + a)), 0)
        dz = np.maximum(np.maximum(k - lz, lz - (k + a)), 0)
        near = dx * dx + dz * dz <= reach * reach
        return i[near], f[near], k[near]

    def _fits(self, i, f, k) -> np.ndarray:
        """Some box holding each footprint is clear at its feet."""
        nx, _, nz = self.clear.shape
        out = np.zeros(len(i), bool)
        for x in (i - 1, i):
            for z in (k - 1, k):
                ok = (x >= 0) & (x < nx) & (z >= 0) & (z < nz)
                out[ok] |= self.clear[x[ok], f[ok], z[ok]]
        return out

    def start(self, door) -> bool:
        """Flood-fill from where the walker stands on the street near ``door``: footprints wholly
        outside the asset's x-z bounds, within DOOR_REACH of it, its feet within REACH_Y of its y.
        False if there are none."""
        i, f, k = self._around(door, DOOR_REACH)
        x0, x1, z0, z1 = self.bounds
        a = self.foot
        street = (i + a <= x0) | (i >= x1) | (k + a <= z0) | (k >= z1)
        i, f, k = i[street], f[street], k[street]
        standing = (self.ground[i, f - 1, k] == f) & self._fits(i, f, k)
        if not standing.any():
            return False
        shape = self.reached.shape
        flat = self.reached.reshape(-1)
        cells = np.unique(np.ravel_multi_index((i[standing], f[standing], k[standing]), shape))
        while len(cells):
            flat[cells] = True
            cells = np.unique(np.concatenate(self._moves(*np.unravel_index(cells, shape))))
            cells = cells[~flat[cells]]
        return True

    def _moves(self, xs, fs, zs) -> list[np.ndarray]:
        """Where each state's four moves land, as flat indices: Goblin.cs's MoveAxis, then its
        gravity, once for each box that may hold the move."""
        nx, _, nz = self.clear.shape
        out = []
        for axis, step in _MOVES:
            i, k = (xs + step, zs) if axis == 0 else (xs, zs + step)
            along = np.minimum(xs, i) if axis == 0 else np.minimum(zs, k)
            for side in (-1, 0):
                x, z = (along, zs + side) if axis == 0 else (xs + side, along)
                ok = (x >= 0) & (x < nx) & (z >= 0) & (z < nz)
                x, z, f, ni, nk = x[ok], z[ok], fs[ok], i[ok], k[ok]
                level = self.clear[x, f, z]
                raised = ~level & self.clear[x, f + self.climb, z]
                up = self.ground[ni, f + self.climb - 1, nk]
                to = np.where(level, self.ground[ni, f - 1, nk], 0)
                to = np.where(raised & (up > f), up, to).astype(np.intp)
                keep = to > 0  # 0: it stopped, or fell out of the world
                moved = keep & (to != f)  # where it dropped or stepped, a box must fit too
                keep[moved] = self._fits(ni[moved], to[moved], nk[moved])
                out.append(np.ravel_multi_index((ni[keep], to[keep], nk[keep]), self.reached.shape))
        return out

    def reaches(self, pos) -> bool:
        i, f, k = self._around(pos)
        return bool(self.reached[i, f, k].any())

    def nearest(self, pos) -> tuple[list, float]:
        """The reached place nearest to ``pos`` (the middle of the goblin's feet), and how far."""
        cells = np.argwhere(self.reached).astype(float)
        cells[:, [0, 2]] += self.foot / 2
        cells += self.origin
        d = np.linalg.norm(cells - np.asarray(pos, float), axis=1)
        j = int(np.argmin(d))
        return [_num(v) for v in cells[j]], float(d[j])

    def needs(self) -> str:
        return (
            f"the walker is {self.width} × {self.height} voxels and climbs {self.climb} ([scale] "
            f"walker_*), checked with {MARGIN} voxels to spare, so it needs doorways {self.W} "
            f"wide and {self.H} tall, risers of {self.climb} at most, and stairwells open "
            f"{self.climb + self.H} voxels above each tread"
        )


def check_walk(asset: BuiltAsset, style: StyleBible) -> list[Finding]:
    """A building (or a walkable asset, like a bridge) must let the walker in from the street at
    its ``door`` and up to every ``floor_N``, ``cellar`` and ``tour_NN`` socket. Without a
    ``door`` or ``floor_0`` it can't be checked, and lint warns (an older building no one
    enters)."""
    if not set(WALK_TAGS) & set(asset.tags) or not all(k in style.scale for k in WALKER_KEYS):
        return []
    width, height, climb = (int(style.scale[k]) for k in WALKER_KEYS)
    sockets = {n: Socket.from_value(v, n).position for n, v in asset.sockets.items()}
    ground = f"on the ground (y = ground_y, {asset.ground_y} here)"
    if "building" in asset.tags:
        door_at = f"out on the street in front of the main entrance, {ground}"
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
                    "warning",
                    f"{asset.name} has no {name!r} socket, so the walk lint can't check it can be "
                    f"entered; add one {where}. The walk goes from the street at 'door' to every "
                    "floor_N, cellar and tour_NN socket",
                    {"socket": name},
                )
            )
    grid = asset.visible_grid()
    if "door" not in sockets or grid.count() == 0:
        return out
    walker = Walker(asset, grid, width, height, climb)
    details = {"walker": [width, height, climb], "box": [walker.W, walker.H, walker.W]}
    door = sockets["door"]
    if not walker.start(door):
        out.append(
            Finding(
                "walk",
                "error",
                f"the walker has nowhere to start at 'door' {_at(door)}: it stands on the street "
                f"(outside the asset's x-z bounds) within {DOOR_REACH} voxels of 'door', on "
                f"something solid with its feet within {REACH_Y} voxels of y = {_num(door[1])}, "
                f"and finds no such place. Put 'door' {door_at}; nothing was walked",
                {"socket": "door", **details},
            )
        )
        return out
    for name, pos in sorted(sockets.items()):
        if not walked(name) or walker.reaches(pos):
            continue
        point, distance = walker.nearest(pos)
        out.append(
            Finding(
                "walk",
                "error",
                f"the walker can't reach {name!r} {_at(pos)} from the street at 'door': the "
                f"nearest it gets is {_at(point)}, {distance:.0f} voxels away. Check the stair or "
                f"doorway between them: {walker.needs()}",
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
