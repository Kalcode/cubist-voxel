"""The walk lint and the storey lint, for buildings an engine's character walks through.

The walk lint flood-fills a building from the street at its ``door`` socket with a walker the
size of the game's character (``[scale]`` ``walker_width``, ``walker_height`` and
``walker_climb``, plus MARGIN) and errors on each door, storey, cellar or tour socket it never
reaches, or reaches but can't walk back to the street from, and on each leg of the tour (from
``door``, then stop to stop) it can't walk in a straight line. The storey lint keeps each hideable
part a horizontal band, because an engine hides the storeys above the character by y.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order

from cubist.core.model import BuiltAsset
from cubist.core.socket import Socket
from cubist.core.style import WALKER_KEYS, StyleBible
from cubist.lint.checks import Finding

WALK_TAGS = ("building", "walkable")  # a bridge is walkable: door and floor_0 at its deck ends
MARGIN = 2  # voxels added to the walker's width and height, so what passes has room to spare
GROUND_PAD = 16  # voxels of ground around the asset's footprint
DOOR_REACH = 16  # voxels from 'door' the walk may start on the street (eaves or a jetty over it)
REACH_Y = 2  # a socket is reached by feet within this many voxels of its y
ARRIVE = 2.0  # Kiln's walk arrives at a stop within this many voxels (x-z), then heads for the next
LEG_SLACK = 1.0  # a straight walk on the grid: the footprint's centre stays this close to the line
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
    the street under eaves and jetties. Past the bounds the ground (the street) lies on every
    side, or on the asset's ``street_sides`` only (a quay's river side has none). The walk
    starts out on the street: footprints wholly outside the asset's x-z bounds, within
    DOOR_REACH voxels of 'door'.

    **The way back.** A drop is a move too, so a flight walked down is never climbed. ``back``
    marks the reached states that can walk back to the street: a flood from the starts over the
    moves reversed."""

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
        if asset.street_sides:  # the street lies on those sides only: no ground on the others
            solid &= self._street(asset.street_sides, x1 - x0, z1 - z0)[:, None, :]
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

    @staticmethod
    def _street(sides, nx: int, nz: int) -> np.ndarray:
        """Which columns of the padded x-z grid hold ground: under the asset (Kiln's carve comes
        after), and the pad on each named side, its corners included."""
        p = GROUND_PAD
        street = np.zeros((nx + 2 * p, nz + 2 * p), bool)
        street[p : p + nx, p : p + nz] = True
        pads = {
            "-x": (slice(0, p), slice(None)),
            "+x": (slice(p + nx, None), slice(None)),
            "-z": (slice(None), slice(0, p)),
            "+z": (slice(None), slice(p + nz, None)),
        }
        for side in sides:
            street[pads[side]] = True
        return street

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
        """Flood-fill from where the walker stands on the street near ``door`` (footprints wholly
        outside the asset's x-z bounds, within DOOR_REACH of it, its feet within REACH_Y of its
        y), then find the way back. False if there's nowhere to start."""
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
        starts = np.unique(np.ravel_multi_index((i[standing], f[standing], k[standing]), shape))
        cells, moves = starts, []
        while len(cells):
            flat[cells] = True
            src, dst = self._moves(cells)
            moves.append((src, dst))
            cells = np.unique(dst)
            cells = cells[~flat[cells]]
        self._way_back(starts, moves)
        return True

    def _way_back(self, starts, moves) -> None:
        """``back``: the reached states with a way to the street, by a breadth-first flood from
        the starts over every move turned round."""
        states = np.flatnonzero(self.reached)
        n = len(states)
        src = np.searchsorted(states, np.concatenate([m[0] for m in moves]))
        dst = np.searchsorted(states, np.concatenate([m[1] for m in moves]))
        self._states, self._src, self._dst = states, src, dst  # for the tour's legs
        # Node n stands for the street, with an edge to each start.
        rows = np.concatenate([dst, np.full(len(starts), n)])
        cols = np.concatenate([src, np.searchsorted(states, starts)])
        turned = csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n + 1, n + 1))
        order = breadth_first_order(turned, n, directed=True, return_predecessors=False)
        self.back = np.zeros(self.reached.shape, bool)
        self.back.reshape(-1)[states[order[order < n]]] = True

    def _moves(self, cells) -> tuple[np.ndarray, np.ndarray]:
        """Each state's four moves (Goblin.cs's MoveAxis, then its gravity, once for each box that
        may hold the move), as flat indices: where from and where to."""
        xs, fs, zs = np.unravel_index(cells, self.reached.shape)
        nx, _, nz = self.clear.shape
        froms, tos = [], []
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
                froms.append(cells[ok][keep])
                tos.append(np.ravel_multi_index((ni[keep], to[keep], nk[keep]), self.reached.shape))
        return np.concatenate(froms), np.concatenate(tos)

    def _centres(self):
        """Each reached state's footprint centre (x, z) and feet, in the asset's coordinates."""
        if not hasattr(self, "_cxzf"):
            i, f, k = np.unravel_index(self._states, self.reached.shape)
            half = self.foot / 2
            self._cxzf = (i + self.origin[0] + half, k + self.origin[2] + half, f + self.origin[1])
        return self._cxzf

    def _stands(self, x, z, y) -> bool:
        """Some reached state has its footprint's centre within LEG_SLACK of (x, z), its feet within
        REACH_Y of y: the walker can be there."""
        cx, cz, feet = self._centres()
        return bool(((np.hypot(cx - x, cz - z) <= LEG_SLACK) & (np.abs(feet - y) <= REACH_Y)).any())

    def follows(self, a, b) -> dict | None:
        """Whether the walker makes the leg from stop ``a`` to stop ``b`` as Kiln's walk will:
        having arrived anywhere within ARRIVE of ``a`` (x-z), it heads straight for ``b`` until
        within ARRIVE of it. The starts are ``a`` and 8 points ARRIVE round it, turned to the leg
        so the two square to it, the furthest off any line it may take, are among them. Each must
        make it, its footprint's centre within LEG_SLACK of its straight line and its feet between
        the stops' heights (REACH_Y to spare). None if it does; else what fails:

        - ``{"way": "arrival", ...}``: the walker can't stand at a start. The goblin is narrower
          than the walker, so it might arrive there and walk a line never tried; the stop must
          move (``move`` voxels ``toward``), so the lint stays the stricter of the two.
        - ``{"way": "leg", "nearest", "distance"}``: from a start, the farthest point along its line
          the walker gets to, and how far short of ``b``.

        A short leg (``b`` within ARRIVE of ``a``) is walked too: the walker may arrive 2 from ``a``
        and so up to 4 from ``b``. A leg of length 0 is turned to +x for its starts."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b[[0, 2]] - a[[0, 2]]
        length = float(np.hypot(*d))
        ux, uz = (d / length) if length > 0 else (1.0, 0.0)
        starts = [(a[0], a[2])]
        for angle in range(0, 360, 45):
            c, s_ = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            starts.append((a[0] + ARRIVE * (c * ux - s_ * uz), a[2] + ARRIVE * (s_ * ux + c * uz)))
        blocked = [(sx, sz) for sx, sz in starts if not self._stands(sx, sz, a[1])]
        if blocked:
            return self._room(a, blocked)
        cx, cz, feet = self._centres()
        ylo, yhi = min(a[1], b[1]) - REACH_Y, max(a[1], b[1]) + REACH_Y

        def off(px, pz, sx, sz, ex, ez):
            """Distance from (px, pz) to the segment s-e, and how far along it (0..1)."""
            vx, vz = ex - sx, ez - sz
            t = np.clip(((px - sx) * vx + (pz - sz) * vz) / max(vx * vx + vz * vz, 1e-9), 0, 1)
            return np.hypot(px - (sx + t * vx), pz - (sz + t * vz)), t

        # The states any start's line may use, and the moves between them.
        band, _ = off(cx, cz, a[0], a[2], b[0], b[2])
        cand = np.flatnonzero((band <= ARRIVE + LEG_SLACK) & (feet >= ylo) & (feet <= yhi))
        pos = np.full(len(self._states), -1)
        pos[cand] = np.arange(len(cand))
        keep = (pos[self._src] >= 0) & (pos[self._dst] >= 0)
        esrc, edst = pos[self._src[keep]], pos[self._dst[keep]]
        px, pz, pf = cx[cand], cz[cand], feet[cand]
        goal = (np.hypot(px - b[0], pz - b[2]) <= ARRIVE) & (np.abs(pf - b[1]) <= REACH_Y)
        worst = None
        for sx, sz in starts:
            lateral, t = off(px, pz, sx, sz, b[0], b[2])
            near = lateral <= LEG_SLACK
            at_start = (np.hypot(px - sx, pz - sz) <= LEG_SLACK) & (np.abs(pf - a[1]) <= REACH_Y)
            start = near & at_start
            m = len(cand)
            ok = near[esrc] & near[edst]
            rows = np.concatenate([esrc[ok], np.full(int(start.sum()), m)])
            cols = np.concatenate([edst[ok], np.flatnonzero(start)])
            graph = csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(m + 1, m + 1))
            order = breadth_first_order(graph, m, directed=True, return_predecessors=False)
            seen = np.zeros(m + 1, bool)
            seen[order] = True
            if (seen[:m] & goal & near).any():
                continue
            walked = np.flatnonzero(seen[:m])
            j = walked[np.argmax(t[walked])] if len(walked) else None
            point = [_num(v) for v in ((px[j], pf[j], pz[j]) if j is not None else (sx, a[1], sz))]
            short = float((1 - (t[j] if j is not None else 0)) * np.hypot(b[0] - sx, b[2] - sz))
            if worst is None or short > worst["distance"]:
                worst = {"way": "leg", "nearest": point, "distance": round(short, 2)}
        return worst

    def _room(self, a, blocked) -> dict:
        """What to say of a stop the walker can't stand all round: the start points it can't, and
        how far (whole voxels) to move the stop away from them so it can."""
        move = 0
        sumx = sumz = 0.0
        for sx, sz in blocked:
            wx, wz = (sx - a[0]) / ARRIVE, (sz - a[2]) / ARRIVE  # toward the obstacle (0 at a)
            sumx, sumz = sumx + wx, sumz + wz
            k = next((k for k in range(1, 9) if self._stands(sx - k * wx, sz - k * wz, a[1])), None)
            move = None if k is None or move is None else max(move, k)
        n = math.hypot(sumx, sumz) or 1.0
        away = (("x", -sumx / n), ("z", -sumz / n))
        toward = "".join(f"{'+' if v > 0 else '-'}{axis}" for axis, v in away if abs(v) > 0.38)
        sx, sz = blocked[0]
        at = [_num(sx), _num(a[1]), _num(sz)]
        return {"way": "arrival", "at": at, "move": move, "toward": toward}

    def reaches(self, pos) -> bool:
        i, f, k = self._around(pos)
        return bool(self.reached[i, f, k].any())

    def returns(self, pos) -> bool:
        """Some state the walker reaches at ``pos`` has a way back to the street."""
        i, f, k = self._around(pos)
        return bool(self.back[i, f, k].any())

    def nearest(self, pos, among: np.ndarray) -> tuple[list, float]:
        """The state of ``among`` nearest to ``pos`` (the middle of the goblin's feet), and how
        far."""
        cells = np.argwhere(among).astype(float)
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
    its ``door`` and up to every ``floor_N``, ``cellar`` and ``tour_NN`` socket, and back out, and
    walk its tour in straight legs (``door``, ``tour_01``, ``tour_02``, ...). Without a ``door`` or
    ``floor_0`` it can't be checked, and lint warns (an older building no one enters)."""
    if not set(WALK_TAGS) & set(asset.tags) or not all(k in style.scale for k in WALKER_KEYS):
        return []
    width, height, climb = (int(style.scale[k]) for k in WALKER_KEYS)
    sockets = {n: Socket.from_value(v, n).position for n, v in asset.sockets.items()}
    ground = f"on the ground (y = ground_y, {asset.ground_y} here)"
    street = f"within {DOOR_REACH} voxels of the street (sockets stay inside the model's bounds)"
    if "building" in asset.tags:
        door_at = (
            f"at the main entrance, {ground}: on the front face, or under the overhang, {street}"
        )
        floor_at = "on the ground floor, where someone would stand"
    else:
        door_at = f"where the walk starts, {ground}: one end of a bridge's deck, {street}"
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
        sides = (
            f" on the asset's street_sides ({', '.join(asset.street_sides)}: put 'door' by one of "
            "them, or name its side)"
            if asset.street_sides
            else ""
        )
        out.append(
            Finding(
                "walk",
                "error",
                f"the walker has nowhere to start at 'door' {_at(door)}: it stands on the street "
                f"(outside the asset's x-z bounds{sides}) within {DOOR_REACH} voxels of 'door', on "
                f"something solid with its feet within {REACH_Y} voxels of y = {_num(door[1])}, "
                f"and finds no such place. Put 'door' {door_at}; nothing was walked",
                {"socket": "door", **details},
            )
        )
        return out
    failed = set()
    for name, pos in sorted(sockets.items()):
        if not walked(name):
            continue
        if not walker.reaches(pos):
            way = "in"
            point, distance = walker.nearest(pos, walker.reached)
            what = (
                f"the walker can't reach {name!r} {_at(pos)} from the street at 'door': the "
                f"nearest it gets is {_at(point)}, {distance:.0f} voxels away"
            )
        elif not walker.returns(pos):
            way = "out"
            point, distance = walker.nearest(pos, walker.back)
            what = (
                f"the walker can't get back from {name!r} {_at(pos)} to the street: it gets "
                "there only by dropping down what it can't climb back up, and the nearest place "
                f"it walks back out from is {_at(point)}, {distance:.0f} voxels away"
            )
        else:
            continue
        failed.add(name)
        out.append(
            Finding(
                "walk",
                "error",
                f"{what}. Check the stair or doorway between them: {walker.needs()}",
                {
                    "socket": name,
                    "way": way,
                    "nearest": point,
                    "distance": round(distance, 2),
                    **details,
                },
            )
        )
    stops = ["door"] + sorted(n for n in sockets if n.startswith("tour_"))
    for prev, name in zip(stops, stops[1:], strict=False):
        if prev in failed or name in failed:
            continue  # already an error: it isn't reached, or there's no way back from it
        a, b = sockets[prev], sockets[name]
        fails = walker.follows(a, b)
        if fails is None:
            continue
        if fails["way"] == "arrival":
            failed.add(prev)
            move = fails["move"]
            fix = (
                f"move it at least {move} voxel{'s' if move != 1 else ''} toward {fails['toward']}"
                if move
                else "clear room round it"
            )
            out.append(
                Finding(
                    "walk",
                    "error",
                    f"{prev!r} {_at(a)} is too close to an obstacle for a {ARRIVE:g}-voxel "
                    f"arrival: Kiln's walk is to arrive anywhere within {ARRIVE:g} voxels of a "
                    f"stop and head on to the next ({name!r}), and the walker can't stand at "
                    f"{_at(fails['at'])}. The goblin is narrower and might, then walk a line the "
                    f"lint never tried, so {fix}: {walker.needs()}",
                    {"socket": prev, "to": name, **fails, **details},
                )
            )
            continue
        total = math.hypot(b[0] - a[0], b[2] - a[2])
        out.append(
            Finding(
                "walk",
                "error",
                f"the tour can't walk a straight line from {prev!r} {_at(a)} to {name!r} "
                f"{_at(b)}: arriving within {ARRIVE:g} voxels of {prev!r} and heading straight "
                f"for {name!r}, the walker gets as far as {_at(fails['nearest'])}, "
                f"{fails['distance']:.0f} voxels short of its {total:.0f}. A tour's walk (Kiln's) "
                f"is to arrive within {ARRIVE:g} of each stop and steer straight for the next "
                "('door', then tour_01, tour_02, ...), so put a stop where the way turns, give the "
                f"line room either side, or clear it: {walker.needs()}",
                {"socket": name, "from": prev, **fails, **details},
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
