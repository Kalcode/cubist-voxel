"""World lint: what the map hides (design.md, "Worlds").

Errors for things outside the world; warnings for landmarks near its edge (the engine shows the
world's end from about 400 m), steep ground around landmarks, props on uneven ground or growing into
each other, caves with no way in, and tunnel floors with a step no one climbs; information on how
much ground is too steep to walk.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cubist.lint import LintReport
from cubist.lint.checks import Finding
from cubist.world.caves import Cavern, Tunnel, Worms
from cubist.world.model import LoadedWorld
from cubist.world.props import Place, Scatter, footprint, scatter_points
from cubist.world.sample import heights, slopes

EDGE_M = 400.0  # how far the engine's levels of detail reach: closer than this, the edge shows
WALK_SLOPE_DEG = 45.0
UNEVEN_M = 1.0  # a placed prop's footprint may fall this much before it visibly floats
CLIMB_M = 1.0  # a tunnel's floor may rise or fall this much per metre along it and still be walked
FLOOR_SEARCH_M = 12.0  # how far under a tunnel's bottom its floor is looked for


def _f(check: str, severity: str, message: str, **details) -> Finding:
    return Finding(check, severity, message, details)


def lint_world(loaded: LoadedWorld, slope_map: np.ndarray, export: Path | None) -> LintReport:
    """Lints a loaded world; ``slope_map`` is the map's slopes, ``export`` the project's exports."""
    out: list[Finding] = []
    sx, sz = loaded.size_m

    def inside(x: float, z: float) -> bool:
        return 0 <= x < sx and 0 <= z < sz

    # Bounds and edges.
    big = min(sx, sz) >= 2 * EDGE_M
    for name, (x, z, _) in loaded.landmark_points().items():
        if not inside(x, z):
            out.append(
                _f("bounds", "error", f"landmark {name} at ({x:g}, {z:g}) is outside the world")
            )
        elif big and min(x, z, sx - x, sz - z) < EDGE_M:
            d = min(x, z, sx - x, sz - z)
            out.append(
                _f(
                    "edge",
                    "warning",
                    f"landmark {name} is {d:.0f} m from the world's edge; the engine "
                    f"shows the edge from {EDGE_M:.0f} m (keep players further in)",
                )
            )
    for prop in loaded.world.props:
        if isinstance(prop, Place) and not inside(prop.at[0], prop.at[-1]):
            out.append(
                _f(
                    "bounds",
                    "error",
                    f"{prop.asset} at ({prop.at[0]:g}, {prop.at[-1]:g}) is outside the world",
                )
            )
    for kind, items in (("river", loaded.world.rivers), ("road", loaded.world.roads)):
        for edit in items:
            pts = edit.path.points
            outside = (pts[:, 0] < 0) | (pts[:, 0] >= sx) | (pts[:, 1] < 0) | (pts[:, 1] >= sz)
            if outside.any():
                out.append(
                    _f(
                        "bounds",
                        "info",
                        f"a {kind} leaves the world near ({pts[outside][0][0]:.0f}, "
                        f"{pts[outside][0][1]:.0f})",
                    )
                )

    # Slopes.
    steep = float(np.mean(slope_map > WALK_SLOPE_DEG)) * 100
    out.append(
        _f(
            "slope",
            "info",
            f"{steep:.1f}% of the ground is steeper than {WALK_SLOPE_DEG:g}° (unwalkable)",
            percent=round(steep, 2),
        )
    )
    ring = np.linspace(-20, 20, 9)
    gx, gz = np.meshgrid(ring, ring)
    for name, (x, z, _) in loaded.landmark_points().items():
        if not inside(x, z):
            continue
        worst = float(slopes(loaded, x + gx, z + gz).max())
        if worst > WALK_SLOPE_DEG:
            out.append(
                _f(
                    "slope",
                    "warning",
                    f"the ground within 20 m of {name} reaches {worst:.0f}°; "
                    f"flatten() around it if people walk there",
                )
            )

    out += _props(loaded, export)
    out += _caves(loaded)
    out += _tunnel_floors(loaded)
    return LintReport(out)


def _footprint_box(export: Path, prop: Place) -> tuple[float, float, float, float]:
    w, d = footprint(export, prop.asset, prop.rotate_y)
    x, z = prop.at[0], prop.at[-1]
    return x - w / 2, z - d / 2, x + w / 2, z + d / 2


def _props(loaded: LoadedWorld, export: Path | None) -> list[Finding]:
    out: list[Finding] = []
    placed = [p for p in loaded.world.props if isinstance(p, Place)]
    scattered = [p for p in loaded.world.props if isinstance(p, Scatter)]
    if not placed and not scattered:
        return out
    if export is None or not all(
        (export / f"{p.asset}.cubist.json").is_file() for p in [*placed, *scattered]
    ):
        out.append(
            _f("props", "info", "props not checked against the ground: export their assets first")
        )
        return out
    boxes = [_footprint_box(export, p) for p in placed]
    u = np.linspace(0, 1, 5)
    for p, (x0, z0, x1, z1) in zip(placed, boxes, strict=True):
        if len(p.at) == 3:
            continue
        gx, gz = np.meshgrid(x0 + (x1 - x0) * u, z0 + (z1 - z0) * u)
        h = heights(loaded, gx, gz)
        fall = float(h.max() - h.min())
        if fall > UNEVEN_M:
            out.append(
                _f(
                    "uneven",
                    "warning",
                    f"{p.asset} at ({p.at[0]:g}, {p.at[-1]:g}) stands on ground that "
                    f"falls {fall:.1f} m across it, so it floats on the low side; "
                    "flatten() under it",
                )
            )
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            a, b = boxes[i], boxes[j]
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                out.append(
                    _f(
                        "overlap",
                        "warning",
                        f"{placed[i].asset} at ({placed[i].at[0]:g}, {placed[i].at[-1]:g}) "
                        f"overlaps {placed[j].asset} at "
                        f"({placed[j].at[0]:g}, {placed[j].at[-1]:g})",
                    )
                )
    for s in scattered:
        x, z, _ = scatter_points(loaded, s)
        w, d = footprint(export, s.asset, 0)
        reach = max(w, d) / 2
        for p, (x0, z0, x1, z1) in zip(placed, boxes, strict=True):
            n = int(
                np.sum((x > x0 - reach) & (x < x1 + reach) & (z > z0 - reach) & (z < z1 + reach))
            )
            if n:
                out.append(
                    _f(
                        "overlap",
                        "warning",
                        f"{n} scattered {s.asset} grow into {p.asset} at "
                        f"({p.at[0]:g}, {p.at[-1]:g}); give the scatter avoid=rect(...) around it",
                    )
                )
    return out


def _caves(loaded: LoadedWorld) -> list[Finding]:
    """Caves connect when a tunnel runs into a cavern or another tunnel.

    A group needs one tunnel that opens at the surface (worms count as reachable: they cross the
    ground's tunnels at random).
    """
    caves = list(loaded.world.caves)
    tunnels = [c for c in caves if isinstance(c, Tunnel)]
    caverns = [c for c in caves if isinstance(c, Cavern)]
    if not tunnels and not caverns:
        return []
    nodes = [*tunnels, *caverns]
    parent = list(range(len(nodes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def touches(a, b) -> bool:
        if isinstance(a, Cavern) and isinstance(b, Cavern):
            return False
        if isinstance(a, Cavern):
            a, b = b, a
        if isinstance(b, Cavern):
            q = (((a.points - b.c) / (b.r + a.radius)) ** 2).sum(axis=1)
            return bool((q <= 1).any())
        d = np.linalg.norm(a.points[:, None, :] - b.points[None, :, :], axis=-1)
        return bool((d <= a.radius + b.radius).any())

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if touches(nodes[i], nodes[j]):
                parent[find(i)] = find(j)
    open_roots = set()
    for i, t in enumerate(tunnels):
        ground = heights(loaded, t.points[:, 0], t.points[:, 2])
        if (t.points[:, 1] + t.radius >= ground).any():
            open_roots.add(find(i))
    worms = [c for c in caves if isinstance(c, Worms)]
    out = []
    for i, node in enumerate(nodes):
        if find(i) in open_roots or (
            worms
            and isinstance(node, Cavern)
            and any(w.x0 <= node.c[0] <= w.x1 and w.z0 <= node.c[2] <= w.z1 for w in worms)
        ):
            continue
        if isinstance(node, Cavern):
            x, y, z = node.c
            out.append(
                _f(
                    "cave",
                    "warning",
                    f"the cavern at ({x:g}, {y:g}, {z:g}) has no way in from the surface; "
                    f"run a tunnel() from the ground into it",
                )
            )
        else:
            x, y, z = node.points[0]
            out.append(
                _f(
                    "cave",
                    "warning",
                    f"the tunnel from ({x:.0f}, {y:.0f}, {z:.0f}) never reaches the "
                    f"surface or a cave that does",
                )
            )
    return out


def _tunnel_floors(loaded: LoadedWorld) -> list[Finding]:
    """Walks each tunnel's line a metre at a time and finds the floor under it (the ground, less
    every cave); a floor that rises or drops more than ``CLIMB_M`` in a metre is a step a character
    doesn't climb (a worm under the floor once left such a pit)."""
    caves = list(loaded.world.caves)
    out = []
    for t in (c for c in caves if isinstance(c, Tunnel)):
        pts = t.points
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        along = np.r_[0.0, np.cumsum(seg)]
        s = np.arange(0.0, along[-1], 1.0)
        line = np.stack([np.interp(s, along, pts[:, k]) for k in range(3)], axis=-1)
        # Down from the line in 1/16 m steps, to FLOOR_SEARCH_M under the tunnel's bottom.
        dy = np.arange(0.0, t.radius + FLOOR_SEARCH_M, 1 / 16)
        x = line[:, 0:1] + 0 * dy
        z = line[:, 2:3] + 0 * dy
        y = line[:, 1:2] - dy
        ground = heights(loaded, line[:, 0], line[:, 2])[:, None] + 0 * dy
        air = y >= ground
        for c in caves:
            air |= c.carve(x, y, z, ground, loaded.seed)
        solid = ~air
        found = solid.any(axis=1)
        floor = np.where(found, y[np.arange(len(y)), solid.argmax(axis=1)], np.nan)
        steps = np.abs(np.diff(floor))
        bad = np.flatnonzero(~(steps <= CLIMB_M))  # a missing floor (nan) counts too
        if len(bad) == 0:
            continue
        i = bad[np.argmax(np.nan_to_num(steps[bad], nan=np.inf))]
        x0, y0, z0 = t.points[0]
        px, pz = line[i + 1, 0], line[i + 1, 2]
        if np.isnan(steps[i]):
            what = f"falls away (none within {t.radius + FLOOR_SEARCH_M:g} m of the line)"
        else:
            change = floor[i + 1] - floor[i]
            what = f"{'rises' if change > 0 else 'drops'} {abs(change):.1f} m"
        out.append(
            _f(
                "cave",
                "warning",
                f"the floor of the tunnel from ({x0:.0f}, {y0:.0f}, {z0:.0f}) {what} near "
                f"({px:.0f}, {pz:.0f}): a character doesn't climb that",
            )
        )
    return out


def write_lint(report: LintReport, path: Path) -> None:
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
