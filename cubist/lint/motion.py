"""Motion lint: joint limits, loops that don't close, parts moving through each other, and how
far each clip reaches. Poses come from the same evaluator previews and export use."""

from __future__ import annotations

import math

import numpy as np

from cubist.core.model import BuiltAsset, voxel_sha256
from cubist.lint.checks import Finding
from cubist.motion.clips import Clip
from cubist.motion.joints import Joint
from cubist.motion.pose import moving_parts, posed_bounds, sample_times, world_matrices

LIMIT_TOLERANCE = 1e-6
LOOP_TOLERANCE = 0.05  # voxels a part's corners may drift between a loop's ends
AXIS_FORGIVENESS = 1.5  # voxels near a turning joint's axis that may brush neighbors
COLLISION_SAMPLES_PER_S = 24


def check_motion(asset: BuiltAsset) -> list[Finding]:
    if not asset.clips:
        return []
    return [
        *check_limits(asset),
        *check_loops(asset),
        *check_collisions(asset),
        *motion_bounds(asset),
    ]


def axis_name(axis) -> str:
    """'x', '-y', ... for axis-aligned vectors, else the vector."""
    a = np.asarray(axis, dtype=np.float64)
    i = int(np.argmax(np.abs(a)))
    if np.isclose(abs(a[i]), 1.0):
        return ("-" if a[i] < 0 else "") + "xyz"[i]
    return "(" + ", ".join(f"{c:.2f}" for c in a) + ")"


def check_limits(asset: BuiltAsset) -> list[Finding]:
    out = []
    for c in asset.clips:
        for track in c.tracks:
            joint = asset.part(track.part).joint
            if track.channel != "joint" or joint is None or joint.limits is None:
                continue
            lo, hi = joint.limits
            bad = [v for v in track.values if v < lo - LIMIT_TOLERANCE or v > hi + LIMIT_TOLERANCE]
            if not bad:
                continue
            worst = max(bad, key=lambda v: max(lo - v, v - hi))
            if joint.kind == "slide":
                message = (
                    f"clip {c.name!r} slides {track.part!r} {worst:g} voxels, but its slide "
                    f"distance is {hi:g} (0 to {hi:g}); keep the keys in range or raise "
                    "slide(..., distance=...)"
                )
            else:
                message = (
                    f"clip {c.name!r} turns {track.part!r} to {worst:g} degrees, but its hinge "
                    f"limits are ({lo:g}, {hi:g}); keep the keys inside the limits or widen "
                    "hinge(..., limits=...)"
                )
            out.append(
                Finding(
                    "joint_limits",
                    "error",
                    message,
                    {"clip": c.name, "part": track.part, "value": worst, "limits": [lo, hi]},
                )
            )
    return out


def _corners(asset: BuiltAsset, matrices: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for p in asset.parts:
        b = p.grid.bounds()
        if b is None:
            continue
        lo, hi = np.array(b[0], float), np.array(b[1], float)
        pts = np.array([[(hi if i & k else lo)[j] for j, k in enumerate((1, 2, 4))]
                        for i in range(8)])  # fmt: skip
        m = matrices[p.name]
        out[p.name] = pts @ m[:3, :3].T + m[:3, 3]
    return out


def check_loops(asset: BuiltAsset) -> list[Finding]:
    out = []
    for c in asset.clips:
        if not c.loop:
            continue
        start = _corners(asset, world_matrices(asset, c, 0.0))
        end = _corners(asset, world_matrices(asset, c, c.length))
        drift = {n: float(np.max(np.linalg.norm(end[n] - start[n], axis=1))) for n in start}
        part, worst = max(drift.items(), key=lambda kv: kv[1], default=(None, 0.0))
        if worst > LOOP_TOLERANCE:
            out.append(
                Finding(
                    "loop_closure",
                    "warning",
                    f"loop clip {c.name!r} ends in a different pose than it starts ({part!r} "
                    f"is {worst:.1f} voxels off), so it hitches when it repeats; end each track "
                    "on the value it starts with (spins: whole turns)",
                    {"clip": c.name, "part": part, "drift": worst},
                )
            )
    return out


def _driver(asset: BuiltAsset, name: str, clip: Clip) -> tuple[str, Joint] | None:
    """The part (itself or an ancestor) whose joint moves ``name`` in this clip."""
    driven = {t.part for t in clip.tracks}
    while name is not None:
        p = asset.part(name)
        if name in driven and p.joint is not None:
            return name, p.joint
        name = p.parent
    return None


def _keys(cells: np.ndarray) -> np.ndarray:
    c = cells.astype(np.int64) + (1 << 20)
    return (c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2]


def check_collisions(asset: BuiltAsset) -> list[Finding]:
    out = []
    voxels = {p.name: p.grid.voxels()[0].astype(np.float64) + 0.5 for p in asset.parts}
    rest_keys = {n: _keys(np.floor(v)) for n, v in voxels.items()}
    for c in asset.clips:
        moving = moving_parts(asset, c)
        if not moving:
            continue
        forgiven: dict[str, np.ndarray] = {}
        hint: dict[str, str] = {}
        for name in moving:
            pts = voxels[name]
            near = np.zeros(len(pts), dtype=bool)
            drv = _driver(asset, name, c)
            if drv is not None and drv[1].kind in ("hinge", "spin"):
                pivot = np.asarray(asset.part(drv[0]).pivot, dtype=np.float64)
                axis = np.asarray(drv[1].axis)
                rel = pts - pivot
                dist = np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)
                near = dist < AXIS_FORGIVENESS
            forgiven[name] = near
            if drv is not None and drv[1].kind == "hinge":
                flip = axis_name(-np.asarray(drv[1].axis))
                hint[name] = (
                    f"if the hinge turns the wrong way, flip its axis "
                    f"({axis_name(drv[1].axis)!r} -> {flip!r}), or narrow the keys"
                )
            elif drv is not None and drv[1].kind == "slide":
                hint[name] = "check the slide direction and distance"
            else:
                hint[name] = "check the joint axis and the keys"

        worst: dict[tuple[str, str], tuple[int, float]] = {}
        names = [p.name for p in asset.parts if len(voxels[p.name])]
        for t in sample_times(c, rate=COLLISION_SAMPLES_PER_S):
            m = world_matrices(asset, c, float(t))
            posed = {}
            for n in names:
                pts = voxels[n]
                posed[n] = _keys(np.floor(np.round(pts @ m[n][:3, :3].T + m[n][:3, 3], 6)))
            for a in moving:
                if not len(voxels[a]):
                    continue
                for b in names:
                    if b == a or (b in moving and names.index(b) < names.index(a)):
                        continue
                    hits = np.isin(posed[a], posed[b])
                    hits &= ~np.isin(rest_keys[a], rest_keys[b])  # shared at rest: overlap lint
                    hits &= ~forgiven[a]
                    n_hits = int(hits.sum())
                    if n_hits > worst.get((a, b), (0, 0.0))[0]:
                        worst[(a, b)] = (n_hits, float(t))
        for (a, b), (n_hits, t) in worst.items():
            if n_hits >= max(3, math.ceil(0.01 * len(voxels[a]))):
                out.append(
                    Finding(
                        "motion_collision",
                        "warning",
                        f"clip {c.name!r} moves {a!r} through {b!r} at t={t:.2f} s "
                        f"({n_hits} voxels); {hint[a]}",
                        {"clip": c.name, "parts": [a, b], "time": t, "voxels": n_hits},
                    )
                )
    return out


def motion_bounds(asset: BuiltAsset) -> list[Finding]:
    rest = posed_bounds(asset, {})
    clips, lines = {}, []
    for c in asset.clips:
        boxes = [posed_bounds(asset, world_matrices(asset, c, float(t))) for t in
                 sample_times(c, rate=COLLISION_SAMPLES_PER_S)]  # fmt: skip
        lo = [min(b[0][i] for b in boxes) for i in range(3)]
        hi = [max(b[1][i] for b in boxes) for i in range(3)]
        beyond = {}
        for i, axis in enumerate("xyz"):
            if rest[0][i] - lo[i] > 0:
                beyond[f"-{axis}"] = rest[0][i] - lo[i]
            if hi[i] - rest[1][i] > 0:
                beyond[f"+{axis}"] = hi[i] - rest[1][i]
        clips[c.name] = {"bounds": [lo, hi], "beyond": beyond}
        reach = ", ".join(f"{n} at {side}" for side, n in beyond.items()) or "stays in the rest box"
        lines.append(f"clip {c.name!r} reaches {tuple(lo)}-{tuple(hi)} ({reach})")
    return [Finding("motion_bounds", "info", "; ".join(lines), {"rest": rest, "clips": clips})]


def check_hand_edits(asset: BuiltAsset, summary: dict) -> list[Finding]:
    """Animated assets come from their script: a .vox that no longer matches its build would
    export motion made for other voxels."""
    if not asset.is_animated() or not summary.get("parts"):
        return []
    recorded = {p["name"]: p.get("voxel_sha256") for p in summary["parts"]}
    names = [p.name for p in asset.parts]
    changed = [p.name for p in asset.parts
               if recorded.get(p.name) and recorded[p.name] != voxel_sha256(p.grid)]  # fmt: skip
    added = [n for n in names if n not in recorded]
    removed = [n for n in recorded if n not in names]
    if not (changed or added or removed):
        return []
    what = ", ".join(
        [f"part {n!r} changed" for n in changed]
        + [f"part {n!r} added" for n in added]
        + [f"part {n!r} removed" for n in removed]
    )
    name = summary.get("name", asset.name)
    return [
        Finding(
            "hand_edit",
            "error",
            f"out/{name}/{name}.vox no longer matches its build ({what}). Animated assets come "
            "from their script: rebuild, or export with --force to use the edited voxels with "
            "the build's motion",
            {"changed": changed, "added": added, "removed": removed},
        )
    ]
