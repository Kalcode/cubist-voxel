"""Lint: palette, bounds, pivots, part overlap, motion, connectivity, symmetry, grounding, scale,
a building's walk and storey bands, budgets, thin features, and hidden interior. Errors block
export; warnings and infos are advice."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

import numpy as np

from cubist.core.grid import VoxelGrid
from cubist.core.model import BuiltAsset
from cubist.core.style import StyleBible
from cubist.lint import checks, motion, walk
from cubist.lint.checks import Finding

SEVERITIES = ("error", "warning", "info")


@dataclass
class LintReport:
    findings: list[Finding]

    def count(self, severity: str) -> int:
        return sum(1 for f in self.findings if f.severity == severity)

    @property
    def ok(self) -> bool:
        return self.count("error") == 0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "errors": self.count("error"),
            "warnings": self.count("warning"),
            "infos": self.count("info"),
            "findings": [f.to_dict() for f in self.findings],
        }

    def summary(self) -> str:
        def plural(n, word):
            return f"{n} {word}{'' if n == 1 else 's'}"

        head = (
            f"lint: {plural(self.count('error'), 'error')}, "
            f"{plural(self.count('warning'), 'warning')}, {plural(self.count('info'), 'info')}"
        )
        lines = [head] + [f"{f.severity.upper()} {f.check}: {f.message}" for f in self.findings]
        return "\n".join(lines)


def with_frames(asset: BuiltAsset) -> BuiltAsset:
    """The asset with each flipbook's later frames as extra parts (``flame@1``, ...), for checks
    that must hold on every frame (palette, bounds)."""
    if not any(p.frames for p in asset.parts):
        return asset
    extra = [
        replace(p, name=f"{p.name}@{k}", grid=g, frames=None)
        for p in asset.parts
        for k, g in enumerate((p.frames or [])[1:], start=1)
    ]
    return replace(asset, parts=[*asset.parts, *extra])


def _floating(asset: BuiltAsset, others: list, grid) -> tuple[list[Finding], list]:
    found = checks.check_floating(asset, VoxelGrid.merge([*others, grid]))
    return found, [f.details.get("list") for f in found]


def frame_checks(asset: BuiltAsset) -> list[Finding]:
    """Floating pieces on each later flipbook frame, and frames that pop instead of flowing."""
    out = []
    for p in asset.parts:
        if not p.frames:
            continue
        others = [q.grid for q in asset.parts if q.name != p.name]
        _, rest = _floating(asset, others, p.frames[0])
        for k, grid in enumerate(p.frames[1:], start=1):
            found, listed = _floating(asset, others, grid)
            if listed == rest:
                continue  # the same pieces as the rest pose, already reported
            for f in found:
                f.message = f"frame {k} of {p.name!r}: {f.message}"
                out.append(f)
        pops = []
        n = len(p.frames)
        for k in range(n):
            a, b = p.frames[k], p.frames[(k + 1) % n]
            ka = {tuple(c) for c in a.voxels()[0].tolist()}
            kb = {tuple(c) for c in b.voxels()[0].tolist()}
            if max(len(ka), len(kb)) and len(ka ^ kb) / max(len(ka), len(kb)) > 0.6:
                pops.append(f"{k}->{(k + 1) % n}")
        if pops:
            out.append(
                Finding(
                    "frame_pop",
                    "info",
                    f"flipbook {p.name!r} changes most of its voxels between frames "
                    f"{', '.join(pops)}; small changes per frame read as motion, big ones as "
                    "flashing",
                    {"part": p.name, "frames": pops},
                )
            )
    return out


def lint(
    asset: BuiltAsset,
    style: StyleBible,
    rgba: np.ndarray | None = None,
    extra: Iterable[Finding] = (),
) -> LintReport:
    """Run every check. ``rgba`` is the asset's color table when it came from a file whose
    indices may not match the style palette; by default the style palette's own table."""
    table = style.palette.rgba_table() if rgba is None else np.asarray(rgba)
    merged = VoxelGrid.merge([p.grid for p in asset.parts])
    every_frame = with_frames(asset)
    findings = [
        *checks.check_palette(every_frame, style, table),
        *checks.check_bounds(every_frame),
        *checks.check_pivots(asset),
        *extra,
        *checks.check_overlap(asset),
        *motion.check_motion(asset),
        *checks.check_floating(asset, merged),
        *frame_checks(asset),
        *checks.check_symmetry(asset, merged, style),
        *checks.check_grounding(asset, merged),
        *checks.check_light_reach(asset, style, merged),
        *checks.check_scale(asset, style),
        *checks.check_occlusion(asset),
        *walk.check_walk(asset, style),
        *walk.check_storeys(asset),
        *checks.check_budgets(asset, merged, style),
        *checks.check_thin(merged),
        *checks.check_hidden(merged),
    ]
    findings.sort(key=lambda f: SEVERITIES.index(f.severity))
    return LintReport(findings)


__all__ = ["Finding", "LintReport", "lint"]
