"""Lint: palette, bounds, pivots, part overlap, connectivity, symmetry, grounding, budgets,
thin features, and hidden interior. Errors block export; warnings and infos are advice."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from cubist.core.grid import VoxelGrid
from cubist.core.model import BuiltAsset
from cubist.core.style import StyleBible
from cubist.lint import checks
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
    findings = [
        *checks.check_palette(asset, style, table),
        *checks.check_bounds(asset),
        *checks.check_pivots(asset),
        *extra,
        *checks.check_overlap(asset),
        *checks.check_floating(asset, merged),
        *checks.check_symmetry(asset, merged, style),
        *checks.check_grounding(asset, merged),
        *checks.check_budgets(asset, merged, style),
        *checks.check_thin(merged),
        *checks.check_hidden(merged),
    ]
    findings.sort(key=lambda f: SEVERITIES.index(f.severity))
    return LintReport(findings)


__all__ = ["Finding", "LintReport", "lint"]
