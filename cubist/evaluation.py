"""Evaluation (milestones 6 and 11): how far briefs got.

For every brief in a project: was it built, does it lint without errors, how many iterations
(distinct script versions built, from ``out/<name>/history.jsonl``), and the person's 1-5
score from ``eval/scores.toml`` (0 = not scored yet). An asset is accepted when it lints without
errors within 5 iterations and scores at least 3. Animated assets list their clips, and their
GIFs are copied to ``eval/gifs/`` so the report can show them.
"""

from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path

import numpy as np
from PIL import Image

from cubist.build.runner import iteration_count, load_project_style
from cubist.export import lint_loaded, load_asset
from cubist.motion.pose import preview_clips
from cubist.project import load_brief
from cubist.render.contact_sheet import INK, MUTED, SHEET_BG
from cubist.render.font import draw_text, text_height
from cubist.render.raster import iso

MAX_ITERATIONS = 5
ACCEPT_SCORE = 3


def iterations(project: str | Path, name: str) -> int:
    return iteration_count(Path(project) / "out" / name)


def _scores(project: Path) -> dict[str, int]:
    path = project / "eval" / "scores.toml"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return {k: int(v) for k, v in tomllib.load(f).items() if isinstance(v, int)}


def evaluate(project: str | Path) -> dict:
    project = Path(project)
    style = load_project_style(project / "style.toml")
    scores = _scores(project)
    rows = []
    for brief_file in sorted((project / "briefs").glob("*.toml")):
        brief = load_brief(brief_file)
        name = brief.get("name", brief_file.stem)
        row = {
            "name": name,
            "category": brief.get("category", "prop"),
            "built": False,
            "lint_ok": False,
            "errors": None,
            "warnings": None,
            "iterations": iterations(project, name),
            "score": scores.get(name, 0),
            "clips": [],
        }
        asset_dir = project / "out" / name
        if (asset_dir / f"{name}.vox").exists():
            loaded = load_asset(asset_dir)
            built, report = lint_loaded(loaded, style)
            row.update(
                built=True,
                lint_ok=report.ok,
                errors=report.count("error"),
                warnings=report.count("warning"),
                clips=[c.name for c in preview_clips(built, with_previews=True)],
            )
        passing = row["built"] and row["lint_ok"] and row["iterations"] <= MAX_ITERATIONS
        if not passing:
            row["accepted"] = False
        elif row["score"] == 0:
            row["accepted"] = None  # waiting for a person's score
        else:
            row["accepted"] = row["score"] >= ACCEPT_SCORE
        rows.append(row)
    return {"assets": rows, "summary": _summarize(rows)}


def _summarize(rows: list[dict]) -> dict:
    n = len(rows)
    built = [r for r in rows if r["built"]]
    scored = [r for r in rows if r["score"] > 0]
    by_category: dict[str, dict] = {}
    for r in rows:
        c = by_category.setdefault(
            r["category"], {"briefs": 0, "built": 0, "lint_ok": 0, "accepted": 0, "pending": 0}
        )
        c["briefs"] += 1
        c["built"] += r["built"]
        c["lint_ok"] += r["lint_ok"]
        c["accepted"] += r["accepted"] is True
        c["pending"] += r["accepted"] is None

    def rate(k):
        return k / n if n else 0.0

    return {
        "briefs": n,
        "built": len(built),
        "lint_pass_rate": rate(sum(r["lint_ok"] for r in rows)),
        "lint_clean_rate": rate(sum(r["lint_ok"] and r["warnings"] == 0 for r in rows)),
        "mean_iterations": float(np.mean([r["iterations"] for r in built])) if built else 0.0,
        "within_iterations": rate(sum(0 < r["iterations"] <= MAX_ITERATIONS for r in rows)),
        "scored": len(scored),
        "mean_score": float(np.mean([r["score"] for r in scored])) if scored else None,
        "accepted": sum(r["accepted"] is True for r in rows),
        "pending_score": sum(r["accepted"] is None for r in rows),
        "by_category": by_category,
    }


def write_report(project: str | Path, report: dict) -> dict[str, Path]:
    """eval/report.md, eval/report.json, eval/gallery.png, and a scores.toml template."""
    project = Path(project)
    out = project / "eval"
    out.mkdir(exist_ok=True)
    scores_path = out / "scores.toml"
    existing = _scores(project)
    missing = [r["name"] for r in report["assets"] if r["name"] not in existing]
    if missing or not scores_path.exists():
        header = (
            ""
            if scores_path.exists()
            else (
                "# Score each asset 1-5 after looking at out/<name>/contact.png and, for moving\n"
                "# assets, its GIFs in eval/gifs/ (0 = not scored).\n"
                "# 1 unusable, 2 needs major work, 3 usable, 4 good, 5 excellent.\n"
            )
        )
        with open(scores_path, "a") as f:
            f.write(header + "".join(f"{name} = 0\n" for name in missing))

    gifs_dir = out / "gifs"
    for r in report["assets"]:
        r["gifs"] = []
        for clip_name in r.get("clips", []):
            source = project / "out" / r["name"] / f"{clip_name}.gif"
            if source.exists():
                gifs_dir.mkdir(exist_ok=True)
                target = gifs_dir / f"{r['name']}-{clip_name}.gif"
                shutil.copyfile(source, target)
                r["gifs"].append(f"gifs/{target.name}")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    s = report["summary"]
    lines = [
        "# Evaluation report",
        "",
        f"- Briefs: {s['briefs']}; built: {s['built']}",
        f"- Lint pass rate (no errors): {s['lint_pass_rate']:.0%}; "
        f"lint clean (no warnings either): {s['lint_clean_rate']:.0%}",
        f"- Mean iterations: {s['mean_iterations']:.1f}; "
        f"within {MAX_ITERATIONS}: {s['within_iterations']:.0%}",
        f"- Scored: {s['scored']}; mean score: "
        + (f"{s['mean_score']:.1f}" if s["mean_score"] is not None else "n/a"),
        f"- Accepted: {s['accepted']}; waiting for a score: {s['pending_score']}",
        "",
        "| Category | Briefs | Built | Lint OK | Accepted | Pending |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for cat, c in sorted(s["by_category"].items()):
        lines.append(
            f"| {cat} | {c['briefs']} | {c['built']} | {c['lint_ok']} | "
            f"{c['accepted']} | {c['pending']} |"
        )
    lines += [
        "",
        "| Asset | Category | Built | Lint | Iterations | Score | Accepted |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in report["assets"]:
        lint = (
            "—"
            if not r["built"]
            else (f"ok ({r['warnings']} warnings)" if r["lint_ok"] else f"{r['errors']} errors")
        )
        accepted = {True: "yes", False: "no", None: "pending"}[r["accepted"]]
        lines.append(
            f"| {r['name']} | {r['category']} | {'yes' if r['built'] else 'no'} | "
            f"{lint} | {r['iterations']} | {r['score'] or '—'} | {accepted} |"
        )
    moving = [r for r in report["assets"] if r.get("gifs")]
    if moving:
        lines += ["", "## Motion", ""]
        for r in moving:
            lines += [f"### {r['name']}", ""]
            lines += [f"![{r['name']} {g}]({g})" for g in r["gifs"]]
            lines.append("")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    _gallery(project, report).save(out / "gallery.png")
    return {
        "report": out / "report.md",
        "json": out / "report.json",
        "gallery": out / "gallery.png",
        "scores": scores_path,
    }


def _thumbnail(asset_dir: Path, fit: int) -> np.ndarray:
    """The asset's iso view scaled to fit a ``fit``-pixel square, whole and centered."""
    asset = load_asset(asset_dir)
    grid = asset.visible_grid
    lo, hi = grid.bounds()
    nx, ny, nz = (b - a for a, b in zip(lo, hi, strict=True))
    scale = 2
    for s in range(12, 1, -2):  # largest even scale whose image fits in twice the cell
        if max((nx + nz) * s, (nx + nz) * s // 2 + ny * round(1.2247 * s)) <= 2 * fit:
            scale = s
            break
    view = iso(grid, asset.scene.palette, scale).rgb
    img = Image.fromarray(view)
    img.thumbnail((fit, fit), Image.LANCZOS)
    return np.asarray(img.convert("RGB"))


def _gallery(project: Path, report: dict, cell: int = 220, columns: int = 5) -> Image.Image:
    rows = report["assets"]
    label_h = 2 * text_height() + 10
    n_rows = max(1, -(-len(rows) // columns))
    sheet = np.full(
        (n_rows * (cell + label_h) + 16, columns * cell + 16, 3), SHEET_BG, dtype=np.uint8
    )
    for i, r in enumerate(rows):
        x0 = 8 + (i % columns) * cell
        y0 = 8 + (i // columns) * (cell + label_h)
        if r["built"]:
            thumb = _thumbnail(project / "out" / r["name"], cell - 8)
            ox, oy = x0 + (cell - thumb.shape[1]) // 2, y0 + (cell - thumb.shape[0]) // 2
            sheet[oy : oy + thumb.shape[0], ox : ox + thumb.shape[1]] = thumb
        status = "NOT BUILT" if not r["built"] else ("LINT OK" if r["lint_ok"] else "LINT ERR")
        draw_text(sheet, x0, y0 + cell + 2, r["name"], INK)
        draw_text(
            sheet,
            x0,
            y0 + cell + 4 + text_height(),
            f"{status}  IT {r['iterations']}  SCORE {r['score'] or '-'}",
            MUTED,
        )
    return Image.fromarray(sheet)
