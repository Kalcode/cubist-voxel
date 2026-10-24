"""Builds asset scripts in a subprocess with a timeout.

The parent process never imports the script. A worker (``python -m cubist.build.runner``)
runs it, voxelizes each ``@asset``, and writes into ``out/<name>/``:

* ``<name>.vox`` — the model (the previous build moves to ``<name>.prev.vox``),
* ``contact.png`` — the contact sheet,
* ``lint.json`` — the lint report,
* ``build.json`` — a summary including the lint result,
* ``history.jsonl`` — one line appended per build (for counting iterations),
* ``manifest.json`` — parts, joints and clips as an engine reads them (see docs/manifest.md),
* for assets with clips: ``motion.png`` (the filmstrip; the previous one moves to
  ``motion.prev.png``) and ``<clip>.gif`` per clip.

It reports back through a result file so script output cannot corrupt it.

Not yet enforced: the design's "no file or network access" rule for scripts.
"""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cubist.core.model import AssetDef, BuiltAsset
from cubist.core.palette import Palette
from cubist.core.style import StyleBible, StyleError, load_style
from cubist.io.vox import encode_vox
from cubist.lint import Finding, lint
from cubist.manifest import build_manifest
from cubist.render.contact_sheet import contact_sheet
from cubist.render.filmstrip import clip_frames, filmstrip, save_gif

DEFAULT_TIMEOUT_S = 30.0


class BuildError(RuntimeError):
    """A script failed to build; the message is written for the agent to act on."""


@dataclass
class BuildResult:
    name: str
    vox_path: Path
    contact_path: Path
    lint_path: Path
    summary_path: Path
    summary: dict


def find_style(start: Path) -> Path:
    for directory in Path(start).resolve().parents:
        candidate = directory / "style.toml"
        if candidate.is_file():
            return candidate
    raise BuildError(f"no style.toml found in any directory above {start}; pass --style")


def load_project_style(style: Path) -> StyleBible:
    try:
        return load_style(style)
    except StyleError as e:
        raise BuildError(str(e)) from e


def load_palette(style: Path) -> Palette:
    return load_project_style(style).palette


def _run_worker(script: Path, style: Path, out: Path, seed: int, timeout: float) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        result_file = Path(tmp) / "result.json"
        cmd = [
            sys.executable,
            "-m",
            "cubist.build.runner",
            str(script),
            str(style),
            str(out),
            str(result_file),
            str(seed),
        ]
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired as e:
            raise BuildError(f"{script.name}: build timed out after {timeout:g} s") from e
        if not result_file.exists():
            raise BuildError(f"{script.name}: build worker crashed\n{proc.stderr.strip()}")
        result = json.loads(result_file.read_text())
    if "error" in result:
        raise BuildError(result["error"])
    return result


def _add_finding(result: dict, finding: Finding) -> None:
    """Append a finding found outside the worker to lint.json and build.json."""
    lint_path, summary_path = Path(result["lint"]), Path(result["summary"])
    report = json.loads(lint_path.read_text())
    report["findings"].insert(0, finding.to_dict())
    report["errors"] += finding.severity == "error"
    report["ok"] = report["errors"] == 0
    lint_path.write_text(json.dumps(report, indent=2) + "\n")
    summary = result["summary_data"]
    summary["lint"]["ok"] = report["ok"]
    summary["lint"]["errors"] = report["errors"]
    lines = summary["lint"]["summary"].splitlines()
    lines.insert(1, f"{finding.severity.upper()} {finding.check}: {finding.message}")
    summary["lint"]["summary"] = "\n".join(lines)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


def build_script(
    script: str | os.PathLike,
    style: str | os.PathLike | None = None,
    out: str | os.PathLike | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    seed: int = 0,
    check_determinism: bool = False,
) -> list[BuildResult]:
    script = Path(script).resolve()
    style = Path(style).resolve() if style else find_style(script)
    out = Path(out).resolve() if out else style.parent / "out"
    result = _run_worker(script, style, out, seed, timeout)
    if check_determinism:
        with tempfile.TemporaryDirectory() as tmp:
            again = _run_worker(script, style, Path(tmp), seed, timeout)
        hashes = {r["name"]: r["vox_sha256"] for r in again["assets"]}
        for r in result["assets"]:
            if hashes.get(r["name"]) != r["vox_sha256"]:
                _add_finding(
                    r,
                    Finding(
                        "determinism",
                        "error",
                        f"two builds with seed {seed} produced different voxels; randomness must "
                        "come only from seeded ops",
                    ),
                )
    return [
        BuildResult(
            r["name"],
            Path(r["vox"]),
            Path(r["contact"]),
            Path(r["lint"]),
            Path(r["summary"]),
            r["summary_data"],
        )
        for r in result["assets"]
    ]


# --- worker -----------------------------------------------------------------


def _script_traceback(exc: BaseException, script: Path) -> str:
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if Path(f.filename) == script]
    lines = traceback.format_list(frames)
    return "".join(lines) + "".join(traceback.format_exception_only(exc)).strip()


def write_outputs(built: BuiltAsset, style: StyleBible, asset_dir: Path, script: Path) -> dict:
    asset_dir.mkdir(parents=True, exist_ok=True)
    vox_bytes = encode_vox(built.to_vox_scene())
    vox_path = asset_dir / f"{built.name}.vox"
    if vox_path.exists():  # keep the last build for `cubist diff`
        vox_path.replace(asset_dir / f"{built.name}.prev.vox")
    vox_path.write_bytes(vox_bytes)
    contact_path = asset_dir / "contact.png"
    contact_sheet(built.merged_grid(), built.palette, name=built.name, size=built.size).save(
        contact_path
    )
    report = lint(built, style)
    lint_path = asset_dir / "lint.json"
    lint_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    outputs = {"vox": str(vox_path), "contact_sheet": str(contact_path), "lint": str(lint_path)}
    outputs.update(_write_motion(built, asset_dir))
    manifest_path = asset_dir / "manifest.json"
    manifest = build_manifest(built, style.voxel_size_m)
    del manifest["files"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    outputs["manifest"] = str(manifest_path)
    summary = {
        **built.summary(),
        "script": str(script),
        "outputs": outputs,
        "lint": {
            "ok": report.ok,
            "errors": report.count("error"),
            "warnings": report.count("warning"),
            "infos": report.count("info"),
            "summary": report.summary(),
        },
    }
    summary_path = asset_dir / "build.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    _log_history(asset_dir, script, built, report)
    return {
        "name": built.name,
        "vox": str(vox_path),
        "vox_sha256": hashlib.sha256(vox_bytes).hexdigest(),
        "contact": str(contact_path),
        "lint": str(lint_path),
        "summary": str(summary_path),
        "summary_data": summary,
    }


def _write_motion(built: BuiltAsset, asset_dir: Path) -> dict:
    """Filmstrip and GIFs for an asset with clips; clears stale ones either way."""
    motion_path = asset_dir / "motion.png"
    keep = {f"{c.name}.gif" for c in built.clips}
    for old in asset_dir.glob("*.gif"):
        if old.name not in keep:
            old.unlink()
    if not built.clips:
        for old in (motion_path, asset_dir / "motion.prev.png"):
            old.unlink(missing_ok=True)
        return {}
    table = built.palette.rgba_table()
    if motion_path.exists():
        motion_path.replace(asset_dir / "motion.prev.png")
    filmstrip(built, table, built.clips).save(motion_path)
    gifs = {}
    for c in built.clips:
        frames, durations = clip_frames(built, table, c)
        save_gif(frames, durations, asset_dir / f"{c.name}.gif")
        gifs[c.name] = str(asset_dir / f"{c.name}.gif")
    return {"motion": str(motion_path), "gifs": gifs}


def _append_history(asset_dir: Path, script: Path, **fields) -> None:
    """One line per build attempt in history.jsonl; distinct script hashes count as
    iterations, failed attempts included."""
    asset_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        **fields,
    }
    with open(asset_dir / "history.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def _log_history(asset_dir: Path, script: Path, built: BuiltAsset, report) -> None:
    _append_history(
        asset_dir,
        script,
        ok=True,
        voxels=built.merged_grid().count(),
        errors=report.count("error"),
        warnings=report.count("warning"),
        seed=built.seed,
    )


def _build_in_process(script: Path, style_path: Path, out: Path, seed: int) -> list[dict]:
    style = load_project_style(style_path)
    try:
        namespace = runpy.run_path(str(script), run_name="__cubist_asset__")
    except Exception as e:
        message = f"{script.name} failed to load:\n{_script_traceback(e, script)}"
        _append_history(out / script.stem, script, ok=False, error=message, seed=seed)
        raise BuildError(message) from e
    assets = list({id(v): v for v in namespace.values() if isinstance(v, AssetDef)}.values())
    if not assets:
        raise BuildError(f"{script.name} defines no @asset functions")
    results = []
    for a in assets:
        try:
            built = a.build(style.palette, seed=seed)
        except Exception as e:
            message = f"{script.name}: asset {a.name!r} failed:\n{_script_traceback(e, script)}"
            _append_history(out / a.name, script, ok=False, error=message, seed=seed)
            raise BuildError(message) from e
        results.append(write_outputs(built, style, out / a.name, script))
    return results


def _worker(argv: list[str]) -> None:
    script, style, out, result_file = (Path(a) for a in argv[:4])
    seed = int(argv[4]) if len(argv) > 4 else 0
    try:
        payload = {"assets": _build_in_process(script, style, out, seed)}
    except BuildError as e:
        payload = {"error": str(e)}
    except Exception as e:  # anything else is a bug in cubist, not the script
        payload = {"error": "internal error:\n" + "".join(traceback.format_exception(e))}
    result_file.write_text(json.dumps(payload))


if __name__ == "__main__":
    _worker(sys.argv[1:])
