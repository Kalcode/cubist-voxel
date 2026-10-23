"""Builds asset scripts in a subprocess with a timeout.

The parent process never imports the script. A worker (``python -m cubist.build.runner``)
runs it, voxelizes each ``@asset``, writes ``out/<name>/<name>.vox`` and ``build.json``,
and reports back through a result file so script output cannot corrupt it.

Not yet enforced: the design's "no file or network access" rule for scripts.
"""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
import tempfile
import tomllib
import traceback
from dataclasses import dataclass
from pathlib import Path

from cubist.core.model import AssetDef
from cubist.core.palette import Palette, PaletteError
from cubist.io.vox import write_vox

DEFAULT_TIMEOUT_S = 30.0


class BuildError(RuntimeError):
    """A script failed to build; the message is written for the agent to act on."""


@dataclass
class BuildResult:
    name: str
    vox_path: Path
    summary_path: Path
    summary: dict


def find_style(script: Path) -> Path:
    for directory in Path(script).resolve().parents:
        candidate = directory / "style.toml"
        if candidate.is_file():
            return candidate
    raise BuildError(f"no style.toml found in any directory above {script}; pass --style")


def load_palette(style: Path) -> Palette:
    try:
        with open(style, "rb") as f:
            entries = tomllib.load(f).get("palette", {})
    except tomllib.TOMLDecodeError as e:
        raise BuildError(f"{style}: {e}") from e
    if "source" in entries:
        raise BuildError(f"{style}: palette 'source' files are not supported yet; define ramps")
    try:
        return Palette(entries)
    except PaletteError as e:
        raise BuildError(f"{style}: {e}") from e


def build_script(
    script: str | os.PathLike,
    style: str | os.PathLike | None = None,
    out: str | os.PathLike | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> list[BuildResult]:
    script = Path(script).resolve()
    style = Path(style).resolve() if style else find_style(script)
    out = Path(out).resolve() if out else style.parent / "out"
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
    return [
        BuildResult(r["name"], Path(r["vox"]), Path(r["summary"]), r["summary_data"])
        for r in result["assets"]
    ]


# --- worker -----------------------------------------------------------------


def _script_traceback(exc: BaseException, script: Path) -> str:
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if Path(f.filename) == script]
    lines = traceback.format_list(frames)
    return "".join(lines) + "".join(traceback.format_exception_only(exc)).strip()


def _build_in_process(script: Path, style: Path, out: Path) -> list[dict]:
    palette = load_palette(style)
    try:
        namespace = runpy.run_path(str(script), run_name="__cubist_asset__")
    except Exception as e:
        raise BuildError(f"{script.name} failed to load:\n{_script_traceback(e, script)}") from e
    assets = list({id(v): v for v in namespace.values() if isinstance(v, AssetDef)}.values())
    if not assets:
        raise BuildError(f"{script.name} defines no @asset functions")
    results = []
    for a in assets:
        try:
            built = a.build(palette)
        except Exception as e:
            raise BuildError(
                f"{script.name}: asset {a.name!r} failed:\n{_script_traceback(e, script)}"
            ) from e
        asset_dir = out / a.name
        asset_dir.mkdir(parents=True, exist_ok=True)
        vox_path = asset_dir / f"{a.name}.vox"
        write_vox(vox_path, built.to_vox_scene())
        summary = {**built.summary(), "script": str(script), "outputs": {"vox": str(vox_path)}}
        summary_path = asset_dir / "build.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        results.append(
            {
                "name": a.name,
                "vox": str(vox_path),
                "summary": str(summary_path),
                "summary_data": summary,
            }
        )
    return results


def _worker(argv: list[str]) -> None:
    script, style, out, result_file = (Path(a) for a in argv)
    try:
        payload = {"assets": _build_in_process(script, style, out)}
    except BuildError as e:
        payload = {"error": str(e)}
    except Exception as e:  # anything else is a bug in cubist, not the script
        payload = {"error": "internal error:\n" + "".join(traceback.format_exception(e))}
    result_file.write_text(json.dumps(payload))


if __name__ == "__main__":
    _worker(sys.argv[1:])
