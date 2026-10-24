"""Bridge to vengi's ``voxconvert`` CLI for formats Cubist doesn't write itself (OBJ,
Qubicle, Minecraft schematics, ...). Cubist does not reimplement them."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

EXECUTABLE = "vengi-voxconvert"


class VengiError(RuntimeError):
    """vengi-voxconvert is missing or failed."""


def convert(vox: Path, out: Path, timeout: float = 120.0) -> Path:
    exe = shutil.which(EXECUTABLE)
    if exe is None:
        raise VengiError(
            f"{EXECUTABLE} is not on PATH; install vengi (https://vengi-voxel.github.io/vengi/) "
            "to export formats other than .vox and .glb"
        )
    out = Path(out)
    out.unlink(missing_ok=True)
    proc = subprocess.run(
        [exe, "--input", str(vox), "--output", str(out)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0 or not out.exists():
        detail = (proc.stderr or proc.stdout).strip()
        raise VengiError(f"{EXECUTABLE} failed converting {vox.name} to {out.suffix}: {detail}")
    return out
