"""The manifest: everything an engine needs to place, move and play an asset, beside its
``.glb`` and ``.vox``. ``docs/manifest.md`` documents it; the JSON Schema is checked in at
``cubist/schema/manifest-1.schema.json``. New optional fields keep version 1; a field whose
meaning changes bumps the version."""

from __future__ import annotations

from pathlib import Path

from cubist.core.model import BuiltAsset
from cubist.core.socket import Socket
from cubist.io.gltf import glb_origin, nearest_part, used_colors
from cubist.motion.pose import used_cycles

MANIFEST_VERSION = 1
SCHEMA_PATH = Path(__file__).parent / "schema" / "manifest-1.schema.json"
UNITS = {"length": "voxel", "angle": "degree", "time": "second", "spin": "turn"}


def parents_first(asset: BuiltAsset) -> list:
    """Parts depth-first from the roots, each part's children in model order."""
    children: dict[str | None, list] = {}
    names = {p.name for p in asset.parts}
    for p in asset.parts:
        children.setdefault(p.parent if p.parent in names else None, []).append(p)
    out = []

    def visit(parent: str | None) -> None:
        for p in children.get(parent, []):
            out.append(p)
            visit(p.name)

    visit(None)
    return out


def _color_name(asset: BuiltAsset, index: int) -> str:
    try:
        return asset.palette.name_of(index) if asset.palette else f"#{index}"
    except IndexError:
        return f"#{index}"


def build_manifest(
    asset: BuiltAsset,
    voxel_size_m: float,
    files: dict[str, str] | None = None,
    center: bool = True,
) -> dict:
    sockets = {}
    for name, value in asset.sockets.items():
        sock = Socket.from_value(value, name)
        sockets[name] = {
            "position": list(sock.position),
            "part": sock.part or nearest_part(asset, sock.position),
            "forward": list(sock.forward) if sock.forward else None,
            "up": list(sock.up) if sock.up else None,
        }
    return {
        "cubist_manifest": MANIFEST_VERSION,
        "name": asset.name,
        "size": list(asset.size) if asset.size else None,
        "voxel_size_m": voxel_size_m,
        "glb_origin": glb_origin(asset, center),
        "units": dict(UNITS),
        "parts": [
            {
                "name": p.name,
                "parent": p.parent,
                "pivot": list(p.pivot),
                "joint": p.joint.to_dict() if p.joint else None,
                "hidden": p.hidden,
                "frames": {
                    "count": len(p.frames),
                    "fps": p.fps,
                    "vox_models": [p.name] + [f"{p.name}@{k}" for k in range(1, len(p.frames))],
                }
                if p.frames
                else None,
                "voxels": p.grid.count(),
                "vox_model": p.name,
            }
            for p in parents_first(asset)
        ],
        "sockets": sockets,
        "lights": {
            name: {
                "position": list(lt.position),
                "part": lt.part or nearest_part(asset, lt.position),
                "color": _color_name(asset, lt.color),
                "rgb": "#" + "".join(f"{v:02x}" for v in lt.rgb),
                "range": lt.range,
                "intensity": lt.intensity,
                "flicker": lt.flicker,
            }
            for name, lt in asset.lights.items()
        },
        "clips": [c.to_dict() for c in asset.clips],
        "cycles": [
            {
                "name": c.name,
                "ramp": c.ramp,
                "fps": c.fps,
                "colors": [_color_name(asset, i) for i in c.indices],
            }
            for c in used_cycles(asset)
        ],
        "glow": {
            _color_name(asset, i): s
            for i, s in sorted(asset.glow.items())
            if i in used_colors(asset)
        },
        "files": dict(files or {}),
    }
