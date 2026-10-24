"""The manifest: everything an engine needs to place, move and play an asset, beside its
``.glb`` and ``.vox``. ``docs/manifest.md`` documents it; the JSON Schema is checked in at
``cubist/schema/manifest-1.schema.json``. New optional fields keep version 1; a field whose
meaning changes bumps the version."""

from __future__ import annotations

from pathlib import Path

from cubist.core.model import BuiltAsset
from cubist.io.gltf import glb_origin, nearest_part, socket_part, socket_position

MANIFEST_VERSION = 1
SCHEMA_PATH = Path(__file__).parent / "schema" / "manifest-1.schema.json"
UNITS = {"length": "voxel", "angle": "degree", "time": "second", "spin": "turn"}


def build_manifest(
    asset: BuiltAsset,
    voxel_size_m: float,
    files: dict[str, str] | None = None,
    center: bool = True,
) -> dict:
    sockets = {}
    for name, value in asset.sockets.items():
        point = socket_position(value)
        sockets[name] = {
            "position": list(point),
            "part": socket_part(value) or nearest_part(asset, point),
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
                "voxels": p.grid.count(),
                "vox_model": p.name,
            }
            for p in asset.parts
        ],
        "sockets": sockets,
        "clips": [c.to_dict() for c in asset.clips],
        "files": dict(files or {}),
    }
