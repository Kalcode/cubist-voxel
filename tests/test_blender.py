"""Exported animations play in Blender the way Cubist previews them.

Blender imports each .glb headless; the world-space bounding box of every part at several
times must match Cubist's pose evaluator. Skipped when Blender isn't installed (set
CUBIST_BLENDER to its binary, or put ``blender`` on PATH).
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cubist import Model, Part, asset, box, clip, hinge, slide
from cubist.core.palette import Palette
from cubist.io.gltf import encode_glb, glb_origin
from cubist.motion.pose import world_matrices
from cubist.render.posed import _Mesh

MAC_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
BLENDER = (
    os.environ.get("CUBIST_BLENDER")
    or shutil.which("blender")
    or (MAC_BLENDER if os.path.exists(MAC_BLENDER) else None)
)
pytestmark = pytest.mark.skipif(BLENDER is None, reason="Blender is not installed")

P = Palette({"wood": ["#3b2414", "#8a5a32"], "metal": "#7d8491"})
S = 0.25
SCRIPT = Path(__file__).parent / "blender_pose.py"


def chest(p):
    body = box((8, 4, 6), color=p.wood.dark) - box((6, 3, 4)).at(1, 1, 1)
    return Model(
        [
            Part("body", body, pivot=(4, 0, 3)),
            Part(
                "lid",
                box((8, 2, 6), color=p.wood.light).at(0, 4, 0),
                parent="body",
                pivot=(4, 4, 0),
                joint=hinge("-x", (0, 120)),
            ),
        ],
        clips=[clip("open", 0.5, {"lid": [0, 115, 100]})],
    )


def desk(p):
    top = Part("desk", box((8, 4, 8), color=p.wood.dark), pivot=(4, 0, 4))
    drawer = Part(
        "drawer",
        box((6, 2, 6), color=p.wood.light).at(1, 1, 1),
        parent="desk",
        pivot=(4, 1, 7),
        joint=slide("+z", 4),
    )
    knob = Part(
        "knob",
        box((2, 1, 1), color=p.metal).at(3, 1, 7),
        parent="drawer",
        pivot=(4, 1, 7),
        joint=hinge("x", (0, 90)),
    )
    return Model(
        [top, drawer, knob], clips=[clip("pull", 1.0, {"drawer": [0, 4], "knob": [0, 90]})]
    )


def blender_boxes(glb: Path, out: Path, times: list[float]) -> dict:
    subprocess.run(
        [BLENDER, "-b", "--factory-startup", "--python", str(SCRIPT), "--", str(glb), str(out),
         *map(str, times)],
        check=True, capture_output=True, timeout=120,
    )  # fmt: skip
    return json.loads(out.read_text())["poses"]


@pytest.mark.parametrize(
    "fn,name,size,times",
    [
        (chest, "chest", (8, 7, 6), [0.0, 0.2, 0.35, 0.5]),
        (desk, "desk", (8, 4, 12), [0.0, 0.4, 1.0]),
    ],
)
def test_blender_poses_match_cubist(fn, name, size, times, tmp_path):
    a = asset(name=name, size=size)(fn).build(P)
    glb = tmp_path / f"{name}.glb"
    glb.write_bytes(encode_glb(a, P.rgba_table(), voxel_size_m=S))
    poses = blender_boxes(glb, tmp_path / "poses.json", times)
    origin = np.array(glb_origin(a))
    meshes = {p.name: _Mesh(p.grid) for p in a.parts}
    for t in times:
        got = poses[repr(t)]
        mats = world_matrices(a, a.clips[0], t)
        for part, mesh in meshes.items():
            m = mats[part]
            pts = mesh.corners.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
            metres = (pts - origin) * S
            blender = np.stack([metres[:, 0], -metres[:, 2], metres[:, 1]], axis=1)
            want = [blender.min(axis=0), blender.max(axis=0)]
            assert np.allclose(got[part], want, atol=1e-3), (part, t, got[part], want)
    moved = a.clips[0].tracks[0].part
    assert not np.allclose(poses[repr(times[0])][moved], poses[repr(times[-1])][moved])
