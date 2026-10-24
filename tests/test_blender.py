"""Exported animations play in Blender the way Cubist previews them.

Blender imports each .glb headless; every part's world matrix at several times must match
Cubist's pose evaluator (which catches a spin turning the wrong way, not just its bounding box).
Skipped when Blender isn't installed (set CUBIST_BLENDER to its binary, or put ``blender`` on
PATH).
"""

import json
import os
import runpy
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cubist import Model, Part, asset, box, clip, hinge, slide
from cubist.build.runner import load_project_style
from cubist.core.model import AssetDef
from cubist.core.palette import Palette
from cubist.io.gltf import encode_glb, glb_origin
from cubist.motion.pose import frame_index, hidden_at, world_matrices

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
DEMO = Path(__file__).parent.parent / "projects" / "demo"
TO_BLENDER = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)  # glTF Y-up -> Z-up


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


def demo(name, clip_name=None):
    from cubist.motion.pose import preview_clips

    style = load_project_style(DEMO / "style.toml")
    [a] = [v for v in runpy.run_path(str(DEMO / "assets" / f"{name}.py")).values()
           if isinstance(v, AssetDef)]  # fmt: skip
    built = a.build(style.palette, glow=style.glow, cycles=style.cycles)
    clips = preview_clips(built)
    clip = next(c for c in clips if clip_name in (None, c.name))
    return built, style.palette.rgba_table(), clip


def cases():
    def nested():
        a = asset(name="desk", size=(8, 4, 12))(desk).build(P)
        return a, P.rgba_table(), a.clips[0]

    yield "nested", nested
    for name in ("chest", "door", "window", "desk", "biplane", "barrel", "blaster", "torch"):
        yield name, lambda name=name: demo(name)
    for clip_name in ("walk", "idle"):
        yield f"goblin-{clip_name}", lambda c=clip_name: demo("goblin", c)


def blender_poses(glb: Path, out: Path, clip_name: str, times: list[float]) -> dict:
    subprocess.run(
        [BLENDER, "-b", "--factory-startup", "--python", str(SCRIPT), "--", str(glb), str(out),
         clip_name, *map(str, times)],
        check=True, capture_output=True, timeout=120,
    )  # fmt: skip
    return json.loads(out.read_text())["poses"]


@pytest.mark.parametrize("make", [m for _, m in cases()], ids=[n for n, _ in cases()])
def test_blender_poses_match_cubist(make, tmp_path):
    a, table, c = make()
    glb = tmp_path / "a.glb"
    glb.write_bytes(encode_glb(a, table, voxel_size_m=S))
    times = [0.0, round(c.length * 0.3, 4), round(c.length * 0.55, 4), c.length]
    poses = blender_poses(glb, tmp_path / "poses.json", c.name, times)
    origin = np.array(glb_origin(a))
    moved = False
    for t in times:
        got = poses[repr(t)]
        mats, hide = world_matrices(a, c, t), hidden_at(a, c, t)
        for p in a.parts:
            meshes = (
                [(f"{p.name}@{k}", k == frame_index(a, c, p.name, t)) for k in range(len(p.frames))]
                if p.frames
                else [(p.name, True)]
            )
            for mesh, current in meshes:
                if mesh not in got:
                    continue  # empty parts have no mesh
                shown = current and p.name not in hide
                m, pivot = mats[p.name], np.array(p.pivot, dtype=float)
                # node-local glTF points: the origin and one voxel along each axis
                local = np.array([[0, 0, 0], [S, 0, 0], [0, S, 0], [0, 0, S]])
                points = local if shown else np.zeros_like(local)  # scaled to nothing
                world = ((points / S + pivot) @ m[:3, :3].T + m[:3, 3] - origin) * S
                want = world @ TO_BLENDER.T
                mw = np.array(got[mesh]["matrix"])
                have = (local @ TO_BLENDER.T) @ mw[:3, :3].T + mw[:3, 3]
                # sampled curves play back linearly between keys: a few mm at S = 0.25
                assert np.allclose(have, want, atol=6e-3), (mesh, t)
        moved |= any(not np.allclose(poses[repr(times[0])][n]["matrix"], got[n]["matrix"])
                     for n in got)  # fmt: skip
    assert moved
