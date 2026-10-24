"""Milestone 4 acceptance: GLBs import into Godot 4 with the right hierarchy, scale, and colors.

Runs a headless Godot that loads the GLB with GLTFDocument and prints its scene tree.
Skipped when ``godot`` is not on PATH.
"""

import json
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest

from cubist import Model, Part, asset, box
from cubist.build.runner import load_project_style
from cubist.core.model import AssetDef
from cubist.core.palette import Palette
from cubist.io.gltf import write_glb

GODOT = shutil.which("godot")
pytestmark = pytest.mark.skipif(GODOT is None, reason="godot not on PATH")
DEMO = Path(__file__).parent.parent / "projects" / "demo"

INSPECT = """
extends SceneTree

func _init():
    var path = OS.get_cmdline_user_args()[0]
    var doc = GLTFDocument.new()
    var state = GLTFState.new()
    var err = doc.append_from_file(path, state)
    if err != OK:
        print("ERROR ", err)
        quit(1)
        return
    var scene = doc.generate_scene(state)
    _dump(scene, "")
    scene.free()
    quit()

func _dump(node, parent):
    var info = {"name": str(node.name), "parent": parent, "type": node.get_class()}
    if node is Node3D:
        info["position"] = [node.position.x, node.position.y, node.position.z]
    if node is MeshInstance3D and node.mesh:
        var aabb = node.mesh.get_aabb()
        info["aabb_size"] = [aabb.size.x, aabb.size.y, aabb.size.z]
        var mat = node.mesh.surface_get_material(0)
        info["texture"] = mat != null and mat.albedo_texture != null
        info["filter"] = mat.texture_filter if mat != null else -1
        info["triangles"] = node.mesh.get_faces().size() / 3
        info["emission"] = mat != null and mat.emission_enabled
    if node is AnimationPlayer:
        info["animations"] = node.get_animation_list()
    print("NODE ", JSON.stringify(info))
    for child in node.get_children():
        _dump(child, str(node.name))
"""


def godot_tree(glb: Path, tmp_path: Path) -> dict[str, dict]:
    project = tmp_path / "godot"
    project.mkdir()
    (project / "project.godot").write_text("config_version=5\n")
    (project / "inspect.gd").write_text(INSPECT)
    proc = subprocess.run(
        [GODOT, "--headless", "--path", str(project), "-s", "inspect.gd", "--", str(glb)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    nodes = [json.loads(line[5:]) for line in proc.stdout.splitlines() if line.startswith("NODE ")]
    assert nodes, proc.stdout + proc.stderr
    return {n["name"]: n for n in nodes}


def close(a, b, tol=1e-4):
    return all(abs(x - y) <= tol for x, y in zip(a, b, strict=True))


def test_chest_hierarchy_scale_and_texture(tmp_path):
    p = Palette({"wood": ["#3b2414", "#8a5a32"]})

    @asset(name="chest", size=(4, 5, 4))
    def chest(p):
        return Model(
            parts=[
                Part("body", box((4, 4, 4), color=p.wood.dark), pivot=(2, 0, 2)),
                Part(
                    "lid",
                    box((4, 1, 4), color=p.wood.light).at(0, 4, 0),
                    parent="body",
                    pivot=(0, 4, 0),
                ),
            ],
            sockets={"handle": (2, 5, 2)},
        )

    glb = tmp_path / "chest.glb"
    write_glb(glb, chest.build(p), p.rgba_table(), voxel_size_m=0.25)
    tree = godot_tree(glb, tmp_path)
    assert tree["body"]["parent"] == "chest"
    assert tree["lid"]["parent"] == "body"
    assert tree["handle"]["parent"] == "lid"
    assert close(tree["chest"]["position"], [-0.5, 0, -0.5])
    assert close(tree["body"]["position"], [0.5, 0, 0.5])
    assert close(tree["lid"]["position"], [-0.5, 1.0, -0.5])
    assert close(tree["body"]["aabb_size"], [1.0, 1.0, 1.0])
    assert tree["body"]["texture"] is True
    assert tree["body"]["triangles"] == 12


def test_goblin_imports_with_every_part_under_the_torso(tmp_path):
    style = load_project_style(DEMO / "style.toml")
    [goblin] = [
        v
        for v in runpy.run_path(str(DEMO / "assets" / "goblin.py")).values()
        if isinstance(v, AssetDef)
    ]
    built = goblin.build(style.palette)
    glb = tmp_path / "goblin.glb"
    write_glb(glb, built, style.palette.rgba_table(), voxel_size_m=style.voxel_size_m)
    tree = godot_tree(glb, tmp_path)
    for part in ("head", "leg_l", "leg_r", "arm_l", "arm_r"):
        assert tree[part]["parent"] == "torso", part
    assert tree["hand_r"]["parent"] == "arm_r"  # the goblin's own right, at low x
    assert tree["arm_r"]["position"][0] < tree["arm_l"]["position"][0]
    # 24 voxels tall at 1/16 m: the head's top sits 22 voxels up.
    head = tree["head"]
    top = tree["torso"]["position"][1] + head["position"][1] + head["aabb_size"][1]
    assert abs(top - 22 / 16) < 1e-4


@pytest.mark.parametrize(
    "name,clip,light",
    [
        ("torch", "flame_frames", "flame"),
        ("blaster", "fire", None),
        ("goblin", "walk", None),
        ("campfire", "flames_frames", "fire"),
        ("barrel", "explode", None),
    ],
)
def test_moving_demo_assets_import_with_clips_lights_and_glow(name, clip, light, tmp_path):
    """Clips arrive as animations, lights as OmniLight3D, glowing colors as emission."""
    import runpy

    from cubist.build.runner import load_project_style
    from cubist.core.model import AssetDef
    from cubist.io.gltf import encode_glb

    style = load_project_style(DEMO / "style.toml")
    [a] = [v for v in runpy.run_path(str(DEMO / "assets" / f"{name}.py")).values()
           if isinstance(v, AssetDef)]  # fmt: skip
    built = a.build(style.palette, glow=style.glow, cycles=style.cycles)
    glb = tmp_path / f"{name}.glb"
    glb.write_bytes(encode_glb(built, style.palette.rgba_table(), voxel_size_m=0.0625))
    tree = godot_tree(glb, tmp_path)
    players = [n for n in tree.values() if n["type"] == "AnimationPlayer"]
    assert players and clip in players[0]["animations"]
    if light:
        assert any(n["type"] == "OmniLight3D" for n in tree.values())
    meshes = [n for n in tree.values() if n["type"] == "MeshInstance3D"]
    glows = any(c in built.glow for p in built.parts for g in (p.frames or [p.grid])
                for c in g.colors_used())  # fmt: skip
    assert any(m["emission"] for m in meshes) == glows
