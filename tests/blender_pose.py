"""Runs inside Blender: import a .glb, pose its animation at the given times, and write each
mesh's world-space vertex bounding box (Blender axes, metres) as JSON, keyed by mesh name
(object names can collide with the root node, which Cubist names after the asset).

blender -b --factory-startup --python tests/blender_pose.py -- <glb> <out.json> <t> [<t> ...]
"""

import json
import sys

import bpy  # type: ignore[import-not-found]

args = sys.argv[sys.argv.index("--") + 1 :]
glb, out_path, times = args[0], args[1], [float(t) for t in args[2:]]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
scene = bpy.context.scene
fps = scene.render.fps / scene.render.fps_base

result = {"fps": fps, "poses": {}}
for t in times:
    frame = t * fps
    scene.frame_set(int(frame), subframe=frame - int(frame))
    boxes = {}
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        pts = [obj.matrix_world @ v.co for v in obj.data.vertices]
        boxes[obj.data.name] = [
            [min(p[i] for p in pts) for i in range(3)],
            [max(p[i] for p in pts) for i in range(3)],
        ]
    result["poses"][repr(t)] = boxes

with open(out_path, "w") as f:
    json.dump(result, f)
