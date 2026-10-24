"""Runs inside Blender: import a .glb, pose its animation at the given times, and write each
mesh's world-space vertex bounding box and world matrix (Blender axes, metres) as JSON, keyed
by mesh name (object names can collide with the root node, which Cubist names after the asset).
With several animations, the one named after ``--clip`` is played.

blender -b --factory-startup --python tests/blender_pose.py -- <glb> <out.json> <t> [<t> ...]
"""

import json
import sys

import bpy  # type: ignore[import-not-found]

args = sys.argv[sys.argv.index("--") + 1 :]
glb, out_path, clip_name, times = args[0], args[1], args[2], [float(t) for t in args[3:]]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
scene = bpy.context.scene
# The importer keeps each glTF animation as an action per object, named "<clip>_<object>"
# or similar; make the requested clip's actions active.
for obj in scene.objects:
    data = obj.animation_data
    if data is None:
        continue
    for track in list(data.nla_tracks):
        for strip in track.strips:
            if strip.action and strip.action.name.startswith(clip_name):
                data.action = strip.action
        data.nla_tracks.remove(track)
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
        boxes[obj.data.name] = {
            "box": [
                [min(p[i] for p in pts) for i in range(3)],
                [max(p[i] for p in pts) for i in range(3)],
            ],
            "matrix": [list(row) for row in obj.matrix_world],
        }
    result["poses"][repr(t)] = boxes

with open(out_path, "w") as f:
    json.dump(result, f)
