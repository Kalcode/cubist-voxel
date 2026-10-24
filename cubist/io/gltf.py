"""glTF 2.0 binary (.glb) export.

* One node per part, placed at its pivot and parented as in the model, so engines can rotate
  parts without extra setup. Mesh vertices are in the node's space. Sockets become empty
  nodes under the part that owns them (named, or else the nearest part).
* Greedy-meshed, with faces only where they touch outside air (sealed cavities are dropped).
* Colors come from a 256x1 palette texture (nearest filtering, UVs at palette cells; one
  material per asset), or from vertex colors. Optional ambient occlusion is baked into
  vertex colors, which glTF multiplies with the texture.
* Y-up and +Z front, like the DSL; metres from ``voxel_size_m``. By default the root node
  shifts the asset so its footprint is centered on the origin with its base at y = 0.
* Clips become animations on the part nodes, sampled from the same pose evaluator the
  previews use: 30 samples a second (more where a part turns fast, so no step between keys
  turns 60 degrees or more), linear; tracks that are all ``step`` keep just their keys.

Written by hand (JSON + binary chunks) so output is byte-for-byte deterministic.
"""

from __future__ import annotations

import io
import json
import os
import struct

import numpy as np
from PIL import Image

from cubist.core.grid import VoxelGrid
from cubist.core.model import BuiltAsset
from cubist.core.socket import Socket
from cubist.io.mesh import Quad, greedy_mesh
from cubist.motion.clips import STEPPED, Clip
from cubist.motion.pose import dense_times, frame_index, local_motion, preview_clips, track_value

AO_LEVELS = (0.5, 0.7, 0.85, 1.0)
_FLOAT, _UINT, _USHORT = 5126, 5125, 5123
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963
_NEAREST, _CLAMP = 9728, 33071


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def socket_position(value) -> tuple[float, float, float]:
    return Socket.from_value(value).position


def socket_part(value) -> str | None:
    return Socket.from_value(value).part


class _Binary:
    """Accumulates the BIN chunk, buffer views, and accessors."""

    def __init__(self):
        self.data = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def view(self, payload: bytes, target: int | None = None) -> int:
        self.data.extend(b"\x00" * (-len(self.data) % 4))
        view = {"buffer": 0, "byteOffset": len(self.data), "byteLength": len(payload)}
        if target is not None:
            view["target"] = target
        self.data.extend(payload)
        self.views.append(view)
        return len(self.views) - 1

    def accessor(
        self,
        array: np.ndarray,
        kind: str,
        component: int,
        target: int | None,
        normalized: bool = False,
        bounds: bool = False,
    ) -> int:
        acc = {
            "bufferView": self.view(np.ascontiguousarray(array).tobytes(), target),
            "componentType": component,
            "count": int(len(array)),
            "type": kind,
        }
        if normalized:
            acc["normalized"] = True
        if bounds:
            acc["min"] = [float(v) for v in array.min(axis=0)]
            acc["max"] = [float(v) for v in array.max(axis=0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1


def _quad_geometry(quads: list[Quad], pivot: np.ndarray, scale: float):
    """Positions, normals, color indices, AO factors (per vertex) and triangle indices."""
    n = len(quads)
    pos = np.zeros((n, 4, 3))
    normals = np.zeros((n, 4, 3))
    color = np.zeros((n, 4), dtype=np.int64)
    ao = np.ones((n, 4))
    tris = np.zeros((n, 6), dtype=np.int64)
    for i, q in enumerate(quads):
        u_axis, v_axis = [a for a in range(3) if a != q.axis]
        for c, (u, v) in enumerate(((q.u0, q.v0), (q.u1, q.v0), (q.u0, q.v1), (q.u1, q.v1))):
            pos[i, c, q.axis] = q.layer
            pos[i, c, u_axis] = u
            pos[i, c, v_axis] = v
        normals[i, :, q.axis] = q.sign
        color[i] = q.color
        if q.ao is not None:
            ao[i] = [AO_LEVELS[a] for a in q.ao]
        # e_u x e_v points along +axis for x and z, -axis for y.
        front = q.sign * (-1 if q.axis == 1 else 1) > 0
        # Split along the brighter diagonal so AO interpolates without a crease.
        flip = q.ao is not None and ao[i, 0] + ao[i, 3] < ao[i, 1] + ao[i, 2]
        if flip:
            local = (0, 1, 2, 1, 3, 2) if front else (0, 2, 1, 1, 2, 3)
        else:
            local = (0, 1, 3, 0, 3, 2) if front else (0, 3, 1, 0, 2, 3)
        tris[i] = np.array(local) + 4 * i
    pos = (pos.reshape(-1, 3) - pivot) * scale
    return pos, normals.reshape(-1, 3), color.reshape(-1), ao.reshape(-1), tris.reshape(-1)


def glb_origin(asset: BuiltAsset, center: bool = True) -> list[float]:
    """The DSL point that lands on the GLB origin: the footprint center at y = 0 when
    centered (the declared size's, else the voxels'), else the DSL origin."""
    if not center:
        return [0, 0, 0]
    if asset.size is not None:
        return [asset.size[0] / 2.0, 0, asset.size[2] / 2.0]
    b = VoxelGrid.merge([p.grid for p in asset.parts]).bounds() or ((0, 0, 0), (0, 0, 0))
    return [(b[0][0] + b[1][0]) / 2.0, 0, (b[0][2] + b[1][2]) / 2.0]


def nearest_part(asset: BuiltAsset, point: tuple[float, float, float]) -> str | None:
    """The part with the voxel nearest ``point``, preferring parts visible at rest (sockets,
    lights and emitters shouldn't vanish with a hidden muzzle flash)."""
    hidden = set(asset.hidden_parts())
    candidates = [p for p in asset.parts if p.name not in hidden] or asset.parts
    best, best_d = None, np.inf
    for part in candidates:
        coords, _ = part.grid.voxels()
        if len(coords) == 0:
            continue
        d = float(np.min(np.sum((coords + 0.5 - np.array(point)) ** 2, axis=1)))
        if d < best_d:
            best, best_d = part.name, d
    return best


SAMPLE_RATE = 30.0
MAX_TURN_PER_KEY = 60.0  # degrees; slerp between keys must never take the short way round
STEP_LEAD = 1e-3  # seconds: the key placed just before a step track's jump
MAX_MOVE_PER_KEY = 0.5  # voxels; snaps like a slide's recoil get more keys


def _quat(rot: np.ndarray) -> np.ndarray:
    """A 3x3 rotation as an (x, y, z, w) unit quaternion."""
    m = rot
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        k = 2.0 * np.sqrt(tr + 1.0)
        q = [(m[2, 1] - m[1, 2]) / k, (m[0, 2] - m[2, 0]) / k, (m[1, 0] - m[0, 1]) / k, k / 4]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        k = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [k / 4, (m[0, 1] + m[1, 0]) / k, (m[0, 2] + m[2, 0]) / k, (m[2, 1] - m[1, 2]) / k]
    elif m[1, 1] > m[2, 2]:
        k = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / k, k / 4, (m[1, 2] + m[2, 1]) / k, (m[0, 2] - m[2, 0]) / k]
    else:
        k = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / k, (m[1, 2] + m[2, 1]) / k, k / 4, (m[1, 0] - m[0, 1]) / k]
    q = np.array(q)
    return q / np.linalg.norm(q)


def animation_samples(asset: BuiltAsset, clip: Clip, voxel_size_m: float = 0.0625) -> dict:
    """Per driven part: ``times``, node ``translation`` (n, 3) in metres, ``rotation`` (n, 4)
    xyzw, and the glTF ``interpolation``."""
    pivots = {p.name: np.array(p.pivot, dtype=np.float64) for p in asset.parts}
    out = {}
    motion = [t for t in clip.tracks if t.channel not in STEPPED]
    motions: dict[float, dict] = {}  # local_motion covers every part; reuse it across parts
    for name in dict.fromkeys(t.part for t in motion):
        tracks = [t for t in motion if t.part == name]
        part = asset.part(name)
        steps = [t for t in tracks if clip.track_ease(t) == "step"]
        if len(steps) == len(tracks):
            # Keys only, plus the clip's end so the animation lasts the whole clip.
            times = sorted({0.0, clip.length, *(t for tr in tracks for t in tr.times)})
            interpolation = "STEP"
        else:
            times = dense_times(
                asset, clip, SAMPLE_RATE, MAX_TURN_PER_KEY * 0.75, [name], max_move=MAX_MOVE_PER_KEY
            )
            # A step track among smooth ones: a key just before each jump keeps it sharp.
            jumps = [t - STEP_LEAD for tr in steps for t in tr.times if t - STEP_LEAD > 0]
            times = sorted({*times.tolist(), *jumps})
            interpolation = "LINEAR"
        # glTF stores float32 times, which must strictly increase.
        times = _f32_times(times)
        offset = pivots[name] - (pivots[part.parent] if part.parent in pivots else 0.0)
        trans, rots = [], []
        for t in times:
            if t not in motions:
                motions[t] = local_motion(asset, clip, float(t))
            tr, rot = motions[t][name]
            trans.append((offset + tr) * voxel_size_m)
            q = _quat(rot)
            if rots and np.dot(q, rots[-1]) < 0:
                q = -q
            rots.append(q)
        out[name] = {
            "times": np.array(times, dtype=np.float64),
            "translation": np.array(trans),
            "rotation": np.array(rots),
            "rest_translation": offset * voxel_size_m,
            "interpolation": interpolation,
        }
    return out


def _f32_times(times) -> list[float]:
    """Sorted times that stay strictly increasing once glTF stores them as float32. The
    float64 times are kept: values must be sampled at the exact key times (a float32 time can
    round to just before a key, and a step would then show the previous value)."""
    out: list[float] = []
    last = None
    for t in sorted(times):
        t32 = np.float32(t)
        if last is None or t32 > last:
            out.append(float(t))
            last = t32
    return out


def _animation(
    asset: BuiltAsset,
    clip: Clip,
    binary: _Binary,
    index_of: dict[str, int],
    s: float,
    frame_nodes: dict[tuple[str, int], int] | None = None,
) -> dict:
    samples = animation_samples(asset, clip, s)
    channels: list[dict] = []
    samplers: list[dict] = []
    inputs: dict[tuple[str, bytes], int] = {}

    def add(
        name: str, path: str, values: np.ndarray, sample: dict, node: int | None = None
    ) -> None:
        times = sample["times"].astype(np.float32).reshape(-1, 1)
        key = (sample["interpolation"], times.tobytes())
        if key not in inputs:
            inputs[key] = binary.accessor(times, "SCALAR", _FLOAT, None, bounds=True)
        kind = "VEC4" if path == "rotation" else "VEC3"
        output = binary.accessor(values.astype(np.float32), kind, _FLOAT, None)
        samplers.append(
            {"input": inputs[key], "output": output, "interpolation": sample["interpolation"]}
        )
        target = index_of[name] if node is None else node
        channels.append({"sampler": len(samplers) - 1, "target": {"node": target, "path": path}})

    identity = np.array([0.0, 0.0, 0.0, 1.0])
    for name, sample in samples.items():
        if np.abs(sample["translation"] - sample["rest_translation"]).max() > 1e-9:
            add(name, "translation", sample["translation"], sample)
        if np.abs(np.abs(sample["rotation"] @ identity) - 1.0).max() > 1e-12:
            add(name, "rotation", sample["rotation"], sample)
    for track in clip.tracks:
        if track.channel == "visible":
            times = _f32_times({0.0, clip.length, *track.times})
            values = [track_value(track, t, clip) for t in times]
            shown = np.array([[v, v, v] for v in values], dtype=np.float64)
            steps = {"times": np.array(times), "interpolation": "STEP"}
            add(track.part, "scale", shown, steps)
        elif track.channel == "frame" and frame_nodes:
            times = _f32_times({0.0, clip.length, *track.times})
            steps = {"times": np.array(times), "interpolation": "STEP"}
            n = len(asset.part(track.part).frames or [])
            for k in range(n):
                on = [1.0 if frame_index(asset, clip, track.part, t) == k else 0.0 for t in times]
                values = np.array([[v, v, v] for v in on])
                add(track.part, "scale", values, steps, frame_nodes[(track.part, k)])
    if not channels and samples:
        # A clip that moves nothing still exists, so engines can find it by name.
        name, sample = next(iter(samples.items()))
        add(name, "rotation", sample["rotation"], sample)
    return {
        "name": clip.name,
        "channels": channels,
        "samplers": samplers,
        "extras": {
            "loop": clip.loop,
            "length": clip.length,
            **({"travel": [v * s for v in clip.travel]} if clip.travel else {}),
        },
    }


def _lights(asset, nodes, index_of, pivots, root, s) -> None:
    """A node per light under the part it moves with (named, or the nearest)."""
    for i, (name, lt) in enumerate(asset.lights.items()):
        owner_name = lt.part or nearest_part(asset, lt.position)
        offset = np.array(lt.position) - (pivots[owner_name] if owner_name in pivots else 0.0)
        nodes.append(
            {
                "name": name,
                "translation": [float(v) * s for v in offset],
                "extensions": {"KHR_lights_punctual": {"light": i}},
            }
        )
        owner = nodes[index_of[owner_name]] if owner_name in index_of else root
        owner.setdefault("children", []).append(len(nodes) - 1)


def used_colors(asset: BuiltAsset) -> set[int]:
    used: set[int] = set()
    for part in asset.parts:
        for grid in getattr(part, "frames", None) or [part.grid]:
            used.update(grid.colors_used())
    return used


def _emissive(asset: BuiltAsset, rgba: np.ndarray, doc: dict, binary, material: dict) -> None:
    """Glowing colors as an emissive palette texture (black elsewhere); strengths above 1 go
    in KHR_materials_emissive_strength. Assets with no glowing color are left as they were."""
    glow = {i: s for i, s in asset.glow.items() if i in used_colors(asset)}
    if not glow:
        return
    peak = max(glow.values())
    scale = max(1.0, peak)
    table = np.zeros((256, 4), dtype=np.uint8)
    table[:, 3] = 255
    for index, strength in glow.items():
        table[index, :3] = np.rint(rgba[index, :3].astype(np.float64) * min(1.0, strength / scale))
    png = io.BytesIO()
    Image.fromarray(table[None, :, :], "RGBA").save(png, format="PNG")
    doc["images"].append(
        {"name": "glow", "mimeType": "image/png", "bufferView": binary.view(png.getvalue())}
    )
    doc["textures"].append({"sampler": 0, "source": len(doc["images"]) - 1})
    material["emissiveTexture"] = {"index": len(doc["textures"]) - 1}
    material["emissiveFactor"] = [1.0, 1.0, 1.0]
    if peak > 1.0:
        material["extensions"] = {"KHR_materials_emissive_strength": {"emissiveStrength": peak}}
        doc.setdefault("extensionsUsed", []).append("KHR_materials_emissive_strength")


def encode_glb(
    asset: BuiltAsset,
    rgba: np.ndarray,
    voxel_size_m: float = 0.0625,
    vertex_colors: bool = False,
    ao: bool = False,
    center: bool = True,
    hollow: bool = True,
) -> bytes:
    if not voxel_size_m > 0:
        raise ValueError(f"voxel_size_m must be > 0, got {voxel_size_m}")
    s = float(voxel_size_m)
    rgba = np.asarray(rgba, dtype=np.uint8)
    binary = _Binary()
    use_texture = not vertex_colors

    root = {"name": asset.name, "children": []}
    if center:
        cx, _, cz = glb_origin(asset)
        root["translation"] = [-cx * s, 0.0, -cz * s]
    nodes: list[dict] = [root]
    meshes: list[dict] = []
    index_of: dict[str, int] = {}
    pivots = {p.name: np.array(p.pivot, dtype=np.float64) for p in asset.parts}

    def mesh_of(grid: VoxelGrid, pivot: np.ndarray, name: str) -> int | None:
        quads = greedy_mesh(grid, ao=ao, seal=hollow)
        if not quads:
            return None
        pos, nrm, color, shade, tris = _quad_geometry(quads, pivot, s)
        attributes = {
            "POSITION": binary.accessor(
                pos.astype(np.float32), "VEC3", _FLOAT, _ARRAY_BUFFER, bounds=True
            ),
            "NORMAL": binary.accessor(nrm.astype(np.float32), "VEC3", _FLOAT, _ARRAY_BUFFER),
        }
        if use_texture:
            uv = np.stack([(color + 0.5) / 256.0, np.full(len(color), 0.5)], axis=1)
            attributes["TEXCOORD_0"] = binary.accessor(
                uv.astype(np.float32), "VEC2", _FLOAT, _ARRAY_BUFFER
            )
        if vertex_colors or ao:
            # COLOR_0 is linear in glTF; the palette is sRGB.
            rgb = srgb_to_linear(rgba[color, :3]) if vertex_colors else np.ones((len(color), 3))
            col = np.concatenate([rgb * shade[:, None], np.ones((len(rgb), 1))], axis=1)
            attributes["COLOR_0"] = binary.accessor(
                col.astype(np.float32), "VEC4", _FLOAT, _ARRAY_BUFFER
            )
        big = len(pos) > 65535
        indices = binary.accessor(
            tris.astype(np.uint32 if big else np.uint16),
            "SCALAR",
            _UINT if big else _USHORT,
            _ELEMENT_ARRAY_BUFFER,
        )
        meshes.append(
            {
                "name": name,
                "primitives": [
                    {"attributes": attributes, "indices": indices, "material": 0, "mode": 4}
                ],
            }
        )
        return len(meshes) - 1

    frame_nodes: dict[tuple[str, int], int] = {}
    for part in asset.parts:
        parent = part.parent if part.parent in pivots else None
        offset = pivots[part.name] - (pivots[parent] if parent else 0.0)
        node: dict = {"name": part.name, "translation": [float(v) * s for v in offset]}
        if part.hidden:
            node["scale"] = [0.0, 0.0, 0.0]  # shown by a clip's visibility track
        index_of[part.name] = len(nodes)
        nodes.append(node)
        if part.frames:
            # A flipbook: one child node per frame, all but the first scaled to nothing.
            for k, grid in enumerate(part.frames):
                fnode: dict = {"name": f"{part.name}@{k}"}
                mesh = mesh_of(grid, pivots[part.name], f"{part.name}@{k}")
                if mesh is not None:
                    fnode["mesh"] = mesh
                if k:
                    fnode["scale"] = [0.0, 0.0, 0.0]
                frame_nodes[(part.name, k)] = len(nodes)
                node.setdefault("children", []).append(len(nodes))
                nodes.append(fnode)
            continue
        mesh = mesh_of(part.grid, pivots[part.name], part.name)
        if mesh is not None:
            node["mesh"] = mesh

    for part in asset.parts:
        parent = part.parent if part.parent in index_of else None
        owner = nodes[index_of[parent]] if parent else root
        owner.setdefault("children", []).append(index_of[part.name])

    for name, value in asset.sockets.items():
        point = socket_position(value)
        owner_name = socket_part(value) or nearest_part(asset, point)
        offset = np.array(point) - (pivots[owner_name] if owner_name in pivots else 0.0)
        snode: dict = {"name": name, "translation": [float(v) * s for v in offset]}
        sock = Socket.from_value(value, name)
        if sock.forward is not None:
            # local +Z forward, +Y up
            up, fwd = np.array(sock.up), np.array(sock.forward)
            snode["rotation"] = [
                float(c) for c in _quat(np.column_stack([np.cross(up, fwd), up, fwd]))
            ]
        nodes.append(snode)
        owner = nodes[index_of[owner_name]] if owner_name in index_of else root
        owner.setdefault("children", []).append(len(nodes) - 1)

    _lights(asset, nodes, index_of, pivots, root, s)
    for name, e in asset.emitters.items():  # empty nodes the engine attaches its particles to
        owner_name = e.part or nearest_part(asset, e.position)
        offset = np.array(e.position) - (pivots[owner_name] if owner_name in pivots else 0.0)
        fwd = np.array(e.direction)
        up = np.array([0.0, 1.0, 0.0]) if abs(fwd[1]) < 0.99 else np.array([0.0, 0.0, 1.0])
        up = up - (up @ fwd) * fwd
        up /= np.linalg.norm(up)
        basis = np.column_stack([np.cross(up, fwd), up, fwd])
        nodes.append(
            {
                "name": name,
                "translation": [float(v) * s for v in offset],
                "rotation": [float(c) for c in _quat(basis)],
            }
        )
        owner = nodes[index_of[owner_name]] if owner_name in index_of else root
        owner.setdefault("children", []).append(len(nodes) - 1)
    material: dict = {
        "name": "palette",
        "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 1.0},
    }
    doc: dict = {
        "asset": {"version": "2.0", "generator": "cubist"},
        "scene": 0,
        "scenes": [{"name": asset.name, "nodes": [0]}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": [material],
    }
    if use_texture:
        png = io.BytesIO()
        Image.fromarray(rgba[None, :, :], "RGBA").save(png, format="PNG")
        doc["images"] = [
            {"name": "palette", "mimeType": "image/png", "bufferView": binary.view(png.getvalue())}
        ]
        doc["samplers"] = [
            {"magFilter": _NEAREST, "minFilter": _NEAREST, "wrapS": _CLAMP, "wrapT": _CLAMP}
        ]
        doc["textures"] = [{"sampler": 0, "source": 0}]
        material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
        _emissive(asset, rgba, doc, binary, material)
    if asset.lights:
        doc.setdefault("extensions", {})["KHR_lights_punctual"] = {
            "lights": [
                {
                    "name": name,
                    "type": "point",
                    "color": [float(c) for c in srgb_to_linear(np.array(lt.rgb))],
                    "intensity": lt.intensity,
                    "range": lt.range * s,
                }
                for name, lt in asset.lights.items()
            ]
        }
        doc.setdefault("extensionsUsed", []).append("KHR_lights_punctual")
    clips = preview_clips(asset)
    if clips:
        doc["animations"] = [_animation(asset, c, binary, index_of, s, frame_nodes) for c in clips]
    doc["extras"] = {
        "cubist": {"voxel_size_m": s, "size": list(asset.size) if asset.size else None}
    }
    doc["bufferViews"] = binary.views
    doc["accessors"] = binary.accessors
    bin_chunk = bytes(binary.data) + b"\x00" * (-len(binary.data) % 4)
    doc["buffers"] = [{"byteLength": len(bin_chunk)}]

    json_chunk = json.dumps(doc, separators=(",", ":"), sort_keys=True).encode()
    json_chunk += b" " * (-len(json_chunk) % 4)
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    return b"".join(
        [
            struct.pack("<4sII", b"glTF", 2, total),
            struct.pack("<I4s", len(json_chunk), b"JSON"),
            json_chunk,
            struct.pack("<I4s", len(bin_chunk), b"BIN\x00"),
            bin_chunk,
        ]
    )


def write_glb(path: str | os.PathLike, asset: BuiltAsset, rgba: np.ndarray, **options) -> None:
    with open(path, "wb") as f:
        f.write(encode_glb(asset, rgba, **options))
