"""MagicaVoxel ``.vox`` reader and writer.

Follows the public format spec (github.com/ephtracy/voxel-model). Parts are written as
separate models placed by the scene graph (``nTRN`` / ``nGRP`` / ``nSHP``), so they keep
their names and world positions; the palette goes in the ``RGBA`` chunk.

Cubist is Y-up with +Z toward the front; MagicaVoxel is Z-up with the front at -Y.
A DSL voxel cell (x, y, z) maps to the MagicaVoxel cell (x, -1 - z, y). MagicaVoxel
places a model by its center, ``floor(size / 2)`` from the model's min corner.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import numpy as np

from cubist.core.grid import VoxelGrid

VERSION = 200
MAX_MODEL_AXIS = 256
_IDENTITY_ROTATION = "4"  # row 0 -> x, row 1 -> y, no sign flips


class VoxError(ValueError):
    """Invalid or unsupported .vox data."""


@dataclass
class VoxPart:
    name: str
    grid: VoxelGrid
    parent: str | None = None
    hidden: bool = False  # written as the scene graph's _hidden attribute


@dataclass
class VoxScene:
    parts: list[VoxPart]
    palette: np.ndarray  # (256, 4) uint8, indexed by color index; row 0 unused


def default_palette() -> np.ndarray:
    """MagicaVoxel's built-in palette, used when a file has no RGBA chunk."""
    levels = [0xFF, 0xCC, 0x99, 0x66, 0x33, 0x00]
    cube = [(r, g, b) for r in levels for g in levels for b in levels][:-1]  # drop black
    ramp = [0xEE, 0xDD, 0xBB, 0xAA, 0x88, 0x77, 0x55, 0x44, 0x22, 0x11]
    tail = [(v, 0, 0) for v in ramp] + [(0, v, 0) for v in ramp] + [(0, 0, v) for v in ramp]
    tail += [(v, v, v) for v in ramp]
    table = np.zeros((256, 4), dtype=np.uint8)
    table[1:, :3] = cube + tail
    table[1:, 3] = 255
    return table


# --- writing ----------------------------------------------------------------


def _chunk(cid: bytes, content: bytes, children: bytes = b"") -> bytes:
    return cid + struct.pack("<ii", len(content), len(children)) + content + children


def _string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<i", len(b)) + b


def _dict(d: dict[str, str]) -> bytes:
    return struct.pack("<i", len(d)) + b"".join(_string(k) + _string(v) for k, v in d.items())


def _to_mv(grid: VoxelGrid) -> tuple[np.ndarray, tuple[int, int, int]]:
    """A cropped DSL grid as a MagicaVoxel-oriented array and its world translation."""
    ox, oy, oz = grid.origin
    sx, sy, sz = grid.shape
    # [x, y, z] -> [x, depth, height] with depth reversed so the DSL front lands at low y.
    mv = np.transpose(grid.data, (0, 2, 1))[:, ::-1, :]
    mv_min = np.array([ox, -oz - sz, oy])
    t = mv_min + np.array(mv.shape) // 2
    return mv, (int(t[0]), int(t[1]), int(t[2]))


def _trn(
    node_id: int, name: str | None, child: int, layer: int, t=None, hidden: bool = False
) -> bytes:
    attrs = {"_name": name} if name else {}
    if hidden:
        attrs["_hidden"] = "1"
    frame = {"_t": f"{t[0]} {t[1]} {t[2]}"} if t is not None else {}
    return _chunk(
        b"nTRN",
        struct.pack("<i", node_id)
        + _dict(attrs)
        + struct.pack("<iiii", child, -1, layer, 1)
        + _dict(frame),
    )


def _grp(node_id: int, children: list[int]) -> bytes:
    return _chunk(
        b"nGRP",
        struct.pack("<i", node_id)
        + _dict({})
        + struct.pack(f"<i{len(children)}i", len(children), *children),
    )


def _shp(node_id: int, model: int) -> bytes:
    return _chunk(
        b"nSHP", struct.pack("<i", node_id) + _dict({}) + struct.pack("<ii", 1, model) + _dict({})
    )


def encode_vox(scene: VoxScene) -> bytes:
    """Parts with children become named groups holding their own shape and their child parts,
    so editors show the part hierarchy."""
    models, translations = [], []
    for part in scene.parts:
        grid = part.grid.cropped()
        if grid.count() == 0:
            raise VoxError(f"part {part.name!r} is empty")
        if any(s > MAX_MODEL_AXIS for s in grid.shape):
            raise VoxError(
                f"part {part.name!r} is {grid.shape}; .vox models are limited to "
                f"{MAX_MODEL_AXIS} voxels per axis"
            )
        mv, t = _to_mv(grid)
        filled = np.argwhere(mv)
        xyzi = np.column_stack([filled, mv[tuple(filled.T)]]).astype(np.uint8)
        models.append(_chunk(b"SIZE", struct.pack("<iii", *mv.shape)))
        models.append(_chunk(b"XYZI", struct.pack("<i", len(xyzi)) + xyzi.tobytes()))
        translations.append(t)

    names = {p.name for p in scene.parts}
    kids: dict[str, list[int]] = {}
    roots = []
    for i, part in enumerate(scene.parts):
        if part.parent in names and part.parent != part.name:
            kids.setdefault(part.parent, []).append(i)
        else:
            roots.append(i)

    nodes: list[bytes] = []
    next_id = [2]
    placed: set[int] = set()

    def new_id() -> int:
        next_id[0] += 1
        return next_id[0] - 1

    def subtree(i: int) -> int:
        placed.add(i)
        part = scene.parts[i]
        children = [k for k in kids.get(part.name, []) if k not in placed]
        if not children:
            trn_id, shp_id = new_id(), new_id()
            nodes.append(_trn(trn_id, part.name, shp_id, 0, translations[i], part.hidden))
            nodes.append(_shp(shp_id, i))
            return trn_id
        group_trn, group, own_trn, own_shp = new_id(), new_id(), new_id(), new_id()
        nodes.append(_trn(group_trn, part.name, group, 0, None, part.hidden))
        nodes.append(_trn(own_trn, part.name, own_shp, 0, translations[i]))
        nodes.append(_shp(own_shp, i))
        child_ids = [own_trn]
        for k in children:
            if k not in placed:
                child_ids.append(subtree(k))
        nodes.append(_grp(group, child_ids))
        return group_trn

    top = [subtree(i) for i in roots]
    # Parts caught in a parent cycle never hang off a root; put them at the top level.
    top += [subtree(i) for i in range(len(scene.parts)) if i not in placed]
    root = _trn(0, None, 1, -1)
    layer = _chunk(b"LAYR", struct.pack("<i", 0) + _dict({}) + struct.pack("<i", -1))
    palette = np.asarray(scene.palette, dtype=np.uint8)
    rgba = _chunk(b"RGBA", palette[1:256].tobytes() + bytes(4))
    children = b"".join(models) + root + _grp(1, top) + b"".join(nodes) + layer + rgba
    return b"VOX " + struct.pack("<i", VERSION) + _chunk(b"MAIN", b"", children)


def write_vox(path: str | os.PathLike, scene: VoxScene) -> None:
    with open(path, "wb") as f:
        f.write(encode_vox(scene))


# --- reading ----------------------------------------------------------------


class _Reader:
    def __init__(self, data: bytes, pos: int = 0):
        self.data, self.pos = data, pos

    def int32(self) -> int:
        (v,) = struct.unpack_from("<i", self.data, self.pos)
        self.pos += 4
        return v

    def string(self) -> str:
        n = self.int32()
        s = self.data[self.pos : self.pos + n].decode("utf-8", errors="replace")
        self.pos += n
        return s

    def dict(self) -> dict[str, str]:
        return {self.string(): self.string() for _ in range(self.int32())}


def _iter_chunks(data: bytes, pos: int, end: int):
    while pos < end:
        if pos + 12 > end:
            raise VoxError(f"truncated chunk header at byte {pos}")
        cid = data[pos : pos + 4]
        n, m = struct.unpack_from("<ii", data, pos + 4)
        if n < 0 or m < 0 or pos + 12 + n + m > end:
            raise VoxError(f"chunk {cid!r} at byte {pos} overruns the file")
        yield cid, pos + 12, n
        pos += 12 + n + m


def decode_vox(data: bytes) -> VoxScene:
    try:
        return _decode(data)
    except VoxError:
        raise
    except (struct.error, ValueError, IndexError, KeyError, RecursionError) as e:
        raise VoxError(f"malformed .vox data: {e!r}") from e


def _decode(data: bytes) -> VoxScene:
    if len(data) < 20 or data[:4] != b"VOX " or data[8:12] != b"MAIN":
        raise VoxError("not a .vox file")
    main_content, main_children = struct.unpack_from("<ii", data, 12)
    end = 20 + main_content + main_children
    if end > len(data):
        raise VoxError(f"file is truncated: MAIN declares {end} bytes, file has {len(data)}")
    models: list[np.ndarray] = []
    size = None
    nodes: dict[int, tuple[bytes, dict]] = {}
    palette = None
    for cid, start, n in _iter_chunks(data, 20 + main_content, end):
        r = _Reader(data, start)
        if cid == b"SIZE":
            size = (r.int32(), r.int32(), r.int32())
            if any(s <= 0 or s > MAX_MODEL_AXIS for s in size):
                raise VoxError(f"invalid model size {size}")
        elif cid == b"XYZI":
            if size is None:
                raise VoxError("XYZI chunk without a preceding SIZE")
            count = r.int32()
            if count < 0 or 4 + 4 * count > n:
                raise VoxError(f"XYZI declares {count} voxels but holds {(n - 4) // 4}")
            v = np.frombuffer(data, dtype=np.uint8, count=4 * count, offset=r.pos).reshape(-1, 4)
            grid = np.zeros(size, dtype=np.uint8)
            v = v[(v[:, 3] != 0) & np.all(v[:, :3] < np.array(size), axis=1)]
            grid[v[:, 0], v[:, 1], v[:, 2]] = v[:, 3]
            models.append(grid)
            size = None
        elif cid == b"RGBA":
            entries = np.frombuffer(data, dtype=np.uint8, count=1024, offset=start).reshape(256, 4)
            palette = np.zeros((256, 4), dtype=np.uint8)
            palette[1:] = entries[:255]
        elif cid in (b"nTRN", b"nGRP", b"nSHP"):
            node_id = r.int32()
            attrs = r.dict()
            if cid == b"nTRN":
                child, _reserved, _layer, n_frames = r.int32(), r.int32(), r.int32(), r.int32()
                frames = [r.dict() for _ in range(n_frames)]
                nodes[node_id] = (cid, {"attrs": attrs, "child": child, "frames": frames})
            elif cid == b"nGRP":
                children = [r.int32() for _ in range(r.int32())]
                nodes[node_id] = (cid, {"attrs": attrs, "children": children})
            else:
                shape_models = []
                for _ in range(r.int32()):
                    shape_models.append(r.int32())
                    r.dict()
                nodes[node_id] = (cid, {"attrs": attrs, "models": shape_models})
    if palette is None:
        palette = default_palette()

    if not nodes:
        parts = [VoxPart(f"model{i}", _from_mv(m, None)) for i, m in enumerate(models)]
        return VoxScene(parts, palette)

    parts: list[VoxPart] = []

    def visit(
        node_id: int, offset: np.ndarray, name: str | None, owners: tuple, hidden: bool = False
    ) -> None:
        kind, node = nodes[node_id]
        if kind == b"nTRN":
            frame = node["frames"][0] if node["frames"] else {}
            if frame.get("_r", _IDENTITY_ROTATION) != _IDENTITY_ROTATION:
                raise VoxError(f"rotated node {node_id} is not supported yet")
            t = np.array([int(v) for v in frame.get("_t", "0 0 0").split()])
            own = node["attrs"].get("_name") or name
            child_kind = nodes[node["child"]][0]
            inner = owners + (own,) if child_kind == b"nGRP" and own else owners
            hide = hidden or node["attrs"].get("_hidden") == "1"
            visit(node["child"], offset + t, own, inner, hide)
        elif kind == b"nGRP":
            for child in node["children"]:
                visit(child, offset, None, owners, hidden)
        else:
            model_id = node["models"][0]
            part_name = name or f"model{model_id}"
            # A shape named like the group it sits in is that group's own shape.
            chain = owners[:-1] if owners and owners[-1] == part_name else owners
            parent = chain[-1] if chain else None
            parts.append(VoxPart(part_name, _from_mv(models[model_id], offset), parent, hidden))

    visit(0, np.zeros(3, dtype=np.int64), None, ())
    known = {p.name for p in parts}
    for p in parts:
        if p.parent not in known:
            p.parent = None
    return VoxScene(parts, palette)


def _from_mv(mv: np.ndarray, t: np.ndarray | None) -> VoxelGrid:
    """A MagicaVoxel model array placed at translation t (None = min corner at origin)."""
    data = np.ascontiguousarray(np.transpose(mv[:, ::-1, :], (0, 2, 1)))
    if t is None:
        return VoxelGrid(data).cropped()
    mv_min = t - np.array(mv.shape) // 2
    depth = mv.shape[1]
    origin = (int(mv_min[0]), int(mv_min[2]), int(-mv_min[1] - depth))
    return VoxelGrid(data, origin).cropped()


def read_vox(path: str | os.PathLike) -> VoxScene:
    with open(path, "rb") as f:
        return decode_vox(f.read())
