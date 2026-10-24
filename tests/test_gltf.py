import io
import struct

import numpy as np
import pytest
from PIL import Image
from pygltflib import GLTF2

from cubist import Model, Part, asset, box
from cubist.core.palette import Palette
from cubist.io.gltf import encode_glb

P = Palette({"wood": ["#3b2414", "#8a5a32"], "metal": "#7d8491"})
S = 0.5  # metres per voxel, for easy arithmetic


def chest(p, sockets=None):
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
        sockets=sockets if sockets is not None else {"handle": (2, 5, 2)},
    )


def build(fn=chest, **kw):
    return asset(name="chest", size=(4, 5, 4))(fn).build(P, **kw)


def load(data: bytes) -> GLTF2:
    return GLTF2.load_from_bytes(data)


def accessor(g: GLTF2, index: int) -> np.ndarray:
    acc = g.accessors[index]
    view = g.bufferViews[acc.bufferView]
    blob = g.binary_blob()
    dtype = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}[acc.componentType]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc.type]
    start = view.byteOffset + (acc.byteOffset or 0)
    arr = np.frombuffer(blob, dtype=dtype, count=acc.count * width, offset=start)
    return arr.reshape(acc.count, width) if width > 1 else arr


def node(g: GLTF2, name: str):
    return next(n for n in g.nodes if n.name == name)


def children(g: GLTF2, name: str) -> list[str]:
    return [g.nodes[i].name for i in (node(g, name).children or [])]


def test_glb_container_header():
    data = encode_glb(build(), P.rgba_table(), voxel_size_m=S)
    magic, version, length = struct.unpack_from("<4sII", data)
    assert (magic, version, length) == (b"glTF", 2, len(data))


def test_parts_become_a_node_hierarchy_with_sockets():
    g = load(encode_glb(build(), P.rgba_table(), voxel_size_m=S))
    root = g.nodes[g.scenes[g.scene].nodes[0]]
    assert root.name == "chest"
    assert children(g, "chest") == ["body"]
    assert children(g, "body") == ["lid"]
    assert children(g, "lid") == ["handle"]  # nearest part owns the socket
    assert node(g, "handle").mesh is None


def test_nodes_sit_at_their_pivots_relative_to_their_parents():
    g = load(encode_glb(build(), P.rgba_table(), voxel_size_m=S, center=False))
    assert node(g, "body").translation == [1.0, 0.0, 1.0]
    assert node(g, "lid").translation == [-1.0, 2.0, -1.0]
    assert node(g, "handle").translation == [1.0, 0.5, 1.0]


def test_centering_puts_the_asset_footprint_on_the_origin():
    g = load(encode_glb(build(), P.rgba_table(), voxel_size_m=S))
    assert node(g, "chest").translation == [-1.0, 0.0, -1.0]


def test_mesh_vertices_are_in_node_space_and_metres():
    g = load(encode_glb(build(), P.rgba_table(), voxel_size_m=S))
    prim = g.meshes[node(g, "body").mesh].primitives[0]
    pos = accessor(g, prim.attributes.POSITION)
    assert pos.min(axis=0).tolist() == [-1.0, 0.0, -1.0]
    assert pos.max(axis=0).tolist() == [1.0, 2.0, 1.0]
    assert g.accessors[prim.attributes.POSITION].min == [-1.0, 0.0, -1.0]


def test_one_color_box_is_twelve_triangles():
    g = load(encode_glb(build(), P.rgba_table(), voxel_size_m=S))
    prim = g.meshes[node(g, "body").mesh].primitives[0]
    assert len(accessor(g, prim.indices)) == 36


def test_triangles_wind_counter_clockwise_around_their_normals():
    g = load(encode_glb(build(), P.rgba_table(), voxel_size_m=S))
    prim = g.meshes[node(g, "body").mesh].primitives[0]
    pos = accessor(g, prim.attributes.POSITION)
    nrm = accessor(g, prim.attributes.NORMAL)
    tri = accessor(g, prim.indices).reshape(-1, 3)
    cross = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])
    assert np.all(np.sum(cross * nrm[tri[:, 0]], axis=1) > 0)


def test_palette_texture_and_uvs_point_at_palette_cells():
    g = load(encode_glb(build(), P.rgba_table(), voxel_size_m=S))
    image = g.images[0]
    assert image.mimeType == "image/png"
    view = g.bufferViews[image.bufferView]
    png = g.binary_blob()[view.byteOffset : view.byteOffset + view.byteLength]
    pal = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))
    assert pal.shape == (1, 256, 4)
    assert tuple(pal[0, P.wood.light.index]) == (0x8A, 0x5A, 0x32, 255)
    prim = g.meshes[node(g, "lid").mesh].primitives[0]
    uv = accessor(g, prim.attributes.TEXCOORD_0)
    assert np.allclose(uv[:, 0], (P.wood.light.index + 0.5) / 256)
    sampler = g.samplers[0]
    assert (sampler.magFilter, sampler.minFilter) == (9728, 9728)  # nearest


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def test_vertex_colors_are_linear_and_drop_the_texture():
    g = load(encode_glb(build(), P.rgba_table(), voxel_size_m=S, vertex_colors=True))
    assert not g.textures
    prim = g.meshes[node(g, "body").mesh].primitives[0]
    colors = accessor(g, prim.attributes.COLOR_0)
    assert g.accessors[prim.attributes.COLOR_0].componentType == 5126  # float, no banding
    assert np.allclose(colors[:, :3], srgb_to_linear((0x3B, 0x24, 0x14)), atol=1e-6)
    assert np.allclose(colors[:, 3], 1.0)


def test_ao_option_bakes_occlusion_into_vertex_colors():
    def post_on_floor(p):
        return box((5, 1, 5), color=p.wood.light) | box((1, 3, 1), color=p.metal).at(2, 1, 2)

    a = asset(name="post")(post_on_floor).build(P)
    g = load(encode_glb(a, P.rgba_table(), voxel_size_m=S, ao=True))
    prim = g.meshes[0].primitives[0]
    colors = accessor(g, prim.attributes.COLOR_0)
    assert colors[:, 0].min() < 1.0 and colors[:, 0].max() == 1.0
    assert g.textures  # AO multiplies the palette texture


def test_sealed_cavities_are_not_meshed():
    solid = asset(name="s")(lambda p: box((5, 5, 5), color=p.wood.dark)).build(P)
    cavity = asset(name="c")(
        lambda p: box((5, 5, 5), color=p.wood.dark) - box((3, 3, 3)).at(1, 1, 1)
    ).build(P)

    def tris(a, **kw):
        g = load(encode_glb(a, P.rgba_table(), voxel_size_m=S, **kw))
        return len(accessor(g, g.meshes[0].primitives[0].indices)) // 3

    assert tris(cavity) == tris(solid) == 12
    assert tris(cavity, hollow=False) > 12


def test_socket_with_an_explicit_part():
    a = build(lambda p: chest(p, sockets={"keyhole": (2, 2, 4, "body")}))
    g = load(encode_glb(a, P.rgba_table(), voxel_size_m=S, center=False))
    assert "keyhole" in children(g, "body")
    assert node(g, "keyhole").translation == [0.0, 1.0, 1.0]


def test_empty_part_becomes_a_node_without_a_mesh():
    def fn(p):
        return Model(
            parts=[
                Part("body", box((2, 2, 2), color=p.wood.dark)),
                Part("ghost", box((2, 2, 2), color=p.wood.dark) - box((3, 3, 3))),
            ]
        )

    g = load(encode_glb(asset(name="g")(fn).build(P), P.rgba_table(), voxel_size_m=S))
    assert node(g, "ghost").mesh is None


def test_encoding_is_deterministic():
    a = build()
    assert encode_glb(a, P.rgba_table()) == encode_glb(a, P.rgba_table())


def test_bad_voxel_size_is_an_error():
    with pytest.raises(ValueError, match="voxel_size_m"):
        encode_glb(build(), P.rgba_table(), voxel_size_m=0)
