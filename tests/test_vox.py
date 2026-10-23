import struct

import numpy as np
import pytest

from cubist.core.grid import VoxelGrid
from cubist.core.palette import Palette
from cubist.io.vox import (
    VoxError,
    VoxPart,
    VoxScene,
    decode_vox,
    default_palette,
    encode_vox,
    read_vox,
    write_vox,
)

P = Palette({"wood": ["#3b2414", "#5e3a1f", "#8a5a32"], "metal": ["#2a2d34", "#7d8491"]})


def grid(points):
    coords = np.array(list(points.keys()))
    colors = np.array(list(points.values()), dtype=np.uint8)
    return VoxelGrid.from_voxels(coords, colors)


def chunks(data: bytes):
    """Top-level chunks inside MAIN as (id, content) pairs, parsed independently of the reader."""
    assert data[:4] == b"VOX "
    assert data[8:12] == b"MAIN"
    content_size, children_size = struct.unpack_from("<ii", data, 12)
    assert content_size == 0
    assert 20 + children_size == len(data)
    out, pos = [], 20
    while pos < len(data):
        cid = data[pos : pos + 4]
        n, m = struct.unpack_from("<ii", data, pos + 4)
        out.append((cid, data[pos + 12 : pos + 12 + n]))
        pos += 12 + n + m
    return out


def test_header_and_version():
    data = encode_vox(VoxScene([VoxPart("a", grid({(0, 0, 0): 1}))], P.rgba_table()))
    assert data[:4] == b"VOX "
    assert struct.unpack_from("<i", data, 4)[0] == 200


def test_y_up_grid_is_written_z_up_with_front_at_low_y():
    # A column going up (DSL +Y) and one voxel at the front (DSL max Z).
    g = grid({(0, 0, 0): 1, (0, 3, 0): 2, (0, 0, 1): 3})
    data = encode_vox(VoxScene([VoxPart("a", g)], P.rgba_table()))
    found = dict(chunks(data))
    assert struct.unpack("<iii", found[b"SIZE"]) == (1, 2, 4)  # x, y(depth), z(height)
    (n,) = struct.unpack_from("<i", found[b"XYZI"])
    voxels = {
        tuple(found[b"XYZI"][4 + 4 * i : 7 + 4 * i]): found[b"XYZI"][7 + 4 * i] for i in range(n)
    }
    assert voxels == {(0, 1, 0): 1, (0, 1, 3): 2, (0, 0, 0): 3}


def test_rgba_chunk_maps_entry_k_to_color_index_k_plus_one():
    data = encode_vox(VoxScene([VoxPart("a", grid({(0, 0, 0): 1}))], P.rgba_table()))
    rgba = dict(chunks(data))[b"RGBA"]
    assert len(rgba) == 1024
    assert tuple(rgba[0:4]) == (0x3B, 0x24, 0x14, 255)  # color index 1
    assert tuple(rgba[16:20]) == (0x7D, 0x84, 0x91, 255)  # color index 5


def test_round_trip_single_part_is_exact(tmp_path):
    g = grid({(x, y, z): 1 + (x + y + z) % 5 for x in range(3) for y in range(5) for z in range(2)})
    scene = VoxScene([VoxPart("crate", g)], P.rgba_table())
    write_vox(tmp_path / "a.vox", scene)
    back = read_vox(tmp_path / "a.vox")
    assert [p.name for p in back.parts] == ["crate"]
    assert back.parts[0].grid == g
    assert np.array_equal(back.palette, scene.palette)


def test_round_trip_keeps_part_names_and_world_positions():
    body = grid({(x, y, 0): 1 for x in range(4) for y in range(3)})
    lid = grid({(1, 7, -2): 4, (2, 7, -2): 5, (-3, 8, 5): 2})
    scene = VoxScene([VoxPart("body", body), VoxPart("lid", lid)], P.rgba_table())
    back = decode_vox(encode_vox(scene))
    assert [p.name for p in back.parts] == ["body", "lid"]
    assert back.parts[0].grid == body
    assert back.parts[1].grid == lid


def test_odd_and_even_sizes_round_trip_at_every_offset():
    for size in (1, 2, 3, 4):
        for offset in (-3, -2, 0, 1, 2):
            g = grid({(offset, offset, offset + i): 1 for i in range(size)})
            back = decode_vox(encode_vox(VoxScene([VoxPart("a", g)], P.rgba_table())))
            assert back.parts[0].grid == g, (size, offset)


def test_encoding_is_deterministic():
    scene = VoxScene([VoxPart("a", grid({(0, 0, 0): 1, (2, 1, 0): 3}))], P.rgba_table())
    assert encode_vox(scene) == encode_vox(scene)


def test_part_over_256_on_an_axis_is_rejected():
    g = grid({(0, 0, 0): 1, (256, 0, 0): 1})
    with pytest.raises(VoxError, match="256"):
        encode_vox(VoxScene([VoxPart("long", g)], P.rgba_table()))


def test_empty_part_is_rejected():
    empty = VoxelGrid(np.zeros((2, 2, 2), dtype=np.uint8))
    with pytest.raises(VoxError, match="empty"):
        encode_vox(VoxScene([VoxPart("ghost", empty)], P.rgba_table()))


def _chunk(cid: bytes, content: bytes) -> bytes:
    return cid + struct.pack("<ii", len(content), 0) + content


def test_reads_plain_file_without_scene_graph_or_palette():
    size = _chunk(b"SIZE", struct.pack("<iii", 2, 1, 3))
    xyzi = _chunk(b"XYZI", struct.pack("<i", 2) + bytes([0, 0, 0, 7, 1, 0, 2, 9]))
    unknown = _chunk(b"NOTE", b"\x00" * 12)
    body = size + xyzi + unknown
    data = b"VOX " + struct.pack("<i", 150) + b"MAIN" + struct.pack("<ii", 0, len(body)) + body
    scene = decode_vox(data)
    assert len(scene.parts) == 1
    g = scene.parts[0].grid
    # MV (x, y, z) -> DSL (x, z, depth - 1 - y); depth = 1 here.
    assert g == grid({(0, 0, 0): 7, (1, 2, 0): 9})
    assert np.array_equal(scene.palette, default_palette())


def test_default_palette_layout():
    pal = default_palette()
    assert pal.shape == (256, 4)
    assert tuple(pal[0]) == (0, 0, 0, 0)
    assert tuple(pal[1]) == (255, 255, 255, 255)
    assert tuple(pal[2]) == (255, 255, 204, 255)
    assert tuple(pal[215]) == (0, 0, 51, 255)
    assert tuple(pal[216]) == (238, 0, 0, 255)
    assert tuple(pal[255]) == (17, 17, 17, 255)


def test_garbage_is_a_vox_error():
    with pytest.raises(VoxError):
        decode_vox(b"not a vox file at all")


def test_truncated_file_is_a_vox_error():
    data = encode_vox(VoxScene([VoxPart("a", grid({(0, 0, 0): 1, (1, 1, 1): 2}))], P.rgba_table()))
    for cut in (40, 60, 100):
        with pytest.raises(VoxError):
            decode_vox(data[:cut] + data[cut + 1 :])


def test_scene_graph_cycle_is_a_vox_error():
    size = _chunk(b"SIZE", struct.pack("<iii", 1, 1, 1))
    xyzi = _chunk(b"XYZI", struct.pack("<i", 1) + bytes([0, 0, 0, 1]))
    empty = struct.pack("<i", 0)
    # A root transform whose child is itself.
    loop = _chunk(
        b"nTRN", struct.pack("<i", 0) + empty + struct.pack("<iiii", 0, -1, -1, 1) + empty
    )
    body = size + xyzi + loop
    data = b"VOX " + struct.pack("<i", 150) + b"MAIN" + struct.pack("<ii", 0, len(body)) + body
    with pytest.raises(VoxError):
        decode_vox(data)
