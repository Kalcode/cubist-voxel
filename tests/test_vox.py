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


def test_round_trip_keeps_the_part_hierarchy():
    parts = [
        VoxPart("torso", grid({(0, 0, 0): 1})),
        VoxPart("head", grid({(0, 2, 0): 2}), parent="torso"),
        VoxPart("hat", grid({(0, 4, 0): 3}), parent="head"),
        VoxPart("arm", grid({(2, 0, 0): 4}), parent="torso"),
        VoxPart("flag", grid({(5, 0, 0): 5})),
    ]
    back = decode_vox(encode_vox(VoxScene(parts, P.rgba_table())))
    assert {p.name: p.parent for p in back.parts} == {
        "torso": None,
        "head": "torso",
        "hat": "head",
        "arm": "torso",
        "flag": None,
    }
    for a, b in zip(
        sorted(parts, key=lambda p: p.name), sorted(back.parts, key=lambda p: p.name), strict=True
    ):
        assert a.grid == b.grid


def test_parent_parts_are_groups_in_the_scene_graph():
    parts = [
        VoxPart("torso", grid({(0, 0, 0): 1})),
        VoxPart("head", grid({(0, 2, 0): 2}), parent="torso"),
    ]
    found = [cid for cid, _ in chunks(encode_vox(VoxScene(parts, P.rgba_table())))]
    assert found.count(b"nGRP") == 2  # root group plus torso's group


def _matl(material_id: int, props: dict[str, str]) -> bytes:
    def s(text: str) -> bytes:
        return struct.pack("<i", len(text)) + text.encode()

    body = b"".join(s(k) + s(v) for k, v in props.items())
    return _chunk(b"MATL", struct.pack("<ii", material_id, len(props)) + body)


def materials(data: bytes) -> dict[int, dict[str, str]]:
    """MATL chunks as {material id: properties}, parsed independently of the reader."""
    out = {}
    for cid, content in chunks(data):
        if cid != b"MATL":
            continue
        material_id, n = struct.unpack_from("<ii", content)
        pos, props = 8, {}
        for _ in range(n):
            pair = []
            for _ in range(2):
                (size,) = struct.unpack_from("<i", content, pos)
                pair.append(content[pos + 4 : pos + 4 + size].decode())
                pos += 4 + size
            props[pair[0]] = pair[1]
        out[material_id] = props
    return out


def test_glow_is_written_as_emissive_materials_on_its_color_indices():
    # MagicaVoxel's HDR emission is _emit * 10 ^ _flux; _flux is the smallest whole power.
    scene = VoxScene([VoxPart("a", grid({(0, 0, 0): 1}))], P.rgba_table(), glow={2: 0.6, 4: 1.5})
    assert materials(encode_vox(scene)) == {
        2: {"_type": "_emit", "_emit": "0.6", "_flux": "0"},
        4: {"_type": "_emit", "_emit": "0.15", "_flux": "1"},
    }


def test_no_glow_writes_no_materials():
    scene = VoxScene([VoxPart("a", grid({(0, 0, 0): 1}))], P.rgba_table())
    assert materials(encode_vox(scene)) == {}


def test_glow_round_trips_through_materials():
    glow = {1: 0.25, 2: 1.0, 3: 1.5, 5: 12.0}
    scene = VoxScene([VoxPart("a", grid({(0, 0, 0): 1}))], P.rgba_table(), glow=glow)
    assert decode_vox(encode_vox(scene)).glow == pytest.approx(glow)


def test_reads_glow_from_magicavoxel_materials():
    # MagicaVoxel writes a MATL for every palette entry, most of them diffuse. Material ids are
    # color indices, with 256 standing for index 0 (empty). Older files give the amount as _weight.
    size = _chunk(b"SIZE", struct.pack("<iii", 1, 1, 1))
    xyzi = _chunk(b"XYZI", struct.pack("<i", 1) + bytes([0, 0, 0, 7]))
    mats = [
        _matl(3, {"_type": "_diffuse", "_weight": "1", "_rough": "0.1"}),
        _matl(7, {"_type": "_emit", "_emit": "0.5", "_flux": "2", "_ldr": "0"}),
        _matl(9, {"_type": "_emit", "_weight": "0.4"}),
        _matl(11, {"_type": "_emit", "_emit": "0"}),
        _matl(256, {"_type": "_emit", "_emit": "1"}),
    ]
    body = size + xyzi + b"".join(mats)
    data = b"VOX " + struct.pack("<i", 200) + b"MAIN" + struct.pack("<ii", 0, len(body)) + body
    assert decode_vox(data).glow == pytest.approx({7: 50.0, 9: 0.4})


def test_unreadable_material_values_are_skipped_not_fatal():
    size = _chunk(b"SIZE", struct.pack("<iii", 1, 1, 1))
    xyzi = _chunk(b"XYZI", struct.pack("<i", 1) + bytes([0, 0, 0, 4]))
    bad = {"_type": "_emit", "_emit": "bright"}
    huge = {"_type": "_emit", "_emit": "1", "_flux": "1e999"}
    mats = _matl(4, bad) + _matl(5, huge)
    body = size + xyzi + mats
    data = b"VOX " + struct.pack("<i", 200) + b"MAIN" + struct.pack("<ii", 0, len(body)) + body
    scene = decode_vox(data)
    assert scene.glow == {} and scene.parts[0].grid.count() == 1
