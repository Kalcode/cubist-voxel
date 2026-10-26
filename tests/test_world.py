"""Worlds (design.md, "Worlds", milestone 12a): height fields, surface rules, the world map."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cubist.core.palette import Palette
from cubist.world import (
    World,
    WorldError,
    build_world,
    circle,
    flat,
    height_map,
    height_over,
    load_world,
    noise,
    rect,
    ridges,
    slope_over,
    world,
)
from cubist.world.sample import columns, surface_names

ROOT = Path(__file__).parent.parent
DEMO = ROOT / "projects" / "demo"


def _palette() -> Palette:
    return Palette({"grass": "#4a8a3a", "stone": "#6b6b75", "snow": "#f4f1e8", "dirt": "#5e3a1f"})


def _grid(n=16, size=100.0):
    x, z = np.meshgrid(np.linspace(0, size, n), np.linspace(0, size, n))
    return x, z


def test_noise_is_deterministic_seeded_and_scaled():
    x, z = _grid()
    a = noise(scale_m=50, amplitude_m=10, octaves=3).sample(x, z, seed=1)
    b = noise(scale_m=50, amplitude_m=10, octaves=3).sample(x, z, seed=1)
    c = noise(scale_m=50, amplitude_m=10, octaves=3).sample(x, z, seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.shape == x.shape
    assert np.abs(a).max() <= 10.0 + 1e-9
    assert np.abs(a).max() > 1.0  # not flat
    r = ridges(scale_m=50, amplitude_m=10).sample(x, z, seed=1)
    assert r.min() >= 0.0 and r.max() <= 10.0


def test_fields_combine_with_arithmetic_blend_and_flatten():
    x, z = _grid()
    f = flat(5) + flat(3) * 2 - 1
    assert np.allclose(f.sample(x, z, seed=0), 10)
    # Blend by a mask: inside the circle the second field, outside the first.
    blended = flat(0).blend(flat(100), mask=circle((50, 50), radius_m=20, edge_m=0))
    h = blended.sample(np.array([50.0, 95.0]), np.array([50.0, 95.0]), seed=0)
    assert h.tolist() == [100.0, 0.0]
    # A soft edge ramps between them.
    soft = flat(0).blend(flat(100), mask=circle((50, 50), radius_m=20, edge_m=10))
    mid = soft.sample(np.array([75.0]), np.array([50.0]), seed=0)[0]
    assert 0 < mid < 100
    # Flatten sets an area's height and leaves the rest.
    hilly = noise(scale_m=30, amplitude_m=20).flatten(rect((10, 10), (40, 40), edge_m=0), at_m=7)
    inside = hilly.sample(np.array([20.0, 30.0]), np.array([20.0, 25.0]), seed=3)
    assert np.allclose(inside, 7)


def test_height_maps_sample_bilinearly(tmp_path):
    # A 2 × 2 greyscale map: black on the left, white on the right.
    Image.fromarray(np.array([[0, 255], [0, 255]], dtype=np.uint8)).save(tmp_path / "h.png")
    f = height_map(tmp_path / "h.png", low_m=10, high_m=20, size_m=(100, 100))
    h = f.sample(np.array([25.0, 50.0, 75.0]), np.array([50.0, 50.0, 50.0]), seed=0)
    assert np.allclose(h, [10, 15, 20])  # pixel centers at 25 and 75 m
    with pytest.raises(WorldError, match="no such height map"):
        height_map(tmp_path / "missing.png", low_m=0, high_m=1).sample(
            np.zeros(1), np.zeros(1), seed=0
        )


def test_surface_rules_first_match_wins():
    p = _palette()

    @world(name="slope", size_m=(64, 64))
    def slope(p):
        # 1 m up per metre east (45°) for x < 32, then flat at 32 m.
        ground = flat(0) + _ramp()
        return World(
            ground=ground,
            surface=[height_over(31.5, p.snow), slope_over(30, p.stone), p.grass],
            subsoil=[p.dirt],
        )

    w = load_world(slope, p, seed=0)
    h, s, names = columns(w, np.array([10.0, 40.0]), np.array([20.0, 20.0]))
    assert np.allclose(h, [10, 32])
    assert s[0] == pytest.approx(45, abs=1)
    assert surface_names(w, names) == ["stone", "snow"]


class _Ramp:
    def sample(self, x, z, seed):
        return np.minimum(np.asarray(x, dtype=np.float64), 32.0)


def _ramp():
    from cubist.world.field import Custom

    return Custom(_Ramp().sample, "ramp")


def test_world_scripts_need_a_ground_and_a_size():
    with pytest.raises(WorldError, match="size_m"):
        world(name="bad", size_m=(0, 10))(lambda p: None)

    @world(name="nothing", size_m=(64, 64))
    def nothing(p):
        return None

    with pytest.raises(WorldError, match="return a World"):
        load_world(nothing, _palette(), seed=0)


def test_build_writes_a_map_and_stats_deterministically(tmp_path):
    script = DEMO / "worlds" / "islet.py"
    out1 = build_world(script, out_dir=tmp_path / "a")
    out2 = build_world(script, out_dir=tmp_path / "b")
    a = (tmp_path / "a" / "islet" / "map.png").read_bytes()
    b = (tmp_path / "b" / "islet" / "map.png").read_bytes()
    assert a == b
    stats = out1.stats
    assert stats["name"] == "islet"
    assert stats["height_m"]["max"] > stats["height_m"]["min"]
    assert abs(sum(stats["surface_pct"].values()) - 100) < 0.01
    assert set(stats["slope_pct"]) == {"0-15", "15-30", "30-45", "45+"}
    assert out2.map_path.exists()


def test_islet_map_matches_golden():
    from test_golden import GOLDEN, _compare

    out = build_world(DEMO / "worlds" / "islet.py", write=False)
    _compare(out.map_image, GOLDEN / "world_islet.png")


def test_cli_world_build(tmp_path, capsys):
    from cubist.cli import main

    assert main(["world", "build", str(DEMO / "worlds" / "islet.py"), "--out", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert "built world islet" in printed and "surface:" in printed
    assert (tmp_path / "islet" / "map.png").exists() and (
        tmp_path / "islet" / "stats.json"
    ).exists()
    assert main(["world", "build", str(tmp_path / "nope.py")]) == 1
    assert "no such world script" in capsys.readouterr().err


def test_chunk_payloads_round_trip():
    from cubist.world.bake import decode_chunk, encode_chunk

    rng = np.random.default_rng(3)
    cells = np.zeros((32, 32, 32), dtype=np.uint16)
    cells[:10] = 7  # a few uniform bricks
    cells[10:20] = rng.integers(0, 4, size=(10, 32, 32))  # mixed ones, air included
    payload = encode_chunk(cells)
    assert payload[:2] == b"\x01\x00"
    assert np.array_equal(decode_chunk(payload), cells)
    layered = np.zeros((32, 32, 32), dtype=np.uint16)
    layered[:, :8] = 3  # every brick uniform, but not the same
    assert np.array_equal(decode_chunk(encode_chunk(layered)), layered)
    uniform = np.full((32, 32, 32), 9, dtype=np.uint16)
    assert encode_chunk(uniform) == b"\x01\x01\x09\x00"
    assert np.array_equal(decode_chunk(encode_chunk(uniform)), uniform)


def test_bake_writes_kiln_tiles_matching_the_ground(tmp_path):
    import json
    import struct
    import zlib

    from cubist.world.bake import N, bake_world, decode_chunk
    from cubist.world.build import load_script
    from cubist.world.sample import heights

    # A one-tile world (128 m) keeps the test quick.
    script = tmp_path / "tiny.py"
    script.write_text(
        "from cubist.world import *\n\n"
        "@world(name='tiny', size_m=(128, 128), seed=2)\n"
        "def tiny(p):\n"
        "    ground = flat(20) + noise(scale_m=40, amplitude_m=8, octaves=3)\n"
        "    return World(ground=ground, surface=[slope_over(35, p.stone.mid), p.leaf.mid],\n"
        "                 subsoil=[(1, p.wood[1]), p.stone.dark], landmarks={'spawn': (64, 64)})\n"
    )
    style = DEMO / "style.toml"
    summary = bake_world(script, out_dir=tmp_path / "a", levels=2, workers=1, style=style)
    info = json.loads((tmp_path / "a" / "map.json").read_text())
    assert info["levels"] == 2 and info["compression"] == "deflate" and info["max_x"] == 128 * 16
    assert {m["name"] for m in info["material_registry"]} >= {"leaf.mid", "stone.dark"}
    assert len(info["far"]) == 6 and summary.chunks > 0
    world_json = json.loads((tmp_path / "a" / "world.json").read_text())
    assert set(world_json["landmarks"]) == {"spawn"}

    # Column (40, 50) of level 0 (2 m chunks): its top solid cell sits at the ground's height.
    tile = (tmp_path / "a" / "L0" / "0_0.tile").read_bytes()
    assert tile[:4] == b"KTIL"
    cx, cz = 40, 50
    min_cy, max_cy, below, _, off, length, raw_len = struct.unpack_from(
        "<iiHHIII", tile, 16 + 24 * (cx + 64 * cz)
    )
    raw = zlib.decompress(tile[off : off + length])
    loaded = load_script(script, style=style)
    assert len(raw) == raw_len and below == loaded.subsoil_layers()[1].index
    column = []
    at = 0
    for _ in range(min_cy, max_cy):
        n = struct.unpack_from("<I", raw, at)[0]
        column.append(decode_chunk(raw[at + 4 : at + 4 + n]))
        at += 4 + n
    solid = np.concatenate(column, axis=1)[:, :, :] != 0  # [z, y, x] stacked along y
    top = min_cy * N + solid[0, :, 0].nonzero()[0].max() + 1  # the first empty voxel above
    x_m, z_m = (40 * N + 0.5) / 16, (50 * N + 0.5) / 16
    expected = heights(loaded, np.array([x_m]), np.array([z_m]))[0] * 16
    assert abs(top - expected) <= 1

    again = bake_world(script, out_dir=tmp_path / "b", levels=2, workers=1, style=style)
    assert (tmp_path / "b" / "L0" / "0_0.tile").read_bytes() == tile
    assert again.chunks == summary.chunks


def test_paths_measure_distance_and_arc_length():
    from cubist.world import path

    line = path([(0, 0), (100, 0)], smooth=False)
    d, t = line.nearest(
        np.array([50.0, 50.0, 150.0, -5.0]), np.array([0.0, 7.0, 0.0, 3.0]), reach=10
    )
    assert d[:2].tolist() == [0.0, 7.0] and np.isinf(d[2])
    assert t[0] == 50 and t[3] == 0 and d[3] == pytest.approx(np.hypot(5, 3))
    curve = path([(0, 0), (50, 40), (100, 0)])
    assert len(curve.points) > 3 and curve.length > 128  # smoothed through the middle point


def _river_world(p):
    from cubist.world import path, river, road
    from cubist.world.field import Custom

    ground = flat(30) + Custom(
        lambda x, z, s: -np.asarray(x) * 0.05, "tilt"
    )  # falls 5 m per 100 m east
    return World(
        ground=ground,
        surface=[p.leaf.mid],
        subsoil=[p.stone.dark],
        rivers=[
            river(
                path([(10, 60), (240, 60)], smooth=False),
                width_m=8,
                depth_m=2,
                bed=p.stone.light,
                water=p.water.mid,
            )
        ],  # fmt: skip
        roads=[road(path([(60, 0), (60, 50)], smooth=False), width_m=4, color=p.cork)],
    )


def test_rivers_cut_a_channel_that_holds_water_and_roads_level_a_strip():
    from cubist.world.sample import heights, surface_names, water

    w = world(name="river", size_m=(256, 128))(_river_world)
    p = _demo_palette()
    loaded = load_world(w, p, seed=0)
    xs = np.array([50.0, 150.0, 230.0])
    centre = heights(loaded, xs, np.full(3, 60.0))
    bank_far = heights(loaded, xs, np.full(3, 90.0))
    assert np.all(centre < bank_far - 1.5)  # the channel is cut
    level, color = water(loaded, xs, np.full(3, 60.0))
    assert np.all(level > centre) and np.all(np.diff(level) <= 0)  # water in it, flowing downhill
    assert (color == p.water.mid.index).all()
    assert np.isinf(water(loaded, np.array([50.0]), np.array([90.0]))[0]).all()
    # The road is level across its width and painted.
    across = heights(loaded, np.array([58.5, 60.0, 61.5]), np.full(3, 25.0))
    assert np.ptp(across) < 0.05
    _, _, names = columns(loaded, np.array([60.0, 100.0]), np.array([25.0, 25.0]))
    assert surface_names(loaded, names) == ["cork", "leaf.mid"]


def test_baked_rivers_hold_water_voxels(tmp_path):
    import struct
    import zlib

    from cubist.world.bake import N, bake_tile, decode_chunk, layers_of

    p = _demo_palette()
    loaded = load_world(world(name="river", size_m=(256, 128))(_river_world), p, seed=0)
    bake_tile(loaded, tmp_path, 0, 0, 0, layers_of(loaded))
    tile = (tmp_path / "L0" / "0_0.tile").read_bytes()
    cx, cz = (
        int(150 * 16) // N % 64,
        int(60 * 16) // N % 64,
    )  # the chunk column over the river at x 150
    min_cy, max_cy, _, _, off, length, _ = struct.unpack_from(
        "<iiHHIII", tile, 16 + 24 * (cx + 64 * cz)
    )
    raw = zlib.decompress(tile[off : off + length])
    cells, at = [], 0
    for _ in range(min_cy, max_cy):
        n = struct.unpack_from("<I", raw, at)[0]
        cells.append(decode_chunk(raw[at + 4 : at + 4 + n]))
        at += 4 + n
    column = np.concatenate(cells, axis=1)
    assert (column == p.water.mid.index).any()
    assert (column == p.stone.light.index).any()  # the bed under it


def _demo_palette() -> Palette:
    from cubist.build.runner import load_project_style

    return load_project_style(DEMO / "style.toml").palette
