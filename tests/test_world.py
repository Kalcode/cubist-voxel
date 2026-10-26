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
    assert set(summary.level_seconds) == {0, 1} and "L1" in summary.levels_line()
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


def test_ground_steps_snap_baked_heights():
    from cubist.world.bake import sample_columns
    from cubist.world.field import Custom

    def ramp(step_m):
        @world(name="ramp", size_m=(64, 64))
        def ramp(p):
            slope = Custom(lambda x, z, s: 10 + np.asarray(x) * 0.3 + np.asarray(z) * 0.0, "ramp")
            return World(ground=slope, surface=[p.leaf.mid], step_m=step_m)

        return load_world(ramp, _demo_palette())

    fine = sample_columns(ramp(None), 0, 0, 0, 64, 4)[0]
    steps = sample_columns(ramp(0.25), 0, 0, 0, 64, 4)[0]
    assert len(np.unique(fine)) > 60  # one-voxel steps, nearly every column
    assert np.all(steps % 4 == 0) and np.abs(steps - fine).max() <= 2  # 4-voxel steps, rounded
    with pytest.raises(WorldError, match="step_m"):
        ramp(-1)


def test_coarse_levels_sample_fields_every_two_cells():
    from cubist.world.bake import sample_columns
    from cubist.world.field import Custom

    sizes = []

    def ramp_field(x, z, s):
        sizes.append(np.size(x))
        return 10 + np.asarray(x) * 0.3 + np.asarray(z) * 0.1

    @world(name="ramp", size_m=(512, 512))
    def ramp(p):
        return World(ground=Custom(ramp_field, "ramp"), surface=[p.leaf.mid])

    loaded = load_world(ramp, _demo_palette())
    h = sample_columns(loaded, 3, 0, 0, 64, 32)[0]  # cells of 0.5 m
    assert max(sizes) < (64 + 2) * (32 + 2) / 3  # a lattice every 1 m, not every cell
    centers = (np.arange(64) + 0.5) * 0.5
    rows = (np.arange(32) + 0.5) * 0.5
    expected = (10 + centers[None, :] * 0.3 + rows[:, None] * 0.1) * 16
    assert np.allclose(h, expected)  # interpolation keeps a ramp exact


def test_region_bakes_redo_only_the_tiles_over_the_area(tmp_path):
    from cubist.world.bake import bake_world

    script = tmp_path / "pair.py"
    script.write_text(
        "from cubist.world import *\n\n"
        "@world(name='pair', size_m=(256, 256), seed=2)\n"
        "def pair(p):\n"
        "    return World(ground=flat(20) + noise(scale_m=40, amplitude_m=8, octaves=3),\n"
        "                 surface=[p.leaf.mid])\n"
    )
    style, out = DEMO / "style.toml", tmp_path / "out"
    region = ((10, 10), (100, 60))  # inside level 0's tile (0, 0) and level 1's only tile
    with pytest.raises(WorldError, match="existing bake"):
        bake_world(script, out_dir=out, levels=2, workers=1, style=style, region=region)
    bake_world(script, out_dir=out, levels=2, workers=1, style=style)
    tiles = {p: p.read_bytes() for p in out.rglob("*.tile")}
    for p in tiles:
        p.write_bytes(b"stale")
    part = bake_world(script, out_dir=out, levels=2, workers=1, style=style, region=region)
    assert part.tiles == 2 and abs(part.km2 - 128 * 128 / 1e6) < 1e-9  # per level-0 tile baked
    redone = {p for p in tiles if p.read_bytes() != b"stale"}
    assert redone == {out / "L0" / "0_0.tile", out / "L1" / "0_0.tile"}
    assert all(p.read_bytes() == tiles[p] for p in redone)  # the same bytes as the whole bake


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


def test_carvers_say_where_the_air_is():
    from cubist.world import cavern, tunnel, worms

    t = tunnel([(0, 10, 0), (20, 10, 0)], radius_m=2, smooth=False)
    air = t.carve(
        np.array([10.0, 10.0, 10.0]), np.array([10.0, 11.9, 12.1]), np.zeros(3), 0, seed=0
    )
    assert air.tolist() == [True, True, False]
    c = cavern((50, 20, 50), radius_m=(8, 4, 8))
    assert c.carve(
        np.array([50.0, 57.0, 50.0]), np.array([20.0, 20.0, 25.0]), np.full(3, 50.0), 0, 0
    ).tolist() == [True, True, False]
    rough = cavern((50, 20, 50), radius_m=8, roughness_m=2, seed=1)
    probe = np.linspace(40, 60, 41)
    assert (
        rough.carve(probe, np.full(41, 20.0), np.full(41, 50.0), 0, 0).sum()
        != c.carve(probe, np.full(41, 20.0), np.full(41, 50.0), 0, 0).sum()
    )
    w = worms(((0, 0), (100, 100)), depth_m=(4, 20), scale_m=16, thickness=0.2, seed=3)
    xs, ys, zs = np.meshgrid(
        np.linspace(1, 99, 30), np.linspace(-40, 60, 30), np.linspace(1, 99, 30)
    )
    hollow = w.carve(xs, ys, zs, 30.0, seed=0)
    assert hollow.any()
    depth = 30.0 - ys[hollow]
    assert depth.min() >= 4 and depth.max() <= 20  # only in the band under the ground
    assert np.array_equal(hollow, w.carve(xs, ys, zs, 30.0, seed=0))


def test_worms_leave_the_floors_of_tunnels_and_caverns():
    from cubist.world import cavern, tunnel, worms

    @world(name="floors", size_m=(100, 100))
    def floors(p):
        return World(
            ground=flat(40),
            surface=[p.leaf.mid],
            caves=[
                tunnel([(10, 20, 50), (90, 20, 50)], radius_m=2.5, smooth=False),
                cavern((50, 15, 20), radius_m=(8, 4, 8)),
                worms(((0, 0), (100, 100)), depth_m=(4, 36), scale_m=12, thickness=0.5),
            ],
        )

    loaded = load_world(floors, _demo_palette(), seed=0)
    w = loaded.world.caves[2]
    # Under the tunnel's floor, 2 m of rock stay (measured out from the tunnel's wall)...
    x, angle, out = np.meshgrid(
        np.linspace(12, 88, 60), np.linspace(-1, 1, 9), np.linspace(0.1, 1.9, 7)
    )
    r = 2.5 + out
    y, z = 20 - r * np.cos(angle), 50 + r * np.sin(angle)
    assert not w.carve(x, y, z, 40.0, 0).any()
    # ...and worms carve plenty elsewhere, opening into its walls over it.
    x, z = np.meshgrid(np.linspace(12, 88, 60), np.linspace(44, 56, 13))
    assert w.carve(x, np.full_like(x, 25.0), z, 40.0, 0).any()
    cx, cz = np.meshgrid(np.linspace(44, 56, 13), np.linspace(14, 26, 13))
    under = np.hypot(cx - 50, cz - 20) <= 7  # under the cavern's floor
    assert not w.carve(cx, np.full_like(cx, 10.5), cz, 40.0, 0)[under].any()
    # Opting out cuts the floors.
    loose = worms(((0, 0), (100, 100)), depth_m=(4, 36), scale_m=12, thickness=0.5, floors=False)
    loose.keep_floors(loaded.world.caves)
    assert loose.carve(x, np.full_like(x, 16.5), z, 40.0, 0)[np.abs(z - 50) <= 2].any()


def test_baked_caves_are_hollow_below_the_surface(tmp_path):
    import struct
    import zlib

    from cubist.world import cavern, tunnel
    from cubist.world.bake import N, bake_tile, decode_chunk, layers_of

    @world(name="caves", size_m=(128, 128))
    def caves(p):
        return World(
            ground=flat(40),
            surface=[p.leaf.mid],
            subsoil=[p.stone.dark],
            caves=[
                cavern((64, 25, 64), radius_m=6),
                tunnel([(64, 25, 64), (64, 25, 120)], radius_m=2),
            ],
        )

    loaded = load_world(caves, _demo_palette(), seed=0)
    bake_tile(loaded, tmp_path, 0, 0, 0, layers_of(loaded))
    tile = (tmp_path / "L0" / "0_0.tile").read_bytes()
    cx = cz = 64 * 16 // N  # the chunk column over the cavern's middle
    min_cy, max_cy, _, _, off, length, _ = struct.unpack_from(
        "<iiHHIII", tile, 16 + 24 * (cx + 64 * cz)
    )
    assert min_cy * N <= (25 - 6) * 16  # the bounds reach down to the cavern's floor
    raw = zlib.decompress(tile[off : off + length])
    cells, at = [], 0
    for _ in range(min_cy, max_cy):
        n = struct.unpack_from("<I", raw, at)[0]
        cells.append(decode_chunk(raw[at + 4 : at + 4 + n]))
        at += 4 + n
    column = np.concatenate(cells, axis=1)[0, :, 0]  # one voxel column, bottom to top
    y = np.arange(min_cy * N, max_cy * N)
    assert (column[(y > 25 * 16 - 50) & (y < 25 * 16 + 50)] == 0).all()  # hollow around 25 m
    assert (column[(y > 25 * 16 + 120) & (y < 40 * 16)] != 0).all()  # solid rock over it
    assert y[-1] == 40 * 16 - 1 and column[-1] != 0  # the column ends with the ground's top voxel


def _project(tmp_path):
    """A scratch project: the demo style, and an 'oak' export 4 × 4 m (64 × 96 × 64 voxels)."""
    import json
    import shutil

    shutil.copy(DEMO / "style.toml", tmp_path / "style.toml")
    (tmp_path / "export").mkdir()
    (tmp_path / "export" / "oak.cubist.json").write_text(
        json.dumps(
            {"cubist_manifest": 1, "name": "oak", "size": [64, 96, 64], "glb_origin": [32, 0, 32]}
        )
    )
    (tmp_path / "export" / "oak.vox").write_bytes(b"")
    return tmp_path


def _hill(x, z, s):
    return 10 + np.asarray(x) * 0.1  # rises 1 m per 10 m east


def test_props_stand_on_the_ground_and_scatter_by_the_rules(tmp_path):
    from cubist.world import place, scatter
    from cubist.world.field import Custom
    from cubist.world.props import placements, scatter_points

    project = _project(tmp_path)

    @world(name="woods", size_m=(200, 200), seed=4)
    def woods(p):
        return World(
            ground=Custom(_hill),
            surface=[p.leaf.mid],
            props=[
                place("oak", at=(50, 50), name="old_oak"),
                place("oak", at=(80, 30, 120), rotate_y=90),
                scatter(
                    "oak",
                    ((0, 0), (200, 200)),
                    per_ha=40,
                    height_m=(0, 25),
                    avoid=circle((100, 100), 30, 0),
                ),
            ],
        )

    loaded = load_world(woods, _demo_palette(), seed=0)
    props = placements(loaded, project / "export")
    old = props[0]
    assert old.name == "old_oak" and old.at[0] == 50 * 16
    assert old.at[1] == pytest.approx(
        (10 + 52 * 0.1) * 16
    )  # the footprint's high (east) edge, 2 m out
    assert props[1].at == (80 * 16, 30 * 16, 120 * 16) and props[1].rotate_y == 90
    x, z, turn = scatter_points(loaded, loaded.world.props[2])
    assert len(x) == len(props) - 2
    assert np.all(10 + x * 0.1 <= 25)  # height_m keeps them under 25 m: x under 150
    assert np.all(np.hypot(x - 100, z - 100) > 30)  # avoided the clearing
    area_ha = (150 * 200 - np.pi * 30**2) / 10000
    assert 0.7 * 40 * area_ha < len(x) < 1.3 * 40 * area_ha
    assert set(np.unique(turn)) <= {0, 90, 180, 270}
    again = scatter_points(load_world(woods, _demo_palette(), seed=0), loaded.world.props[2])
    assert np.array_equal(x, again[0])
    # Worked out once per loaded world: the map, lint and stats share it.
    assert scatter_points(loaded, loaded.world.props[2])[0] is x and not x.flags.writeable

    @world(name="missing", size_m=(64, 64))
    def missing(p):
        return World(ground=flat(1), surface=[p.leaf.mid], props=[place("barrel", at=(5, 5))])

    with pytest.raises(WorldError, match="isn't exported"):
        placements(load_world(missing, _demo_palette()), project / "export")


def test_bake_writes_props_into_world_json(tmp_path):
    import json

    from cubist.world.bake import bake_world

    project = _project(tmp_path)
    (project / "worlds").mkdir()
    script = project / "worlds" / "grove.py"
    script.write_text(
        "from cubist.world import *\n\n"
        "@world(name='grove', size_m=(128, 128))\n"
        "def grove(p):\n"
        "    trees = scatter('oak', ((0, 0), (128, 128)), per_ha=20)\n"
        "    return World(ground=flat(5), surface=[p.leaf.mid],\n"
        "                 props=[place('oak', at=(64, 64)), trees,\n"
        "                        place('oak', at=(20, 30), below_ground_m=3, name='deep',\n"
        "                              carve=True)],\n"
        "                 landmarks={'grove': (64, 64), 'cellar': 'deep'})\n"
    )
    bake_world(script, levels=1, workers=1)
    out = project / "export" / "worlds" / "grove"
    world_json = json.loads((out / "world.json").read_text())
    # A landmark at a named prop sits at its base: here 3 m under the ground.
    assert world_json["landmarks"] == {"grove": [1024, 80, 1024], "cellar": [320, 32, 480]}
    assert world_json["props"][-1]["carve"] is True and "carve" not in world_json["props"][0]
    assert world_json["props"][0] == {
        "asset": "../../oak.vox",
        "at": [1024.0, 80.0, 1024.0],
        "rotate_y": 0,
    }
    assert len(world_json["props"]) > 10
    assert all(
        p["at"][1] == 80 - 0.25 * 16 for p in world_json["props"][1:-1]
    )  # scattered ones sink 0.25 m
    with pytest.raises(WorldError, match="no prop named"):
        load_world(
            world(name="lost", size_m=(64, 64))(
                lambda p: World(ground=flat(5), surface=[p.leaf.mid], landmarks={"x": "nowhere"})
            ),
            _demo_palette(),
        )


def test_bake_writes_roads_rivers_and_tunnels_as_paths(tmp_path):
    import json

    from cubist.world.bake import bake_world

    script = tmp_path / "ways.py"
    script.write_text(
        "from cubist.world import *\n\n"
        "@world(name='ways', size_m=(128, 128))\n"
        "def ways(p):\n"
        "    lane = road(path([(10, 20), (120, 20)]), width_m=3, color=p.cork, name='lane')\n"
        "    brook = river(path([(10, 90), (120, 100)]), width_m=4, depth_m=1)\n"
        "    dig = tunnel([(60, 3, 60), (100, 6, 60)], radius_m=2, below_ground=True, name='dig')\n"
        "    return World(ground=flat(12), surface=[p.leaf.mid], roads=[lane], rivers=[brook],\n"
        "                 caves=[dig])\n"
    )
    bake_world(script, out_dir=tmp_path / "out", levels=1, workers=1, style=DEMO / "style.toml")
    paths = json.loads((tmp_path / "out" / "world.json").read_text())["paths"]
    assert [(p["kind"], p["name"]) for p in paths] == [
        ("road", "lane"),
        ("river", "river1"),
        ("tunnel", "dig"),
    ]
    lane = np.array(paths[0]["points"])
    assert lane[0, 0] == 10 * 16 and lane[-1, 0] == 120 * 16 and np.all(lane[:, 2] == 20 * 16)
    assert np.allclose(lane[:, 1], 12 * 16, atol=1)  # on the road (the flat ground)
    dig = np.array(paths[2]["points"])
    assert np.isclose(dig[0, 1], (12 - 3 - 2) * 16)  # the tunnel's floor: its line, less its radius


def test_world_lint_finds_edges_slopes_props_and_closed_caves(tmp_path):
    from cubist.world import cavern, place, scatter, tunnel
    from cubist.world.field import Custom
    from cubist.world.lint import lint_world
    from cubist.world.sample import slopes

    project = _project(tmp_path)
    (project / "export" / "cottage.cubist.json").write_text(
        '{"cubist_manifest": 1, "name": "cottage", '
        '"size": [96, 80, 128], "glb_origin": [48, 0, 64]}'
    )

    @world(name="linty", size_m=(1000, 1000))
    def linty(p):
        cliff = Custom(
            lambda x, z, s: np.where(np.asarray(x) > 500, 30.0, 0.0) + np.asarray(z) * 0.0, "cliff"
        )
        return World(
            ground=flat(10) + cliff,
            surface=[p.leaf.mid],
            caves=[
                cavern((200, -20, 200), radius_m=6),  # deep, nothing reaches it
                cavern((700, 25, 700), radius_m=6),
                tunnel(
                    [(700, 41, 640), (700, 25, 700)], radius_m=2, smooth=False
                ),  # opens at the surface (40 m)
            ],
            props=[
                place("cottage", at=(498, 300)),  # straddles the cliff
                place("cottage", at=(300, 300)),
                place("cottage", at=(303, 302)),  # on top of the last one
                scatter("oak", ((280, 280), (320, 320)), per_ha=400, spacing_m=6),
            ],
            landmarks={"spawn": (100, 500), "lost": (1200, 10), "cliff": (500, 600)},
        )

    loaded = load_world(linty, _demo_palette(), seed=0)
    x, z = np.meshgrid(np.arange(0, 1000, 10.0), np.arange(0, 1000, 10.0))
    report = lint_world(loaded, slopes(loaded, x, z), project / "export")
    text = report.summary()
    assert report.count("error") == 1 and "landmark lost" in text
    assert "landmark spawn is 100 m from the world's edge" in text
    assert "within 20 m of cliff" in text
    assert "falls" in text and "(498, 300)" in text
    assert "overlaps cottage at (303, 302)" in text
    assert "scattered oak grow into cottage" in text
    assert "(200, -20, 200) has no way in" in text
    assert "(700, 25, 700)" not in text  # its tunnel opens at the surface
    # The tunnel's floor (2 m under its line) meets the cavern's (6 m under its center): a drop.
    assert "floor of the tunnel from (700, 41, 640) drops" in text and "near (700, 69" in text


def test_world_lint_passes_walkable_tunnels():
    from cubist.world import tunnel
    from cubist.world.lint import lint_world
    from cubist.world.sample import slopes

    @world(name="ramp", size_m=(400, 400))
    def ramp(p):
        line = [(100, 41, 200), (200, 30, 200), (300, 30, 260)]
        return World(ground=flat(40), surface=[p.leaf.mid], caves=[tunnel(line, radius_m=2.5)])

    loaded = load_world(ramp, _demo_palette(), seed=0)
    x, z = np.meshgrid(np.arange(0, 400, 10.0), np.arange(0, 400, 10.0))
    assert "floor of the tunnel" not in lint_world(loaded, slopes(loaded, x, z), None).summary()


def test_region_renders_build_the_area_like_the_bake():
    from cubist.world import cavern
    from cubist.world.region import region_grid, render_region

    @world(name="hollow", size_m=(256, 256))
    def hollow(p):
        return World(ground=flat(20), surface=[p.leaf.mid], subsoil=[(1, p.wood[1]), p.stone.dark],
                     caves=[cavern((128, 12, 128), radius_m=5)])  # fmt: skip

    loaded = load_world(hollow, _demo_palette(), seed=0)
    grid, level, corner = region_grid(loaded, (128, 128), size_m=32)
    assert level == 1 and grid.data.shape[0] == grid.data.shape[2] == 256  # 32 m in 0.125 m cells
    column = grid.data[128, :, 128]  # through the cavern's middle
    y = corner[1] + (np.arange(len(column)) + 0.5) * 0.125
    assert column[(y > 8) & (y < 16)].max() == 0  # hollow
    assert column[(y > 18) & (y < 19.5)].min() > 0 and y[-1] < 20.2  # solid up to the ground
    cut, _, _ = region_grid(loaded, (128, 128), size_m=32, cut_m=12)
    assert cut.data.shape[1] < grid.data.shape[1]
    assert render_region(loaded, (128, 128), size_m=32, cut_m=12).width > 200
    with pytest.raises(WorldError, match="outside the world"):
        region_grid(loaded, (2000, 2000), size_m=32)


def test_region_renders_draw_props_where_kiln_stamps_them(tmp_path):
    import json

    from cubist.core.grid import VoxelGrid
    from cubist.io.vox import VoxPart, VoxScene, write_vox
    from cubist.world import place
    from cubist.world.region import region_grid

    project = _project(tmp_path)
    # A post 2 voxels along x, 32 up, 1 along z, its origin at its base corner.
    post = VoxelGrid(np.full((2, 32, 1), 5, dtype=np.uint8))
    write_vox(
        project / "export" / "post.vox",
        VoxScene([VoxPart("post", post)], _demo_palette().rgba_table()),
    )
    (project / "export" / "post.cubist.json").write_text(
        json.dumps(
            {"cubist_manifest": 1, "name": "post", "size": [2, 32, 1], "glb_origin": [0, 0, 0]}
        )
    )

    @world(name="posts", size_m=(64, 64))
    def posts(p):
        props = [place("post", at=(10, 10)), place("post", at=(20, 10), rotate_y=90)]
        return World(ground=flat(5), surface=[p.leaf.mid], props=props)

    loaded = load_world(posts, _demo_palette(), seed=0)
    grid, level, corner = region_grid(loaded, (16, 16), size_m=16, export=project / "export")
    assert level == 0
    x0, y0, z0 = (int(round(c * 16)) for c in corner)

    def at(x, y, z):  # world voxel → the grid's color there
        return int(grid.data[x - x0, y - y0, z - z0])

    # Unturned: voxels (160..161, 80..111, 160), standing on the ground (5 m = voxel 80).
    assert at(160, 80, 160) == at(161, 111, 160) == 5 and at(162, 80, 160) == 0
    # A quarter turn maps (x, y, z) to (z, y, -x - 1), as Kiln stamps it: along -z from 159.
    assert at(320, 80, 159) == at(320, 111, 158) == 5 and at(321, 80, 159) == at(320, 80, 160) == 0
    assert at(200, 79, 200) != 0 and at(200, 80, 200) == 0  # the ground below them


def test_scattered_points_find_the_same_nearest_point_in_buckets():
    from cubist.world import path

    line = path([(0, 0), (900, 300), (1800, -200), (2600, 400)])
    rng = np.random.default_rng(3)
    x, z = rng.uniform(-100, 2700, 5000), rng.uniform(-600, 800, 5000)
    for reach in (40.0, 2000.0):
        d, t = line.nearest(x, z, reach)  # bucketed by area
        d0, t0 = line._nearest(x, z, reach)  # every point against every segment near them all
        assert np.array_equal(d, d0) and np.allclose(t[np.isfinite(d)], t0[np.isfinite(d0)])


def test_path_distance_fields_cap_far_away():
    from cubist.world import path

    d = path([(0, 0), (100, 0)], smooth=False).distance(max_m=50)
    assert d.sample(
        np.array([50.0, 50.0, 300.0]), np.array([10.0, 80.0, 0.0]), seed=0
    ).tolist() == [10.0, 50.0, 50.0]


def test_caves_and_props_can_be_placed_below_the_ground():
    from cubist.world import cavern, place, tunnel
    from cubist.world.props import placements

    @world(name="deep", size_m=(256, 256))
    def deep(p):
        ground = flat(30) + Custom(lambda x, z, s: np.asarray(x) * 0.1, "tilt")
        return World(
            ground=ground,
            surface=[p.leaf.mid],
            caves=[tunnel([(50, -1, 50), (100, 10, 50)], radius_m=2, smooth=False,
                          below_ground=True),
                   cavern((100, 12, 50), radius_m=4, below_ground=True)],
            props=[place("torch", at=(100, 50), below_ground_m=15.5)],
        )  # fmt: skip

    from cubist.world.field import Custom

    loaded = load_world(deep, _demo_palette(), seed=0)
    t, c = loaded.world.caves
    assert t.points[0, 1] == pytest.approx(30 + 5 + 1) and t.points[-1, 1] == pytest.approx(
        30 + 10 - 10
    )
    assert c.c[1] == pytest.approx(40 - 12)
    [torch] = placements(loaded, Path("/nonexistent"))
    assert torch.at[1] == pytest.approx((40 - 15.5) * 16)


def test_level_flattens_to_the_ground_at_the_center():
    f = (flat(10) + Custom(lambda x, z, s: np.asarray(x) * 0.5, "slope")).level(
        (40, 40), radius_m=5, edge_m=0
    )
    h = f.sample(np.array([36.0, 44.0, 80.0]), np.array([40.0, 40.0, 40.0]), seed=0)
    assert h[:2].tolist() == [30.0, 30.0] and h[2] == 50.0


from cubist.world.field import Custom  # noqa: E402


def test_coarse_path_distance_stays_close_away_from_the_path():
    from cubist.world import path

    line = path([(0, 0), (300, 200), (600, 0)])
    exact, coarse = line.distance(max_m=800), line.distance(max_m=800, step_m=8)
    x, z = np.meshgrid(np.linspace(0, 600, 300), np.linspace(-400, 400, 300))
    a, b = exact.sample(x, z, 0), coarse.sample(x, z, 0)
    far = a > 50
    # Off by at most a few metres (where the nearest part of the path switches, and next to it).
    assert np.abs(a - b)[far].max() < 3 and np.abs(a - b).max() < 4.5
    # A point on its own reads what it reads in a dense batch (props stand on the baked ground).
    for i, j in ((0, 0), (150, 77), (299, 299)):
        alone = coarse.sample(np.array([x[i, j]]), np.array([z[i, j]]), 0)[0]
        assert alone == b[i, j]
