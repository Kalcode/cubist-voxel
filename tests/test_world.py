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
