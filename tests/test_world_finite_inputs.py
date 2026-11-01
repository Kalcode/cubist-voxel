"""The world DSL refuses NaN and infinity, as the asset DSL does (test_finite_inputs.py). NaN
passes a "> 0" check, and a ground, path or placement built from it makes a NaN height, a hole
the bake can't explain, or a NaN written into world.json. Every function that takes a number is
listed here with each of its numeric arguments: add a row for a new one."""

import numpy as np
import pytest

from cubist.core.palette import Palette
from cubist.world import (
    World,
    WorldError,
    cavern,
    circle,
    flat,
    height_map,
    height_over,
    height_under,
    load_world,
    mask_map,
    noise,
    path,
    place,
    rect,
    ridges,
    river,
    road,
    scatter,
    slope_over,
    tunnel,
    where,
    world,
    worms,
)

P = Palette({"grass": "#4a8a3a", "stone": "#6b6b75"})
GRASS = P.grass
BAD = [float("nan"), float("inf"), float("-inf")]
LINE = path([(0, 0), (10, 0), (20, 10)])
GROUND = noise(10, 2)

FIELDS = [
    ("flat", "flat height_m", lambda b: flat(b)),
    ("noise scale_m", "noise scale_m", lambda b: noise(b)),
    ("noise amplitude_m", "noise amplitude_m", lambda b: noise(10, b)),
    ("noise octaves", "noise octaves", lambda b: noise(10, 1, octaves=b)),
    ("noise seed", "noise seed", lambda b: noise(10, 1, seed=b)),
    ("ridges scale_m", "ridges scale_m", lambda b: ridges(b)),
    ("ridges amplitude_m", "ridges amplitude_m", lambda b: ridges(10, b)),
    ("ridges octaves", "ridges octaves", lambda b: ridges(10, 1, octaves=b)),
    ("ridges sharpness", "ridges sharpness", lambda b: ridges(10, 1, sharpness=b)),
    ("ridges seed", "ridges seed", lambda b: ridges(10, 1, seed=b)),
    ("height_map low_m", "height_map low_m", lambda b: height_map("a.png", b, 1)),
    ("height_map high_m", "height_map high_m", lambda b: height_map("a.png", 0, b)),
    ("height_map origin_m", "height_map origin_m", lambda b: height_map("a.png", 0, 1, (b, 0))),
    ("height_map size_m", "height_map size_m", lambda b: height_map("a.png", 0, 1, size_m=(b, 5))),
    ("mask_map origin_m", "mask_map origin_m", lambda b: mask_map("a.png", (0, b))),
    ("mask_map size_m", "mask_map size_m", lambda b: mask_map("a.png", size_m=(5, b))),
    ("circle center", "circle center", lambda b: circle((b, 0), 5)),
    ("circle radius_m", "circle radius_m", lambda b: circle((0, 0), b)),
    ("circle edge_m", "circle edge_m", lambda b: circle((0, 0), 5, b)),
    ("rect lo", "rect lo", lambda b: rect((b, 0), (5, 5))),
    ("rect hi", "rect hi", lambda b: rect((0, 0), (5, b))),
    ("rect edge_m", "rect edge_m", lambda b: rect((0, 0), (5, 5), b)),
    ("field + number", "field arithmetic", lambda b: GROUND + b),
    ("number + field", "field arithmetic", lambda b: b + GROUND),
    ("field - number", "field arithmetic", lambda b: GROUND - b),
    ("number - field", "field arithmetic", lambda b: b - GROUND),
    ("field * number", "field arithmetic", lambda b: GROUND * b),
    ("number * field", "field arithmetic", lambda b: b * GROUND),
    ("max", "field arithmetic", lambda b: GROUND.max(b)),
    ("min", "field arithmetic", lambda b: GROUND.min(b)),
    ("blend", "field arithmetic", lambda b: GROUND.blend(b, circle((0, 0), 5))),
    ("flatten at_m", "flatten at_m", lambda b: GROUND.flatten(circle((0, 0), 5), b)),
    ("level center", "level center", lambda b: GROUND.level((0, b), 5)),
    ("level radius_m", "level radius_m", lambda b: GROUND.level((0, 0), b)),
    ("level edge_m", "level edge_m", lambda b: GROUND.level((0, 0), 5, b)),
    ("terrace step_m", "terrace step_m", lambda b: GROUND.terrace(b)),
    ("terrace softness", "terrace softness", lambda b: GROUND.terrace(2, b)),
]

PATHS = [
    ("path point", "path points", lambda b: path([(0, 0), (b, 5)])),
    ("path point 3", "path points", lambda b: path([(0, 0), (5, 5), (10, b)])),
    ("distance max_m", "distance max_m", lambda b: LINE.distance(b)),
    ("distance step_m", "distance step_m", lambda b: LINE.distance(10, b)),
    ("river width_m", "river width_m", lambda b: river(LINE, b)),
    ("river depth_m", "river depth_m", lambda b: river(LINE, 4, b)),
    ("river bank_m", "river bank_m", lambda b: river(LINE, 4, bank_m=b)),
    ("road width_m", "road width_m", lambda b: road(LINE, b, GRASS)),
    ("road bank_m", "road bank_m", lambda b: road(LINE, 4, GRASS, bank_m=b)),
    ("road smooth_m", "road smooth_m", lambda b: road(LINE, 4, GRASS, smooth_m=b)),
]

CAVES = [
    ("tunnel point", "tunnel points", lambda b: tunnel([(0, 0, 0), (5, b, 0)])),
    ("tunnel radius_m", "tunnel radius_m", lambda b: tunnel([(0, 0, 0), (5, 0, 0)], b)),
    ("cavern center", "cavern center", lambda b: cavern((0, b, 0), 5)),
    ("cavern radius_m", "cavern radius_m", lambda b: cavern((0, 0, 0), b)),
    ("cavern radii", "cavern radius_m", lambda b: cavern((0, 0, 0), (5, b, 5))),
    ("cavern roughness_m", "cavern roughness_m", lambda b: cavern((0, 0, 0), 5, b)),
    ("cavern seed", "cavern seed", lambda b: cavern((0, 0, 0), 5, seed=b)),
    ("worms region", "worms region", lambda b: worms(((0, 0), (b, 5)))),
    ("worms depth_m", "worms depth_m", lambda b: worms(((0, 0), (5, 5)), depth_m=(1, b))),
    ("worms scale_m", "worms scale_m", lambda b: worms(((0, 0), (5, 5)), scale_m=b)),
    ("worms thickness", "worms thickness", lambda b: worms(((0, 0), (5, 5)), thickness=b)),
    ("worms seed", "worms seed", lambda b: worms(((0, 0), (5, 5)), seed=b)),
]

PROPS = [
    ("place x", "place at", lambda b: place("tree", (b, 0))),
    ("place y", "place at", lambda b: place("tree", (0, b, 0))),
    ("place sink_m", "place sink_m", lambda b: place("tree", (0, 0), sink_m=b)),
    (
        "place below_ground_m",
        "place below_ground_m",
        lambda b: place("tree", (0, 0), below_ground_m=b),
    ),
    ("place rotate_y", "rotate_y", lambda b: place("tree", (0, 0), rotate_y=b)),
    ("scatter per_ha", "scatter per_ha", lambda b: scatter("tree", ((0, 0), (9, 9)), b)),
    ("scatter region", "scatter region", lambda b: scatter("tree", ((0, 0), (b, 9)), 5)),
    ("scatter seed", "scatter seed", lambda b: scatter("tree", ((0, 0), (9, 9)), 5, seed=b)),
    (
        "scatter spacing_m",
        "scatter spacing_m",
        lambda b: scatter("tree", ((0, 0), (9, 9)), 5, spacing_m=b),
    ),
    (
        "scatter slope_max",
        "scatter slope_max",
        lambda b: scatter("tree", ((0, 0), (9, 9)), 5, slope_max=b),
    ),
    (
        "scatter height_m",
        "scatter height_m",
        lambda b: scatter("tree", ((0, 0), (9, 9)), 5, height_m=(0, b)),
    ),
    ("scatter sink_m", "scatter sink_m", lambda b: scatter("tree", ((0, 0), (9, 9)), 5, sink_m=b)),
]

RULES = [
    ("height_over", "height_over m", lambda b: height_over(b, GRASS)),
    ("height_under", "height_under m", lambda b: height_under(b, GRASS)),
    ("slope_over", "slope_over degrees", lambda b: slope_over(b, GRASS)),
    ("where over", "where over", lambda b: where(circle((0, 0), 5), GRASS, over=b)),
    ("world size_m", "world size_m", lambda b: world("w", (b, 100))),
    ("world seed", "world seed", lambda b: world("w", (100, 100), seed=b)),
]


def _loaded(**kwargs):
    defn = world("w", (100, 100))(lambda p: World(flat(1), [p.grass], **kwargs))
    return load_world(defn, P)


WORLDS = [
    ("surface_depth_m", "surface_depth_m", lambda b: _loaded(surface_depth_m=b)),
    ("step_m", "step_m", lambda b: _loaded(step_m=b)),
    ("subsoil depth", "subsoil depth", lambda b: _loaded(subsoil=[(b, P.stone), P.stone])),
    ("landmark x", "landmark", lambda b: _loaded(landmarks={"a": (b, 5)})),
    ("landmark z", "landmark", lambda b: _loaded(landmarks={"a": (5, b)})),
]

ALL = [*FIELDS, *PATHS, *CAVES, *PROPS, *RULES, *WORLDS]


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.parametrize(("case", "match", "make"), ALL, ids=[c[0] for c in ALL])
def test_world_dsl_refuses_nan_and_infinity(case, match, make, bad):
    with pytest.raises(WorldError, match=match):
        make(bad)


def test_the_error_says_what_to_do():
    with pytest.raises(WorldError, match=r"noise scale_m must be a finite number, got nan"):
        noise(float("nan"))


def test_finite_world_inputs_still_build():
    assert (flat(1.5) + 2 - 0.5).sample(np.zeros(2), np.zeros(2), 0).tolist() == [3.0, 3.0]
    (GROUND * 2 + 1).max(0).min(9).terrace(2, 0.5).level((5, 5), 4)
    tunnel([(0, 0, 0), (5, 1, 0)], 2)
    cavern((0, 0, 0), (5, 4, 5), 1.0)
    worms(((0, 0), (9, 9)), (1, 5))
    road(LINE, 4, GRASS), river(LINE, 4)
    place("tree", (1, 2, 3), below_ground_m=2)
    scatter("tree", ((0, 0), (9, 9)), 5, height_m=(0, 9), slope_max=30)
    _loaded(step_m=0.5, surface_depth_m=0.5, subsoil=[(2, P.stone), P.stone])
