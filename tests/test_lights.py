"""Point lights: where a torch or a campfire lights its surroundings."""

import json

import jsonschema
import numpy as np
import pytest
from pygltflib import GLTF2

from cubist import Model, ModelError, Part, asset, box, light
from cubist.core.model import BuiltAsset
from cubist.core.palette import Palette
from cubist.io.gltf import encode_glb, srgb_to_linear
from cubist.io.vox import decode_vox, encode_vox
from cubist.manifest import SCHEMA_PATH, build_manifest
from cubist.render.filmstrip import posed_sheet
from cubist.render.posed import PosedScene

P = Palette({"wood": ["#3b2414", "#8a5a32"], "fire": ["#d9480f", "#ffe066"]})


def torch(p, lights=None):
    stick = Part("stick", box((1, 6, 1), color=p.wood.dark), pivot=(0.5, 0, 0.5))
    lights = (
        lights
        if lights is not None
        else {"flame": light((0.5, 7, 0.5), p.fire.light, range=6, intensity=2, flicker=0.3)}
    )
    return Model([stick], lights=lights)


def built(fn=torch):
    return asset(name="torch", size=(1, 8, 1))(fn).build(P)


def test_light_round_trips_through_summary_and_vox():
    a = built()
    [(name, lt)] = a.lights.items()
    assert name == "flame" and lt.color == P.fire.light.index and lt.flicker == 0.3
    again = BuiltAsset.from_vox(decode_vox(encode_vox(a.to_vox_scene())), a.summary(), P)
    assert again.lights == a.lights


def test_glb_has_a_point_light_node_under_its_part():
    g = GLTF2.load_from_bytes(encode_glb(built(), P.rgba_table(), voxel_size_m=0.5))
    assert "KHR_lights_punctual" in g.extensionsUsed
    [lt] = g.extensions["KHR_lights_punctual"]["lights"]
    assert lt["type"] == "point" and lt["intensity"] == 2.0 and lt["range"] == 3.0
    want = srgb_to_linear(np.array([0xFF, 0xE0, 0x66]))
    assert np.allclose(lt["color"], want, atol=1e-6)
    names = [n.name for n in g.nodes]
    node = g.nodes[names.index("flame")]
    assert node.extensions == {"KHR_lights_punctual": {"light": 0}}
    stick = g.nodes[names.index("stick")]
    assert names.index("flame") in stick.children
    assert node.translation == [0.0, 3.5, 0.0]  # (0.5, 7, 0.5) - pivot (0.5, 0, 0.5), metres


def test_manifest_records_light_with_color_name_and_flicker():
    m = build_manifest(built(), voxel_size_m=1)
    jsonschema.validate(m, json.loads(SCHEMA_PATH.read_text()))
    assert m["lights"]["flame"] == {
        "position": [0.5, 7.0, 0.5], "part": "stick", "color": "fire.light", "rgb": "#ffe066",
        "range": 6.0, "intensity": 2.0, "flicker": 0.3,
    }  # fmt: skip
    assert m["colors"]["fire.light"] == {"index": P.fire.light.index, "rgb": "#ffe066"}


def test_bad_flicker_and_unknown_part_are_errors():
    with pytest.raises(ModelError, match="flicker"):
        light((0, 0, 0), P.fire.light, flicker=2)
    with pytest.raises(ModelError, match="unknown part"):
        built(lambda p: torch(p, {"x": light((0, 0, 0), p.fire.light, part="nope")}))
    with pytest.raises(ModelError, match="palette color"):
        light((0, 0, 0), "#ffffff")


def test_range_intensity_and_position_must_be_finite():
    # NaN passes "<= 0" and json.dumps would write NaN or Infinity into the manifest, which Kiln
    # then rejects as invalid JSON, far from the cause: refuse it here.
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ModelError, match="range"):
            light((0, 0, 0), P.fire.light, range=bad)
        with pytest.raises(ModelError, match="intensity"):
            light((0, 0, 0), P.fire.light, intensity=bad)
        with pytest.raises(ModelError, match="position"):
            light((0, bad, 0), P.fire.light)
    assert light((0, 0, 0), P.fire.light, range=1e6, intensity=0.01).range == 1e6


def test_posed_views_mark_the_light():
    a = built()
    scene = PosedScene(a, P.rgba_table())
    region = ((-2, 0, -2), (3, 9, 3))
    plain = scene.render({}, "front", 8, region)
    marked = scene.render({}, "front", 8, region, markers=[((0.5, 7, 0.5), (255, 224, 102))])
    assert (marked.rgb != plain.rgb).any()
    rgb = np.array([255, 224, 102])
    assert np.any(np.all(marked.rgb == rgb, axis=-1))
    assert posed_sheet(a, P.rgba_table(), None, 0.0) is not None


def test_static_assets_have_no_lights_extension():
    from test_gltf_anim import P as GP
    from test_gltf_anim import build as gbuild
    from test_gltf_anim import static

    assert b"KHR_lights" not in encode_glb(gbuild(static), GP.rgba_table(), voxel_size_m=0.5)


# --- lint: lights that can't reach the floor ------------------------------------------

from cubist.core.style import StyleBible  # noqa: E402
from cubist.lint import lint  # noqa: E402


def post(height, range_, grounded=True):
    def fn(p):
        pole = Part("pole", box((1, height, 1), color=p.wood.dark), pivot=(0.5, 0, 0.5))
        lamp = light((0.5, height + 0.5, 0.5), p.fire.light, range=range_)
        return Model([pole], lights={"lamp": lamp})

    return asset(name="post", size=(1, height, 1), grounded=grounded)(fn).build(P)


def reach(a):
    style = StyleBible(path=None, name="t", voxel_size_m=0.0625, palette=P)
    return [f for f in lint(a, style).findings if f.check == "light_reach"]


def test_light_too_short_to_light_the_floor_warns_in_metres():
    # The lamp post that started this: 50 voxels up, range 48.
    [f] = reach(post(50, 48))
    assert f.severity == "warning"
    assert "lamp" in f.message and "3.2 m" in f.message and "3.0 m" in f.message
    assert f.details["min_range"] == 72  # ceil(sqrt(2) * 50.5)


def test_light_reaching_the_floor_around_it_is_fine():
    assert reach(post(50, 120)) == []
    assert reach(post(6, 160)) == []  # a campfire's light, low and wide


def test_light_shorter_than_a_candle_warns_even_when_low():
    # The demo campfire: 6 voxels up, so it reaches the floor, but range 14 is 0.9 m.
    [f] = reach(post(6, 14))
    assert "0.9 m" in f.message and "candle" in f.message
    assert f.details["min_range"] == 32  # 2 m at 1/16 m


def cellared(at, range_):
    """A building with a 48-voxel cellar under the street (ground_y 48) and a light at y = at."""

    def fn(p):
        body = Part("body", box((4, 120, 4), color=p.wood.dark), pivot=(2, 0, 2))
        return Model([body], lights={"lamp": light((2, at, 4.5), p.fire.light, range=range_)})

    spec = asset(name="house", size=(4, 120, 6), tags=["building"], grounded=True, ground_y=48)
    return spec(fn).build(P)


def test_a_building_s_lights_are_measured_from_the_street_or_the_cellar_floor():
    # A door lamp 40 voxels above the street lights the street with 58: the cellar below the
    # street doesn't count.
    assert reach(cellared(88, 58)) == []
    [f] = reach(cellared(88, 56))
    assert f.details["min_range"] == 57  # ceil(sqrt(2) * 40)
    # A light in the cellar is measured from the cellar's floor.
    [f] = reach(cellared(30, 40))
    assert f.details["min_range"] == 43  # ceil(sqrt(2) * 30)


def test_ungrounded_lights_only_need_a_candle_s_reach():
    assert reach(post(50, 40, grounded=False)) == []  # held or hung: no known floor
    [f] = reach(post(50, 10, grounded=False))
    assert f.details["min_range"] == 32


# --- indoor lights: measured from the floor under them ---------------------------------


def storeyed(at, range_, indoor=True):
    """A building with a 48-voxel cellar under the street (ground_y 48), a storey's floor whose
    surface is at y = 150 over x 0..4, and a light at (2, at, 2): over the floor, or at x 6, past
    it, when ``at`` is a tuple."""
    x, y = (6, at[0]) if isinstance(at, tuple) else (2, at)

    def fn(p):
        walls = Part("walls", box((4, 120, 4), color=p.wood.dark), pivot=(2, 0, 2))
        floor = Part("floor", box((4, 4, 4), color=p.wood.mid).at(0, 146, 0), pivot=(2, 146, 2))
        lamp = light((x, y, 2), p.fire.light, range=range_, indoor=indoor)
        return Model([walls, floor], lights={"lamp": lamp})

    spec = asset(name="house", size=(8, 220, 6), tags=["building"], grounded=True, ground_y=48)
    return spec(fn).build(P)


def test_an_indoor_light_is_measured_from_the_floor_under_it():
    # A hearth's light 9 voxels over an upper floor lights its room with 13; measured from the
    # street, 111 voxels down, it would need 157.
    assert reach(storeyed(159, 40)) == []
    [f] = reach(storeyed(159, 40, indoor=False))
    assert f.details["min_range"] == 157  # ceil(sqrt(2) * (159 - 48))
    [f] = reach(storeyed(210, 40))
    assert "above the floor under it" in f.message
    assert f.details["min_range"] == 85  # ceil(sqrt(2) * (210 - 150))
    # The candle's reach still holds indoors.
    [f] = reach(storeyed(159, 10))
    assert f.details["min_range"] == 32 and "candle" in f.message


def fixture_over_floors(fixture, at, range_):
    """A floor (its surface at y = 4, x and z 0..8), a light at (2.5, at, 2.5) indoor, and the
    light's fixture as boxes ((size, corner) pairs) between them."""

    def fn(p):
        floor = Part("floor", box((8, 4, 8), color=p.wood.dark), pivot=(4, 0, 4))
        parts = [floor] + [
            Part(f"fixture_{i}", box(size, color=p.fire.light).at(*corner), pivot=corner)
            for i, (size, corner) in enumerate(fixture)
        ]
        lamp = light((2.5, at, 2.5), p.fire.light, range=range_, indoor=True)
        return Model(parts, lights={"lamp": lamp})

    return asset(name="lantern", size=(8, 220, 8), grounded=True)(fn).build(P)


def test_an_indoor_light_in_its_lantern_is_measured_from_the_floor_under_the_lantern():
    # The light in the middle of its glass (y 150..158), 150 voxels over the floor: the glass is
    # its own fixture, not its floor.
    [f] = reach(fixture_over_floors([((3, 8, 3), (1, 150, 1))], 154, 40))
    assert "above the floor under it" in f.message
    assert f.details["min_range"] == 213  # ceil(sqrt(2) * 150)
    assert reach(fixture_over_floors([((3, 8, 3), (1, 150, 1))], 154, 213)) == []


def test_a_standing_lamp_is_measured_from_the_floor_under_its_own():
    # A lamp on a post standing on an upper floor (its surface at y 104): the post and the floor
    # are one run under the light, so it is measured from the floor below them (y 4), 126.5 down.
    standing = [((8, 4, 8), (0, 100, 0)), ((1, 26, 1), (2, 104, 2))]
    [f] = reach(fixture_over_floors(standing, 130.5, 160))
    assert f.details["min_range"] == 179  # ceil(sqrt(2) * 126.5)


def test_an_indoor_light_with_nothing_under_it_warns_and_keeps_the_ground_line():
    # Past the floor's edge, nothing is under it: it is measured from the street, and says so.
    findings = reach(storeyed((159,), 160))
    [f] = findings
    assert f.severity == "warning" and "'lamp'" in f.message
    assert "nothing straight under it" in f.message
    findings = reach(storeyed((159,), 40))
    assert len(findings) == 2
    assert {f.details.get("min_range") for f in findings} == {None, 157}


def test_indoor_round_trips_through_the_summary_but_not_into_the_manifest():
    a = storeyed(159, 40)
    assert a.lights["lamp"].indoor
    again = BuiltAsset.from_vox(decode_vox(encode_vox(a.to_vox_scene())), a.summary(), P)
    assert again.lights == a.lights
    # A build.json from before the flag reads as outdoor.
    old = {k: v for k, v in a.lights["lamp"].to_summary().items() if k != "indoor"}
    assert not type(a.lights["lamp"]).from_summary(old).indoor
    # Only the lint reads it: the manifest's light is as before.
    m = build_manifest(a, voxel_size_m=1)
    jsonschema.validate(m, json.loads(SCHEMA_PATH.read_text()))
    assert "indoor" not in m["lights"]["lamp"]


def test_indoor_must_be_true_or_false():
    with pytest.raises(ModelError, match="indoor"):
        light((0, 0, 0), P.fire.light, indoor="yes")


# --- a light's hold: past hold_m metres it stops falling off --------------------------


def test_hold_m_goes_in_the_manifest_only_when_set():
    # Kiln holds the light's level past hold_m metres, out to its range (its formats.md, "Lights").
    lamp = {"flame": light((0.5, 7, 0.5), P.fire.light, range=6, hold_m=3)}
    held = built(lambda p: torch(p, lamp))
    m = build_manifest(held, voxel_size_m=1)
    jsonschema.validate(m, json.loads(SCHEMA_PATH.read_text()))
    assert m["lights"]["flame"]["hold_m"] == 3.0
    assert list(m["lights"]["flame"])[-1] == "hold_m"  # after the keys every manifest already has
    # Unset, the light is as before: no key, so every existing manifest is byte-for-byte the same.
    assert "hold_m" not in build_manifest(built(), voxel_size_m=1)["lights"]["flame"]
    schema = json.loads(SCHEMA_PATH.read_text())
    bad = json.loads(json.dumps(m))
    bad["lights"]["flame"]["hold_m"] = 0
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_hold_m_round_trips_through_the_summary():
    held = built(lambda p: torch(p, {"flame": light((0.5, 7, 0.5), p.fire.light, hold_m=2.5)}))
    assert held.lights["flame"].hold_m == 2.5
    again = BuiltAsset.from_vox(decode_vox(encode_vox(held.to_vox_scene())), held.summary(), P)
    assert again.lights == held.lights
    # A build.json from before the hold, or a light without one, reads as none.
    plain = built()
    assert plain.lights["flame"].hold_m is None
    assert "hold_m" not in plain.lights["flame"].to_summary()
    flame = plain.lights["flame"]
    assert type(flame).from_summary(flame.to_summary()).hold_m is None


def test_hold_m_must_be_a_positive_number_or_none():
    for bad in (0, -1, "3", True, float("nan"), float("inf")):
        with pytest.raises(ModelError, match="hold_m"):
            light((0, 0, 0), P.fire.light, hold_m=bad)
    assert light((0, 0, 0), P.fire.light, hold_m=None).hold_m is None
    assert light((0, 0, 0), P.fire.light, hold_m=1).hold_m == 1.0
