"""Emitters and particle previews: smoke, sparks and dust, described for the engine and
previewed with a seeded simulation."""

import json
import textwrap

import jsonschema
import numpy as np
import pytest
from pygltflib import GLTF2

from cubist import Model, ModelError, Part, asset, box, clip, emitter, turn
from cubist.build.runner import build_script
from cubist.core.grid import VoxelGrid
from cubist.core.model import BuiltAsset
from cubist.core.palette import Palette
from cubist.io.gltf import encode_glb
from cubist.io.vox import decode_vox, encode_vox
from cubist.lint.motion import check_motion
from cubist.manifest import SCHEMA_PATH, build_manifest
from cubist.motion.particles import particles_at
from cubist.render.filmstrip import clip_frames

P = Palette({"wood": ["#3b2414", "#8a5a32"], "smoke": ["#5a5a5a", "#9a9a9a"],
             "fire": ["#d9480f", "#ffe066"]})  # fmt: skip


def chimney(p, emitters=None, clips=None):
    body = Part("stack", box((2, 6, 2), color=p.wood.dark), pivot=(1, 0, 1))
    emitters = (
        emitters
        if emitters is not None
        else {
            "smoke": emitter((1, 6.5, 1), "puff", rate=4, life=2, speed=2, gravity=-1, grow=(1, 2))
        }
    )
    return Model([body], emitters=emitters, clips=clips or [])


def built(fn=chimney, size=(2, 6, 2)):
    return asset(name="chimney", size=size)(fn).build(P)


PUFF = VoxelGrid.from_voxels(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]),
                             np.array([3, 3, 4, 4], dtype=np.uint8))  # fmt: skip


def test_emitter_round_trips_through_summary_and_vox():
    a = built()
    again = BuiltAsset.from_vox(decode_vox(encode_vox(a.to_vox_scene())), a.summary(), P)
    assert again.emitters == a.emitters


def test_emitter_needs_rate_or_burst_and_sane_values():
    with pytest.raises(ModelError, match="rate"):
        emitter((0, 0, 0), "puff")
    with pytest.raises(ModelError, match="life"):
        emitter((0, 0, 0), "puff", rate=1, life=0)
    with pytest.raises(ModelError, match="particle"):
        emitter((0, 0, 0), "Bad Name!", rate=1)
    with pytest.raises(ModelError, match="unknown part"):
        built(lambda p: chimney(p, {"s": emitter((0, 0, 0), "puff", rate=1, part="nope")}))


def test_continuous_emitter_shows_a_steady_stream():
    a = built()
    now = particles_at(a, None, 0.0)
    later = particles_at(a, None, 5.0)
    assert len(now) == len(later) == 8  # rate 4 x life 2: running before t = 0 too
    ys = sorted(round(p.position[1], 3) for p in now)
    assert ys[0] >= 6.5 and ys[-1] > 9  # rising out of the stack


def test_particles_are_deterministic_and_seeded():
    a = built()
    one = [p.position.tolist() for p in particles_at(a, None, 1.0)]
    two = [p.position.tolist() for p in particles_at(a, None, 1.0)]
    assert one == two

    def other(p):
        return chimney(p, {"smoke": emitter((1, 6.5, 1), "puff", rate=4, life=2, seed=9)})

    three = [p.position.tolist() for p in particles_at(built(other), None, 1.0)]
    assert one != three


def gun(p):
    body = Part("gun", box((1, 1, 4), color=p.wood.dark), pivot=(0.5, 0, 2))
    sparks = emitter((0.5, 0.5, 4.5), "spark", direction="+z", burst=6, life=0.3, speed=10,
                     spread=30)  # fmt: skip
    fire = clip("fire", 0.5, [turn("gun", "x", [(0, 0), (0.05, -5), (0.5, 0)])],
                events=[(0.1, "emit:sparks")])  # fmt: skip
    return Model([body], emitters={"sparks": sparks}, clips=[fire])


def test_burst_appears_only_after_its_event_and_dies_with_its_life():
    a = built(gun, size=(1, 1, 5))
    c = a.clips[0]
    assert particles_at(a, c, 0.05) == []
    shots = particles_at(a, c, 0.2)
    assert len(shots) == 6 and all(p.position[2] > 4.5 for p in shots)
    assert particles_at(a, c, 0.45) == []


def test_previews_draw_particles_or_a_stand_in_cube():
    a = built()
    a.particle_models = {"puff": PUFF}
    frames, _ = clip_frames(a, P.rgba_table(), clip("x", 0.5, [turn("stack", "y", [0, 0])]), fps=4)
    with_puffs = np.asarray(frames[0])
    a.particle_models = {"puff": None}
    frames, _ = clip_frames(a, P.rgba_table(), clip("x", 0.5, [turn("stack", "y", [0, 0])]), fps=4)
    stand_in = np.asarray(frames[0])
    assert not np.array_equal(with_puffs, stand_in)
    assert with_puffs.shape[0] > 40  # the canvas grew to fit the smoke


def test_lint_warns_about_unbuilt_particles_blocked_emitters_and_stray_events():
    a = built()
    a.particle_models = {"puff": None}
    checks = {f.check for f in check_motion(a)}
    assert "emitter_particle" in checks

    def blocked(p):
        return chimney(p, {"smoke": emitter((1, 2, 1), "puff", direction="+y", rate=1)})

    assert "emitter_blocked" in {f.check for f in check_motion(built(blocked))}

    def stray(p):
        m = gun(p)
        c = clip("fire", 0.5, [turn("gun", "x", [0, 0])], events=[(0.1, "emit:smoke")])
        return Model(m.parts, emitters=m.emitters, clips=[c])

    assert "emit_event" in {f.check for f in check_motion(built(stray, size=(1, 1, 5)))}


def test_manifest_lists_emitters_and_particle_dependencies():
    m = build_manifest(built(gun, size=(1, 1, 5)), voxel_size_m=1)
    jsonschema.validate(m, json.loads(SCHEMA_PATH.read_text()))
    assert m["particles"] == ["spark"]
    e = m["emitters"]["sparks"]
    assert e["burst"] == 6 and e["rate"] is None and e["direction"] == [0.0, 0.0, 1.0]
    assert e["part"] == "gun"


def test_glb_has_an_oriented_empty_node_per_emitter():
    g = GLTF2.load_from_bytes(encode_glb(built(gun, size=(1, 1, 5)), P.rgba_table(),
                                         voxel_size_m=1))  # fmt: skip
    names = [n.name for n in g.nodes]
    node = g.nodes[names.index("sparks")]
    assert node.mesh is None and node.rotation is not None
    assert names.index("sparks") in g.nodes[names.index("gun")].children


@pytest.fixture
def project(tmp_path):
    (tmp_path / "style.toml").write_text(
        '[project]\nvoxel_size_m = 0.25\n[palette]\nwood = ["#3b2414", "#8a5a32"]\n'
        'smoke = ["#5a5a5a", "#9a9a9a"]\n'
    )
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "puff.py").write_text(
        textwrap.dedent("""\
        from cubist import *

        @asset(name="puff", size=(2, 2, 2), tags=["particle"])
        def puff(p):
            return box((2, 2, 2), color=p.smoke.light)
        """)
    )
    (tmp_path / "assets" / "chimney.py").write_text(
        textwrap.dedent("""\
        from cubist import *

        @asset(name="chimney", size=(2, 6, 2), grounded=True)
        def chimney(p):
            stack = Part("stack", box((2, 6, 2), color=p.wood.dark), pivot=(1, 0, 1))
            smoke = emitter((1, 6.5, 1), "puff", rate=4, life=2, gravity=-1)
            return Model([stack], emitters={"smoke": smoke},
                         clips=[clip("idle", 1, [turn("stack", "y", [0, 0])], loop=True)])
        """)
    )
    return tmp_path


def test_build_uses_built_particles_and_warns_when_missing(project):
    [r] = build_script(project / "assets" / "chimney.py")
    assert any("emitter_particle" in f for f in r.summary["lint"]["summary"].splitlines())
    build_script(project / "assets" / "puff.py")
    [r] = build_script(project / "assets" / "chimney.py")
    assert "emitter_particle" not in r.summary["lint"]["summary"]


def test_emitter_numbers_must_be_finite():
    # NaN passes "<= 0" and json.dumps would write NaN or Infinity into the manifest, which an
    # engine's JSON reader rejects far from the cause: refuse them here.
    def make(**kw):
        args = {"rate": 4, **kw}
        return emitter(args.pop("position", (0, 0, 0)), "puff", **args)

    for bad in (float("nan"), float("inf"), float("-inf")):
        for key in ("rate", "life", "speed", "spread", "gravity"):
            with pytest.raises(ModelError, match=key):
                make(**{key: bad})
        with pytest.raises(ModelError, match="grow"):
            make(grow=(1, bad))
        with pytest.raises(ModelError, match="position"):
            make(position=(0, bad, 0))
        with pytest.raises(ModelError, match="direction"):
            make(direction=(0, bad, 0))
        with pytest.raises(ModelError, match="seed"):
            make(seed=bad)
    e = make(speed=-1, gravity=-3, seed=7)
    assert (e.speed, e.gravity, e.seed) == (-1.0, -3.0, 7)
