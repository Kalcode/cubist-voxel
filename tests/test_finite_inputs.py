"""The DSL refuses NaN and infinity. NaN passes a "> 0" check (every comparison with it is
false), and what it reaches either voxelizes to nothing or everything, or goes into the
manifest as NaN or Infinity, which an engine's JSON reader rejects far from the cause. Lights,
sockets and emitters have their own tests (test_lights.py, test_sockets.py, test_emitters.py)."""

import pytest

from cubist import (
    Model,
    ModelError,
    Part,
    ShapeError,
    asset,
    band,
    bob,
    box,
    burst,
    capsule,
    checker,
    chunks,
    cone,
    cylinder,
    ellipsoid,
    extrude,
    fracture,
    heightmap,
    hinge,
    lathe,
    line,
    panel,
    polyline,
    pyramid,
    region,
    shake,
    slide,
    sphere,
    spin,
    spin_track,
    stripes,
    swing,
    torus,
    wedge,
)
from cubist.core.palette import Palette

P = Palette({"wood": ["#3b2414", "#8a5a32"]})
C = P.wood.dark
BAD = [float("nan"), float("inf"), float("-inf")]


JOINTS = [
    ("hinge axis", "axis", lambda b: hinge((0, b, 1), (0, 90))),
    ("hinge limits", "hinge limits", lambda b: hinge("x", (0, b))),
    ("hinge low limit", "hinge limits", lambda b: hinge("x", (b, 0))),
    ("slide direction", "axis", lambda b: slide((b, 0, 0), 4)),
    ("slide distance", "slide distance", lambda b: slide("x", b)),
    ("spin axis", "axis", lambda b: spin((0, b, 0))),
    ("spin rps", "rps", lambda b: spin("y", b)),
]


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.parametrize(("case", "match", "make"), JOINTS, ids=[j[0] for j in JOINTS])
def test_joints_refuse_nan_and_infinity(case, match, make, bad):
    # A hinge's limits are never open: a joint that turns without them is a spin.
    with pytest.raises(ModelError, match=match):
        make(bad)


SHAPES = [
    ("box", "box size", lambda b: box((1, b, 1), color=C)),
    ("ellipsoid", "ellipsoid radii", lambda b: ellipsoid((1, b, 1), color=C)),
    ("sphere", "ellipsoid radii", lambda b: sphere(b, color=C)),
    ("cylinder r", "r and h", lambda b: cylinder(b, 2, color=C)),
    ("cylinder h", "r and h", lambda b: cylinder(1, b, color=C)),
    ("cone r_top", "r_top", lambda b: cone(1, 2, r_top=b, color=C)),
    ("capsule r", "capsule r", lambda b: capsule((0, 0, 0), (0, 4, 0), b, color=C)),
    ("capsule r_end", "capsule r_end", lambda b: capsule((0, 0, 0), (0, 4, 0), 1, C, r_end=b)),
    ("capsule end", "capsule p0 and p1", lambda b: capsule((0, b, 0), (0, 4, 0), 1, color=C)),
    ("torus", "torus R and r", lambda b: torus(b, 1, color=C)),
    ("wedge", "box size", lambda b: wedge((2, b, 2), color=C)),
    ("pyramid", "box size", lambda b: pyramid((3, b, 3), color=C)),
    ("thick line r", "line r", lambda b: line((0, 0, 0), (4, 0, 0), C, r=b)),
    ("thick line end", "line p0 and p1", lambda b: line((0, 0, 0), (4, b, 0), C, r=1)),
    ("line end", "line points", lambda b: line((0, 0, 0), (4, b, 0), C)),
    ("polyline point", "line points", lambda b: polyline([(0, 0, 0), (4, 0, 0), (4, b, 0)], C)),
    ("panel point", "panel points", lambda b: panel([(0, 0, 0), (4, 0, 0), (4, b, 0)], C)),
    (
        "panel thickness",
        "panel thickness",
        lambda b: panel([(0, 0, 0), (4, 0, 0), (4, 4, 0)], C, thickness=b),
    ),
    ("lathe profile", "lathe profile", lambda b: lathe([(0, 1), (4, b)], color=C)),
    ("extrude polygon", "extrude polygon", lambda b: extrude([(0, 0), (4, 0), (b, 4)], 2, color=C)),
    ("extrude depth", "extrude depth", lambda b: extrude([(0, 0), (4, 0), (0, 4)], b, color=C)),
    ("heightmap", "heightmap heights", lambda b: heightmap([[1, b], [1, 1]], color=C)),
]


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.parametrize(("case", "match", "make"), SHAPES, ids=[s[0] for s in SHAPES])
def test_shapes_refuse_nan_and_infinity(case, match, make, bad):
    with pytest.raises(ShapeError, match=match):
        make(bad)


TRANSFORMS = [
    ("at", r"at\(x, y, z\)", lambda s, b: s.at(0, b, 0)),
    ("translate", r"at\(x, y, z\)", lambda s, b: s.translate(b, 0, 0)),
    ("rotate degrees", "rotate degrees", lambda s, b: s.rotate(b)),
    ("rotate axis", "rotate axis", lambda s, b: s.rotate(30, axis=(0, b, 1))),
    ("rotate about", "rotate about", lambda s, b: s.rotate(30, about=(0, b, 0))),
    ("scale factor", "scale factor", lambda s, b: s.scale(b)),
    ("scale about", "scale about", lambda s, b: s.scale(2, about=(b, 0, 0))),
    ("reflect", "reflect about", lambda s, b: s.reflect("x", about=b)),
    ("mirror", "mirror about", lambda s, b: s.mirror("x", about=b)),
    ("array", "array step", lambda s, b: s.array(3, (b, 0, 0))),
    ("radial", "radial about", lambda s, b: s.radial(3, about=(b, 0, 0))),
]


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.parametrize(("case", "match", "make"), TRANSFORMS, ids=[t[0] for t in TRANSFORMS])
def test_transforms_refuse_nan_and_infinity(case, match, make, bad):
    with pytest.raises(ShapeError, match=match):
        make(box((2, 2, 2), color=C), bad)


def test_finite_shapes_and_transforms_still_build():
    shape = (
        box((2, 3, 2), color=C)
        .at(1, 0, 1)
        .rotate(90)
        .scale(2)
        .mirror("x", about=0)
        .union(capsule((0, 0, 0), (0, 4, 0), 1, C, r_end=0.5), cone(1, 2, r_top=0, color=C))
    )
    assert shape.voxelize().count() > 0


def _part(pivot):
    return Model([Part("body", box((2, 2, 2), color=C), pivot=pivot)])


@pytest.mark.parametrize("bad", BAD)
def test_a_part_s_pivot_must_be_finite(bad):
    with pytest.raises(ModelError, match="pivot"):
        Part("body", box((2, 2, 2), color=C), pivot=(1, bad, 1))
    with pytest.raises(ShapeError, match="about_x"):
        Part("arm_l", box((2, 2, 2), color=C), pivot=(1, 1, 1)).mirrored("arm_r", about_x=bad)


@pytest.mark.parametrize("bad", [*BAD, 0, -2])
def test_an_asset_s_size_must_be_finite_and_positive(bad):
    with pytest.raises(ModelError, match="size"):
        asset(name="crate", size=(2, bad, 2))(lambda p: _part((1, 0, 1)))


def test_finite_pivots_and_sizes_still_build():
    built = asset(name="crate", size=(2, 2, 2))(lambda p: _part((1, 0, 1))).build(P)
    assert built.size == (2, 2, 2) and built.parts[0].pivot == (1, 0, 1)
    # ground_y was already a whole number of voxels: NaN and infinity never passed it.
    with pytest.raises(ModelError, match="ground_y"):
        asset(name="crate", size=(2, 2, 2), ground_y=float("nan"))(lambda p: _part(None))


def _solid():
    return box((6, 6, 6), color=C)


def _build(shape):
    return shape.voxelize()


MODIFIERS = [
    ("smooth_union k", "smooth_union k", lambda b: _solid().smooth_union(sphere(2, color=C), b)),
    ("taper top", "taper top", lambda b: _solid().taper(b)),
    ("displace amp", "displace amp", lambda b: _solid().displace(b)),
    ("displace scale", "displace scale", lambda b: _solid().displace(1, scale=b)),
    ("displace seed", "displace seed", lambda b: _solid().displace(1, seed=b)),
    ("displace octaves", "displace octaves", lambda b: _solid().displace(1, octaves=b)),
    ("round", "round radius", lambda b: _solid().round(b)),
    ("chamfer", "chamfer", lambda b: _solid().chamfer(b)),
    ("hollow", "hollow thickness", lambda b: _solid().hollow(b)),
    ("erode", "erode radius", lambda b: _solid().erode(b)),
    ("dilate", "dilate radius", lambda b: _solid().dilate(b)),
    ("paint_faces depth", "paint_faces depth", lambda b: _solid().paint_faces("top", C, depth=b)),
    ("gradient lo", "gradient lo", lambda b: _solid().gradient("y", P.wood, lo=b)),
    ("gradient hi", "gradient hi", lambda b: _solid().gradient("y", P.wood, hi=b)),
    ("dither amount", "dither amount", lambda b: _solid().dither(P.wood, amount=b)),
    ("dither seed", "dither seed", lambda b: _solid().dither(P.wood, seed=b)),
    ("dither mirror", "dither mirror", lambda b: _solid().dither(P.wood, mirror=("x", b))),
    ("darken_crevices", "darken_crevices steps", lambda b: _solid().darken_crevices(b)),
    ("highlight_edges", "highlight_edges steps", lambda b: _solid().highlight_edges(b)),
    ("masonry block", "masonry block", lambda b: _solid().masonry(P.wood, block=(4, b, 4))),
    ("masonry seed", "masonry seed", lambda b: _solid().masonry(P.wood, seed=b)),
    ("decal at", "decal at", lambda b: _solid().decal(["x"], {"x": C}, at=(0, b))),
    ("band at", "band at", lambda b: band("y", b)),
    ("band height", "band height", lambda b: band("y", 0, b)),
    ("region lo", "region lo", lambda b: region((0, b, 0), (1, 1, 1))),
    ("region hi", "region hi", lambda b: region((0, 0, 0), (1, b, 1))),
    ("stripes period", "stripes period", lambda b: stripes("y", b)),
    ("stripes width", "stripes width", lambda b: stripes("y", 4, b)),
    ("stripes offset", "stripes offset", lambda b: stripes("y", 4, 1, b)),
    ("checker size", "checker size", lambda b: checker(b)),
    ("fracture seed", "fracture seed", lambda b: fracture(_solid(), seed=b)),
    ("fracture min_voxels", "fracture min_voxels", lambda b: fracture(_solid(), min_voxels=b)),
    ("fracture pieces", "fracture pieces", lambda b: fracture(_solid(), pieces=b)),
]


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.parametrize(("case", "match", "make"), MODIFIERS, ids=[m[0] for m in MODIFIERS])
def test_modifiers_and_paint_refuse_nan_and_infinity(case, match, make, bad):
    # Selections are lazy about nothing: they check when made. Modifiers check when made too,
    # so no voxelizing is needed to see the error.
    with pytest.raises((ShapeError, ModelError), match=match):
        make(bad)


def test_finite_modifiers_still_build():
    shape = (
        _solid()
        .smooth_union(sphere(2, color=C), 1)
        .taper(0.5)
        .displace(1, seed=3)
        .round(1)
        .hollow(1)
        .erode(0.5)
        .dilate(1)
        .paint_faces("top", C, depth=2)
        .gradient("y", P.wood, lo=0, hi=6)
        .dither(P.wood, amount=0.5, mirror=("x", 3))
        .masonry(P.wood, seed=2)
        .paint(C, where=band("y", 0, 2) & region((0, 0, 0), (9, 9, 9)) & stripes("x", 4, 2))
    )
    assert shape.voxelize().count() > 0


def _debris():
    return chunks("rock", box((6, 6, 6), color=C), pieces=2)


GENERATORS = [
    ("swing amount", "swing amount", lambda b: swing("arm", "x", b)),
    ("swing phase", "swing phase", lambda b: swing("arm", "x", 10, phase=b)),
    ("swing cycles", "swing cycles", lambda b: swing("arm", "x", 10, cycles=b)),
    ("swing center", "swing center", lambda b: swing("arm", "x", 10, center=b)),
    ("bob height", "bob height", lambda b: bob("body", b)),
    ("bob cycles", "bob cycles", lambda b: bob("body", 1, cycles=b)),
    ("bob phase", "bob phase", lambda b: bob("body", 1, phase=b)),
    ("shake amount", "shake amount", lambda b: shake("body", b)),
    ("shake rate", "shake rate", lambda b: shake("body", 1, rate=b)),
    ("shake seed", "shake seed", lambda b: shake("body", 1, seed=b)),
    ("spin_track turns", "spin_track turns", lambda b: spin_track("fan", b)),
    ("burst speed", "burst speed", lambda b: burst(_debris(), speed=b)),
    ("burst up", "burst up", lambda b: burst(_debris(), up=b)),
    ("burst spin", "burst spin", lambda b: burst(_debris(), spin=b)),
    ("burst gravity", "burst gravity", lambda b: burst(_debris(), gravity=b)),
    ("burst rate", "burst rate", lambda b: burst(_debris(), rate=b)),
    ("burst seed", "burst seed", lambda b: burst(_debris(), seed=b)),
    ("burst start", "burst start", lambda b: burst(_debris(), start=b)),
    ("burst origin", "burst origin", lambda b: burst(_debris(), origin=(0, b, 0))),
]


@pytest.mark.parametrize("bad", BAD)
@pytest.mark.parametrize(("case", "match", "make"), GENERATORS, ids=[g[0] for g in GENERATORS])
def test_motion_generators_refuse_nan_and_infinity(case, match, make, bad):
    with pytest.raises(ModelError, match=match):
        make(bad)
