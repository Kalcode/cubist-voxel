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
    box,
    capsule,
    cone,
    cylinder,
    ellipsoid,
    extrude,
    heightmap,
    hinge,
    lathe,
    line,
    panel,
    polyline,
    pyramid,
    slide,
    sphere,
    spin,
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
