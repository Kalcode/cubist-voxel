import pytest

from cubist import Model, ModelError, Part, asset, box, capsule
from cubist.core.palette import Palette

P = Palette({"wood": ["#3b2414", "#8a5a32"], "skin": {"goblin": "#6b8e23"}})


def test_asset_decorator_records_metadata_and_builds():
    @asset(name="crate", size=(4, 4, 4), tags=["prop", "wood"])
    def crate(p):
        return Model(parts=[Part("body", box((4, 4, 4), color=p.wood.dark), pivot=(2, 0, 2))])

    assert (crate.name, crate.size, crate.tags) == ("crate", (4, 4, 4), ("prop", "wood"))
    built = crate.build(P)
    assert built.name == "crate"
    assert [part.name for part in built.parts] == ["body"]
    assert built.parts[0].grid.count() == 64
    assert built.parts[0].pivot == (2, 0, 2)


def test_bare_asset_decorator_works_like_the_called_form():
    @asset
    def crate(p):
        return box((4, 4, 4), color=p.wood.dark)

    assert (crate.name, crate.size, crate.tags) == ("crate", None, ())
    assert crate.build(P).parts[0].grid.count() == 64


def test_unset_pivot_defaults_to_the_bottom_center_of_the_part():
    @asset(name="post", size=(3, 6, 3))
    def post(p):
        return Model(parts=[Part("pole", box((1, 5, 2), color=p.wood.dark).at(1, 1, 0))])

    assert post.build(P).parts[0].pivot == (1.5, 1, 1)


def test_returning_a_bare_shape_makes_a_single_part_model():
    @asset(name="cube", size=(2, 2, 2))
    def cube(p):
        return box((2, 2, 2), color=p.wood.light)

    built = cube.build(P)
    assert [part.name for part in built.parts] == ["cube"]
    assert built.voxel_count() == 8


def test_mirrored_part_returns_original_and_reflected_twin():
    arm = capsule((0, 0, 0), (0, -6, 0), r=1, color=P.skin.goblin).at(4, 15, 6)
    left, right = Part("arm_l", arm, parent="torso", pivot=(4, 15, 6)).mirrored("arm_r", about_x=8)
    assert left.name == "arm_l" and right.name == "arm_r"
    assert right.pivot == (12, 15, 6)
    assert right.parent == "torso"
    lo, hi = right.shape.voxelize().bounds()
    assert (lo[0], hi[0]) == (11, 13)


def test_model_flattens_mirrored_pairs():
    torso = Part("torso", box((6, 7, 4), color=P.wood.dark).at(5, 9, 4))
    arm = Part("arm_l", box((1, 4, 1), color=P.wood.light).at(4, 11, 6), parent="torso")
    model = Model(parts=[torso, arm.mirrored("arm_r", about_x=8)])
    assert [p.name for p in model.parts] == ["torso", "arm_l", "arm_r"]


def test_duplicate_part_names_are_an_error():
    a = Part("body", box((1, 1, 1), color=P.wood.dark))
    with pytest.raises(ModelError, match="body"):
        Model(parts=[a, a])


def test_unknown_parent_is_an_error():
    with pytest.raises(ModelError, match="torso"):
        Model(parts=[Part("head", box((1, 1, 1), color=P.wood.dark), parent="torso")])


def test_asset_returning_something_else_is_an_error():
    @asset(name="bad", size=(1, 1, 1))
    def bad(p):
        return 42

    with pytest.raises(ModelError, match="Model"):
        bad.build(P)


def test_built_asset_converts_to_vox_scene():
    @asset(name="two", size=(4, 4, 4))
    def two(p):
        return Model(
            parts=[
                Part("a", box((1, 1, 1), color=p.wood.dark)),
                Part("b", box((1, 1, 1), color=p.wood.light).at(3, 0, 0)),
            ]
        )

    scene = two.build(P).to_vox_scene()
    assert [part.name for part in scene.parts] == ["a", "b"]
    assert tuple(scene.palette[2]) == (0x8A, 0x5A, 0x32, 255)


def test_star_import_exposes_the_whole_dsl():
    import cubist
    import cubist.dsl

    assert set(cubist.dsl.__all__) - {"building"} <= set(cubist.__all__)


def test_socket_naming_an_unknown_part_is_an_error():
    body = Part("body", box((1, 1, 1), color=P.wood.dark))
    with pytest.raises(ModelError, match="arm"):
        Model(parts=[body], sockets={"hand": (0, 0, 0, "arm")})
    Model(parts=[body], sockets={"hand": (0, 0, 0, "body")})


def test_ground_y_defaults_to_zero_and_must_sit_inside_the_asset():
    @asset(name="crate", size=(4, 4, 4))
    def crate(p):
        return box((4, 4, 4), color=p.wood.dark)

    assert crate.ground_y == 0 and crate.build(P).ground_y == 0

    def make(ground_y):
        return asset(name="crate", size=(4, 4, 4), ground_y=ground_y)(crate.fn)

    assert make(3).build(P).ground_y == 3
    with pytest.raises(ModelError, match="ground_y"):
        make(-1).build(P)
    with pytest.raises(ModelError, match="ground_y.*height"):
        make(4).build(P)
