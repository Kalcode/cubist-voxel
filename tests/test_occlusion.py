"""Occlusion tags: which parts an engine hides when they come between the camera and the player
(a roof, an upper storey, a tree's crown)."""

import json

import jsonschema
import pytest

from cubist import Model, ModelError, Part, asset, box
from cubist.core.model import BuiltAsset
from cubist.core.palette import Palette
from cubist.core.style import StyleBible
from cubist.io.vox import decode_vox, encode_vox
from cubist.lint import lint
from cubist.manifest import SCHEMA_PATH, build_manifest

P = Palette({"wood": ["#3b2414", "#8a5a32"], "stone": ["#4a4a52", "#8e8e98"]})


def house(p, roof_tag="roof", chimney_tag=None):
    walls = Part("walls", box((8, 6, 8), color=p.stone.dark))
    roof = Part(
        "roof", box((8, 3, 8), color=p.wood.dark).at(0, 6, 0), parent="walls", occlusion=roof_tag
    )
    chimney = Part(
        "chimney",
        box((2, 3, 2), color=p.stone.light).at(5, 9, 5),
        parent="roof",
        occlusion=chimney_tag,
    )
    return Model([walls, roof, chimney])


def built(fn=house, tags=("building", "stone")):
    return asset(name="house", size=(8, 12, 8), tags=list(tags))(fn).build(P)


def test_manifest_parts_carry_their_resolved_occlusion():
    m = build_manifest(built(), voxel_size_m=1)
    jsonschema.validate(m, json.loads(SCHEMA_PATH.read_text()))
    tags = {p["name"]: p["occlusion"] for p in m["parts"]}
    # The chimney hangs from the roof, so it hides with it.
    assert tags == {"walls": None, "roof": "roof", "chimney": "roof"}


def test_a_child_s_own_tag_wins_over_its_parent_s():
    m = build_manifest(built(lambda p: house(p, "upper", "roof")), voxel_size_m=1)
    assert {p["name"]: p["occlusion"] for p in m["parts"]}["chimney"] == "roof"


def test_occlusion_survives_the_build_summary_and_vox():
    a = built()
    again = BuiltAsset.from_vox(decode_vox(encode_vox(a.to_vox_scene())), a.summary(), P)
    assert {p.name: p.occlusion for p in again.parts} == {
        "walls": None, "roof": "roof", "chimney": None
    }  # fmt: skip


def test_mirrored_parts_keep_their_tag():
    crown = Part("crown_l", box((2, 2, 2), color=P.wood.light), occlusion="canopy")
    _, twin = crown.mirrored("crown_r", about_x=4)
    assert twin.occlusion == "canopy"


def test_unknown_occlusion_value_suggests_the_known_ones():
    with pytest.raises(ModelError, match="roof.*upper.*canopy"):
        Part("roof", box((2, 2, 2)), occlusion="rooftop")


def lint_info(a):
    style = StyleBible(path=None, name="t", voxel_size_m=0.0625, palette=P)
    return [f for f in lint(a, style).findings if f.check == "occlusion"]


def test_building_with_no_roof_or_upper_tag_gets_an_info():
    [f] = lint_info(built(lambda p: house(p, roof_tag=None)))
    assert f.severity == "info" and "occlusion=" in f.message


def test_tagged_roof_or_non_building_is_quiet():
    assert lint_info(built()) == []
    assert lint_info(built(lambda p: house(p, roof_tag=None), tags=("prop",))) == []


def test_untagged_chimney_above_a_tagged_roof_is_quiet():
    # The smithy: its chimney rises from the hearth, not the roof, and is the highest part.
    def smithy(p):
        m = house(p)
        stack = Part("stack", box((2, 6, 2), color=p.stone.light).at(0, 6, 0), parent="walls")
        roof = Part(
            "roof", box((8, 3, 8), color=p.wood.dark).at(0, 6, 0) - box((2, 3, 2)).at(0, 6, 0),
            parent="walls", occlusion="roof",
        )  # fmt: skip
        return Model([m.parts[0], roof, stack])

    assert lint_info(built(smithy)) == []
