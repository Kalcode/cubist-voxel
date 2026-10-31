"""The manifest: the documented contract between Cubist and an engine."""

import json
import textwrap
from pathlib import Path

import jsonschema
import pytest
from pygltflib import GLTF2

from cubist import Model, Part, asset, box, clip, hinge
from cubist.build.runner import build_script
from cubist.core.palette import Palette
from cubist.export import ExportBlocked, export_asset
from cubist.io.gltf import used_colors
from cubist.io.vox import read_vox, write_vox
from cubist.manifest import MANIFEST_VERSION, SCHEMA_PATH, build_manifest

P = Palette({"wood": ["#3b2414", "#8a5a32"]})
SCHEMA = json.loads(Path(SCHEMA_PATH).read_text())


def chest(p):
    return Model(
        parts=[
            Part("body", box((4, 4, 4), color=p.wood.dark), pivot=(2, 0, 2)),
            Part(
                "lid",
                box((4, 1, 4), color=p.wood.light).at(0, 4, 0),
                parent="body",
                pivot=(2, 4, 0),
                joint=hinge("-x", (0, 110)),
            ),
        ],
        sockets={"handle": (2, 5, 4, "lid")},
        clips=[clip("open", 0.5, {"lid": [0, 100]}, events=[(0.0, "creak")])],
    )


def built(fn=chest, size=(4, 5, 4)):
    return asset(name="chest", size=size)(fn).build(P)


def test_manifest_validates_against_schema():
    m = build_manifest(built(), voxel_size_m=0.0625, files={"glb": "chest.glb"})
    jsonschema.validate(m, SCHEMA)
    assert m["cubist_manifest"] == MANIFEST_VERSION == 1
    assert m["units"] == {"length": "voxel", "angle": "degree", "time": "second", "spin": "turn"}
    lid = m["parts"][1]
    assert lid["joint"] == {"type": "hinge", "axis": [-1.0, 0.0, 0.0], "limits": [0.0, 110.0]}
    assert lid["parent"] == "body" and lid["pivot"] == [2, 4, 0] and lid["vox_model"] == "lid"
    assert m["sockets"] == {
        "handle": {"position": [2.0, 5.0, 4.0], "part": "lid", "forward": None, "up": None}
    }
    assert m["clips"][0]["name"] == "open" and m["clips"][0]["events"] == [[0.0, "creak"]]


def test_manifest_static_asset_has_no_clips_and_null_joints():
    def crate(p):
        return box((4, 4, 4), color=p.wood.dark)

    m = build_manifest(built(crate, (4, 4, 4)), voxel_size_m=0.0625)
    jsonschema.validate(m, SCHEMA)
    assert m["clips"] == [] and m["parts"][0]["joint"] is None


def test_manifest_records_glb_origin_matching_glb_root():
    from cubist.io.gltf import encode_glb

    a = built()
    m = build_manifest(a, voxel_size_m=0.5)
    g = GLTF2.load_from_bytes(encode_glb(a, P.rgba_table(), voxel_size_m=0.5))
    assert [-v * 0.5 for v in m["glb_origin"]] == g.nodes[0].translation
    assert build_manifest(a, voxel_size_m=0.5, center=False)["glb_origin"] == [0, 0, 0]


def test_ground_y_lifts_the_glb_origin_in_the_manifest_and_the_glb():
    from cubist.io.gltf import encode_glb

    @asset(name="house", size=(8, 64, 6), ground_y=48)
    def house(p):
        return box((8, 64, 6), color=p.wood.dark)

    a = house.build(P)
    m = build_manifest(a, voxel_size_m=0.5)
    assert m["glb_origin"] == [4, 48, 3]
    g = GLTF2.load_from_bytes(encode_glb(a, P.rgba_table(), voxel_size_m=0.5))
    assert g.nodes[0].translation == [-2.0, -24.0, -1.5]
    assert build_manifest(a, voxel_size_m=0.5, center=False)["glb_origin"] == [0, 0, 0]


def test_manifest_socket_without_part_names_the_nearest():
    def fn(p):
        m = chest(p)
        return Model(m.parts, sockets={"top": (2, 6, 2)}, clips=m.clips)

    assert build_manifest(built(fn), voxel_size_m=1)["sockets"]["top"]["part"] == "lid"


@pytest.fixture
def project(tmp_path):
    (tmp_path / "style.toml").write_text(
        textwrap.dedent("""\
        [project]
        name = "t"
        voxel_size_m = 0.25
        [palette]
        wood = ["#3b2414", "#8a5a32"]
        """)
    )
    (tmp_path / "assets").mkdir()
    script = tmp_path / "assets" / "chest.py"
    script.write_text(
        textwrap.dedent("""\
        from cubist import *

        @asset(name="chest", size=(4, 5, 4), grounded=True)
        def chest(p):
            return Model(
                parts=[
                    Part("body", box((4, 4, 4), color=p.wood.dark), pivot=(2, 0, 2)),
                    Part("lid", box((4, 1, 4), color=p.wood.light).at(0, 4, 0), parent="body",
                         pivot=(2, 4, 0), joint=hinge("-x", (0, 110))),
                ],
                clips=[clip("open", 0.5, {"lid": [0, 100]})],
            )
        """)
    )
    return tmp_path


def test_export_keeps_the_ground_line_of_the_build(project):
    # Export reads the asset back from out/, so build.json must carry ground_y.
    script = project / "assets" / "house.py"
    script.write_text(
        textwrap.dedent("""\
        from cubist import *

        @asset(name="house", size=(8, 64, 6), ground_y=48)
        def house(p):
            return box((8, 64, 6), color=p.wood.dark)
        """)
    )
    [r] = build_script(script)
    m = json.loads(Path(export_asset(r.vox_path.parent)["manifest"]).read_text())
    assert m["glb_origin"] == [4, 48, 3]


def test_export_writes_manifest_beside_glb(project):
    [r] = build_script(project / "assets" / "chest.py")
    out = export_asset(r.vox_path.parent)
    assert set(out) == {"glb", "vox", "manifest"}
    m = json.loads(Path(out["manifest"]).read_text())
    assert Path(out["manifest"]).name == "chest.cubist.json"
    assert m["files"] == {"glb": "chest.glb", "vox": "chest.vox"}
    assert m["voxel_size_m"] == 0.25 and m["clips"][0]["name"] == "open"
    jsonschema.validate(m, SCHEMA)


def test_hand_edited_animated_vox_blocks_export(project):
    [r] = build_script(project / "assets" / "chest.py")
    scene = read_vox(r.vox_path)
    lid = scene.parts[1].grid
    lid.data[0, 0, 0] = 0  # someone deletes a voxel in an editor
    write_vox(r.vox_path, scene)
    with pytest.raises(ExportBlocked, match="'lid' changed"):
        export_asset(r.vox_path.parent)
    assert "manifest" in export_asset(r.vox_path.parent, force=True)


def test_hand_edited_static_vox_still_exports(project):
    script = project / "assets" / "chest.py"
    script.write_text(
        script.read_text()
        .replace(', joint=hinge("-x", (0, 110))', "")
        .replace('clips=[clip("open", 0.5, {"lid": [0, 100]})],', "")
    )
    [r] = build_script(script)
    scene = read_vox(r.vox_path)
    scene.parts[1].grid.data[0, 0, 0] = 0
    write_vox(r.vox_path, scene)
    export_asset(r.vox_path.parent)


RAMPS = Palette(
    {
        "slate": ["#10141c", "#2a3140", "#4a5468", "#7a8699"],
        "brick": ["#3a1a14", "#7a3a2a", "#b0604a"],
        "moss": ["#1a2a14", "#3a5a2a", "#6a9a4a"],
        "cloth": {"red": "#aa2222", "blue": "#2222aa"},
        "cork": "#c8a070",
    }
)


def ramps_asset(fn, tags=()):
    return asset(name="hut", size=(4, 4, 4), tags=list(tags))(fn).build(RAMPS)


def test_manifest_lists_the_asset_tags():
    def hut(p):
        return box((4, 4, 4), color=p.cork)

    m = build_manifest(ramps_asset(hut, ["building", "town"]), voxel_size_m=0.0625)
    assert m["tags"] == ["building", "town"]
    assert build_manifest(ramps_asset(hut), voxel_size_m=0.0625)["tags"] == []


def test_manifest_ramps_list_each_used_list_ramp_dark_to_light():
    def hut(p):
        return Model(
            parts=[
                Part("a", box((2, 2, 2), color=p.slate.ramp[1]), pivot=(0, 0, 0)),
                Part("b", box((2, 2, 2), color=p.brick.ramp[2]).at(2, 0, 0), pivot=(2, 0, 0)),
                Part("c", box((2, 2, 2), color=p.cloth.red).at(0, 2, 0), pivot=(0, 2, 0)),
                Part("d", box((2, 2, 2), color=p.cork).at(2, 2, 0), pivot=(2, 2, 0)),
            ]
        )

    built_hut = ramps_asset(hut)
    built_hut.glow = {RAMPS.slate.ramp[1].index: 1.5}
    m = build_manifest(built_hut, voxel_size_m=0.0625)
    assert list(m["ramps"]) == ["slate", "brick"]
    assert m["ramps"]["slate"] == [c.index for c in RAMPS.slate.ramp]
    assert m["ramps"]["brick"] == [c.index for c in RAMPS.brick.ramp]
    assert "moss" not in m["ramps"]
    assert "cloth" not in m["ramps"]
    assert "cork" not in m["ramps"]
    # the indices the voxels carry are the ones listed
    for i in used_colors(built_hut):
        owners = [n for n in RAMPS.list_ramps() if i in [c.index for c in getattr(RAMPS, n).ramp]]
        for n in owners:
            assert i in m["ramps"][n]
    # and so are the indices the manifest's named colours carry
    assert m["colors"]
    for name, entry in m["colors"].items():
        ramp_name = name.split(".")[0]
        if ramp_name in m["ramps"]:
            assert entry["index"] in m["ramps"][ramp_name]


def test_manifest_ramps_empty_when_no_list_ramp_is_used():
    def hut(p):
        return box((4, 4, 4), color=p.cork)

    assert build_manifest(ramps_asset(hut), voxel_size_m=0.0625)["ramps"] == {}


def test_manifest_with_tags_and_ramps_validates_against_the_schema():
    def hut(p):
        return box((4, 4, 4), color=p.slate.dark)

    m = build_manifest(
        ramps_asset(hut, ["building"]), voxel_size_m=0.0625, files={"glb": "hut.glb"}
    )
    assert m["tags"] == ["building"]
    assert m["ramps"]
    jsonschema.validate(m, SCHEMA)


@pytest.mark.parametrize("bad", [0, 256])
def test_manifest_schema_rejects_a_ramp_index_out_of_range(bad):
    def hut(p):
        return box((4, 4, 4), color=p.slate.dark)

    m = build_manifest(ramps_asset(hut), voxel_size_m=0.0625, files={"glb": "hut.glb"})
    jsonschema.validate(m, SCHEMA)
    m["ramps"]["slate"][0] = bad
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(m, SCHEMA)
