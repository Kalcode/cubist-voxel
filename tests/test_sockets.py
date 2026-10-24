"""Sockets: attachment points, optionally with a direction (muzzles, wand tips, grips)."""

import json

import jsonschema
import numpy as np
import pytest
from pygltflib import GLTF2

from cubist import Model, ModelError, Part, asset, box, socket
from cubist.core.model import BuiltAsset
from cubist.core.palette import Palette
from cubist.core.socket import Socket
from cubist.io.gltf import encode_glb
from cubist.io.vox import decode_vox, encode_vox
from cubist.manifest import SCHEMA_PATH, build_manifest

P = Palette({"wood": ["#3b2414", "#8a5a32"]})


def gun(p, sockets):
    return Model(
        [Part("body", box((2, 2, 6), color=p.wood.dark), pivot=(1, 0, 3))], sockets=sockets
    )


def built(sockets):
    return asset(name="gun", size=(2, 2, 6))(lambda p: gun(p, sockets)).build(P)


def test_tuple_sockets_still_work_and_keep_list_summary():
    a = built({"grip": (1, 0, 1), "tip": (1, 1, 6, "body")})
    assert a.sockets["grip"] == Socket((1.0, 0.0, 1.0))
    assert a.summary()["sockets"] == {"grip": [1, 0, 1], "tip": [1, 1, 6, "body"]}


def test_socket_with_forward_up_orients_glb_node():
    a = built({"muzzle": socket((1, 1, 6), part="body", forward="+x", up="+y")})
    g = GLTF2.load_from_bytes(encode_glb(a, P.rgba_table(), voxel_size_m=1))
    node = next(n for n in g.nodes if n.name == "muzzle")
    x, y, z, w = node.rotation
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip
    assert np.allclose(rot @ [0, 0, 1], [1, 0, 0]) and np.allclose(rot @ [0, 1, 0], [0, 1, 0])


def test_unoriented_socket_node_has_no_rotation():
    a = built({"grip": (1, 0, 1)})
    g = GLTF2.load_from_bytes(encode_glb(a, P.rgba_table(), voxel_size_m=1))
    assert next(n for n in g.nodes if n.name == "grip").rotation is None


def test_socket_up_is_made_perpendicular():
    s = socket((0, 0, 0), forward="+z", up=(0, 1, 1))
    assert np.allclose(s.up, (0, 1, 0)) and np.allclose(s.forward, (0, 0, 1))


def test_socket_up_must_not_parallel_forward():
    with pytest.raises(ModelError, match="up"):
        socket((0, 0, 0), forward="+y", up="+y")


def test_socket_unknown_part_is_an_error():
    with pytest.raises(ModelError, match="unknown part 'barrel'"):
        built({"muzzle": socket((1, 1, 6), part="barrel", forward="+z")})


def test_oriented_socket_round_trips_through_summary_and_vox():
    a = built({"muzzle": socket((1, 1, 6), part="body", forward="+z")})
    summary = a.summary()
    assert summary["sockets"]["muzzle"] == {
        "position": [1.0, 1.0, 6.0], "part": "body", "forward": [0.0, 0.0, 1.0],
        "up": [0.0, 1.0, 0.0],
    }  # fmt: skip
    again = BuiltAsset.from_vox(decode_vox(encode_vox(a.to_vox_scene())), summary, P)
    assert again.sockets == a.sockets


def test_legacy_summary_sockets_load():
    a = built({"grip": (1, 0, 1)})
    summary = {**a.summary(), "sockets": {"grip": [1, 0, 1], "tip": [1, 1, 6, "body"]}}
    for p in summary["parts"]:
        p.pop("hidden")
    again = BuiltAsset.from_vox(decode_vox(encode_vox(a.to_vox_scene())), summary, P)
    assert again.sockets["tip"] == Socket((1.0, 1.0, 6.0), "body")
    assert again.parts[0].hidden is False


def test_manifest_carries_socket_orientation():
    a = built({"muzzle": socket((1, 1, 6), part="body", forward="+z"), "grip": (1, 0, 1)})
    m = build_manifest(a, voxel_size_m=1)
    jsonschema.validate(m, json.loads(SCHEMA_PATH.read_text()))
    assert m["sockets"]["muzzle"]["forward"] == [0.0, 0.0, 1.0]
    assert m["sockets"]["grip"] == {
        "position": [1.0, 0.0, 1.0],
        "part": "body",
        "forward": None,
        "up": None,
    }
