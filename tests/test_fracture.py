"""Fracture: a shape split into connected chunks that are the asset's parts."""

import numpy as np
import pytest
from scipy import ndimage

from cubist import Model, ModelError, asset, box, chunks, cylinder, fracture
from cubist.core.palette import Palette
from cubist.dsl.context import building

P = Palette({"wood": ["#3b2414", "#5e3a1f", "#8a5a32"], "metal": ["#4b505a", "#c2c8d0"]})


def barrel(p):
    body = cylinder(r=5, h=12, color=p.wood.mid).at(6, 0, 6)
    return body.paint(p.metal.light, where=box((12, 1, 12)).at(0, 3, 0))


def grids(shape_fn, pieces=8, seed=1, build_seed=0):
    with building(P, build_seed):
        return [s.voxelize() for s in fracture(shape_fn(P), pieces=pieces, seed=seed)]


def test_chunks_cover_the_shape_exactly_and_keep_colors():
    with building(P, 0):
        whole = barrel(P).voxelize()
    parts = grids(barrel)
    assert 6 <= len(parts) <= 8
    coords = np.concatenate([g.voxels()[0] for g in parts])
    assert len(coords) == whole.count() == len({tuple(c) for c in coords.tolist()})
    for g in parts:
        c, colors = g.voxels()
        assert np.array_equal(whole.lookup(c), colors)


def test_each_chunk_is_one_connected_piece_and_not_a_sliver():
    parts = grids(barrel)
    total = sum(g.count() for g in parts)
    for g in parts:
        _, n = ndimage.label(g.data > 0)
        assert n == 1 and g.count() >= max(4, total // (8 * 4))


def test_fracture_is_seeded_and_follows_the_build_seed():
    a, b = grids(barrel, seed=1), grids(barrel, seed=1)
    assert [g.voxels()[0].tolist() for g in a] == [g.voxels()[0].tolist() for g in b]
    c = grids(barrel, seed=2)
    assert [g.count() for g in a] != [g.count() for g in c]
    d = grids(barrel, seed=1, build_seed=5)
    assert [g.count() for g in a] != [g.count() for g in d]


def test_chunks_are_parts_pivoted_at_their_centers_of_mass():
    def fn(p):
        return Model(chunks("barrel", barrel(p), pieces=6, seed=3))

    a = asset(name="barrel", size=(12, 12, 12))(fn).build(P)
    assert all(p.name.startswith("barrel_") for p in a.parts)
    for part in a.parts:
        centers = part.grid.voxels()[0] + 0.5
        assert np.allclose(part.pivot, centers.mean(axis=0))


def test_fracture_needs_at_least_two_pieces():
    with building(P, 0), pytest.raises(ModelError, match="pieces"):
        fracture(barrel(P), pieces=1)
