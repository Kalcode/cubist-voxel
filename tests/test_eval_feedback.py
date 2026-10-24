"""Fixes and features from the milestone 6 evaluation agents' reports."""

import json
import textwrap

import numpy as np
import pytest
from scipy import ndimage

from cubist import Model, Part, asset, box, capsule, sphere
from cubist.build.runner import BuildError, build_script
from cubist.core.palette import Palette
from cubist.core.style import StyleBible
from cubist.dsl import building, carve, checker, faces, line, panel, polyline, pyramid, stripes
from cubist.evaluation import iterations
from cubist.examples import promote, search_examples
from cubist.lint import lint

P = Palette(
    {"stone": ["#202020", "#404040", "#606060", "#808080"], "white": "#ffffff", "red": "#ff0000"}
)
ST = P.stone


def coords(grid):
    return sorted(map(tuple, grid.voxels()[0].tolist()))


# --- colorless cutters on painted shapes (bug) -------------------------------


def test_colorless_cutter_subtracts_from_a_painted_shape():
    painted = box((8, 8, 8), color=ST.mid).paint(P.white, where=faces("top"))
    g = (painted - box((2, 2, 2)).at(3, 3, 6)).voxelize()
    assert g.count() == 512 - 8
    assert g.get(3, 3, 6) == 0


def test_colorless_cutter_intersects_a_painted_shape():
    painted = box((8, 8, 8), color=ST.mid).dither(ST, seed=1)
    g = (painted & box((2, 2, 2)).at(3, 3, 3)).voxelize()
    assert g.count() == 8


def test_colorless_cutter_of_a_composite_shape():
    painted = box((8, 8, 8), color=ST.mid).paint(P.white)
    cutter = box((1, 8, 1)).at(1, 0, 1) | sphere(1.5).at(6, 6, 6)
    assert (painted - cutter).voxelize().count() < 512


# --- errors that point at the script ------------------------------------------


@pytest.fixture
def project(tmp_path):
    (tmp_path / "style.toml").write_text(
        '[palette]\nstone = ["#202020", "#404040", "#606060"]\nwhite = "#ffffff"\n'
    )
    (tmp_path / "assets").mkdir()
    return tmp_path


def script(project, name, body):
    path = project / "assets" / f"{name}.py"
    path.write_text(
        textwrap.dedent(f"""\
        from cubist import *


        @asset(name="{name}", size=(8, 8, 8))
        def make(p):
            {body}
    """)
    )
    return path


def test_missing_color_error_names_the_primitive_and_its_line(project):
    path = script(
        project, "nocolor", "return box((2, 2, 2)).at(4, 0, 0) | box((2, 2, 2), color=p.stone.mid)"
    )
    with pytest.raises(BuildError) as err:
        build_script(path)
    message = str(err.value)
    assert "box" in message and "nocolor.py:6" in message
    assert "cutter" in message  # explains when no color is fine


def test_failed_builds_are_logged_and_count_as_iterations(project):
    path = script(project, "flaky", "return box((2, 2, 2), color=p.wod.mid)")
    with pytest.raises(BuildError):
        build_script(path)
    script(project, "flaky", "return box((2, 2, 2), color=p.stone.mid)")
    build_script(path)
    lines = (project / "out" / "flaky" / "history.jsonl").read_text().splitlines()
    first, last = json.loads(lines[0]), json.loads(lines[-1])
    assert first["ok"] is False and "wod" in first["error"]
    assert last["ok"] is True
    assert iterations(project, "flaky") == 2


# --- reporting ---------------------------------------------------------------


def test_summary_voxels_count_overlapping_parts_once():
    @asset(name="overlap")
    def overlap(p):
        return Model(
            parts=[
                Part("a", box((2, 2, 2), color=p.white)),
                Part("b", box((2, 2, 2), color=p.white).at(1, 0, 0)),
            ]
        )

    s = overlap.build(P).summary()
    assert s["voxels"] == 12 and s["part_voxels"] == 16


def test_examples_search_without_criteria_lists_the_library(project):
    path = script(project, "crate", "return box((8, 8, 8), color=p.stone.mid)")
    build_script(path)
    promote(project, "crate", note="ok")
    assert [e["name"] for e in search_examples(project)] == ["crate"]


# --- symmetry about an explicit plane -----------------------------------------


def test_symmetry_can_name_its_plane():
    def spike(p):
        # 7 wide centered on 15.5 inside an even-width asset
        return box((7, 2, 1), color=p.white).at(12, 0, 0) | box((1, 3, 1), color=p.red).at(15, 2, 0)

    style = StyleBible(path=None, name="t", voxel_size_m=0.0625, palette=P)
    a = asset(name="s", size=(32, 5, 1), symmetric=("x", 15.5))(spike).build(P)
    assert not [f for f in lint(a, style).findings if f.check == "symmetry"]
    b = asset(name="s", size=(32, 5, 1), symmetric="x")(spike).build(P)
    assert [f for f in lint(b, style).findings if f.check == "symmetry"]


# --- thin lines --------------------------------------------------------------


def test_line_is_a_six_connected_staircase_by_default():
    g = line((0.5, 0.5, 0.5), (6.5, 6.5, 0.5), color=P.white).voxelize()
    _, pieces = ndimage.label(g.data > 0)  # face connectivity
    assert pieces == 1
    rows = [sum(1 for _, y, _ in coords(g) if y == yy) for yy in range(7)]
    assert max(rows) <= 2
    assert g.get(0, 0, 0) and g.get(6, 6, 0)


def test_line_connect_26_is_one_voxel_per_step():
    g = line((0.5, 0.5, 0.5), (6.5, 6.5, 0.5), color=P.white, connect=26).voxelize()
    assert coords(g) == [(i, i, 0) for i in range(7)]


def test_shallow_lines_have_no_gaps():
    g = line((0.5, 0.5, 0.5), (12.5, 3.5, 2.5), color=P.white).voxelize()
    _, pieces = ndimage.label(g.data > 0)
    assert pieces == 1


def test_polyline_joins_its_segments():
    g = polyline([(0.5, 0.5, 0.5), (4.5, 4.5, 0.5), (8.5, 0.5, 0.5)], color=P.white).voxelize()
    _, pieces = ndimage.label(g.data > 0)
    assert pieces == 1 and g.get(4, 4, 0) and g.get(8, 0, 0)


# --- tapered capsules and panels ---------------------------------------------


def test_capsule_tapers_from_r_to_r_end():
    a, b, r0, r1 = np.array([3.0, 3.0, 3.0]), np.array([3.0, 13.0, 3.0]), 2.5, 0.8

    def inside(c):
        t = np.linspace(0, 1, 2001)
        pts = a + t[:, None] * (b - a)
        return np.min(np.linalg.norm(pts - c, axis=1) - (r0 + t * (r1 - r0))) <= 1e-6

    g = capsule(tuple(a), tuple(b), r=r0, r_end=r1, color=P.white).voxelize()
    expected = [
        (x, y, z)
        for x in range(7)
        for y in range(17)
        for z in range(7)
        if inside(np.array([x + 0.5, y + 0.5, z + 0.5]))
    ]
    assert set(coords(g)) == set(expected)


def test_panel_tilted_45_degrees_is_watertight():
    wing = panel([(0, 0, 0), (12, 0, 12), (12, 10, 12), (0, 10, 0)], color=P.white).voxelize()
    room = box((12, 10, 12)).at(0, 0, 0).occupancy()
    filled = wing.lookup(np.argwhere(room.data > 0)) > 0
    air = np.zeros(room.shape, dtype=bool)
    air[tuple(np.argwhere(room.data > 0)[~filled].T)] = True
    _, sides = ndimage.label(air)  # face-connected air on either side of the plate
    assert sides == 2


def test_axis_aligned_panel_is_one_voxel_thick():
    g = panel([(0, 0, 4), (6, 0, 4), (6, 5, 4)], color=P.white).voxelize()
    lo, hi = g.bounds()
    assert hi[2] - lo[2] == 1


# --- shading on thin geometry --------------------------------------------------


def test_highlight_edges_skips_the_faces_of_thin_plates():
    with building(P):
        g = box((6, 6, 1), color=ST[1]).highlight_edges().voxelize()
    assert g.get(2, 2, 0) == ST[1].index  # plate interior: exposed both sides on z only
    assert g.get(0, 0, 0) == ST[2].index  # rim corner


def test_highlight_edges_facing_the_light():
    with building(P):
        g = box((4, 4, 4), color=ST[1]).highlight_edges(facing=("top",)).voxelize()
    assert g.get(0, 3, 0) == ST[2].index  # top edge
    assert g.get(0, 0, 0) == ST[1].index  # bottom edge stays


def test_darken_crevices_does_not_bleed_through_thin_walls():
    wall = box((6, 6, 1), color=ST[2]).at(0, 0, 3)
    shelf = box((6, 1, 3), color=ST[2]).at(0, 2, 0)
    with building(P):
        g = (wall | shelf).darken_crevices().voxelize()
    assert g.get(2, 3, 3) == ST[2].index  # the wall's far side shows no checkerboard


# --- pattern paint and selections ---------------------------------------------


def test_stripes_and_checker_selections():
    g = box((8, 1, 1), color=P.white).paint(P.red, where=stripes("x", period=4, width=2)).voxelize()
    assert [g.get(x, 0, 0) == P.red.index for x in range(8)] == [True, True, False, False] * 2
    c = box((4, 4, 1), color=P.white).paint(P.red, where=checker(2)).voxelize()
    assert c.get(0, 0, 0) == P.red.index and c.get(2, 0, 0) == P.white.index
    assert c.get(2, 2, 0) == P.red.index


def test_masonry_paints_blocks_with_mortar():
    wall = box((16, 8, 2), color=ST.mid)
    a = wall.masonry(ST, block=(4, 2, 4), mortar=P.white, seed=3).voxelize()
    b = wall.masonry(ST, block=(4, 2, 4), mortar=P.white, seed=3).voxelize()
    assert a == b
    used = a.colors_used()
    assert P.white.index in used and len(used) >= 3
    assert all(a.get(x, 0, 1) == P.white.index for x in range(16))  # bed joint at y = 0


def test_carve_cuts_a_recess_and_colors_its_back():
    wall = box((8, 8, 4), color=ST.mid)
    g = carve(wall, box((2, 3, 2)).at(3, 2, 2), color=P.red).voxelize()
    assert g.get(3, 2, 3) == 0  # cut
    assert g.get(3, 2, 1) == P.red.index  # the back of the recess
    assert g.get(3, 2, 0) == ST.mid.index  # ...with the wall's own back behind it
    assert g.get(0, 0, 2) == ST.mid.index
    assert g == wall.carve(box((2, 3, 2)).at(3, 2, 2), color=P.red).voxelize()


def test_carve_color_stays_off_the_outside_of_thin_walls():
    wall = box((8, 8, 3), color=ST.mid)
    g = carve(wall, box((2, 3, 2)).at(3, 2, 1), color=P.red).voxelize()
    assert g.get(3, 2, 0) == ST.mid.index  # a 1-voxel back would show red outside


def test_decal_paints_the_first_surface_seen_from_a_face():
    eyes = ["wk", "k."]
    g = (
        box((8, 8, 8), color=ST.mid)
        .decal(eyes, {"w": P.white, "k": P.red}, face="front", at=(2, 6))
        .voxelize()
    )
    assert g.get(2, 6, 7) == P.white.index and g.get(3, 6, 7) == P.red.index
    assert g.get(2, 5, 7) == P.red.index and g.get(3, 5, 7) == ST.mid.index
    top = box((4, 4, 4), color=ST.mid).decal(["w"], {"w": P.white}, face="top", at=(1, 2))
    assert top.voxelize().get(1, 3, 2) == P.white.index


def test_decal_on_the_right_face_reads_like_the_right_view():
    # In the right view, u runs along z descending: column 0 is the front (high z).
    g = (
        box((4, 4, 4), color=ST.mid)
        .decal(["wk"], {"w": P.white, "k": P.red}, face="right", at=(3, 3))
        .voxelize()
    )
    assert g.get(3, 3, 3) == P.white.index and g.get(3, 3, 2) == P.red.index


def test_pyramid_narrows_to_a_one_voxel_tip_on_odd_widths():
    g = pyramid((5, 3, 5), color=P.white).voxelize()
    layers = [sum(1 for _, y, _ in coords(g) if y == yy) for yy in range(3)]
    assert layers == sorted(layers, reverse=True)
    assert layers[-1] == 1 and g.get(2, 2, 2)


def test_gradient_over_an_explicit_range():
    g = box((1, 8, 1), color=ST.dark).gradient("y", ST, lo=0, hi=4).voxelize()
    assert [g.get(0, y, 0) for y in range(5)] == [c.index for c in ST.ramp] + [ST.light.index]


def test_dither_can_be_mirror_symmetric():
    g = box((8, 4, 4), color=ST[1]).dither(ST, seed=2, mirror=("x", 4)).voxelize()
    voxels = dict(zip(map(tuple, g.voxels()[0].tolist()), g.voxels()[1].tolist(), strict=True))
    assert len(set(voxels.values())) > 1
    assert all(voxels[(7 - x, y, z)] == c for (x, y, z), c in voxels.items())
