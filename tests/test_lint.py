from cubist import Model, Part, asset, box
from cubist.core.palette import Palette
from cubist.core.style import Budgets, Rules, StyleBible
from cubist.lint import lint

P = Palette({"wood": ["#3b2414", "#5e3a1f", "#8a5a32"], "ink": "#000000", "leaf": "#3f6b2a"})


def style(**kw):
    budgets = Budgets(**{k: v for k, v in kw.items() if k.startswith("max_")})
    rules = Rules(**{k: v for k, v in kw.items() if not k.startswith("max_")})
    return StyleBible(
        path=None, name="t", voxel_size_m=0.0625, palette=P, budgets=budgets, rules=rules
    )


def build(fn, **meta):
    return asset(name="a", **meta)(fn).build(P)


def checks(report, severity=None):
    return {f.check for f in report.findings if severity is None or f.severity == severity}


def test_clean_asset_has_no_errors_or_warnings():
    a = build(lambda p: box((4, 4, 4), color=p.wood.mid), size=(4, 4, 4), grounded=True)
    report = lint(a, style())
    assert report.ok
    assert checks(report, "error") == set() and checks(report, "warning") == set()


def test_voxels_outside_declared_size_are_an_error():
    a = build(lambda p: box((5, 4, 4), color=p.wood.mid), size=(4, 4, 4))
    report = lint(a, style())
    assert "bounds" in checks(report, "error")
    assert not report.ok
    finding = next(f for f in report.findings if f.check == "bounds")
    assert finding.details["outside"] == 16


def test_part_over_256_is_an_error():
    a = build(lambda p: box((300, 1, 1), color=p.wood.mid))
    assert "bounds" in checks(lint(a, style()), "error")


def test_pivot_outside_its_part_is_an_error():
    def fn(p):
        return Model(parts=[Part("body", box((2, 2, 2), color=p.wood.mid), pivot=(5, 0, 0))])

    report = lint(build(fn), style())
    assert "pivots" in checks(report, "error")


def test_pivot_on_the_part_boundary_is_fine():
    def fn(p):
        return Model(parts=[Part("lid", box((2, 2, 2), color=p.wood.mid), pivot=(2, 2, 0))])

    assert "pivots" not in checks(lint(build(fn), style()))


def test_socket_outside_the_asset_is_an_error():
    def fn(p):
        return Model(
            parts=[Part("b", box((2, 2, 2), color=p.wood.mid))], sockets={"hand": (9, 0, 0)}
        )

    assert "pivots" in checks(lint(build(fn), style()), "error")


def test_floating_pieces_warn_unless_allowed():
    def fn(p):
        return box((3, 3, 3), color=p.wood.mid) | box((1, 1, 1), color=p.leaf).at(6, 0, 0)

    report = lint(build(fn), style())
    assert "floating" in checks(report, "warning")
    finding = next(f for f in report.findings if f.check == "floating")
    assert finding.details["pieces"] == 1 and finding.details["voxels"] == 1
    assert "floating" not in checks(lint(build(fn, allow_floating=True), style()))


def test_diagonal_contact_counts_as_floating():
    def fn(p):
        return box((2, 2, 2), color=p.wood.mid) | box((1, 1, 1), color=p.leaf).at(2, 2, 2)

    assert "floating" in checks(lint(build(fn), style()))


def test_symmetry_warns_when_asset_differs_from_its_mirror():
    def lopsided(p):
        return box((8, 2, 2), color=p.wood.mid) | box((1, 1, 1), color=p.leaf).at(0, 2, 0)

    def even(p):
        return box((8, 2, 2), color=p.wood.mid) | box((1, 1, 1), color=p.leaf).at(0, 2, 0).mirror(
            "x", about=4
        )

    assert "symmetry" in checks(lint(build(lopsided, size=(8, 3, 2), symmetric="x"), style()))
    assert "symmetry" not in checks(lint(build(even, size=(8, 3, 2), symmetric="x"), style()))
    tolerant = style(symmetry_tolerance=1)
    assert "symmetry" not in checks(lint(build(lopsided, size=(8, 3, 2), symmetric="x"), tolerant))


def test_symmetry_counts_color_differences():
    def fn(p):
        return box((4, 1, 1), color=p.wood.mid) | box((1, 1, 1), color=p.leaf)

    report = lint(build(fn, size=(4, 1, 1), symmetric="x"), style())
    finding = next(f for f in report.findings if f.check == "symmetry")
    assert finding.details["mismatched"] == 2


def test_part_level_symmetry_uses_the_asset_center():
    def fn(p):
        return Model(
            parts=[
                Part("body", box((6, 2, 2), color=p.wood.mid).at(1, 0, 0), symmetric="x"),
                Part("flag", box((1, 3, 1), color=p.leaf).at(0, 2, 0)),
            ]
        )

    report = lint(build(fn, size=(8, 5, 2)), style())
    assert "symmetry" not in checks(report)


def test_grounded_asset_must_touch_the_ground():
    a = build(lambda p: box((2, 2, 2), color=p.wood.mid).at(0, 2, 0), grounded=True)
    assert "grounding" in checks(lint(a, style()), "warning")


def test_budgets_warn():
    a = build(lambda p: box((4, 4, 4), color=p.wood.mid) | box((1, 1, 1), color=p.leaf).at(0, 4, 0))
    report = lint(a, style(max_voxels=10, max_colors=1, max_triangles=8, max_parts=0))
    found = [f for f in report.findings if f.check == "budgets"]
    assert {f.details["budget"] for f in found} == {
        "max_voxels",
        "max_colors",
        "max_triangles",
        "max_parts",
    }
    assert all(f.severity == "warning" for f in found)


def test_thin_features_are_info():
    a = build(lambda p: box((6, 6, 1), color=p.wood.mid))
    report = lint(a, style())
    finding = next(f for f in report.findings if f.check == "thin")
    assert finding.severity == "info" and finding.details["voxels"] == 36


def test_hidden_interior_is_info():
    a = build(lambda p: box((5, 5, 5), color=p.wood.mid))
    finding = next(f for f in lint(a, style()).findings if f.check == "hidden")
    assert finding.severity == "info" and finding.details["voxels"] == 27


def test_enclosed_cavity_walls_are_hidden_too():
    a = build(lambda p: box((5, 5, 5), color=p.wood.mid) - box((1, 1, 1)).at(2, 2, 2))
    finding = next(f for f in lint(a, style()).findings if f.check == "hidden")
    assert finding.details["voxels"] == 26


def test_pure_black_breaks_the_no_pure_black_rule():
    a = build(lambda p: box((2, 2, 2), color=p.ink))
    assert "palette" in checks(lint(a, style(no_pure_black=True)), "error")
    assert "palette" not in checks(lint(a, style()))


def test_colors_outside_the_palette_are_an_error():
    a = build(lambda p: box((2, 2, 2), color=p.wood.mid))
    foreign = P.rgba_table().copy()
    foreign[P.wood.mid.index] = (1, 2, 3, 255)
    report = lint(a, style(), rgba=foreign)
    finding = next(f for f in report.findings if f.check == "palette")
    assert finding.severity == "error" and "#010203" in finding.message


def test_forbidden_dither_is_an_error():
    a = build(lambda p: box((4, 4, 4), color=p.wood[1]).dither(p.wood, seed=1))
    assert "palette" in checks(lint(a, style(dither="forbidden")), "error")
    assert "palette" not in checks(lint(a, style()))


def test_report_summary_and_dict():
    a = build(lambda p: box((5, 4, 4), color=p.wood.mid), size=(4, 4, 4))
    report = lint(a, style())
    d = report.to_dict()
    assert d["errors"] == 1 and d["ok"] is False
    text = report.summary()
    assert text.splitlines()[0].startswith("lint: 1 error")
    assert "ERROR bounds:" in text


def test_empty_asset_does_not_crash():
    a = build(lambda p: box((2, 2, 2), color=p.wood.mid) - box((3, 3, 3)).at(-0.5, -0.5, -0.5))
    assert a.voxel_count() == 0
    assert lint(a, style()).ok


def test_closed_shell_walls_are_not_thin():
    a = build(lambda p: box((6, 6, 6), color=p.wood.mid).hollow(1))
    assert "thin" not in checks(lint(a, style()))


def test_socket_with_a_named_part_is_checked_by_position():
    def fn(p):
        return Model(
            parts=[Part("b", box((2, 2, 2), color=p.wood.mid))],
            sockets={"tip": (1, 2, 1, "b"), "far": (9, 9, 9, "b")},
        )

    report = lint(build(fn), style())
    messages = [f.message for f in report.findings if f.check == "pivots"]
    assert len(messages) == 1 and "far" in messages[0]


def test_overlapping_parts_warn():
    def fn(p):
        return Model(
            parts=[
                Part("a", box((2, 2, 2), color=p.wood.mid)),
                Part("b", box((2, 2, 2), color=p.leaf).at(1, 0, 0)),
            ]
        )

    finding = next(f for f in lint(build(fn), style()).findings if f.check == "overlap")
    assert finding.severity == "warning"
    assert finding.details["pairs"] == [{"parts": ["a", "b"], "voxels": 4}]


def test_thin_info_lists_clusters():
    def fn(p):
        return box((6, 6, 1), color=p.wood.mid) | box((1, 1, 4), color=p.wood.mid).at(9, 0, 0)

    finding = next(
        f for f in lint(build(fn, allow_floating=True), style()).findings if f.check == "thin"
    )
    assert [c["voxels"] for c in finding.details["clusters"]] == [36, 4]


# --- scale against the style bible's [scale] --------------------------------------------


def scaled(**scale):
    return StyleBible(path=None, name="t", voxel_size_m=0.0625, palette=P, scale=scale)


def tall(height, tags):
    def fn(p):
        return box((4, height, 2), color=p.wood.mid)

    return build(fn, size=(4, height, 2), tags=tags)


def scale_findings(a, style):
    return [f for f in lint(a, style).findings if f.check == "scale"]


def test_tagged_asset_far_from_its_scale_height_warns():
    # The convention: a [scale] key <tag>_height applies to assets tagged <tag>.
    style = scaled(door_height=34, table_height=14)
    [f] = scale_findings(tall(20, ["prop", "wood", "door"]), style)
    assert f.severity == "warning"
    assert "door" in f.message and "20 voxels (1.2 m)" in f.message and "34" in f.message
    assert scale_findings(tall(36, ["prop", "door"]), style) == []  # a frame a bit over is fine
    assert len(scale_findings(tall(30, ["table"]), style)) == 1  # twice table height


def test_building_shorter_than_a_door_warns_as_a_doll_house():
    style = scaled(door_height=34)
    [f] = scale_findings(tall(20, ["building", "wood"]), style)
    assert "door" in f.message and "doll" in f.message
    assert scale_findings(tall(48, ["building"]), style) == []


def test_no_scale_section_or_untagged_assets_are_not_checked():
    assert scale_findings(tall(20, ["building", "door"]), scaled()) == []
    assert scale_findings(tall(20, ["prop"]), scaled(door_height=34)) == []
