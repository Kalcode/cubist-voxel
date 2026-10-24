"""Motion lint: limits, loops, collisions, reach."""

from cubist import Model, Part, box, clip, hinge, slide, spin
from cubist.core.model import AssetDef
from cubist.core.palette import Palette
from cubist.lint.motion import check_motion

P = Palette({"wood": ["#3b2414", "#5e3a1f", "#8a5a32"]})


def build(fn, size=None):
    return AssetDef(fn, "t", size, ()).build(P)


def only(findings, check):
    found = [f for f in findings if f.check == check]
    assert len(found) == 1, findings
    return found[0]


def none(findings, check):
    return not [f for f in findings if f.check == check]


def chest(lid_axis="-x", keys=(0, 110), loop=False, limits=(0, 120)):
    def fn(p):
        body = box((14, 7, 10), color=p.wood.mid) - box((12, 6, 8)).at(1, 1, 1)
        lid = box((14, 5, 10), color=p.wood.light).at(0, 7, 0)
        return Model(
            [
                Part("body", body, pivot=(7, 0, 5)),
                Part("lid", lid, parent="body", pivot=(7, 7, 0), joint=hinge(lid_axis, limits)),
            ],
            clips=[clip("open", 0.6, {"lid": list(keys)}, loop=loop)],
        )

    return build(fn, (14, 12, 10))


def test_keys_past_limits_are_errors_naming_clip_part_and_fix():
    f = only(check_motion(chest(keys=(0, 130))), "joint_limits")
    assert f.severity == "error" and "'open'" in f.message and "'lid'" in f.message
    assert "130" in f.message and "limits" in f.message


def test_keys_exactly_at_the_limits_are_fine():
    assert none(check_motion(chest(keys=(0, 120))), "joint_limits")


def test_lid_opening_110_does_not_collide():
    assert none(check_motion(chest()), "motion_collision")


def test_lid_opening_the_wrong_way_collides_and_hints_direction():
    f = only(check_motion(chest(lid_axis="x", keys=(0, 30))), "motion_collision")
    assert f.severity == "warning" and "'lid'" in f.message and "'body'" in f.message
    assert "hinge" in f.message


def goblin(swing):
    def fn(p):
        torso = Part("torso", box((6, 8, 4), color=p.wood.mid).at(3, 6, 2), pivot=(6, 6, 4))
        arm = Part(
            "arm_r",
            box((2, 7, 2), color=p.wood.light).at(1, 7, 3),
            parent="torso",
            pivot=(2, 13, 4),
            joint=hinge("x", (-100, 100)),
        )
        return Model([torso, arm], clips=[clip("walk", 0.8, {"arm_r": swing}, loop=True)])

    return build(fn)


def test_arm_swinging_beside_the_torso_does_not_collide():
    assert none(check_motion(goblin([0, 20, 0, -20, 0])), "motion_collision")


def test_arm_swinging_through_the_torso_collides():
    def fn(p):
        torso = Part("torso", box((6, 8, 4), color=p.wood.mid).at(3, 6, 2), pivot=(6, 6, 4))
        arm = Part(  # inside the torso's x range, in front of it, swinging back through it
            "arm",
            box((2, 7, 2), color=p.wood.light).at(5, 7, 6),
            parent="torso",
            pivot=(6, 13, 7),
            joint=hinge("x", (-100, 100)),
        )
        return Model([torso, arm], clips=[clip("swing", 1.0, {"arm": [0, 90]})])

    assert not none(check_motion(build(fn)), "motion_collision")


def test_open_loop_warns():
    f = only(check_motion(chest(keys=(0, 110), loop=True)), "loop_closure")
    assert f.severity == "warning" and "'open'" in f.message and "end" in f.message


def test_closed_loop_and_full_turn_spin_are_fine():
    assert none(check_motion(goblin([0, 20, 0, -20, 0])), "loop_closure")

    def prop(p):
        blade = Part("prop", box((6, 1, 1), color=p.wood.mid), pivot=(3, 0.5, 0.5), joint=spin("z"))
        return Model([blade], clips=[clip("spin", 1, {"prop": [0, 3]}, loop=True, ease="linear")])

    assert none(check_motion(build(prop)), "loop_closure")


def test_motion_bounds_info_reports_reach():
    f = only(check_motion(chest()), "motion_bounds")
    assert f.severity == "info" and "'open'" in f.message
    lo, hi = f.details["clips"]["open"]["bounds"]
    assert lo[2] < 0  # the lid swings back behind the chest
    assert f.details["clips"]["open"]["beyond"]["-z"] > 0


def test_constant_clip_lints_clean():
    findings = check_motion(chest(keys=(0,)))
    assert [f.severity for f in findings if f.severity != "info"] == []


def test_slide_keys_past_distance_are_errors():
    def fn(p):
        d = Part("drawer", box((6, 2, 6), color=p.wood.mid), joint=slide("+z", 4))
        return Model([d], clips=[clip("open", 1, {"drawer": [0, 5]})])

    f = only(check_motion(build(fn)), "joint_limits")
    assert "'drawer'" in f.message and "distance" in f.message


def test_static_assets_get_no_motion_findings():
    assert check_motion(build(lambda p: box((4, 4, 4), color=p.wood.mid))) == []
