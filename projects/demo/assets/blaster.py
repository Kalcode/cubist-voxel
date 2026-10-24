from cubist import *


@asset(name="blaster", size=(3, 10, 17), tags=["prop", "weapon", "metal"], symmetric="x")
def blaster(p):
    # Faces +Z. Three voxels wide, centered on x = 1.5.
    body = box((3, 3, 10), color=p.metal.mid).at(0, 5, 2)
    grip = extrude([(2, 0), (5, 0), (6, 5), (3, 5)], depth=3, axis="x", color=p.wood[1])
    grip = grip.paint(p.wood.dark, where=stripes("y", period=2, width=1))
    guard = box((1, 1, 3), color=p.metal.dark).at(1, 4, 6)
    # A barrel shroud bored through, so the muzzle reads as a ring from the front.
    shroud = box((3, 3, 4), color=p.metal.dark).at(0, 5, 12) - box((1, 1, 4)).at(1, 6, 12)
    frame = (body | grip | guard | shroud).highlight_edges(facing=("top",))

    slide_top = box((3, 2, 9), color=p.metal.light).at(0, 8, 2)
    slide_top = slide_top.paint(p.metal.mid, where=stripes("z", period=2, width=1) & band("y", 9))

    star = box((1, 3, 1), color=p.fire.light).at(1, 5, 0) | box((3, 1, 1), color=p.fire.light).at(
        0, 6, 0
    )
    flash = (star | box((1, 1, 1), color=p.fire.dark).at(1, 6, 0)).at(0, 0, 16)
    return Model(
        parts=[
            Part("frame", frame, pivot=(1.5, 2, 4)),
            Part("slide", slide_top, parent="frame", pivot=(1.5, 8, 6), joint=slide("-z", 3)),
            Part("flash", flash, parent="frame", pivot=(1.5, 6.5, 16.5), hidden=True),
        ],
        sockets={
            "grip": socket((1.5, 2, 4), part="frame", forward="+z"),
            "muzzle": socket((1.5, 6.5, 16), part="frame", forward="+z"),
        },
        clips=[
            clip(
                "fire",
                0.35,
                [
                    TrackSpec("slide", "joint", [(0, 0), (0.04, 3), (0.3, 0)]),
                    visible("flash", [(0, 1), (0.06, 0)]),
                    turn("frame", "x", [(0, 0), (0.05, -10), (0.35, 0)]),  # muzzle kicks up
                ],
                events=[(0.0, "fire")],
            )
        ],
    )
