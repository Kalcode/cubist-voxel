from cubist import *


@asset(name="door", size=(14, 21, 4), tags=["prop", "wood", "door"], grounded=True)
def door(p):
    # Frame: two posts and a lintel, three voxels deep.
    frame = box((2, 19, 3), color=p.wood.dark).mirror("x", 7) | box(
        (14, 2, 3), color=p.wood.dark
    ).at(0, 19, 0)
    frame = frame.dither(p.wood.ramp, seed=2, amount=0.25).darken_crevices()

    # Plank leaf flush with the frame's front face, with iron straps and a ring pull.
    leaf = box((10, 19, 1), color=p.wood.mid).at(2, 0, 2)
    leaf = leaf.paint(p.wood.light, where=stripes("x", period=6, width=3, offset=2))
    leaf = leaf.dither(p.wood.ramp, seed=7, amount=0.12)
    leaf = leaf.paint(p.wood.dark, where=stripes("x", period=3, width=1, offset=1))  # seams
    for y in (3, 15):
        leaf = leaf.paint(p.metal.dark, where=band("y", y, 1))
    pull = box((1, 2, 1), color=p.metal.light).at(10, 8, 3)
    leaf = leaf | pull
    return Model(
        parts=[
            Part("frame", frame, pivot=(7, 0, 1.5)),
            # Hinged on its left edge at the front face: swings out toward the viewer.
            Part("leaf", leaf, parent="frame", pivot=(2, 0, 3), joint=hinge("y", (-110, 0))),
        ],
        clips=[
            clip("open", 1.0, {"leaf": [0, -100, -92]}),
            clip("close", 0.5, {"leaf": [-92, 0]}),
        ],
    )
