from cubist import *

# The floor is at y = 3: the frame's top and the closed hatch's top are flush with it.
FLOOR = 3


@asset(name="cellar_hatch", size=(14, 4, 14), tags=["prop", "wood", "stone", "door"], grounded=True)
def cellar_hatch(p):
    # Stone frame: a two-voxel border of blocks around a dark shaft going down.
    frame = box((14, FLOOR, 14), color=p.stone.mid).masonry(p.stone, block=(4, 3, 3), seed=3)
    frame = frame.carve(box((10, FLOOR, 10)).at(2, 1, 2), color=p.stone.dark)
    frame = frame.highlight_edges(facing=("top", "left"))
    # Iron hinge leaves set into the back of the frame, where the straps cross the gap.
    for x in (3, 10):
        frame = frame.paint(p.metal[1], where=region((x, FLOOR - 1, 0), (x + 1, FLOOR, 2)))

    # Plank hatch, two voxels thick, filling the opening.
    hatch = box((10, 2, 10), color=p.wood.mid).at(2, FLOOR - 2, 2)
    # Five planks running front to back, alternating shades, lightly grained.
    hatch = hatch.paint(p.wood.light, where=stripes("x", period=4, width=2, offset=2))
    hatch = hatch.dither(p.wood.ramp, seed=8, amount=0.1, mirror=("x", 7))
    # Underside: dark planks with two battens and a diagonal brace, seen when it swings up.
    under = faces("bottom")
    hatch = hatch.paint(p.wood[1], where=under)
    brace = line((3.5, 1.5, 4.5), (10.5, 1.5, 9.5), r=0.9)
    battens = band("z", 3, 2) | band("z", 9, 2) | inside(brace)
    hatch = hatch.paint(p.wood.mid, where=under & battens)
    # Iron straps run from the hinge edge toward the front, and a band crosses near the front.
    straps = region((3, FLOOR - 1, 2), (4, FLOOR, 11)) | region((10, FLOOR - 1, 2), (11, FLOOR, 11))
    hatch = hatch.paint(p.metal[1], where=straps)
    hatch = hatch.paint(p.metal.dark, where=region((6, FLOOR - 1, 10), (8, FLOOR, 11)))  # staple
    # Ring pull lying flat on top, near the front edge.
    ring = torus(1.6, 0.6, color=p.metal.mid).at(7, FLOOR + 0.5, 9)
    hatch = hatch | ring.highlight_edges(facing=("top", "left"))

    return Model(
        parts=[
            Part("frame", frame, pivot=(7, 0, 7)),
            # Hinged along its back edge at the top face: hinge("-x") lifts the front edge up and
            # over toward the back, so the hatch swings away from the frame.
            Part(
                "hatch",
                hatch,
                parent="frame",
                pivot=(7, FLOOR, 2),
                joint=hinge("-x", (0, 120)),
            ),
        ],
        clips=[
            # A tug that rattles it in the frame, then heaved up past vertical, swinging back to
            # rest against its hinges.
            clip(
                "open",
                1.1,
                {"hatch": [(0, 0), (0.08, 4), (0.16, 0), (0.8, 118), (0.98, 104), (1.1, 106)]},
                events=[(0.08, "rattle"), (0.2, "creak")],
            ),
            # Pushed back over vertical, then falls faster and faster and slams with a bounce.
            clip(
                "close",
                0.65,
                {"hatch": [(0, 106), (0.2, 92), (0.38, 55), (0.5, 0), (0.57, 4), (0.65, 0)]},
                ease="linear",
                events=[(0.5, "slam"), (0.5, "emit:dust")],
            ),
        ],
        # Dust blows out of the gap at the front edge when it slams shut, fanning along the floor.
        emitters={
            "dust": emitter(
                (7, FLOOR + 0.5, 12.5),
                "cellar_hatch_dust",
                burst=18,
                life=0.7,
                speed=5,
                spread=85,
                gravity=-0.5,
                grow=(0.6, 1.6),
                part="frame",
            )
        },
    )
