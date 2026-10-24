from cubist import *


@asset(name="biplane", size=(21, 11, 22), tags=["vehicle", "toy"], grounded=True, symmetric="x")
def biplane(p):
    # Facing +Z. Everything is centered on x = 10.5, so widths are odd.
    body = box((3, 3, 15), color=p.cap.mid).at(9, 4, 4)
    body = body.carve(box((1, 1, 2)).at(10, 6, 9), color=p.metal.dark)  # cockpit
    cowl = box((5, 5, 3), color=p.metal.mid).at(8, 3, 17).chamfer(1)
    tips = region((0, 0, 0), (2, 11, 22)) | region((19, 0, 0), (21, 11, 22))
    wings = [
        box((21, 1, 5), color=p.stem.dark).at(0, y, 11).paint(p.cap.mid, where=tips) for y in (4, 8)
    ]
    struts = [box((1, 3, 1), color=p.wood.dark).at(x, 5, z) for x in (2, 18) for z in (12, 14)]
    tail = box((9, 1, 3), color=p.stem.dark).at(6, 5, 4) | box((1, 4, 3), color=p.cap.mid).at(
        10, 7, 4
    )
    gear = [box((1, 3, 1), color=p.metal.dark).at(x, 1, 14) for x in (8, 12)]
    wheels = [box((1, 2, 2), color=p.metal.dark).at(x, 0, 13) for x in (7, 13)]
    plane = body | cowl
    plane = plane.union(*wings, *struts, tail, *gear, *wheels).highlight_edges(facing=("top",))

    blade = box((1, 9, 1), color=p.wood.mid).at(10, 1, 20)
    spinner = box((1, 1, 1), color=p.cap.light).at(10, 5, 21)
    return Model(
        parts=[
            Part("plane", plane, pivot=(10.5, 0, 11)),
            Part(
                "prop",
                blade | spinner,
                parent="plane",
                pivot=(10.5, 5.5, 20.5),
                joint=spin("z", rps=2),
            ),
        ],
        clips=[clip("spin", 1.0, {"prop": [0, 2]}, loop=True, ease="linear")],
    )
