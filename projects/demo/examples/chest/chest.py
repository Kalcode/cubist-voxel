from cubist import *


@asset(name="chest", size=(14, 12, 10), tags=["prop", "wood", "container"], grounded=True)
def chest(p):
    bands = band(axis="x", at=2, height=1) | band(axis="x", at=11, height=1)

    # Open-topped box so the inside shows when the lid swings up.
    body = box((14, 7, 10), color=p.wood.mid) - box((12, 6, 8)).at(1, 1, 1)
    body = body.paint(p.metal.mid, where=bands)
    body = body.paint(p.metal.light, where=region((6, 4, 9), (8, 6, 10)))  # lock plate
    body = body.dither(p.wood.ramp, seed=5, amount=0.3).darken_crevices()

    # Barrel-vaulted lid: the top half of a cylinder lying along x.
    lid = cylinder(r=5, h=14, axis="x", color=p.wood.mid).at(0, 7, 5) & box((14, 5, 10)).at(0, 7, 0)
    lid = lid.paint(p.metal.mid, where=bands).highlight_edges()
    return Model(
        parts=[
            Part("body", body, pivot=(7, 0, 5)),
            # Hinged along the back edge. hinge("-x") lifts the front edge up and over.
            Part("lid", lid, parent="body", pivot=(7, 7, 0), joint=hinge("-x", (0, 120))),
        ],
        sockets={"lock": (7, 5, 10, "body")},
        clips=[
            clip("open", 0.6, {"lid": [0, 115, 105]}),  # overshoot, then settle back
            clip("close", 0.35, {"lid": [105, 0]}, ease="linear"),  # falls shut
        ],
    )
