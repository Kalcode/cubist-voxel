from cubist import *


@asset(name="desk", size=(16, 12, 11), tags=["prop", "wood", "furniture"], grounded=True)
def desk(p):
    top = box((16, 1, 10), color=p.wood.light).at(0, 11, 0).highlight_edges()
    legs = [box((2, 11, 2), color=p.wood.dark).at(x, 0, z) for x in (0, 14) for z in (0, 8)]
    # Apron under the top, with a pocket on the right for the drawer.
    apron = box((12, 3, 10), color=p.wood.mid).at(2, 8, 0) - box((5, 2, 9)).at(8, 8, 1)
    body = (top | apron).union(*legs).dither(p.wood.ramp, seed=3, amount=0.25).darken_crevices()

    # The drawer: an open tray filling the pocket, with a lighter front and a knob.
    drawer = box((5, 2, 9), color=p.wood.mid).at(8, 8, 1) - box((3, 1, 7)).at(9, 9, 2)
    drawer = drawer.paint(p.wood.light, where=band("z", 9, 1))
    drawer = drawer | box((1, 1, 1), color=p.metal.light).at(10, 8, 10)
    return Model(
        parts=[
            Part("desk", body, pivot=(8, 0, 5)),
            Part("drawer", drawer, parent="desk", pivot=(10, 8, 5), joint=slide("+z", 6)),
        ],
        clips=[
            clip("open", 0.7, {"drawer": [0, 6]}),
            clip("close", 0.5, {"drawer": [6, 0]}),
        ],
    )
