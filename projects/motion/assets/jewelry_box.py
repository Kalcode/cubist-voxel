from cubist import *


@asset(name="jewelry_box", size=(10, 8, 8), tags=["prop", "wood", "gold", "container"], grounded=True)
def jewelry_box(p):
    corners = (band("x", 0) | band("x", 9)) & (band("z", 0) | band("z", 7))

    # Body: gold feet, open-topped box with a velvet cushion so the inside reads when the lid lifts.
    feet = (box((2, 1, 2), color=p.gold[1]) | box((2, 1, 2), color=p.gold[1]).at(8, 0, 0)).mirror("z", about=4)
    body = box((10, 4, 8), color=p.wood[1]).at(0, 1, 0) - box((8, 3, 6)).at(1, 2, 1)
    body = body.dither(p.wood.ramp, seed=3, amount=0.12)
    body = body.paint(p.gold.mid, where=corners)
    body = body.paint(p.gold.mid, where=region((4, 3, 7), (6, 5, 8)))  # clasp plate
    # A thick cushion brings the velvet up near the rim, where it shows.
    cushion = box((8, 2, 6), color=p.red[1]).at(1, 2, 1).paint(p.red.mid, where=faces("top"))
    # The treasure: a gem standing proud of the rim and a gold ring, set off-center on purpose.
    gem = box((2, 2, 2), color=p.blue.mid).at(4, 4, 3).paint(p.blue.light, where=band("y", 5))
    gem = gem.paint(p.glow.light, where=region((4, 5, 3), (5, 6, 4)))  # catches the light
    ring = box((3, 1, 3), color=p.gold.mid).at(1, 4, 4) - box((1, 1, 1)).at(2, 4, 5)
    base = (feet | body | cushion | gem | ring).darken_crevices()

    # Lid: a hollow slab with a raised panel; gold corners carry the body's trim up.
    lid = box((10, 2, 8), color=p.wood[1]).at(0, 5, 0) | box((8, 1, 6), color=p.wood.mid).at(1, 7, 1)
    lid = lid - box((8, 2, 6)).at(1, 5, 1)
    lid = lid.paint(p.gold.mid, where=corners)
    lid = lid.paint(p.gold.light, where=region((4, 5, 7), (6, 6, 8)))  # hasp over the clasp
    lid = lid.paint(p.gold.mid, where=region((4, 7, 3), (6, 8, 5)))  # emblem on top
    lid = lid.highlight_edges(facing=("top", "left"))

    # A four-point sparkle that flashes beside the gem as the lid comes up.
    glint = box((3, 1, 1), color=p.glow.light).at(6, 5, 4) | box((1, 3, 1), color=p.glow.light).at(7, 4, 4)

    return Model(
        parts=[
            Part("body", base, pivot=(5, 0, 4)),
            # Hinged along the back edge: hinge("-x") lifts the front edge up and over.
            Part("lid", lid, parent="body", pivot=(5, 5, 0), joint=hinge("-x", (0, 120))),
            Part("glint", glint, parent="body", pivot=(7, 4, 4.5), hidden=True),
        ],
        clips=[
            clip(
                "open",
                0.6,
                [
                    # Pops past open, swings back, settles at 105.
                    TrackSpec("lid", "joint", [(0, 0), (0.22, 118), (0.36, 98), (0.48, 108), (0.6, 105)]),
                    visible("glint", [(0, 0), (0.26, 1), (0.36, 0), (0.44, 1), (0.52, 0)]),
                ],
                events=[(0.0, "click"), (0.26, "sparkle")],
            ),
            clip(
                "close",
                0.3,
                {"lid": [(0, 105), (0.1, 85), (0.18, 45), (0.23, 0), (0.26, 6), (0.3, 0)]},
                ease="linear",
                events=[(0.23, "snap")],
            ),
        ],
    )
