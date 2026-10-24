from cubist import *

# Fourteen wide, symmetric about x = 7. The carcass is solid except for a pocket behind the
# middle drawer; that drawer is a real open tray that slides +Z out of the pocket. The top and
# bottom drawers are fixed fronts standing one voxel proud of the carcass.
DRAWERS = (2, 6, 10)  # bottom y of each drawer front, bottom to top
MIDDLE = DRAWERS[1]


def front(p, y):
    """A drawer front at height y: mid wood with its top row catching the light."""
    plate = box((10, 3, 1), color=p.wood.mid).at(2, y, 7)
    return plate.paint(p.wood.light, where=band("y", y + 2))


def knobs(p, y):
    return box((1, 1, 1), color=p.gold.mid).at(4, y + 1, 8) | box((1, 1, 1), color=p.gold.mid).at(
        9, y + 1, 8
    )


@asset(
    name="dresser",
    size=(14, 14, 9),
    tags=["prop", "wood", "furniture"],
    grounded=True,
    symmetric="x",
)
def dresser(p):
    pocket = box((10, 3, 6)).at(2, MIDDLE, 1)  # z 1..7: through the front, back panel kept
    # Plain subtract, not carve: the side and back walls are one voxel thick, so a carved color
    # shows through on the outside. Only the pocket's floor and ceiling go dark.
    carcass = box((12, 12, 7), color=p.wood[1]).at(1, 1, 0) - pocket
    carcass = carcass.paint(p.wood.dark, where=region((2, MIDDLE - 1, 1), (12, MIDDLE + 4, 6)))
    top = box((14, 1, 9), color=p.wood.mid).at(0, 13, 0)
    top = top.dither(p.wood, seed=7, amount=0.15, mirror=("x", 7))
    top = top.highlight_edges(facing=("top", "left"))
    feet = [box((2, 1, 2), color=p.wood.dark).at(x, 0, z) for x in (1, 11) for z in (0, 5)]
    fixed = [front(p, y) | knobs(p, y) for y in (DRAWERS[0], DRAWERS[2])]
    body = carcass.union(*feet).dither(p.wood, seed=4, amount=0.15, mirror=("x", 7))
    body = (body | top).union(*fixed).darken_crevices()

    # The sliding drawer: an open tray exactly filling the pocket, its front and knobs.
    tray = box((10, 3, 6), color=p.wood.mid).at(2, MIDDLE, 1) - box((8, 2, 4)).at(3, MIDDLE + 1, 2)
    tray = tray.paint(p.wood[1], where=region((3, MIDDLE, 2), (11, MIDDLE + 1, 6)))
    drawer = tray | front(p, MIDDLE) | knobs(p, MIDDLE)

    return Model(
        parts=[
            Part("body", body, pivot=(7, 0, 4.5)),
            Part("drawer", drawer, parent="body", pivot=(7, MIDDLE, 4), joint=slide("+z", 5)),
        ],
        sockets={"pull": socket((7, MIDDLE + 1.5, 9), part="drawer", forward="+z")},
        clips=[
            # Eases out of the pocket and slows to a stop five voxels out.
            clip("open", 0.55, {"drawer": [(0, 0), (0.45, 5), (0.55, 5)]}, events=[(0, "slide")]),
            # Pushed shut: slow at first, then fast into the stop.
            clip(
                "close",
                0.35,
                {"drawer": [(0, 5), (0.15, 3.8), (0.3, 0), (0.35, 0)]},
                ease="linear",
                events=[(0.3, "thud")],
            ),
        ],
    )
