from cubist import *

# Faces +Z, symmetric about x = 12: the fuselage has an even width, and the nacelles sit at
# 12 -/+ 6.5 so each is centered on a voxel center (odd widths, so the hubs have a column).
HUB_Y, HUB_Z = 4.5, 17.5
NACELLE_X = {"prop_r": 5.5, "prop_l": 18.5}  # the plane's right is at low x


def airframe(p):
    # Fuselage: a side profile (z, y) intersected with a top profile (x, z).
    side = extrude(
        [(0.5, 5.5), (4, 3), (17.5, 3), (19.5, 4), (19.5, 6), (16, 7), (1, 7)],
        24,
        axis="x",
        color=p.metal.mid,
    )
    top = extrude(
        [(11, 0.5), (13, 0.5), (14, 5), (14, 16.5), (13, 19.5), (11, 19.5), (10, 16.5), (10, 5)],
        9,
        axis="y",
        color=p.metal.mid,
    )
    body = side & top
    body = body.paint(p.red.mid, where=band("y", 4) & (faces("left") | faces("right")))
    windows = band("y", 5) & stripes("z", 2, 1) & region((0, 0, 8), (24, 9, 14))
    body = body.paint(p.blue.mid, where=windows & (faces("left") | faces("right")))
    body = body.paint(p.blue.light, where=region((10, 6, 14), (14, 7, 16)))  # windshield
    body = body.paint_faces("bottom", p.metal[1])

    # Two voxels thick inboard, one at the outer panels.
    wing = extrude(
        [(0, 10.5), (9, 9), (15, 9), (24, 10.5), (24, 13), (0, 13)], 2, axis="y", color=p.metal.mid
    ).at(0, 3, 0)
    wing = wing - box((3, 1, 20)).at(0, 3, 0) - box((3, 1, 20)).at(21, 3, 0)
    wing = wing.paint(p.metal[1], where=band("y", 3))
    wing = wing.paint(p.red.mid, where=region((0, 0, 0), (2, 9, 20)) | region((22, 0, 0), (24, 9, 20)))
    stab = extrude([(7, 0.5), (17, 0.5), (16, 3.5), (8, 3.5)], 1, axis="y", color=p.metal.mid).at(0, 6, 0)
    fin = extrude([(0.5, 6), (4, 6), (2, 9), (0.5, 9)], 2, axis="x", color=p.red.mid).at(11, 0, 0)

    nacelles = []
    for x in NACELLE_X.values():
        nac = lathe([(0, 0.8), (2, 1.5), (7, 1.6), (9, 1.4)], axis="z", color=p.metal.mid)
        nac = nac.at(x, HUB_Y, 8).paint(p.metal.dark, where=band("z", 15))  # cowl ring
        nac = nac.paint_faces("bottom", p.metal[1])
        strut = box((1, 1, 1), color=p.metal[1]).at(x - 0.5, 2, 11)
        wheel = box((1, 2, 2), color=p.metal.dark).at(x - 0.5, 0, 11)
        nacelles += [nac, strut, wheel]
    nose_gear = box((2, 1, 1), color=p.metal[1]).at(11, 2, 16) | box(
        (2, 2, 2), color=p.metal.dark
    ).at(11, 0, 15)

    plane = body.union(wing, stab, fin, *nacelles, nose_gear)
    return plane.highlight_edges(facing=("top",))


def prop(p, x):
    """Two blades with yellow tips around a red spinner, hub at (x, HUB_Y, HUB_Z)."""
    blade = box((1, 7, 1), color=p.metal.dark).at(x - 0.5, 1, 17)
    blade = blade.paint(p.gold.mid, where=band("y", 1) | band("y", 7))
    spinner = box((1, 1, 2), color=p.red.mid).at(x - 0.5, 4, 17)
    return blade | spinner


@asset(name="twin_prop_plane", size=(24, 9, 20), tags=["vehicle", "metal"], grounded=True, symmetric="x")
def twin_prop_plane(p):
    props = [
        Part(name, prop(p, x), parent="plane", pivot=(x, HUB_Y, HUB_Z), joint=spin("z"))
        for name, x in NACELLE_X.items()
    ]
    return Model(
        parts=[Part("plane", airframe(p), pivot=(12, 0, 10)), *props],
        clips=[
            # One whole turn each, in opposite directions: the tops sweep outboard.
            clip(
                "spin",
                0.5,
                [spin_track("prop_r", 1), spin_track("prop_l", -1)],
                loop=True,
                ease="linear",
            )
        ],
    )
