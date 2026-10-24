import math

from cubist import *

C = 12  # tower axis (x and z)
HUB = (12, 28, 21.5)  # sail hub centre; sails lie in the z = 21 voxel layer


def sail(p, degrees):
    """One lattice sail: a spar from the hub plus a frame offset to one side of it."""
    a = math.radians(degrees)
    u = (math.cos(a), math.sin(a))  # along the spar
    v = (-math.sin(a), math.cos(a))  # across the lattice, same turn for every sail

    def pt(s, t):
        return (HUB[0] + u[0] * s + v[0] * t, HUB[1] + u[1] * s + v[1] * t, HUB[2])

    # The hub sits on a voxel corner, so the spar is two voxels wide; the lattice bars sit
    # on voxel centres (offsets of .5) so each is one voxel wide with 2x2 openings.
    bars = [line(pt(0, 0), pt(10.5, 0), r=0.5, color=p.wood[1])]
    for t in (1.5, 4.5):
        bars.append(line(pt(3.5, t), pt(9.5, t), color=p.wood.mid))
    for s in (3.5, 6.5, 9.5):
        bars.append(line(pt(s, 1.5), pt(s, 4.5), color=p.wood.mid))
    return bars[0].union(*bars[1:])


@asset(name="windmill", size=(24, 40, 24), tags=["building", "wood", "stone"], grounded=True)
def windmill(p):
    tower = lathe([(0, 7.5), (26, 5.5)], color=p.stone.mid).at(C, 0, C)
    plinth = cylinder(r=8.1, h=2, color=p.stone[1]).at(C, 0, C)
    # Door and windows: recess into the wall, then set a wooden slab at the back.
    cuts = [
        box((4, 7, 6)).at(10, 2, 17.5),  # door
        box((2, 3, 6)).at(11, 12, 16.5),  # front window, between the door and the low sail
        box((6, 3, 2)).at(16.5, 12, 11),  # side window
    ]
    tower = (tower | plinth).subtract(*cuts)
    for y in range(5, 26, 4):  # mortar courses
        tower = tower.paint(p.stone[1], where=band("y", y, 1))
    tower = tower.dither(p.stone, seed=1, amount=0.2)
    door = box((4, 7, 1), color=p.wood[1]).at(10, 2, 17.5)
    door = door.paint(p.wood.dark, where=band("x", 12, 1))
    door = door.paint(p.metal.mid, where=region((13, 5, 0), (14, 6, 24)))
    windows = box((2, 3, 1), color=p.wood.dark).at(11, 12, 16.5) | box(
        (1, 3, 2), color=p.wood.dark
    ).at(16.5, 12, 11)
    lintels = [
        box((6, 1, 3), color=p.stone.light).at(9, 9, 17),
        box((4, 1, 3), color=p.stone.light).at(10, 15, 16),
        box((3, 1, 4), color=p.stone.light).at(16, 15, 10),
    ]
    step = box((6, 1, 2), color=p.stone[1]).at(9, 0, 19)
    tower = tower.union(door, windows, *lintels, step).darken_crevices()

    roof = cone(r=7.6, h=12, color=p.wood.mid).at(C, 26, C)
    for y in range(26, 38, 2):
        roof = roof.paint(p.wood[1], where=band("y", y, 1))
    roof = roof.highlight_edges()
    finial = cylinder(r=1, h=2, color=p.metal.mid).at(C, 37, C)
    # A wooden curb ring where the cap turns on the tower.
    curb = cylinder(r=6.4, h=1, color=p.wood.dark).at(C, 25, C)

    # The axle runs out of the roof to the hub; sails turn in the z = 21 layer.
    hub = cylinder(r=1.6, h=7.5, axis="z", color=p.wood.dark).at(HUB[0], HUB[1], 15)
    cap = cylinder(r=1, h=1, axis="z", color=p.metal.mid).at(HUB[0], HUB[1], 22.5)
    sails = [sail(p, d) for d in (0, 90, 180, 270)]
    return tower | curb | roof | finial | hub.union(*sails, cap)
