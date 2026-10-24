import math

from cubist import *

# The cart runs along x; handles point to +x. The axle runs along z at (AX, AY).
AX, AY = 6.5, 5.5  # on voxel centres so the hub and the straight spokes are one voxel wide
R = 5.5  # wheel radius


def wheel(p, z0):
    """A two-voxel-thick spoked wheel with an iron tyre, its back face at z = z0."""
    tyre = cylinder(r=R, h=2, axis="z", color=p.metal.mid).at(AX, AY, z0)
    tyre = tyre - cylinder(r=R - 1, h=2, axis="z").at(AX, AY, z0)
    felloe = cylinder(r=R - 1, h=2, axis="z", color=p.wood.mid).at(AX, AY, z0)
    felloe = felloe - cylinder(r=R - 2, h=2, axis="z").at(AX, AY, z0)
    # Four straight spokes: at this size diagonal spokes sample three voxels wide and
    # fill the wheel in.
    spokes = []
    for k in range(4):
        a = math.radians(90 * k)
        tip = (AX + math.cos(a) * (R - 1.5), AY + math.sin(a) * (R - 1.5))
        for z in (z0 + 0.5, z0 + 1.5):
            spokes.append(line((AX, AY, z), (tip[0], tip[1], z), color=p.wood[1]))
    hub = cylinder(r=1.2, h=2, axis="z", color=p.wood.dark).at(AX, AY, z0)
    cap = box((1, 1, 2), color=p.metal.light).at(AX - 0.5, AY - 0.5, z0)
    # Shade by hand: on a two-voxel-thick wheel every voxel is an edge, so
    # highlight_edges would lighten all of it.
    tyre = tyre.paint(p.metal.light, where=faces("top"))
    return tyre.union(felloe, *spokes, hub, cap)


@asset(name="cart", size=(20, 12, 12), tags=["prop", "wood", "vehicle"], grounded=True)
def cart(p):
    floor = box((12, 1, 8), color=p.wood[1]).at(1, 7, 2)
    sides = box((12, 3, 1), color=p.wood.mid).at(1, 8, 2).mirror("z", about=6)
    ends = box((1, 3, 8), color=p.wood.mid).at(1, 8, 2) | box((1, 3, 8), color=p.wood.mid).at(
        12, 8, 2
    )
    # Plank seams along the boards.
    boards = (sides | ends).paint(p.wood[1], where=band("y", 9, 1))
    posts = [box((1, 5, 1), color=p.wood.dark).at(x, 7, z) for x in (1, 12) for z in (2, 9)]
    bed = floor.union(boards, *posts).dither(p.wood, seed=2, amount=0.15).darken_crevices()
    bed = bed.paint(p.wood.light, where=faces("top") & band("y", 10, 1))  # sunlit board tops

    handles = []
    for z in (3.5, 8.5):
        handles.append(line((1.5, 6.5, z), (19.5, 6.5, z), color=p.wood.mid))
        handles.append(line((17.5, 6.5, z), (19.5, 6.5, z), color=p.leather.dark))  # grips
    legs = [box((1, 6, 1), color=p.wood[1]).at(12, 0, z) for z in (3, 8)]
    axle = cylinder(r=0.6, h=12, axis="z", color=p.metal.dark).at(AX, AY, 0)

    return bed.union(*handles, *legs, axle, wheel(p, 0), wheel(p, 10))
