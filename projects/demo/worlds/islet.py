"""A small island: rolling hills on a low shelf, rocky ridges, snow on the tops, a quarry cut
level into a hill."""

from cubist.world import *


@world(name="islet", size_m=(512, 512), seed=7)
def islet(p):
    land = circle((256, 256), radius_m=150, edge_m=90)
    hills = (
        noise(scale_m=120, amplitude_m=14, octaves=4)
        + ridges(scale_m=200, amplitude_m=30, octaves=3) * land
    )
    ground = (flat(2) + hills * land + land * 12).flatten(
        rect((150, 300), (200, 340), edge_m=24), at_m=19
    )
    return World(
        ground=ground,
        surface=[
            height_under(3, p.cork),
            height_over(44, p.white),
            slope_over(35, p.stone.mid),
            p.leaf.mid,
        ],
        subsoil=[(1, p.wood[1]), p.stone.dark],
        landmarks={"quarry": (175, 320), "centre": (256, 256)},
    )
