from cubist import *


@asset(name="tree", size=(16, 20, 16), tags=["prop", "nature"], grounded=True)
def tree(p):
    trunk = cylinder(r=2.1, h=11, color=p.wood[1]).at(8, 0, 8).taper(top=0.75)
    flare = cone(r=3.2, h=2, r_top=2.1, color=p.wood[1]).at(8, 0, 8)

    canopy = sphere(5.2, color=p.leaf.mid).at(8, 12.5, 8)
    for x, y, z, r in [(5, 11, 9.5, 3.8), (11, 11.5, 7, 4.0), (8, 15.5, 8, 3.4)]:
        canopy = canopy.smooth_union(sphere(r, color=p.leaf.mid).at(x, y, z), k=2)
    canopy = canopy.displace(amp=0.7, scale=2.5, seed=7)
    canopy = canopy.gradient("y", p.leaf).dither(p.leaf, seed=2, amount=0.3).darken_crevices()
    return trunk | flare | canopy
