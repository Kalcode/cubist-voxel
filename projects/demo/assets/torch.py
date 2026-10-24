from cubist import *

# Odd width, centered on a voxel center (3.5): pointed shapes need a middle column for
# their tips. Centered between columns (an even width), a cone's tip vanishes.
C = 3.5


@asset(name="torch", size=(7, 18, 7), tags=["prop", "light", "wood"], grounded=True)
def torch(p):
    handle = cylinder(r=1, h=11, color=p.wood[1]).at(C, 0, C)
    wrap = cylinder(r=1.5, h=2, color=p.cloth.brown).at(C, 7, C)
    cup = cone(r=1.5, h=2, r_top=2.3, color=p.metal.dark).at(C, 9, C)
    # A tall teardrop: round at the base, licking up to a point.
    flame = sphere(2.0, color=p.fire.mid).at(C, 12.5, C)
    flame = flame.smooth_union(cone(r=2.0, h=6, color=p.fire.mid).at(C, 11.5, C), k=1.5)
    flame = flame.displace(amp=0.4, scale=1.5, seed=3)
    # Hot at the base, cooler at the tips.
    flame = flame.gradient("y", p.fire, reverse=True)
    return handle | wrap | cup | flame
