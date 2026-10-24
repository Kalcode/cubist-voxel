from cubist import *


@asset(name="sword", size=(7, 27, 3), tags=["prop", "weapon", "metal"])
def sword(p):
    blade = box((3, 16, 1), color=p.metal.light).at(2, 8, 1)
    fuller = box((1, 14, 1), color=p.metal.mid).at(3, 9, 1)
    tip = cone(r=1.5, h=3, color=p.metal.light).at(3.5, 24, 1.5) & box((3, 3, 1)).at(2, 24, 1)
    guard = box((7, 1, 3), color=p.metal.dark).at(0, 7, 0)
    grip = cylinder(r=1, h=5, color=p.wood.dark).at(3.5, 2, 1.5)
    pommel = sphere(1.2, color=p.metal.mid).at(3.5, 1.2, 1.5)
    return blade | fuller | tip | guard | grip | pommel
