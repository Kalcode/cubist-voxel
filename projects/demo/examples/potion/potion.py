from cubist import *


@asset(name="potion", size=(8, 12, 8), tags=["prop", "glass"], grounded=True)
def potion(p):
    bottle = sphere(3.6, color=p.glass.light).at(4, 3.5, 4)
    # Intersect keeps the left shape's color, so this is the bottle's lower half in red.
    liquid = sphere(3.6, color=p.potion.mid).at(4, 3.5, 4) & box((8, 4, 8)).at(0, 0, 0)
    neck = cylinder(r=1.6, h=3, color=p.glass.dark).at(4, 6.5, 4)
    rim = cylinder(r=2.6, h=1, color=p.glass.dark).at(4, 9, 4)
    cork = cylinder(r=1.2, h=2, color=p.cork).at(4, 10, 4)
    return bottle | liquid | neck | rim | cork
