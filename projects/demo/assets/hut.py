from cubist import *


@asset(name="hut", size=(24, 20, 20), tags=["building", "wood", "stone"], grounded=True)
def hut(p):
    foundation = box((24, 2, 20), color=p.stone.mid).dither(p.stone, seed=1, amount=0.4)

    walls = box((22, 9, 18), color=p.wood.mid).at(1, 2, 1).hollow(1)
    posts = [box((2, 9, 2), color=p.wood.dark).at(x, 2, z) for x in (1, 21) for z in (1, 17)]
    door = box((4, 6, 1), color=p.wood.dark).at(10, 2, 18)
    knob = box((1, 1, 1), color=p.metal.light).at(12, 4, 18)
    windows = [box((3, 3, 1), color=p.glass.dark).at(x, 6, 18) for x in (4, 17)]
    walls = walls.union(*posts, door, knob, *windows).darken_crevices()

    # A gabled roof: a triangle in the x-y plane pushed back along z, overhanging the walls.
    roof = extrude([(0, 0), (12, 8), (24, 0)], depth=20, color=p.roof.mid).at(0, 11, 0)
    roof = roof.dither(p.roof, seed=4, amount=0.4).highlight_edges()
    chimney = box((3, 7, 3), color=p.stone[1]).at(17, 12, 6)
    return foundation | walls | roof | chimney
