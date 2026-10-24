from cubist import *


@asset(name="barrel", size=(12, 16, 12), tags=["prop", "wood"], grounded=True)
def barrel(p):
    body = lathe(profile=[(0, 5), (3, 6), (13, 6), (16, 5)], color=p.wood.mid).at(6, 0, 6)
    body = body.hollow(1).union(cylinder(r=5, h=1, color=p.wood.dark).at(6, 15, 6))
    for y in (2, 13):
        body = body.paint(p.metal.dark, where=band(axis="y", at=y, height=1))
    body = body.dither(p.wood.ramp, seed=1).darken_crevices()
    return Model(parts=[Part("body", body, pivot=(6, 0, 6))])
