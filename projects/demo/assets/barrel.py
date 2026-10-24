from cubist import *


@asset(name="barrel", size=(12, 16, 12), tags=["prop", "wood"], grounded=True)
def barrel(p):
    body = lathe(profile=[(0, 5), (3, 6), (13, 6), (16, 5)], color=p.wood.mid).at(6, 0, 6)
    body = body.hollow(1).union(cylinder(r=5, h=1, color=p.wood.dark).at(6, 15, 6))
    for y in (2, 13):
        body = body.paint(p.metal.dark, where=band(axis="y", at=y, height=1))
    body = body.dither(p.wood.ramp, seed=1).darken_crevices()
    # Built from chunks so it can blow apart: together they are the whole barrel.
    pieces = chunks("stave", body, pieces=8, seed=2)
    boom = clip(
        "explode",
        1.6,
        burst(pieces, speed=11, up=10, spin=600, seed=1),
        events=[(0.0, "emit:smoke")],
        collisions=False,  # debris tumbles through debris
    )
    smoke = emitter((6, 8, 6), "smoke_puff", burst=10, life=1.2, speed=5, spread=80,
                    gravity=-2, grow=(1.2, 2.5), part="stave_0")  # fmt: skip
    return Model(parts=pieces, clips=[boom], emitters={"smoke": smoke})
