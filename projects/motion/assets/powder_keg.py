from cubist import *


@asset(name="powder_keg", size=(10, 12, 10), tags=["prop", "wood", "metal", "explosive"], grounded=True)
def powder_keg(p):
    # A hollow keg so the fracture makes curved stave-like shards and the flash has room inside.
    keg = lathe(profile=[(0, 4), (2, 4.7), (4.5, 5), (7, 4.7), (9, 4)], color=p.wood.mid).at(5, 0, 5)
    keg = keg.hollow(1) - cylinder(r=3.2, h=1).at(5, 8, 5)  # open inside the rim (the chime)
    keg = keg | cylinder(r=3.6, h=1, color=p.wood[1]).at(5, 7, 5)  # recessed head
    # Stave seams: vertical lines on every side.
    seams = (stripes("x", 3) & (faces("front") | faces("back"))) | (stripes("z", 3) & (faces("left") | faces("right")))
    keg = keg.paint(p.wood[1], where=seams & ~band("y", 7, 2)).dither(p.wood.ramp, seed=1, amount=0.2)
    hoops = band("y", 1) | band("y", 6)
    keg = keg.paint(p.metal[1], where=hoops).paint(p.metal.mid, where=hoops & faces("left"))
    # A gray cord out of the bung, bent over, its end burning with a spark on top.
    fuse = box((1, 3, 1), color=p.stone[1]).at(6, 8, 4)
    tip = box((1, 1, 1), color=p.fire.mid).at(7, 10, 4) | box((1, 1, 1), color=p.fire.light).at(7, 11, 4)
    keg = (keg | fuse | tip).darken_crevices()

    # Built from chunks so it can blow apart: together they are the whole keg, fuse and all.
    pieces = chunks("stave", keg, pieces=10, seed=2)

    # A fireball inside the keg, seen for an instant as the staves fly off.
    def fireball(r):
        ball = ellipsoid((r, r, r), color=p.fire.mid).at(5, 1 + r, 5)
        return ball.paint(p.fire.light, where=sphere(r - 1).at(5, 1 + r, 5))

    flash = Part("flash", frames=[fireball(2), fireball(3)], parent=None, hidden=True)

    boom = clip(
        "explode",
        1.6,
        [
            # up=20 against gravity 40 lifts the staves about 5 voxels; all land within ~1.1 s.
            # rate=60 keys/s: at 20, fast-spinning chunks dip through the floor between keys.
            burst(pieces, speed=11, up=20, spin=720, rate=60, seed=1),
            visible("flash", [(0, 1), (0.2, 0)]),
            frames("flash", [(0, 0), (0.05, 1)]),
        ],
        events=[(0.0, "boom"), (0.0, "emit:smoke")],
        collisions=False,  # debris tumbles through debris
    )
    # From the open top of the keg (inside the rim), so it doesn't spray into the staves.
    smoke = emitter((5, 9, 5), "powder_keg_smoke", burst=16, life=1.4, speed=8, spread=50,
                    gravity=-3, grow=(0.6, 1.6))  # fmt: skip
    return Model(parts=[*pieces, flash], clips=[boom], emitters={"smoke": smoke})
