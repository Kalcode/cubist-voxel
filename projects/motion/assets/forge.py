from cubist import *

# Faces +Z. The stonework is symmetric about x = 8; the fire and the tools are not.
# Fire tongues: (x, z, height, radius), on voxel centers so each tip has a column.
TONGUES = [(8.5, 8.5, 5.0, 1.6), (6.5, 9.5, 4.0, 1.3), (10.5, 9.5, 3.5, 1.3), (7.5, 10.5, 3.0, 1.1)]
FIRE_Y = 4  # the flames stand on the coals at y = 3


def flames(p, k):
    """Frame k of the fire: the same tongues, licked a little taller or shorter each frame."""
    out = None
    for i, (x, z, h, r) in enumerate(TONGUES):
        hh = h + (0.0, 0.8, -0.5, 0.4, -0.7, 0.5)[(k + 2 * i) % 6]
        tongue = cone(r=r, h=hh, color=p.fire.mid).at(x, FIRE_Y, z)
        out = tongue if out is None else out | tongue
    out = out.displace(amp=0.3, scale=1.4, seed=21 + k)
    # Yellow at the base to orange at the tips (the darkest red is for the coals).
    return out.gradient("y", p.fire.ramp[1:], reverse=True, lo=FIRE_Y, hi=FIRE_Y + 5)


def stonework(p):
    hearth = box((14, 5, 12), color=p.stone.mid).at(1, 0, 1)
    hearth = hearth - box((8, 2, 6)).at(4, 3, 6)  # fire bed
    hearth = hearth.carve(box((4, 2, 2)).at(6, 1, 11), color=p.stone.dark)  # ash pit
    back = box((10, 5, 3), color=p.stone.mid).at(3, 5, 1)
    # The hood: a front trapezoid (x, y) intersected with a side one (z, y).
    hood = extrude([(1.5, 10), (14.5, 10), (11, 14), (5, 14)], 14, axis="z", color=p.stone.mid) & extrude(
        [(1, 10), (9, 10), (5, 14), (1, 14)], 16, axis="x", color=p.stone.mid
    )
    chimney = box((6, 4, 4), color=p.stone.mid).at(5, 14, 1) | box((8, 1, 6), color=p.stone.mid).at(4, 17, 0)
    chimney = chimney.carve(box((4, 3, 2)).at(6, 15, 2), color=p.stone.dark)  # flue
    stone = hearth | back | hood | chimney
    stone = stone.masonry(p.stone.ramp[1:3], block=(3, 2, 3), seed=3, where=colored(p.stone.mid))
    stone = stone.paint(p.metal[1], where=band("y", 10) & region((0, 0, 0), (16, 11, 10)))  # iron hood band
    stone = stone.darken_crevices().highlight_edges(facing=("top", "left"))
    coals = box((8, 1, 6), color=p.fire.dark).at(4, 3, 6).dither(p.fire.ramp[:3], seed=7, amount=0.6)
    embers = box((4, 1, 2), color=p.fire.dark).at(6, 0, 11).dither(p.fire.ramp[:2], seed=8)
    return (stone - box((4, 1, 2)).at(6, 0, 11)) | coals | embers | tools(p)


def tools(p):
    """An iron bar heating in the coals (its tip glows), and a hammer on the left rim."""
    bar = polyline([(15.5, 5.5, 11.5), (12.5, 5.5, 11.5), (11.5, 4.5, 11.5)], p.metal[1])
    # The end in the coals glows, fading to red-hot along the bar; clear of the flames.
    bar = bar.paint(p.fire.mid, where=band("x", 11)).paint(p.fire[1], where=band("x", 12))
    handle = line((2.5, 5.5, 3.5), (2.5, 5.5, 9.5), p.wood.mid)
    head = box((3, 1, 2), color=p.metal.dark).at(1, 5, 10)
    return bar | handle | head


@asset(name="forge", size=(16, 18, 14), tags=["building", "stone", "fire", "metal"], grounded=True)
def forge(p):
    return Model(
        parts=[
            Part("forge", stonework(p), pivot=(8, 0, 7)),
            # Six frames at 10 fps: the fire flickers on its own, no clip needed.
            Part(
                "fire",
                frames=[flames(p, k) for k in range(6)],
                fps=10,
                parent="forge",
                pivot=(8.5, FIRE_Y, 9.5),
            ),
        ],
        lights={"fire": light((8.5, 6, 9.5), p.fire.light, range=12, intensity=2, flicker=0.5, part="fire")},
        emitters={
            "smoke": emitter(
                (8, 18.5, 3),
                "forge_smoke",
                rate=3,
                life=2.5,
                speed=2.5,
                spread=15,
                gravity=-0.8,
                grow=(0.5, 1.4),
                part="forge",
            )
        },
    )
