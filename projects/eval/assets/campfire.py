import math

from cubist import *

C = 6  # fire center on x and z

# Stone ring: (angle in degrees, radii). Angles sit between the log diagonals so the log ends
# poke out through the gaps; radii vary so the stones read as field stones, not beads.
# The logs run at about 42, 138, 222 and 318 degrees.
STONES = [
    (18, (1.7, 1.7, 1.6)),
    (68, (1.6, 1.5, 1.7)),
    (112, (1.8, 1.8, 1.6)),
    (162, (1.6, 1.6, 1.7)),
    (198, (1.7, 1.5, 1.6)),
    (248, (1.6, 1.8, 1.7)),
    (292, (1.8, 1.6, 1.6)),
    (342, (1.6, 1.7, 1.7)),
]


@asset(name="campfire", size=(12, 10, 12), tags=["prop", "fire", "stone", "wood"], grounded=True)
def campfire(p):
    stones = []
    for deg, radii in STONES:
        a = math.radians(deg)
        x, z = C + 4.3 * math.cos(a), C + 4.3 * math.sin(a)
        stones.append(ellipsoid(radii, color=p.stone.mid).at(x, radii[1] - 0.4, z))
    ring = stones[0].union(*stones[1:]).paint_faces("+y", p.stone.light)
    ring = ring.paint(p.stone[1], where=band("y", 0, 1))
    ring = ring.dither(p.stone, seed=2, amount=0.2).darken_crevices()

    # Ash bed, mounded in the middle, with glowing embers in the gaps between the logs.
    ash = cylinder(r=3.6, h=1, color=p.stone[1]).at(C, 0, C)
    ash = ash | cylinder(r=2.6, h=1, color=p.stone.dark).at(C, 1, C)
    ash = ash.dither(p.stone, seed=5, amount=0.3)
    embers = [
        box((1, 1, 1), color=c).at(x, y, z)
        for x, y, z, c in (
            (6, 1, 8, p.fire.mid),
            (5, 1, 3, p.fire[1]),
            (6, 1, 3, p.fire.mid),
            (3, 1, 5, p.fire[1]),
            (3, 1, 6, p.fire.mid),
        )
    ]
    ash = ash.union(*embers)

    # Two logs crossed in an X, lying on the ash, ends poking out past the stones.
    log_a = capsule((1.0, 1.7, 1.5), (11.0, 1.7, 10.5), r=1.1, color=p.wood[1])
    log_b = capsule((11.0, 1.7, 1.5), (1.0, 1.7, 10.5), r=1.1, color=p.wood[1])
    logs = (log_a | log_b).paint_faces("+y", p.wood.mid)
    for x in (1.0, 11.0):
        for z in (1.5, 10.5):
            logs = logs.paint(p.wood.light, where=inside(sphere(0.9).at(x, 1.7, z)))
    # Charred where the fire licks the middle.
    logs = logs.paint(p.wood.dark, where=inside(cylinder(r=2.6, h=4).at(C, 0, C)))

    # Flames: a few tongues of different heights merged into one body, hot at the base and
    # cooling toward the tips.
    flame = sphere(1.8, color=p.fire.mid).at(C, 2.8, C)
    for x, z, r, h in (
        (5.5, 5.5, 2.0, 7.5),
        (7.5, 6.5, 1.7, 5.5),
        (5.5, 7.5, 1.5, 4.5),
        (7.0, 4.5, 1.4, 4.5),
    ):
        flame = flame.smooth_union(cone(r=r, h=h, color=p.fire.mid).at(x, 2.0, z), k=1.0)
    flame = flame.displace(amp=0.35, scale=1.5, seed=3)
    # Hot to cool, one entry per flame layer (y = 2..9); the darkest red only on the tips.
    hot_to_cool = [p.fire.light] * 2 + [p.fire.mid] * 2 + [p.fire[1]] * 3 + [p.fire.dark]
    flame = flame.gradient("y", hot_to_cool)

    return ash | ring | logs | flame
