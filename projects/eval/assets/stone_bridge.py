import math

from cubist import *


def courses(p, x0, x1, y0, y1, z0, z1, w=5, h=3, start=0):
    """Running-bond blocks filling a box. Each w*h cell is a block with a lit top row and a
    dark mortar seam along its bottom and right edges; blocks vary between mid and light."""
    seam, lit = p.stone[1], p.stone.light
    bases = [p.stone.mid, p.stone.mid, p.stone.light, p.stone.mid, p.stone[1]]
    d = z1 - z0
    blocks = []
    for j, y in enumerate(range(y0, y1, h)):
        for i, x in enumerate(range(x0 - (w // 2) * ((j + start) % 2), x1, w)):
            base = bases[(i * 3 + j * 2 + start) % len(bases)]
            blocks.append(box((w, h, d), color=base).at(x, y, z0))
            blocks.append(box((w - 1, 1, d), color=lit).at(x, y + h - 1, z0))
            blocks.append(box((w, 1, d), color=seam).at(x, y, z0))
            blocks.append(box((1, h, d), color=seam).at(x + w - 1, y, z0))
    return blocks[0].union(*blocks[1:])


@asset(name="stone_bridge", size=(40, 12, 12), tags=["building", "stone"], grounded=True)
def stone_bridge(p):
    arch = cylinder(r=7, h=12, axis="z").at(20, 0, 0)

    # Masonry body, three courses up to the walkway at y = 9, one round arch through it.
    body = courses(p, 0, 40, 0, 9, 0, 12) & (box((40, 9, 12)) - arch)

    # Voussoirs: alternating wedge stones ringing the arch, keystone at the crown.
    ring = (cylinder(r=9, h=12, axis="z").at(20, 0, 0) - arch) & box((40, 9, 12))
    stones = []
    for k in range(9):
        a0, a1 = math.radians(20 * k), math.radians(20 * (k + 1))
        tri = [
            (20, 0),
            (20 + 12 * math.cos(a0), 12 * math.sin(a0)),
            (20 + 12 * math.cos(a1), 12 * math.sin(a1)),
        ]
        color = p.stone.light if k % 2 == 0 else p.stone[1]
        stones.append(extrude(tri, depth=12, color=color))
    voussoirs = stones[0].union(*stones[1:]) & ring

    # Flagstone walkway between the walls: two rows of slabs, seams between them.
    flags = []
    for j, z in enumerate((2, 6)):
        for i, x in enumerate(range(-3 * j, 40, 6)):
            base = [p.stone.light, p.stone.mid, p.stone.light][(i + j) % 3]
            flags.append(box((6, 1, 4), color=base).at(x, 8, z))
            flags.append(box((1, 1, 4), color=p.stone[1]).at(x + 5, 8, z))
    flags.append(box((40, 1, 1), color=p.stone[1]).at(0, 8, 6))
    paving = flags[0].union(*flags[1:]) & box((40, 9, 12))

    # Low parapet walls on both edges, one course above the walkway.
    wall_form = box((40, 3, 2)).at(0, 9, 0) | box((40, 3, 2)).at(0, 9, 10)
    walls = courses(p, 0, 40, 9, 12, 0, 12, start=1) & wall_form

    # Weathering: a light speckle across the masonry, moss creeping up from the footings
    # and onto a couple of coping stones.
    bridge = (body | voussoirs | paving | walls).dither(p.stone, seed=9, amount=0.1)
    moss = (
        ellipsoid((3, 2, 7)).at(3, 0, 6)
        | ellipsoid((2.5, 2.5, 7)).at(11.5, 0, 6)
        | ellipsoid((2.5, 2.5, 7)).at(28.5, 0, 6)
        | ellipsoid((3, 1.5, 7)).at(37, 0, 6)
        | ellipsoid((3, 1, 1.5)).at(8, 12, 1)
        | ellipsoid((3.5, 1, 1.5)).at(30, 12, 11)
    )
    outside = faces("+z") | faces("-z") | faces("+y")
    bridge = bridge.paint(p.leaf[1], where=inside(moss) & outside)
    bridge = bridge.dither(p.leaf, seed=10, amount=0.4)
    return bridge.darken_crevices()
