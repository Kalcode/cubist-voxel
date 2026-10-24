from cubist import *

C = 8  # ring center (even width, so it sits between voxel columns)


def stone_courses(ring, p, courses, n=10):
    """Paint the ring as staggered blocks of varied shade. Blocks are slightly narrower than
    their sector, so the base color shows between them as mortar joints."""
    shades = [p.stone.mid, p.stone.light, p.stone.mid, p.stone[1]]
    for c, (y, h) in enumerate(courses):
        for k in range(n):
            angle = 360.0 * (k + 0.5 * (c % 2)) / n
            block = box((4, h, 2.6)).at(C + 3.5, y, C - 1.3).rotate(angle, "y", about=(C, 0, C))
            ring = ring.paint(shades[(k * 3 + c) % 4], where=block)
    return ring


@asset(name="well", size=(16, 20, 16), tags=["prop", "stone", "wood", "water"], grounded=True)
def well(p):
    # Stone ring: a thick round wall with a capstone course, water sunk two layers below the rim.
    ring = cylinder(r=6.6, h=5, color=p.stone.dark).at(C, 0, C) - cylinder(r=4.6, h=5).at(C, 1, C)
    ring = stone_courses(ring, p, [(0, 2), (2, 2), (4, 1)])
    cap = cylinder(r=7.2, h=1, color=p.stone.dark).at(C, 5, C) - cylinder(r=4.6, h=1).at(C, 5, C)
    cap = stone_courses(cap, p, [(5, 1)], n=8).paint_faces("top", p.stone.light)
    ring = (ring | cap).dither(p.stone, seed=3, amount=0.12)
    # The inner face of the wall is in shadow, so the opening reads as depth.
    shaft = cylinder(r=5.5, h=5, color=p.stone.dark).at(C, 1, C)
    ring = ring.paint(p.stone[1], where=inside(shaft) - colored(p.stone.dark))
    water = cylinder(r=4.7, h=1, color=p.water.mid).at(C, 3, C)
    water = water.dither(p.water, seed=8, amount=0.3)

    # Two posts standing beside the wall, a windlass axle with a crank on the right.
    posts = [box((2, 18, 2), color=p.wood[1]).at(x, 0, 7) for x in (1, 13)]
    posts = [post.paint(p.wood.dark, where=faces("right") | faces("back")) for post in posts]
    axle = cylinder(r=1.0, h=14, axis="x", color=p.wood.mid).at(1, 12, C)
    axle = axle.paint(p.wood.light, where=faces("top"))
    crank = box((1, 3, 1), color=p.wood.dark).at(15, 10, 7) | box((1, 1, 2), color=p.wood.dark).at(
        15, 10, 8
    )

    # Rope from the axle down to a hanging bucket.
    rope = line((8.5, 11.5, 8.5), (8.5, 9.5, 8.5), color=p.cloth.straw)
    bucket = cylinder(r=2.0, h=3, color=p.wood.mid).at(C, 7, C) - cylinder(r=1.0, h=1).at(C, 9, C)
    bucket = bucket.paint(p.metal.mid, where=band(axis="y", at=8, height=1))

    # Pitched roof: a triangular shell along x, ridge over the axle, overhanging the posts.
    outer = extrude([(1, 0), (8, 5), (15, 0)], depth=16, axis="x", color=p.wood.light).at(0, 14, 0)
    inner = extrude([(1, -2), (8, 3), (15, -2)], depth=16, axis="x").at(0, 14, 0)
    roof = outer - inner
    for y in (15, 17):
        roof = roof.paint(p.wood.mid, where=band(axis="y", at=y, height=1))
    roof = roof.paint(p.wood[1], where=faces("bottom"))
    roof = roof.dither(p.wood, seed=5, amount=0.2)
    ridge = box((16, 1, 2), color=p.wood.dark).at(0, 19, 7)

    return ring | water | posts[0] | posts[1] | axle | crank | rope | bucket | roof | ridge
