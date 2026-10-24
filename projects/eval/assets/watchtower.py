from cubist import *

# Even width: symmetric about x = 8. The ladder is on the front (+Z) face.
POSTS = (2, 12)  # min corner of the 2x2 legs on x and z
LEG_Y0, LEG_Y1 = 5, 25  # legs stand on the stone base and carry the platform
FLOOR_Y = 25
ROOF_Y = 32


def braces(p, y0, y1):
    """An X brace on each side between the legs, from y0 to y1, plus a girt on top.

    A 45-degree line through voxel centers samples three voxels wide; nudged 0.2 off center
    it becomes a clean two-wide staircase. Mirror the nudge so the X stays symmetric.
    """
    lo, hi = POSTS[0] + 1.5, POSTS[1] + 0.5  # voxel centers just inside the posts
    out = []
    for plane in (POSTS[0] + 0.5, POSTS[1] + 1.5):  # outer layer of the posts
        for a, b in ((lo - 0.2, hi - 0.2), (hi + 0.2, lo + 0.2)):
            out.append(line((a, y0 + 0.5, plane), (b, y1 - 0.5, plane), color=p.wood[1]))
            out.append(line((plane, y0 + 0.5, a), (plane, y1 - 0.5, b), color=p.wood[1]))
    girt = box((12, 1, 12), color=p.wood[1]).at(2, y1, 2) - box((8, 1, 8)).at(4, y1, 4)
    return out + [girt]


@asset(
    name="watchtower", size=(16, 40, 16), tags=["building", "wood", "stone", "tower"], grounded=True
)
def watchtower(p):
    # Stone base: a wide footing course and a stepped plinth.
    base = box((16, 2, 16), color=p.stone.mid) | box((14, 3, 14), color=p.stone.mid).at(1, 2, 1)
    base = base.dither(p.stone, seed=1, amount=0.3).darken_crevices().highlight_edges()

    # Tall legs with two tiers of cross bracing.
    # Legs are lighter than the braces so they read first.
    legs = [
        box((2, LEG_Y1 - LEG_Y0, 2), color=p.wood.mid).at(x, LEG_Y0, z)
        for x in POSTS
        for z in POSTS
    ]
    frame = legs[0].union(*legs[1:], *braces(p, LEG_Y0, 15), *braces(p, 15, LEG_Y1))

    # Lookout platform: a plank floor overhanging the legs, with a railing (gap for the ladder).
    floor = box((14, 2, 14), color=p.wood.mid).at(1, FLOOR_Y, 1)
    for x in range(1, 15, 2):
        floor = floor.paint(
            p.wood.light, where=region((x, FLOOR_Y + 1, 1), (x + 1, FLOOR_Y + 2, 15))
        )
    floor = floor.paint(p.wood[1], where=band("y", FLOOR_Y, 1))
    ring = box((14, 3, 14), color=p.wood.mid).at(1, 27, 1) - box((12, 3, 12)).at(2, 27, 2)
    balusters = ring & box((16, 2, 16)).at(0, 27, 0)
    for i in range(2, 14, 2):
        balusters = balusters - box((1, 2, 16)).at(i, 27, 0) - box((16, 2, 1)).at(0, 27, i)
    rail = (ring & box((16, 1, 16)).at(0, 29, 0)).paint(p.wood.light)
    # Cutter colored: grid-stage minus an uncolored shape fails (see knight.py).
    railing = (balusters | rail) - box((2, 3, 1), color=p.wood.mid).at(7, 27, 14)

    # Corner posts holding up a stepped, pointed roof with red shingles.
    roof_posts = [box((2, 5, 2), color=p.wood[1]).at(x, 27, z) for x in (1, 13) for z in (1, 13)]
    roof = box((16, 1, 16), color=p.red[1]).at(0, ROOF_Y, 0)
    for i in range(1, 8):
        roof = roof | box((16 - 2 * i, 1, 16 - 2 * i), color=p.red.mid).at(i, ROOF_Y + i, i)
    roof = roof.dither(p.red, seed=7, amount=0.3).highlight_edges()

    # Ladder up the front face, from the footing to above the floor; light rails so it stands
    # out against the dark braces behind it.
    rails = [box((1, 27, 1), color=p.wood.light).at(x, 2, 15) for x in (6, 9)]
    ladder = rails[0] | rails[1]
    for y in range(4, 28, 3):
        ladder = ladder | box((2, 1, 1), color=p.wood.mid).at(7, y, 15)

    tower = frame.union(floor, railing, *roof_posts).darken_crevices()
    return base | tower | roof | ladder
