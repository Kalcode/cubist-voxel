import math

from cubist import *

# Sixteen wide and round about the vertical axis through (8, 8): a whole-number center suits an
# even width. The style's water cycle steps every water voxel one shade along the ramp each tick,
# so shades laid out by radius flow outward and shades laid out by height flow downward.
C = 8
CROWN = (((6, 7), (9, 8)), ((7, 6), (8, 9)), ((6, 8), (9, 7)), ((8, 6), (7, 9)))  # per frame


def rings(water, p, r_max):
    """Paint water in rings one voxel wide, darkest-first outward, so the cycle carries them out."""
    for k in range(r_max, -1, -1):
        water = water.paint(p.water[-k % 3], where=cylinder(r=k + 1, h=10).at(C, 0, C))
    return water


def splash(p, k):
    """Frame k of the spout: a jet with a crown of spray that swirls round, and eight arcs of
    four drops that each move a quarter step further along their fall into the basin, so four
    frames loop seamlessly. Landing foam pops up on alternate arcs every other frame.

    Drops start as white spray and turn to water as they fall. Falling drops take the three
    water shades in turn, so the cycle twinkles them instead of turning them all navy at once
    (which read as dark bars). The solid jet is shaded by height, so its shades climb."""
    foam = p.cloth.white
    out = box((2, 1, 2), color=foam).at(7, 9, 7)
    for y in (7, 8):
        out = out | box((2, 1, 2), color=p.water[-y % 3]).at(7, y, 7)
    for x, z in CROWN[k]:
        out = out | box((1, 1, 1), color=foam).at(x, 9, z)
    for i in range(8):
        a = math.radians(45 * i)
        for j in range(4):
            u = (j + k / 4) / 4
            r, y = 1.5 + 3.9 * u, 9.6 - 7.4 * u * u
            x, z = math.floor(C + r * math.cos(a)), math.floor(C + r * math.sin(a))
            color = foam if u < 0.5 else p.water[(i + j) % 3]
            out = out | box((1, 1, 1), color=color).at(x, math.floor(y), z)
        if (i + k) % 2 == 0:
            x, z = math.floor(C + 5.2 * math.cos(a)), math.floor(C + 5.2 * math.sin(a))
            out = out | box((1, 1, 1), color=foam).at(x, 2, z)
    return out & box((16, 10, 16))


@asset(
    name="fountain",
    size=(16, 10, 16),
    tags=["building", "stone", "water"],
    grounded=True,
    allow_floating=True,  # the splash's drops
)
def fountain(p):
    # Basin: a floor and a low wall, capped by a wider, lighter rim that overhangs it.
    wall = cylinder(r=7.4, h=2, color=p.stone.mid).at(C, 0, C) - cylinder(r=6.4, h=1).at(C, 1, C)
    wall = wall.masonry(p.stone, block=(3, 1, 3), seed=2)
    rim = cylinder(r=8, h=1, color=p.stone.light).at(C, 2, C) - cylinder(r=6.2, h=1).at(C, 2, C)
    rim = rim.masonry(p.stone[2:], block=(3, 1, 3), seed=5)

    # The spout: a plinth in the water, a shaft, a flared capital and a brass nozzle.
    plinth = cylinder(r=2.2, h=3, color=p.stone[1]).at(C, 0, C)
    shaft = cylinder(r=1.6, h=3, color=p.stone.mid).at(C, 2, C)
    capital = cylinder(r=2.6, h=1, color=p.stone.light).at(C, 5, C)
    nozzle = box((2, 1, 2), color=p.gold[1]).at(7, 6, 7)
    stone = (wall | rim | plinth | shaft | capital).dither(p.stone, seed=3, amount=0.15)
    stone = stone.darken_crevices() | nozzle

    # Water in rings, so the cycle sends ripples out from the spout to the rim.
    water = rings(cylinder(r=6.4, h=1, color=p.water.mid).at(C, 1, C), p, 7) - plinth

    return Model(
        parts=[
            Part("fountain", stone | water, pivot=(C, 0, C)),
            # Four frames at 8 fps: the splash plays on its own, no clip needed.
            Part(
                "splash",
                frames=[splash(p, k) for k in range(4)],
                fps=8,
                parent="fountain",
                pivot=(C, 7, C),
            ),
        ],
    )
