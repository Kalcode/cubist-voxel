from cubist import *

CX = 20  # center plane: the left half is built and shaded, then mirrored


def masonry(shape, p):
    """Shade about a quarter of staggered 4x2x4 blocks lighter or darker so walls read as cut
    stone on every face without turning into noise."""
    groups = {0: [], 1: []}
    for cy in range(16):
        off = 2 * (cy % 2)
        for cx in range(-1, 6):
            for cz in range(-1, 5):
                k = ((cx * 73856093) ^ (cy * 19349663) ^ (cz * 83492791)) % 8
                if k in groups:
                    groups[k].append(box((4, 2, 4)).at(4 * cx + off, 2 * cy, 4 * cz + off))
    for k, color in ((0, p.stone.light), (1, p.stone[1])):
        blocks = groups[k][0].union(*groups[k][1:])
        shape = shape.paint(color, where=inside(blocks) & colored(p.stone.mid))
    return shape


def slit(x, y, face_z):
    """A cross-shaped arrow slit one voxel deep on a face at z = face_z."""
    return box((1, 5, 1)).at(x, y, face_z) | box((3, 1, 1)).at(x - 1, y + 3, face_z)


def side_slit(face_x, y, z):
    """The same cross on a face at x = face_x."""
    return box((1, 5, 1)).at(face_x, y, z) | box((1, 1, 3)).at(face_x, y + 3, z - 1)


def crenels_x(starts, y, z, depth):
    return [box((2, 2, depth)).at(s, y, z) for s in starts]


def crenels_z(starts, y, x, depth):
    return [box((depth, 2, 2)).at(x, y, s) for s in starts]


@asset(
    name="gatehouse",
    size=(40, 32, 16),
    tags=["building", "stone", "metal", "castle"],
    grounded=True,
    symmetric="x",
)
def gatehouse(p):
    # Left tower: a dark plinth, a tall body, corbels, and a crenellated parapet overhanging by one.
    plinth = box((12, 2, 12), color=p.stone[1]).at(0, 0, 2)
    body = box((10, 27, 10), color=p.stone.mid).at(1, 0, 3)
    corbels = box((12, 1, 12), color=p.stone.mid).at(0, 26, 2)
    corbels = corbels.subtract(*[box((1, 1, 14)).at(x, 26, 1) for x in (1, 3, 5, 7, 9)])
    corbels = corbels.subtract(*[box((14, 1, 1)).at(-1, 26, z) for z in (3, 5, 7, 9, 11)])
    parapet = box((12, 5, 12), color=p.stone.mid).at(0, 27, 2) - box((10, 5, 10)).at(1, 28, 3)
    parapet = parapet.subtract(*crenels_x((3, 7), 30, 1, 14), *crenels_z((5, 9), 30, -1, 14))
    tower = plinth | body | corbels | parapet

    # Gate wall between the towers, recessed behind their fronts, with its own battlement.
    wall = box((18, 20, 8), color=p.stone.mid).at(11, 0, 4)
    walk = box((18, 3, 1), color=p.stone.mid).at(11, 20, 11) | box(
        (18, 3, 1), color=p.stone.mid
    ).at(11, 20, 4)
    walk = walk.subtract(*crenels_x((13, 17, 21, 25), 21, 3, 10))
    opening = box((8, 9, 10)).at(16, 0, 3) | cylinder(r=4, h=10, axis="z").at(CX, 9, 3)

    # Arrow slits: tower front, back and outer side, and one each side of the arch.
    slits = [
        slit(5, 8, 12),
        slit(5, 17, 12),
        slit(5, 17, 3),
        side_slit(1, 9, 7),
        side_slit(1, 18, 7),
        slit(16, 15, 11),
    ]
    slit_backs = [
        slit(5, 8, 11),
        slit(5, 17, 11),
        slit(5, 17, 4),
        side_slit(2, 9, 7),
        side_slit(2, 18, 7),
        slit(16, 15, 10),
    ]

    # Cut and shade the left half, then mirror it so form and color are exactly symmetric.
    half = (tower | wall | walk).subtract(opening, *slits) & box((CX, 32, 16))
    half = masonry(half, p).dither(p.stone, seed=2, amount=0.06)
    reveal = box((10, 10, 10)).at(15, 0, 3) | cylinder(r=5, h=10, axis="z").at(CX, 9, 3)
    half = half.paint(p.stone[1], where=inside(reveal))
    half = half.paint(p.eye.dark, where=slit_backs[0].union(*slit_backs[1:]))
    stone = half.mirror("x", about=CX)

    # A raised arch ring with jambs and a keystone, in pale dressed stone.
    ring = cylinder(r=5.6, h=1, axis="z", color=p.stone.light).at(CX, 9, 12) - cylinder(
        r=4, h=1, axis="z"
    ).at(CX, 9, 12)
    ring = (ring & box((12, 6, 1)).at(14, 9, 12)) | box((2, 9, 1), color=p.stone.light).at(
        14, 0, 12
    )
    ring = ring.mirror("x", about=CX) | box((2, 2, 1), color=p.stone.light).at(19, 14, 12)

    # Red banners with a gold cross flanking the arch.
    banner = box((3, 9, 1), color=p.red.mid).at(11, 8, 12) - box((1, 1, 1)).at(12, 8, 12)
    banner = (
        banner
        | box((1, 3, 1), color=p.gold.mid).at(12, 11, 12)
        | box((3, 1, 1), color=p.gold.mid).at(11, 12, 12)
    )
    banner = banner | box((4, 1, 1), color=p.metal.mid).at(11, 17, 12)
    banners = banner.paint(
        p.red[1], where=band(axis="x", at=11, height=1) & colored(p.red.mid)
    ).mirror("x", about=CX)

    # Wooden doors closing the back of the passage.
    doors = box((8, 13, 1), color=p.wood[1]).at(16, 0, 4) & opening
    for x in (17, 19, 20, 22):
        doors = doors.paint(p.wood.dark, where=band(axis="x", at=x, height=1))
    doors = doors.paint(
        p.metal[1], where=band(axis="y", at=3, height=1) | band(axis="y", at=9, height=1)
    )

    # Portcullis just inside the arch: bars down to spiked tips, lowered most of the way.
    grille = None
    for x in (17, 19, 20, 22):
        bar = box((1, 11, 1), color=p.metal[1]).at(x, 3, 10) | box(
            (1, 1, 1), color=p.metal.light
        ).at(x, 2, 10)
        grille = bar if grille is None else grille | bar
    for y in (4, 7, 10):
        grille = grille | box((8, 1, 1), color=p.metal.mid).at(16, y, 10)
    grille = grille & opening

    return stone | ring | banners | doors | grille
