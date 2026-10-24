from cubist import *

W = 14  # stall width along x; the crate stands to its right
TOP = 6  # counter surface height


def stripes(width, step=4):
    sel = band(axis="x", at=0, height=step // 2)
    for x in range(step, width, step):
        sel = sel | band(axis="x", at=x, height=step // 2)
    return sel


@asset(name="market_stall", size=(20, 16, 12), tags=["building", "wood", "cloth"], grounded=True)
def market_stall(p):
    # Counter: vertical front planks under an overhanging top, open at the back with a shelf.
    counter = box((W, TOP - 1, 5), color=p.wood.mid).at(0, 0, 5)
    counter = counter - box((W - 2, TOP - 2, 4)).at(1, 0, 5)
    counter = counter | box((W - 2, 1, 4), color=p.wood[1]).at(1, 2, 5)
    counter = counter.paint(p.wood[1], where=stripes(W, step=3) & faces("front"))
    top = box((W, 1, 6), color=p.wood.light).at(0, TOP - 1, 5)
    # The front wall is one voxel thick: keep shelf crevices from showing through it.
    counter = (counter | top).darken_crevices(where=~faces("front"))

    # Posts: taller at the back so the awning slopes down toward the customer.
    posts = []
    for x in (0, W - 2):
        posts.append(box((2, 12, 2), color=p.wood[1]).at(x, 0, 9))
        posts.append(box((2, 15, 2), color=p.wood[1]).at(x, 0, 1))
    posts = [post.paint(p.wood.dark, where=faces("right")) for post in posts]

    # Striped awning: a sloped slab plus a scalloped valance along the front edge.
    slab = extrude(
        [(0, 14), (11, 11.5), (11, 13.5), (0, 16)], depth=W, axis="x", color=p.cloth.white
    )
    valance = box((W, 2, 1), color=p.cloth.white).at(0, 10, 10)
    for x in range(0, W, 4):
        valance = valance - box((2, 1, 1)).at(x + 2, 10, 10)
    awning = (slab | valance).paint(p.red.mid, where=stripes(W))
    awning = awning.paint(p.red.dark, where=faces("bottom") & colored(p.red.mid))
    awning = awning.paint(p.bone.mid, where=faces("bottom") & colored(p.cloth.white))

    # Goods on the counter: a heap of apples, two cabbages, and stoppered jars.
    goods = []
    for x, y, z in [(2.5, 7.5, 8.5), (4.5, 7.5, 8.5), (3.5, 7.5, 6.5), (3.5, 8.5, 7.5)]:
        apple = sphere(1.1, color=p.red.mid).at(x, y, z)
        goods.append(apple.paint(p.red.light, where=band(axis="y", at=y + 0.5, height=1)))
    for x, z in [(6.5, 8.5), (8.5, 6.5)]:
        cabbage = sphere(1.6, color=p.leaf.mid).at(x, TOP + 1.5, z)
        cabbage = cabbage.paint(p.leaf.light, where=faces("top")).paint(
            p.leaf[1], where=faces("bottom")
        )
        goods.append(cabbage)
    for x, z, ramp in [(10, 9, p.blue), (11, 6.5, p.stone), (11.5, 9, p.purple)]:
        jar = cylinder(r=1.0, h=2, color=ramp.mid).at(x, TOP, z)
        goods.append(jar.paint(ramp.light, where=band(axis="y", at=TOP + 1, height=1)))
        goods.append(box((1, 1, 1), color=p.leather.light).at(x - 0.5, TOP + 2, z - 0.5))

    # A crate of oranges beside the counter.
    crate = box((5, 5, 5), color=p.wood.mid).at(W, 0, 6) - box((3, 1, 3)).at(W + 1, 4, 7)
    crate = crate.paint(p.wood.dark, where=band(axis="y", at=2, height=1))
    crate = crate.paint(
        p.wood[1], where=(band(axis="x", at=W, height=1) | band(axis="x", at=W + 4, height=1))
    )
    oranges = [
        sphere(0.9, color=p.fire.mid).at(x, 4.5, z) for x in (W + 1.5, W + 3.5) for z in (7.5, 9.5)
    ]
    oranges = (
        oranges[0].union(*oranges[1:]).paint(p.fire.light, where=band(axis="x", at=W + 1, height=1))
    )

    return counter.union(*posts, awning, *goods, crate, oranges)
