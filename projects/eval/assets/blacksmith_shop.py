from cubist import *


def stripes(axis, start, stop, step):
    """One-voxel bands every `step` voxels along `axis` (plank seams, shingle rows)."""
    sel = band(axis, start, 1)
    for v in range(start + step, stop, step):
        sel = sel | band(axis, v, 1)
    return sel


@asset(
    name="blacksmith_shop",
    size=(32, 24, 24),
    tags=["building", "wood", "stone", "fire"],
    grounded=True,
)
def blacksmith_shop(p):
    # Stone plinth and floor.
    plinth = box((32, 1, 24), color=p.stone.mid).dither(p.stone, seed=1, amount=0.4)

    # Lean-to workshop: tall open front (door height 18), roof sloping down to the back.
    # Side walls follow the roof line; stone footing, vertical planks, dark posts.
    back = box((28, 13, 2), color=p.wood.mid).at(2, 1, 2)
    side = extrude([(2, 1), (22, 1), (22, 19), (2, 13)], depth=2, axis="x", color=p.wood.mid)
    walls = back | side.at(2, 0, 0) | side.at(28, 0, 0)
    walls = walls.paint(p.wood[1], where=stripes("x", 5, 28, 3) | stripes("z", 5, 22, 3))
    walls = walls.paint(p.stone.mid, where=band("y", 1, 3)).dither(p.stone, seed=2, amount=0.4)
    posts = [box((2, 18, 2), color=p.wood.dark).at(x, 1, 20) for x in (2, 28)]
    posts += [box((2, 13, 2), color=p.wood.dark).at(x, 1, 2) for x in (2, 28)]
    lintel = box((28, 1, 2), color=p.wood.dark).at(2, 18, 20)
    shell = walls.union(*posts, lintel).darken_crevices()

    # Shingled roof slab, overhanging the sides and the front.
    roof = extrude([(0, 13), (24, 19), (24, 21), (0, 15)], depth=32, axis="x", color=p.wood[1])
    roof = roof.paint(p.wood.dark, where=stripes("z", 1, 24, 3)).highlight_edges()

    # Forge: stone hearth against the back wall with a firebox open to the front and top,
    # a hood above it, and a chimney rising well clear of the roof.
    hearth = box((10, 6, 7), color=p.stone.mid).at(4, 1, 4)
    firebox = box((6, 4, 6)).at(6, 3, 5)
    hearth = (hearth - firebox).dither(p.stone, seed=3, amount=0.4)
    hearth = hearth.paint(p.stone.light, where=band("y", 6, 1))
    coals = box((6, 1, 5), color=p.fire.dark).at(6, 3, 5)
    coals = coals | box((4, 1, 3), color=p.fire.mid).at(7, 3, 6)
    flames = box((2, 1, 1), color=p.glow.light).at(8, 3, 7)
    flames = (
        flames
        | box((1, 3, 1), color=p.fire.light).at(8, 4, 7)
        | box((1, 2, 1), color=p.fire[2]).at(9, 4, 7)
    )
    flames = (
        flames
        | box((1, 2, 1), color=p.fire[2]).at(7, 4, 8)
        | box((1, 1, 1), color=p.fire.light).at(10, 4, 8)
    )
    flames = (
        flames
        | box((1, 1, 1), color=p.fire.mid).at(9, 4, 9)
        | box((1, 4, 1), color=p.fire[2]).at(9, 4, 6)
    )
    hood = box((10, 5, 7), color=p.stone[1]).at(4, 9, 3).taper(top=0.6)
    chimney = box((6, 23, 4), color=p.stone.mid).at(6, 1, 2)
    chimney = chimney.paint(p.stone[1], where=stripes("y", 2, 24, 3))
    cap = box((8, 1, 5), color=p.stone.dark).at(5, 23, 1.5)
    forge = (hearth | hood | chimney | cap).darken_crevices() | coals | flames

    # Anvil on a stump out front under the eave, horn to the right, flat light face on top.
    stump = cylinder(r=2, h=3, color=p.wood[1]).at(19, 1, 22)
    stump = stump.paint(p.wood.light, where=band("y", 3, 1))
    anvil = box((4, 1, 2), color=p.metal[1]).at(17, 4, 21)
    anvil = anvil | box((2, 1, 2), color=p.metal.dark).at(18, 5, 21)
    anvil = anvil | box((6, 2, 2), color=p.metal.mid).at(16, 6, 21)
    anvil = anvil | cone(r=1, h=2.5, axis="x", color=p.metal.mid).at(22, 7, 22)
    anvil = anvil.paint_faces("+y", p.metal.light, where=band("y", 7, 1))

    # Quench barrel by the right wall and tools hanging on the back wall.
    barrel = cylinder(r=2.5, h=5, color=p.wood.mid).at(25, 1, 8)
    barrel = barrel.paint(p.metal[1], where=band("y", 2, 1) | band("y", 4, 1))
    barrel = barrel | cylinder(r=1.5, h=1, color=p.water.mid).at(25, 5, 8)
    tools = box((1, 4, 1), color=p.wood.dark).at(17, 7, 4) | box((3, 1, 1), color=p.metal[1]).at(
        16, 11, 4
    )
    tools = (
        tools
        | box((1, 5, 1), color=p.metal[1]).at(20, 6, 4)
        | box((1, 5, 1), color=p.metal[1]).at(22, 6, 4)
    )
    tools = tools | box((3, 1, 1), color=p.metal[1]).at(20, 10, 4)

    return plinth | shell | roof | forge | stump | anvil | barrel | tools
