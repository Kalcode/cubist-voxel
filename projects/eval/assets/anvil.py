from cubist import *


@asset(name="anvil", size=(16, 12, 8), tags=["prop", "metal", "wood", "smithing"], grounded=True)
def anvil(p):
    # Tree stump, slightly oval so growth rings show around the anvil's foot.
    def disc(r, h=4):
        return cylinder(r=r, h=h).at(7, 0, 4).scale((1.25, 1, 1), about=(7, 0, 4))

    stump = cylinder(r=4, h=4, color=p.wood.mid).at(7, 0, 4).scale((1.25, 1, 1), about=(7, 0, 4))
    top = band("y", 3, 1)
    grooves = (
        band("x", 3, 1) | band("x", 7, 1) | band("x", 10, 1) | band("z", 2, 1) | band("z", 5, 1)
    )
    stump = stump.paint(p.wood[1], where=grooves & ~top)
    stump = stump.dither(p.wood, seed=2, amount=0.15, where=~top)
    stump = stump.paint(p.wood[1], where=top)  # bark lip
    stump = stump.paint(p.wood.light, where=top & inside(disc(3.4)))
    stump = stump.paint(p.wood.mid, where=top & inside(disc(2.6)))
    stump = stump.paint(p.wood.light, where=top & inside(disc(1.8)))
    roots = wedge((2, 2, 2), rise="+x", color=p.wood.mid).at(0, 0, 2)
    roots = roots | wedge((2, 2, 2), rise="-x", color=p.wood.mid).at(12, 0, 4)
    stump = (stump | roots).darken_crevices()

    # Anvil carved from three orthographic profiles: side (horn, heel, waist, foot),
    # top (horn narrows to a point), and front (waist pinches in).
    side = [
        (3, 4),
        (11, 4),
        (11, 5),
        (9, 6),
        (9, 7.5),
        (11, 8.5),
        (13.5, 9.5),
        (16, 11),
        (16, 12),
        (1, 12),
        (1, 9.5),
        (3, 9),
        (5, 7.5),
        (5, 6),
        (3, 5),
    ]
    top_view = [(0, 2), (10, 2), (16, 3), (16, 5), (10, 6), (0, 6)]
    front = [
        (2, 4),
        (6, 4),
        (6, 5),
        (5, 6),
        (5, 8),
        (6, 8),
        (6, 12),
        (2, 12),
        (2, 8),
        (3, 8),
        (3, 6),
        (2, 5),
    ]
    body = extrude(side, depth=8, color=p.metal.mid)
    body = body & extrude(top_view, depth=12, axis="y") & extrude(front, depth=16, axis="x")
    body = body - box((1, 1, 2)).at(2, 11, 3)  # hardy hole near the heel
    body = body.paint(p.metal[1], where=band("y", 4, 4))
    body = body.paint_faces("+y", p.metal.light, where=band("y", 11, 1))
    body = body.paint_faces("-y", p.metal.dark, where=~band("y", 11, 1))  # keep the horn tip lit
    body = body.darken_crevices()
    return stump | body
