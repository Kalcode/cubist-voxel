import math

from cubist import *

HEAP = ((7, 7.5), (6.6, 3.8, 6.2))
SPILL = ((11, 10.5), (4.6, 2.6, 4.4))


def surface_y(x, z):
    """Height of the undisplaced pile at (x, z), for seating things on it."""
    best = 0.0
    for (cx, cz), (rx, ry, rz) in (HEAP, SPILL):
        t = 1 - ((x - cx) / rx) ** 2 - ((z - cz) / rz) ** 2
        if t > 0:
            best = max(best, ry * math.sqrt(t))
    return best


@asset(name="treasure_pile", size=(16, 8, 16), tags=["prop", "gold", "loot"], grounded=True)
def treasure_pile(p):
    # A main heap and a lower spill to one side, melted together and made lumpy like coins.
    (hx, hz), hr = HEAP
    (sx, sz), sr = SPILL
    heap = ellipsoid(hr, color=p.gold.mid).at(hx, 0, hz)
    spill = ellipsoid(sr, color=p.gold.mid).at(sx, 0, sz)
    pile = heap.smooth_union(spill, k=2.0).displace(amp=0.6, scale=2.0, seed=11)
    pile = pile & box((16, 8, 16))
    # Coins: darker deep down, bright on top, with dithered glints.
    pile = pile.gradient("y", [p.gold[1], p.gold.mid, p.gold.mid, p.gold.light])
    pile = pile.dither(p.gold, seed=4, amount=0.45).darken_crevices()

    # A short stack of coins leaning against the heap.
    stack = cylinder(r=1.2, h=4, color=p.gold.mid).at(2.5, 0, 12.5)
    for y in (0, 2):
        stack = stack.paint(p.gold[1], where=band(axis="y", at=y, height=1))
    stack = stack.paint_faces("top", p.gold.light)

    # A silver chalice so it reads against the gold: foot, stem, bowl, open rim.
    cx, cz, y0 = 9.5, 7.5, 3
    foot = cylinder(r=1.6, h=1, color=p.metal.mid).at(cx, y0, cz)
    stem = cylinder(r=0.6, h=1, color=p.metal.mid).at(cx, y0 + 1, cz)
    bowl = cylinder(r=1.6, h=1, color=p.metal.mid).at(cx, y0 + 2, cz)
    cup = cylinder(r=2.0, h=1, color=p.metal.light).at(cx, y0 + 3, cz)
    cup = cup | cylinder(r=2.3, h=1, color=p.metal.light).at(cx, y0 + 4, cz)
    cup = cup - cylinder(r=1.2, h=1).at(cx, y0 + 4, cz)
    cup = cup.paint(p.metal[1], where=region((cx - 1, y0 + 3, cz - 1), (cx + 1, y0 + 4, cz + 1)))
    ruby = box((1, 1, 1), color=p.red.mid).at(cx - 0.5, y0 + 3, cz + 1.5)
    sapphire = box((1, 1, 1), color=p.blue.mid).at(cx + 1.5, y0 + 3, cz - 0.5)
    chalice = foot | stem | bowl | cup | ruby | sapphire

    # Gems seated on the coins: a bright top over a mid body.
    gems = []
    for x, z, ramp in [
        (4.5, 7.5, p.red),
        (7.5, 11.5, p.blue),
        (12.5, 8.5, p.slime),
        (6.5, 3.5, p.purple),
        (13.5, 12.5, p.red),
        (2.5, 4.5, p.blue),
    ]:
        y = math.floor(surface_y(x, z)) + 0.5
        gem = sphere(1.1, color=ramp.mid).at(x, y, z)
        gems.append(gem.paint(ramp.light, where=band(axis="y", at=y + 0.5, height=1)))

    return pile.union(stack, *gems, chalice)
