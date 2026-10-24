from cubist import *

# Faces +Z, three voxels wide and centered on x = 1.5. The barrel ends at z = 14 so the muzzle
# flash fits in the last layer of the declared box (z = 15). Small parts (hammer, frizzen,
# trigger, guard) sit on the center column x = 1.
X = 1.5
MUZZLE = (X, 5.5, 15)


def row(y, z0, z1, color):
    """One three-wide row of the profile, from z0 up to (not including) z1."""
    return box((3, 1, z1 - z0), color=color).at(0, y, z0)


def flash(p, big):
    """The muzzle flash in the layer in front of the bore: a plus, or a tall star."""
    star = box((1, 3, 1), color=p.fire.mid).at(1, 4, 15) | box((3, 1, 1), color=p.fire.mid).at(0, 5, 15)
    if big:
        star = (
            box((1, 7, 1), color=p.fire.mid).at(1, 2, 15)
            | box((3, 3, 1), color=p.fire.mid).at(0, 4, 15)
            | box((1, 3, 1), color=p.fire.light).at(1, 4, 15)
            | box((3, 1, 1), color=p.fire.light).at(0, 5, 15)
        )
    return star | box((1, 1, 1), color=p.fire.light).at(1, 5, 15)


@asset(name="flintlock", size=(3, 9, 16), tags=["prop", "weapon", "wood", "metal"])
def flintlock(p):
    # Stock and grip, row by row: wood runs under the barrel nearly to the muzzle, and the grip
    # slants down and back to a brass butt cap.
    wood = p.wood.mid
    stock = row(5, 3, 6, wood) | row(4, 2, 13, wood) | row(3, 2, 13, wood)
    stock = stock | row(2, 1, 4, wood) | row(1, 0, 3, wood)
    stock = stock.dither(p.wood, seed=2, amount=0.2).paint_faces("bottom", p.wood[1])
    stock = stock.highlight_edges(facing=("top",))
    stock = stock.paint(p.metal[1], where=region((0, 4, 3), (3, 5, 7)))  # lock plate
    brass = (
        box((3, 1, 2), color=p.gold.mid).at(0, 0, 0)  # butt cap
        | box((3, 2, 1), color=p.gold.mid).at(0, 3, 13)  # forend cap
        | box((1, 1, 4), color=p.gold[1]).at(1, 1, 3)  # trigger guard
        | box((1, 1, 1), color=p.gold[1]).at(1, 2, 6)
    )
    trigger = box((1, 1, 1), color=p.metal.light).at(1, 2, 5)

    # A half-round barrel bedded in the stock (three wide, one more on top) and a dark muzzle
    # band bored through, so the muzzle reads as a ring from the front.
    barrel = box((3, 1, 8), color=p.metal.mid).at(0, 5, 6) | box((1, 1, 8), color=p.metal.mid).at(
        1, 6, 6
    )
    barrel = barrel.paint_faces("top", p.metal.light, where=band("x", 1))  # light down the spine
    barrel = (barrel | box((3, 2, 1), color=p.metal.dark).at(0, 5, 14)) - box((1, 1, 2)).at(1, 5, 13)
    barrel = barrel | box((1, 1, 1), color=p.gold.mid).at(1, 7, 13)  # brass front sight
    body = stock | brass | trigger | barrel

    # The cock: a neck with jaws forward and a spur back, holding a flint.
    hammer = box((1, 3, 1), color=p.metal.mid).at(1, 6, 3) | box((1, 1, 3), color=p.metal.mid).at(
        1, 8, 2
    )
    hammer = hammer.paint(p.stone.dark, where=region((1, 8, 4), (2, 9, 5)))
    frizzen = box((1, 2, 1), color=p.metal.light).at(1, 7, 6)

    return Model(
        parts=[
            Part("stock", body, pivot=(X, 2, 2)),
            # hinge("x"): positive angles turn +Y toward +Z, so the top snaps forward.
            Part("hammer", hammer, parent="stock", pivot=(X, 6.5, 3.5), joint=hinge("x", (0, 60))),
            # Struck by the flint, the frizzen flips forward off the pan.
            Part("frizzen", frizzen, parent="stock", pivot=(X, 7, 7), joint=hinge("x", (0, 70))),
            Part(
                "flash",
                frames=[flash(p, True), flash(p, False)],
                parent="stock",
                pivot=(X, 5.5, 15.5),
                hidden=True,
            ),
        ],
        sockets={
            "grip": socket((X, 2, 2), part="stock", forward="+z"),
            "muzzle": socket(MUZZLE, part="stock", forward="+z"),
        },
        clips=[
            clip(
                "fire",
                0.4,
                [
                    TrackSpec("hammer", "joint", [(0, 0), (0.03, 55), (0.08, 48)]),
                    TrackSpec("frizzen", "joint", [(0, 0), (0.03, 0), (0.06, 65), (0.12, 60)]),
                    visible("flash", [(0, 0), (0.04, 1), (0.1, 0)]),
                    frames("flash", [(0, 0), (0.07, 1)]),  # big star, then a small one
                    turn("stock", "x", [(0, 0), (0.04, 0), (0.08, -12), (0.4, 0)]),  # kick up
                ],
                events=[(0.0, "hammer"), (0.04, "fire"), (0.04, "emit:smoke")],
            ),
            clip(
                "cock",
                0.35,
                {"hammer": [48, 0], "frizzen": [60, 0]},
                events=[(0.3, "click")],
            ),
        ],
        emitters={
            "smoke": emitter(
                (X, 5.5, 15.5),
                "flintlock_smoke",
                direction="+z",
                burst=8,
                life=1.2,
                speed=7,
                spread=30,
                gravity=-3,
                grow=(0.5, 1.6),
                part="stock",
            )
        },
    )
