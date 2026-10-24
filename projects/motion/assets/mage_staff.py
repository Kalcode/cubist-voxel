from cubist import *

C = 2.5  # odd footprint, centered on a voxel center
ORB_Y = 21.5
GRIP_Y = 8

# Diagonal directions for the claw's four prongs: corners leave the orb's faces open.
CORNERS = ((-1, -1), (1, -1), (-1, 1), (1, 1))


def shaft(p):
    """A crooked pole: thick capsules through wobbling points, with knots."""
    pts = [(C, 1.3, C), (C + 0.3, 5, C - 0.2), (C - 0.3, 10, C + 0.2), (C + 0.2, 14, C - 0.2), (C, 17, C)]
    pole = None
    for a, b in zip(pts, pts[1:]):
        seg = capsule(a, b, r=1.25, color=p.wood[1])
        pole = seg if pole is None else pole | seg
    for x, y, z in ((C + 0.9, 6, C + 0.4), (C - 0.8, 12, C - 0.6), (C + 0.5, 15.5, C + 0.9)):
        pole = pole | sphere(1.15, color=p.wood[1]).at(x, y, z)
    # A dark bark seam spiraling up the pole, one turn every 8 voxels.
    seam = polyline(
        [(C + 1.4 * ARC[y % 8][0], y + 0.5, C + 1.4 * ARC[y % 8][1]) for y in range(2, 17)],
        p.wood.dark,
        connect=26,
    )
    pole = pole.paint(p.wood.dark, where=seam)
    pole = pole.highlight_edges(facing=("top", "left")).darken_crevices()
    ferrule = cylinder(r=1.25, h=2, color=p.metal.mid).at(C, 0, C).paint(
        p.metal.dark, where=band("y", 0)
    )
    return pole | ferrule


ARC = [(1, 0), (0.7, 0.7), (0, 1), (-0.7, 0.7), (-1, 0), (-0.7, -0.7), (0, -1), (0.7, -0.7)]


def claw(p):
    """A gold collar and cup, and four root prongs gripping the orb's waist with bone talons."""
    collar = cylinder(r=1.5, h=1, color=p.gold.mid).at(C, 17, C).paint(
        p.gold.dark, where=band("x", 3, 2) | band("z", 3, 2)
    )
    cup = cylinder(r=1, h=2, color=p.gold.light).at(C, 18, C)
    prongs = None
    for dx, dz in CORNERS:
        x0, z0 = 2.5 + dx, 2.5 + dz
        x1, z1 = 2.5 + 2 * dx, 2.5 + 2 * dz
        pts = [(x0, 18.5, z0), (x1, 19.5, z1), (x1, 20.5, z1), (x0, 21.5, z0)]
        prong = polyline(pts, p.wood.dark, connect=6)
        prongs = prong if prongs is None else prongs | prong
    prongs = prongs.paint(p.wood[1], where=band("y", 19, 2))
    prongs = prongs.paint(p.bone.light, where=region((1, 21, 1), (4, 22, 4)))  # talon tips
    return collar | cup | prongs


def orb(p, k):
    """Frame k of the orb's pulse: 0 dim, 1 bright, 2 blazing (the cast)."""
    ramp = p.glow.ramp
    ball = sphere(1.5, color=ramp[min(k + 1, 2)]).at(C, ORB_Y, C)
    ball = ball.paint(ramp[min(k, 2)], where=band("y", 20))
    return ball.paint(ramp[2], where=band("y", 22) | region((1, 21, 1), (3, 22, 3)))


@asset(name="mage_staff", size=(5, 24, 5), tags=["prop", "magic", "wood"])
def mage_staff(p):
    return Model(
        parts=[
            Part("staff", shaft(p) | claw(p), pivot=(C, GRIP_Y, C)),
            # The talons take the orb's waist corners, so subtract the claw from each frame.
            # Pulses on its own (dim, bright, blazing, bright); the cast clip flares it.
            Part(
                "orb",
                frames=[orb(p, k) - claw(p) for k in (0, 1, 2, 1)],
                fps=3,
                parent="staff",
                pivot=(C, ORB_Y, C),
            ),
        ],
        sockets={
            "grip": socket((C, GRIP_Y, C), part="staff", forward="+z"),
            "orb": socket((C, 23, C), part="staff", forward="+y", up="+z"),
        },
        clips=[
            # A slow pulse while held: dim, bright, blazing, bright.
            clip(
                "idle",
                1.6,
                [frames("orb", [(0, 0), (0.4, 1), (0.8, 2), (1.2, 3), (1.6, 0)])],
                loop=True,
            ),
            # Rise and wind back (anticipation), thrust past the target, settle.
            clip(
                "cast",
                1.0,
                [
                    turn("staff", "x", [(0, 0), (0.3, -28), (0.38, -30), (0.5, 38), (0.62, 32), (1.0, 0)]),
                    move("staff", "z", [(0, 0), (0.3, -1), (0.38, -1), (0.5, 3), (0.62, 2), (1.0, 0)]),
                    move("staff", "y", [(0, 0), (0.3, 1), (0.5, 0), (1.0, 0)]),
                    frames("orb", [(0, 0), (0.3, 1), (0.48, 2), (0.75, 1), (0.9, 0)]),
                ],
                events=[(0.5, "cast"), (0.5, "emit:sparks")],
            )
        ],
        emitters={
            "sparks": emitter(
                (C, 23.5, C),
                "mage_staff_spark",
                direction="+y",
                burst=18,
                life=0.6,
                speed=11,
                spread=70,
                gravity=8,
                part="staff",
            )
        },
        lights={"orb": light((C, ORB_Y, C), p.glow.light, range=8, flicker=0.15, part="orb")},
    )
