from cubist import *

# Even width: everything is symmetric about x = z = 4.
C = 4


@asset(
    name="lantern", size=(8, 14, 8), tags=["prop", "metal", "light"], grounded=True, symmetric="x"
)
def lantern(p):
    # Solid base: a stepped plinth.
    base = box((6, 1, 6), color=p.metal.dark).at(1, 0, 1)
    base = base | box((8, 1, 8), color=p.metal[1]).at(0, 1, 0)

    # Glowing glass core, set back one voxel behind the corner posts: a bright 4x4 heart in
    # every pane with a warmer rim, darkest where it meets the base.
    glass = box((6, 6, 6), color=p.glow.mid).at(1, 2, 1)
    glass = glass.paint(
        p.glow.light, where=region((2, 3, 0), (6, 7, 8)) | region((0, 3, 2), (8, 7, 6))
    )
    glass = glass.paint(p.glow.dark, where=band("y", 2, 1))

    posts = [box((1, 6, 1), color=p.metal[1]).at(x, 2, z) for x in (0, 7) for z in (0, 7)]

    # Cap plate, a stepped roof, and a hanging ring (r = 0.85 so its corners join up).
    cap = box((8, 1, 8), color=p.metal[1]).at(0, 8, 0)
    roof = box((6, 1, 6), color=p.metal[1]).at(1, 9, 1) | box((4, 1, 4), color=p.metal[1]).at(
        2, 10, 2
    )
    ring = torus(R=1.5, r=0.85, axis="z", color=p.metal.mid).at(C, 12, C)

    frame = base.union(*posts, cap, roof).highlight_edges().darken_crevices()
    return glass | frame | ring
