from cubist import *

# Odd widths centered on voxel centers (3.5, 4.5) so the flame's tip has a middle column.
C = 3.5  # x center of the plate and torch
TZ = 4.5  # z center of the torch, out from the wall


def cup_shape(p):
    """The iron cup at the top of the torch that the flame sits in."""
    return cone(r=1.6, h=1, r_top=2.4, color=p.metal.dark).at(C, 9, TZ)


def flame(p, k):
    """Frame k of the flame: one teardrop and a side tongue, licked into new shapes each frame."""
    lean = (0.0, 0.45, 0.2, -0.4, -0.2, 0.35)[k]
    tall = (5.2, 4.6, 5.5, 4.8, 5.4, 4.5)[k]
    side = (1, 1, -1, -1, 1, -1)[k]  # which side the small tongue licks up on
    body = sphere(2.1, color=p.fire.mid).at(C, 11.4, TZ)
    tip = cone(r=1.9, h=tall, color=p.fire.mid).at(C + lean, 11, TZ)
    tongue = cone(r=1.1, h=2.6 + 0.3 * (k % 3), color=p.fire.mid).at(C + 1.3 * side, 12, TZ)
    shape = body.smooth_union(tip.union(tongue), k=1.2)
    shape = shape.displace(amp=0.4, scale=1.5, seed=5 + k)
    # Sits on the cup without sharing its voxels.
    shape = shape - box((8, 10, 8))
    # Pale yellow at the heart where it leaves the cup, orange outside, red-orange tips.
    core = [p.fire[1], p.fire.mid, p.fire.light, p.glow.light]
    rim = [p.fire[1], p.fire[1], p.fire.mid, p.fire.light]
    shape = shape.gradient("y", core, reverse=True, lo=10, hi=16)
    outside = ~inside(cylinder(r=1.45, h=8).at(C, 9, TZ))
    return shape.gradient("y", rim, reverse=True, lo=10, hi=16, where=outside)


@asset(name="wall_sconce", size=(8, 16, 8), tags=["prop", "metal", "light", "fire"])
def wall_sconce(p):
    # Iron wall plate: a shield with a pointed top and a rivet at each corner.
    # A raised rim around a recessed panel, so the plate reads as forged and not flat.
    outline = [(0, 2), (1, 1), (6, 1), (7, 2), (7, 9), (3.5, 12.5), (0, 9)]
    recess = [(1, 2), (6, 2), (6, 9), (3.5, 11.5), (1, 9)]
    plate = extrude(outline, 2, color=p.metal.mid) - extrude(recess, 1).at(0, 0, 1)
    plate = plate.paint(p.metal[1], where=band("z", 0, 1))
    plate = plate.highlight_edges(facing=("top", "left"))
    for x, y in ((1, 2), (5, 2), (1, 8), (5, 8)):
        plate = plate | box((1, 1, 1), color=p.metal.light).at(x, y, 1)

    # Bracket: an arm out from the plate to a collar around the handle, braced from below.
    arm = box((1, 1, 2), color=p.metal[1]).at(3, 5, 1)
    brace = line((3.5, 2.5, 1.5), (3.5, 4.5, 3.5), color=p.metal[1])
    collar = torus(1.5, 0.9, color=p.metal.mid).at(C, 5.5, TZ)
    iron = (arm | brace | collar).highlight_edges(facing=("top", "left"))

    # Wooden torch: a handle thickening toward the top, a pitch-soaked wrap, the iron cup.
    handle = capsule((C, 0.8, TZ), (C, 8, TZ), r=1.0, r_end=1.5, color=p.wood.mid)
    handle = handle.gradient("y", [p.wood[1], p.wood.mid, p.wood.light])
    wrap = cylinder(r=1.5, h=2, color=p.leather.mid).at(C, 7, TZ)
    wrap = wrap.paint(p.leather.dark, where=band("y", 7))
    torch = handle | wrap | cup_shape(p)

    # The collar goes on last so it grips the handle (iron wins where they meet).
    body = plate | torch | iron
    return Model(
        parts=[
            Part("sconce", body, pivot=(C, 1, 0.5)),
            # Six frames at 10 fps: the flame flickers on its own, no clip needed.
            Part(
                "flame",
                frames=[flame(p, k) for k in range(6)],
                fps=10,
                parent="sconce",
                pivot=(C, 10, TZ),
            ),
        ],
        lights={"flame": light((C, 12.5, TZ), p.fire.light, range=10, flicker=0.4)},
    )
