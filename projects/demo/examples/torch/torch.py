from cubist import *

# Odd width, centered on a voxel center (3.5): pointed shapes need a middle column for
# their tips. Centered between columns (an even width), a cone's tip vanishes.
C = 3.5


def flame(p, k):
    """Frame k of the flame: the same teardrop, licked into a different shape each frame."""
    lean = (0.3, -0.2, 0.0, 0.25, -0.3, 0.1)[k]
    tall = (6, 5, 6.5, 5.5, 6, 5)[k]
    body = sphere(2.0, color=p.fire.mid).at(C, 12.5, C)
    tip = cone(r=2.0, h=tall, color=p.fire.mid).at(C + lean, 11.5, C)
    shape = body.smooth_union(tip, k=1.5).displace(amp=0.45, scale=1.5, seed=3 + k)
    # Hot at the base, cooler at the tips.
    return shape.gradient("y", p.fire, reverse=True)


@asset(name="torch", size=(7, 18, 7), tags=["prop", "light", "wood"], grounded=True)
def torch(p):
    handle = cylinder(r=1, h=11, color=p.wood[1]).at(C, 0, C)
    wrap = cylinder(r=1.5, h=2, color=p.cloth.brown).at(C, 7, C)
    cup = cone(r=1.5, h=2, r_top=2.3, color=p.metal.dark).at(C, 9, C)
    return Model(
        parts=[
            Part("torch", handle | wrap | cup, pivot=(C, 0, C)),
            # Six frames at 10 fps: the flame flickers on its own, no clip needed.
            Part(
                "flame",
                frames=[flame(p, k) for k in range(6)],
                fps=10,
                parent="torch",
                pivot=(C, 11, C),
            ),
        ],
        lights={"flame": light((C, 14, C), p.fire.light, range=96, flicker=0.35)},
    )
