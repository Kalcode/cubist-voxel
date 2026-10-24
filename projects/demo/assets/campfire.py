from cubist import *

C = 7.5  # odd footprint, centered on a voxel center


def flames(p, k):
    """Three tongues around the center, each frame reshuffled."""
    out = None
    for i, (dx, dz) in enumerate(((0, 0), (-1.5, 1), (1.5, -1))):
        h = (6.5, 4.5, 5)[i] + (0.0, 0.9, -0.7, 0.5)[(k + i) % 4]
        tongue = cone(r=1.6 if i == 0 else 1.2, h=h, color=p.fire.mid).at(C + dx, 3, C + dz)
        out = tongue if out is None else out | tongue
    out = out.displace(amp=0.3, scale=1.4, seed=11 + k)
    # Yellow at the base to orange at the tips (the darkest red is for embers).
    return out.gradient("y", p.fire.ramp[1:], reverse=True, lo=3, hi=9)


@asset(name="campfire", size=(15, 11, 15), tags=["prop", "light", "fire", "stone"], grounded=True)
def campfire(p):
    stones = None
    for i in range(10):
        stone = (
            sphere(1.3, color=p.stone.mid)
            .at(C, 1, C)
            .at(5.2, 0, 0)
            .rotate(36 * i, "y", about=(C, 0, C))
        )
        stones = stone if stones is None else stones | stone
    stones = stones.dither(p.stone, seed=4, amount=0.5).darken_crevices()
    logs = [
        capsule((C - 4, 1.5, C), (C + 4, 2.5, C), r=1, color=p.wood[1]).rotate(
            a, "y", about=(C, 0, C)
        )
        for a in (0, 60, 120)
    ]
    wood = logs[0].union(*logs[1:]).paint(p.wood.dark, where=faces("bottom")).highlight_edges()
    ash = cylinder(r=5.5, h=1, color=p.stone.dark).at(C, 0, C)
    embers = ash.paint(p.fire.dark, where=cylinder(r=2.6, h=1).at(C, 0, C))
    return Model(
        parts=[
            Part("pit", stones | embers | wood, pivot=(C, 0, C)),
            Part(
                "flames",
                frames=[flames(p, k) for k in range(4)],
                fps=8,
                parent="pit",
                pivot=(C, 3, C),
            ),
        ],
        lights={"fire": light((C, 6, C), p.fire.light, range=14, intensity=2, flicker=0.5)},
    )
