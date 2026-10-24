import math

from cubist import *

# Faces +z. The spine runs down the x = 15 voxel column (centre 15.5), so the snout, spines
# and tail tip have a middle column; the model is 31 voxels wide inside the 32 box.
CX = 15.5
BODY = (CX, 9.5, 16)  # body ellipsoid centre
BODY_R = (5, 4.5, 7.5)


def body_top(z):
    """Height of the body ellipsoid's top surface at depth z."""
    t = (z - BODY[2]) / BODY_R[2]
    return BODY[1] + BODY_R[1] * math.sqrt(max(0.0, 1 - t * t))


def wing(p):
    """The right wing, spread out to +x from the shoulder, tilted and swept back."""
    root, elbow, wrist = (19, 13), (24, 19), (29.5, 22.5)
    tips = [(30.8, 14), (27.5, 9.5), (23.5, 9)]
    # Deep notches between the finger tips give the trailing edge its scallops.
    outline = [
        root,
        elbow,
        wrist,
        tips[0],
        (28, 16),
        tips[1],
        (25.5, 14),
        tips[2],
        (21.5, 12.5),
        (19.5, 10),
    ]
    membrane = extrude(outline, depth=1.6, color=p.red.light).at(0, 0, 14.7)
    bones = [
        capsule((*root, 15.5), (*elbow, 15.5), r=0.9, color=p.red.dark),
        capsule((*elbow, 15.5), (*wrist, 15.5), r=0.7, color=p.red.dark),
    ]
    for start, tip in zip((wrist, wrist, elbow), tips):
        # Capsules, not lines: a rotated max-norm line samples as scattered voxels.
        bones.append(capsule((*start, 15.5), (*tip, 15.5), r=0.6, color=p.red.dark))
    shape = membrane.union(*bones)
    # Tip the top edge back so the membrane faces up and forward (it reads from the top
    # and iso views, not only head-on), then sweep the tip back.
    shape = shape.rotate(-35, axis="x", about=(19, 13, 15.5))
    return shape.rotate(15, axis="y", about=(19, 13, 15.5))


def leg(p, hip, foot, r):
    """A leg from hip to foot (x, z on the right side) with a clawed foot."""
    limb = capsule(hip, (foot[0], r, foot[2]), r=r, color=p.red.mid)  # stops at the ground
    paw = box((3, 2, 4), color=p.red[1]).at(foot[0] - 1.5, 0, foot[2] - 2)
    claws = box((3, 1, 1), color=p.bone.light).at(foot[0] - 1.5, 0, foot[2] + 2)
    return (limb | paw | claws).darken_crevices()


@asset(
    name="dragon_whelp",
    size=(32, 24, 32),
    tags=["creature", "red", "creature", "dragon"],
    grounded=True,
)
def dragon_whelp(p):
    body = ellipsoid(BODY_R, color=p.red.mid).at(*BODY)
    chest = sphere(4.5, color=p.red.mid).at(CX, 10, 21)
    neck = capsule((CX, 11, 21), (CX, 16, 25), r=2.6, color=p.red.mid)
    # The tail sweeps back and curls round to the right.
    tail_pts = [
        (CX, 9, 10),
        (CX, 6.5, 5),
        (CX + 3, 4, 2.4),
        (CX + 8, 3, 2),
        (CX + 12, 2.5, 3.8),
        (CX + 13.5, 2.5, 7),
    ]
    radii = [3.0, 2.4, 1.9, 1.5, 1.1, 0.8]
    tail = [capsule(a, b, r=r, color=p.red.mid) for a, b, r in zip(tail_pts, tail_pts[1:], radii)]
    spade = extrude([(-1.5, 0), (0, -1), (1.5, 0), (0, 3)], depth=1, axis="y", color=p.red.dark).at(
        CX + 13.5, 2, 7
    )
    torso = body.union(chest, neck, *tail)

    belly = inside(ellipsoid((3.8, 3, 7)).at(CX, 6.8, 16.5)) | inside(
        capsule((CX, 9.5, 22.5), (CX, 14.5, 26.5), r=1.9)
    )
    torso = torso.paint(p.gold.light, where=belly)
    for z in range(9, 28, 2):  # scale plates
        torso = torso.paint(p.gold.mid, where=belly & band("z", z, 1))
    spines = []
    for z in range(10, 23, 3):
        spines.append(cone(r=1, h=2, color=p.red.dark).at(CX, body_top(z + 0.5) - 0.5, z + 0.5))
    torso = torso.union(*spines, spade).darken_crevices().highlight_edges()

    head = ellipsoid((3.6, 3.2, 3.4), color=p.red.mid).at(CX, 17.5, 25.5)
    snout = box((5, 3, 5), color=p.red.mid).at(CX - 2.5, 15, 27)
    jaw = box((3, 1, 4), color=p.gold.light).at(14, 14, 28)
    nostril = box((1, 1, 1), color=p.red.dark).at(14, 17, 31)
    # Gold eye with a dark pupil at the front, set into the side of the head.
    eye = box((1, 1, 2), color=p.gold.light).at(12, 18, 26) | box((1, 1, 1), color=p.eye.dark).at(
        12, 18, 27
    )
    brow = box((2, 1, 3), color=p.red[1]).at(12, 19, 25)
    horn = capsule((13.5, 20, 25), (12.8, 21.5, 22.5), r=0.9, color=p.bone.mid)
    horn_tip = line((12.5, 22, 22.5), (12.5, 23, 20.5), color=p.bone.light)
    head = head.union(snout, jaw, nostril, eye, brow, horn, horn_tip).mirror("x", about=CX)
    head = head.darken_crevices()

    wing_r = wing(p)
    return Model(
        parts=[
            Part("body", torso, pivot=(CX, 5, 16)),
            Part("head", head, parent="body", pivot=(CX, 16, 25)),
            Part("wing_r", wing_r, parent="body", pivot=(19, 13, 15.5)).mirrored(
                "wing_l", about_x=CX
            ),
            Part(
                "leg_fr",
                leg(p, (CX + 3.5, 8, 20), (CX + 4, 1, 21), 1.6),
                parent="body",
                pivot=(CX + 3.5, 8, 20),
            ).mirrored("leg_fl", about_x=CX),
            Part(
                "leg_br",
                leg(p, (CX + 4, 8, 11), (CX + 4.5, 1, 12), 1.8),
                parent="body",
                pivot=(CX + 4, 8, 11),
            ).mirrored("leg_bl", about_x=CX),
        ],
        sockets={"mouth": (CX, 15, 31, "head")},
    )
