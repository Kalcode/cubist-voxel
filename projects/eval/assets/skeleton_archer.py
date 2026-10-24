from cubist import *

CX = 7  # center plane of the body


def bone(points, color):
    """A one-voxel bone through a polyline of voxel centers."""
    segs = [line(a, b, color=color) for a, b in zip(points, points[1:])]
    return segs[0].union(*segs[1:])


@asset(
    name="skeleton_archer", size=(14, 24, 10), tags=["character", "bone", "undead"], grounded=True
)
def skeleton_archer(p):
    # Torso: pelvis, a lumbar spine, three rib hoops around it, a sternum, and a shoulder bar.
    pelvis = box((6, 2, 3), color=p.bone.mid).at(4, 7, 3) - box((2, 1, 1)).at(6, 7, 5)
    spine = box((2, 8, 2), color=p.bone.dark).at(6, 9, 3)
    ribs = [
        torus(R=2.6, r=0.7, axis="y", color=p.bone.light).at(CX, y + 0.5, 5) for y in (10, 12, 14)
    ]
    sternum = box((2, 5, 1), color=p.bone.mid).at(6, 10, 7)
    shoulders = box((10, 1, 2), color=p.bone.mid).at(2, 15, 4)
    torso = pelvis.union(spine, *ribs, sternum, shoulders)

    # Oversized skull: a rounded cranium with deep dark sockets, a nasal hole, and a toothy jaw.
    cranium = box((8, 6, 7), color=p.bone.light).at(3, 18, 1).round(1)
    # Hollow cheeks under the sockets narrow the face toward the jaw. The cutters carry a color
    # only because round() made the cranium grid-stage, where colorless cutters are refused.
    cheek = box((1, 2, 3), color=p.bone.mid)
    cranium = cranium - cheek.at(3, 18, 5) - cheek.at(10, 18, 5)
    socket = box((2, 2, 2), color=p.eye.dark).at(4, 20, 6)
    nose = box((2, 1, 1), color=p.eye.dark).at(6, 19, 7)
    jaw = box((6, 2, 4), color=p.bone.mid).at(4, 16, 3)
    teeth = box((6, 1, 1), color=p.bone.light).at(4, 17, 6)
    gaps = box((1, 1, 1), color=p.eye.dark).at(5, 17, 6) | box((1, 1, 1), color=p.eye.dark).at(
        8, 17, 6
    )
    head = cranium.union(socket.mirror("x", about=CX), nose, jaw, teeth, gaps)
    head = head.paint(p.bone.mid, where=faces("bottom") & colored(p.bone.light))

    # Legs: thin bones with knobbly knees and chunky feet.
    leg = bone([(4.5, 1.5, 5.5), (4.5, 6.5, 5.5)], p.bone.mid)
    leg = (
        leg
        | box((2, 2, 2), color=p.bone.light).at(4, 3, 5)
        | box((2, 1, 3), color=p.bone.light).at(4, 0, 5)
    )

    # Right arm hangs; left arm brings the bow forward.
    arm_r = bone([(2.5, 14.5, 4.5), (2.5, 7.5, 4.5)], p.bone.mid)
    arm_r = (
        arm_r
        | box((2, 2, 2), color=p.bone.light).at(2, 10, 4)
        | box((2, 3, 2), color=p.bone.light).at(2, 5, 4)
    )
    arm_l = bone([(11.5, 14.5, 4.5), (11.5, 11.5, 4.5), (11.5, 10.5, 7.5)], p.bone.mid)
    arm_l = (
        arm_l
        | box((2, 2, 2), color=p.bone.light).at(11, 11, 4)
        | box((3, 3, 2), color=p.bone.light).at(11, 9, 7)
    )

    # Bow: a curved stave in the y-z plane bulging forward, with a straight string behind it.
    # It sits in the outermost column so the side and iso views show the full curve.
    curve = [
        (2.5, 4.5),
        (4.5, 6.5),
        (6.5, 7.5),
        (8.5, 8.5),
        (11.5, 8.5),
        (13.5, 7.5),
        (15.5, 6.5),
        (17.5, 4.5),
    ]
    stave = bone([(13.5, y, z) for y, z in curve], p.wood.mid)
    stave = stave.paint(p.wood.light, where=faces("top") | faces("front"))
    string = line((13.5, 3.5, 4.5), (13.5, 16.5, 4.5), color=p.cloth.white)
    bow = stave | string

    # Quiver on the back with red fletching poking over the shoulder.
    quiver = (
        box((3, 7, 2), color=p.leather.mid)
        .at(8, 9, 0)
        .paint(p.leather.dark, where=band(axis="y", at=9, height=1))
    )
    for x in (8.5, 10.5):
        quiver = quiver | line((x, 16, 0.5), (x, 17, 0.5), color=p.red.mid)

    return Model(
        parts=[
            Part("torso", torso, pivot=(CX, 7, 5), symmetric="x"),
            Part("head", head, parent="torso", pivot=(CX, 16, 5), symmetric="x"),
            Part("leg_l", leg, parent="torso", pivot=(4.5, 7, 5.5)).mirrored("leg_r", about_x=CX),
            Part("arm_r", arm_r, parent="torso", pivot=(2.5, 15, 4.5)),
            Part("arm_l", arm_l, parent="torso", pivot=(11.5, 15, 4.5)),
            Part("bow", bow, parent="arm_l", pivot=(13.5, 10, 8)),
            Part("quiver", quiver, parent="torso", pivot=(9.5, 9, 1)),
        ],
        sockets={"hand_l": (13, 10, 8, "arm_l"), "hand_r": (3, 6, 5, "arm_r")},
    )
