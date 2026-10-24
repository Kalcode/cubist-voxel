from cubist import *


@asset(name="orc_brute", size=(20, 28, 14), tags=["character", "orc", "leather"], grounded=True)
def orc_brute(p):
    # Head: oversized and low between the shoulders; a wide underbite jaw juts forward.
    skull = box((8, 6, 7), color=p.orc.mid).at(6, 21, 5).round(1)
    jaw = box((10, 3, 8), color=p.orc.mid).at(5, 19, 5)
    brow = box((8, 1, 1), color=p.orc[1]).at(6, 24, 12)
    ear = cone(r=1.5, h=2.5, axis="x", color=p.orc.mid).at(14, 23.5, 8.5)
    head = (skull | jaw | brow | ear.mirror("x", about=10)).highlight_edges().darken_crevices()
    eyes = box((1, 1, 1), color=p.red.light).at(8, 23, 11).mirror("x", about=10)
    nose = box((2, 1, 1), color=p.orc[1]).at(9, 23, 12)
    tusks = box((1, 2, 1), color=p.bone.light).at(7, 22, 12).mirror("x", about=10)
    head = head | eyes | nose | tusks
    head_space = box((10, 9, 9), color=p.orc.mid).at(5, 19, 5)

    # Torso: barrel chest in an open leather vest (green shows down the middle), studded,
    # with a belt, gold buckle, and a strip kilt.
    chest = ellipsoid((5.5, 5.5, 4.5), color=p.orc.mid).at(10, 16, 7)
    opening = region((8, 0, 7), (12, 28, 14))
    chest = chest.paint(p.leather.mid, where=band("y", 10, 9) - opening)
    chest = chest.paint(
        p.leather.dark, where=(band("x", 7, 1) | band("x", 12, 1)) & band("y", 10, 9)
    )
    chest = chest.paint(
        p.metal.light, where=region((5, 13, 9), (7, 14, 14)) | region((13, 13, 9), (15, 14, 14))
    )
    belt = box((12, 2, 9), color=p.leather.dark).at(4, 9, 2)
    buckle = box((2, 2, 1), color=p.gold.mid).at(9, 9, 11)
    kilt = box((10, 3, 8), color=p.leather.mid).at(5, 6, 3)
    kilt = kilt.paint(p.leather.dark, where=band("x", 7, 1) | band("x", 12, 1))
    torso = (chest | belt | kilt).highlight_edges().darken_crevices() | buckle
    torso = torso - head_space

    # Legs: short, thick, booted.
    leg = box((4, 4, 5), color=p.orc[1]).at(5, 2, 4)
    boot = box((5, 2, 7), color=p.leather.dark).at(4, 0, 4)
    leg = (leg | boot).darken_crevices()

    # Arms: long and heavy, oversized fists, leather bracers and pauldrons. The orc faces +z,
    # so its right side is -x: the club arm is on the low-x side.
    pauldron = sphere(3.2, color=p.leather.mid).at(3, 20.5, 7) & box((7, 4, 8)).at(0, 19, 3)
    pauldron = pauldron.highlight_edges() | box((1, 1, 1), color=p.metal.light).at(2, 23, 7)

    # Left arm hangs to the knee.
    upper = capsule((17, 19, 7), (17, 13, 7), r=2.4, color=p.orc.mid)
    fore = capsule((17, 13, 7), (17, 8, 7.5), r=2.4, color=p.orc.mid)
    fore = fore.paint(p.leather.dark, where=band("y", 9, 3))
    fist = box((6, 6, 6), color=p.orc.mid).at(14, 2, 4).round(1)
    arm_l = (upper | fore | fist).highlight_edges().darken_crevices()
    arm_l = (arm_l | pauldron.reflect("x", about=10)) - chest

    # Right arm is bent: fist raised in front of the shoulder, gripping the club upright.
    r_upper = capsule((3, 19, 7), (2.5, 14, 8), r=2.4, color=p.orc.mid)
    r_fore = capsule((2.5, 14, 8), (3, 13, 11), r=2.4, color=p.orc.mid)
    r_fore = r_fore.paint(p.leather.dark, where=band("z", 8, 2))
    r_fist = box((6, 6, 6), color=p.orc.mid).at(0, 10, 8).round(1)
    arm_r = (r_upper | r_fore | r_fist).highlight_edges().darken_crevices()
    arm_r = (arm_r | pauldron) - chest

    # Club: knotted head flaring well above the fist, iron studs, pommel knob below.
    club = lathe(
        [(0, 1.6), (1, 1.6), (1.5, 1.2), (9, 1.2), (10.5, 2.2), (14, 2.9), (18, 2.9), (20, 2.3)],
        color=p.wood.mid,
    ).at(3, 7, 11.5)
    club = club.paint(p.wood.dark, where=band("y", 20, 1) | band("y", 24, 1))
    club = club.dither(p.wood, seed=7, amount=0.25).highlight_edges()
    for x, y, z in [(5, 22, 11), (0, 22, 11), (3, 25, 13), (2, 19, 13), (1, 26, 10)]:
        club = club | box((1, 1, 1), color=p.metal.light).at(x, y, z)
    club = club - r_fist - pauldron

    return Model(
        parts=[
            Part("torso", torso, pivot=(10, 9, 7), symmetric="x"),
            Part("head", head, parent="torso", pivot=(10, 19, 8), symmetric="x"),
            Part("leg_r", leg, parent="torso", pivot=(7, 6, 6)).mirrored("leg_l", about_x=10),
            Part("arm_r", arm_r, parent="torso", pivot=(3, 19, 7)),
            Part("arm_l", arm_l, parent="torso", pivot=(17, 19, 7)),
            Part("club", club, parent="arm_r", pivot=(3, 13, 11)),
        ],
        sockets={"hand_r": (3, 13, 11)},
    )
