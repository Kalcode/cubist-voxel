from cubist import *

CX, CZ = 7, 6  # body axis
# Staff axis (x, z): on voxel centres so it is one voxel wide, forward of the hat brim.
STAFF = (11.5, 9.5)


@asset(name="wizard", size=(14, 28, 12), tags=["character", "purple", "magic"], grounded=True)
def wizard(p):
    robe = lathe([(0, 5.4), (2, 4.6), (8, 3.4), (12, 3.6), (14, 3.0)], color=p.purple.mid)
    robe = robe.at(CX, 0, CZ)
    robe = robe.paint(p.gold.mid, where=band("y", 0, 1) | region((6, 0, 8), (8, 9, 12)))
    robe = robe.paint(p.leather.mid, where=band("y", 9, 1))
    robe = robe.darken_crevices().highlight_edges()

    head = box((6, 6, 6), color=p.skin.mid).at(4, 14, 3).round(1)
    eye = box((1, 1, 1), color=p.eye.dark).at(5, 17, 8)
    brow = box((2, 1, 1), color=p.bone.light).at(4, 18, 8)
    head = head.union(eye, brow).mirror("x", about=CX)
    # White hair round the back and sides of the head.
    hair = (
        region((0, 14, 0), (14, 20, 5))
        | region((0, 14, 0), (5, 18, 7))
        | region((9, 14, 0), (14, 18, 7))
    )
    head = head.paint(p.bone.light, where=hair & ~colored(p.eye.dark, p.bone.light))
    nose = box((2, 2, 1), color=p.skin[1]).at(6, 16, 9)
    # Beard: a downward cone pushed to the front of the face, under a full moustache.
    beard = cone(r=0.9, h=9, r_top=3.4, color=p.bone.light).at(CX, 7, 7.5)
    beard = beard & box((14, 9, 6)).at(0, 7, 6)
    moustache = box((6, 1, 1), color=p.bone.light).at(4, 15, 9)
    beard = (beard | moustache).paint(p.bone.mid, where=band("y", 7, 2)).darken_crevices()
    head = head.union(nose, beard)

    brim = cylinder(r=5.2, h=1, color=p.purple[1]).at(CX, 19, CZ)
    crown = cone(r=3.4, h=8, color=p.purple.mid).at(CX, 20, CZ)
    crown = crown.paint(p.gold.mid, where=band("y", 20, 1))
    tip = capsule((7, 25.5, 6), (6, 27.4, 4.2), r=0.7, color=p.purple.mid)
    hat = (brim | crown | tip).highlight_edges()
    head = head | hat

    sleeve_l = cone(r=2.0, h=5, r_top=1.3, color=p.purple.mid).at(2.5, 8, 6)
    hand_l = sphere(1.6, color=p.skin.mid).at(2.5, 7, 7)
    arm_l = (hand_l | sleeve_l).darken_crevices()
    sleeve_r = capsule((10, 13, 6), (STAFF[0], 10.5, STAFF[1] - 1), r=1.4, color=p.purple.mid)
    hand_r = sphere(1.6, color=p.skin.mid).at(STAFF[0], 10.5, STAFF[1])
    arm_r = (sleeve_r | hand_r).darken_crevices()

    pole = cylinder(r=0.5, h=23, color=p.wood[1]).at(STAFF[0], 0, STAFF[1])
    claw = cone(r=0.8, h=2, r_top=1.8, color=p.gold[1]).at(STAFF[0], 21, STAFF[1])
    # Saturated yellow with a pale hot spot toward the light (top-left-front).
    orb = sphere(2.1, color=p.glow.mid).at(STAFF[0], 24.5, STAFF[1])
    orb = orb.paint(
        p.glow.light, where=region((9, 24, 9), (12, 27, 12)) | region((9, 25, 0), (12, 27, 12))
    )
    staff = pole | claw | orb

    return Model(
        parts=[
            Part("body", robe, pivot=(CX, 0, CZ)),
            Part("head", head, parent="body", pivot=(CX, 14, CZ)),
            Part("arm_l", arm_l, parent="body", pivot=(3, 13, CZ)),
            Part("arm_r", arm_r, parent="body", pivot=(10, 13, CZ)),
            Part("staff", staff, parent="arm_r", pivot=(11, 10, 9)),
        ],
        sockets={"hand_r": (11, 10, 9), "staff_tip": (11, 24, 9)},
    )
