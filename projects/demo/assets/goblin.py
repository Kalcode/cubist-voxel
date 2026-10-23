from cubist import *


@asset(name="goblin_scout", size=(16, 24, 12), tags=["character"])
def goblin(p):
    torso = box((6, 7, 4), color=p.cloth.green).at(5, 9, 4)
    torso = torso | box((6, 1, 4), color=p.cloth.brown).at(5, 9, 4)

    head = box((8, 6, 6), color=p.skin.goblin).at(4, 16, 3)
    ear = cone(r=1.5, h=3, axis="x", color=p.skin.goblin).at(12, 19, 6)
    eye = box((2, 1, 1), color=p.eye.yellow).at(5, 18, 8)
    head = head.union(ear, eye).mirror("x", about=8)
    head = head | box((2, 2, 1), color=p.skin.goblin_dark).at(7, 16, 9)

    leg = box((2, 9, 2), color=p.skin.goblin_dark).at(5, 0, 5)
    arm = capsule((0, 0, 0), (0, -6, 0), r=1, color=p.skin.goblin).at(4, 15, 6)
    return Model(
        parts=[
            Part("torso", torso, pivot=(8, 9, 6)),
            Part("head", head, parent="torso", pivot=(8, 16, 6)),
            Part("leg_l", leg, parent="torso", pivot=(6, 9, 6)).mirrored("leg_r", about_x=8),
            Part("arm_l", arm, parent="torso", pivot=(4, 15, 6)).mirrored("arm_r", about_x=8),
        ],
        sockets={"hand_r": (12, 8, 6)},
    )
