from cubist import *

# Faces +Z; its right side is at low x. Center plane x = 6, so symmetric pieces have even widths.
X = 6
HIP = 5  # top of the legs, bottom of the torso


@asset(name="kobold", size=(12, 18, 10), tags=["character", "creature"], grounded=True)
def kobold(p):
    # Torso in a leather vest, open down the front to show the pale belly scales.
    torso_box = box((6, 5, 4), color=p.leather.mid).at(3, HIP, 3)
    torso = torso_box.paint(p.leather.dark, where=band("y", HIP))  # belt
    torso = torso.paint(p.cloth.straw, where=region((5, HIP + 1, 6), (7, HIP + 5, 7)))
    torso = torso.paint(p.leather.dark, where=region((4, HIP + 1, 6), (8, HIP + 5, 7)) & ~inside(
        box((2, 4, 1)).at(5, HIP + 1, 6)
    ))
    torso = torso.highlight_edges(facing=("top", "left"))

    # Head: a rounded skull, a long snout with a pale lower jaw, fangs, gold eyes, horns.
    skull = box((6, 5, 5), color=p.red.mid).at(3, 10, 1).round(1)
    snout = box((4, 2, 4), color=p.red.mid).at(4, 11, 6)
    head = (skull | snout).gradient("y", [p.red[1], p.red.mid, p.red.light], lo=10, hi=15)
    head = head.paint(p.gold.light, where=region((4, 13, 5), (5, 14, 6)) | region((7, 13, 5), (8, 14, 6)))
    horn = capsule((4, 13.5, 3), (3.2, 16, 0.4), r=1.0, r_end=0.5, color=p.bone.mid)
    horns = horn.mirror("x", X).gradient("y", p.bone, lo=14, hi=18)
    head = head | horns

    # Lower jaw: hinged at the back of the snout so it can drop open to hiss.
    jaw = box((4, 1, 4), color=p.cloth.straw).at(4, 10, 6)
    jaw = jaw | box((1, 1, 1), color=p.bone.light).at(4, 11, 9).mirror("x", X)  # fangs
    head = head - box((1, 1, 1)).at(4, 11, 9).mirror("x", X)

    # Legs, with feet as their own parts that counter-rotate so the soles stay flat.
    leg = box((2, 4, 2), color=p.red[1]).at(3, 1, 4)
    foot = box((2, 1, 4), color=p.red[1]).at(3, 0, 4)
    foot = foot.paint(p.bone.mid, where=band("z", 7))  # claws

    # Arms hang from the shoulders and end in big clawed hands.
    arm = capsule((2, 9.5, 5), (2, 6.5, 5.8), r=0.75, color=p.red.mid)
    hand = sphere(1.5, color=p.red[1]).at(2, 5.2, 6.2)
    hand = hand.paint(p.bone.mid, where=region((1, 4, 6), (3, 5, 8)))  # claws
    arm = (arm | hand).highlight_edges(facing=("top", "left"))

    # Long tail: down and back from the hips, then sweeping out along the floor to one side
    # and curling forward at the end.
    tip = capsule((6.5, 3.2, 1.0), (10.3, 1.6, 0.7), r=1.0, r_end=0.6, color=p.red[1])
    tip = tip | line((10.5, 1.5, 1.5), (11.5, 2.5, 3.5), color=p.red[1])  # face-connected curl
    tail = capsule((X, 6.5, 3.4), (6.5, 3.2, 1.0), r=1.3, r_end=1.0, color=p.red.mid)
    tail = (tail - torso_box - tip).gradient("y", [p.red[1], p.red.mid], lo=3, hi=8)

    return Model(
        parts=[
            Part("torso", torso, pivot=(X, HIP, 5)),
            Part("head", head, parent="torso", pivot=(X, 10, 5)),
            Part("jaw", jaw, parent="head", pivot=(X, 11, 6)),
            Part("leg_r", leg, parent="torso", pivot=(4, HIP, 5)).mirrored("leg_l", about_x=X),
            # Not .mirrored(): a mirrored twin keeps the original's parent (leg_r).
            Part("foot_r", foot, parent="leg_r", pivot=(4, 1, 5)),
            Part("foot_l", foot.reflect("x", X), parent="leg_l", pivot=(8, 1, 5)),
            Part("arm_r", arm, parent="torso", pivot=(2, 9.5, 5)).mirrored("arm_l", about_x=X),
            Part("tail", tail, parent="torso", pivot=(X, 5.5, 2.5)),
            Part("tail_tip", tip, parent="tail", pivot=(6.5, 3.2, 1.0)),
        ],
        clips=[
            # A quick scurry: legs swing in opposite phases, feet counter-rotate to stay flat,
            # the torso dips as the legs spread so the feet stay planted, the head sways with
            # each step, and the tail lashes side to side with the tip trailing behind.
            clip(
                "walk",
                0.7,
                [
                    swing("leg_r", "x", 28),
                    swing("leg_l", "x", 28, phase=0.5),
                    swing("foot_r", "x", 28, phase=0.5),
                    swing("foot_l", "x", 28),
                    swing("arm_r", "x", 24, phase=0.5),
                    swing("arm_l", "x", 24),
                    bob("torso", -0.45),
                    swing("head", "y", 5, phase=0.5),
                    swing("tail", "y", 10),
                    swing("tail_tip", "y", 16, phase=-0.12),
                ],
                loop=True,
                events=[(0.175, "footstep"), (0.525, "footstep")],
                travel=(0, 0, 10.7),  # in place; the ankles sweep 3.76 voxels per step, two steps in 0.7 s
            ),
            # Looks around, tail swaying slowly, and hisses once.
            clip(
                "idle",
                2.4,
                [
                    swing("head", "y", 15),
                    turn("jaw", "x", [(0, 0), (1.3, 0), (1.45, 25), (1.75, 25), (1.95, 0), (2.4, 0)]),
                    swing("arm_r", "x", 4, cycles=2),
                    swing("arm_l", "x", 4, cycles=2, phase=0.5),
                    swing("tail", "y", 6),
                    swing("tail_tip", "y", 12, phase=-0.1),
                ],
                loop=True,
                events=[(1.45, "hiss")],
            ),
        ],
    )
