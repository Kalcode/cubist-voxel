from cubist import *

# Faces +Z, so the knight's right side is at low x. Symmetric parts mirror about x = 8.
MID = 8


@asset(name="knight", size=(16, 24, 12), tags=["character", "metal", "human"], grounded=True)
def knight(p):
    # Torso: breastplate with a raised chest plate and a bright keel, belt, and faulds.
    torso = box((8, 7, 6), color=p.metal.mid).at(4, 8, 3)
    chest = box((6, 4, 1), color=p.metal.mid).at(5, 10, 9)
    keel = box((2, 4, 1), color=p.metal.light).at(7, 10, 9)
    belt = box((8, 1, 6), color=p.leather.mid).at(4, 8, 3)
    buckle = box((2, 1, 1), color=p.gold.mid).at(7, 8, 9)
    faulds = box((8, 2, 6), color=p.metal[1]).at(4, 6, 3)
    torso = torso.union(chest, keel, belt, faulds).highlight_edges().darken_crevices()
    torso = torso | buckle

    # Head: an oversized, rounded great helm with a recessed visor slit and a red plume.
    helm = box((8, 7, 8), color=p.metal.mid).at(4, 15, 2).round(1)
    # The cutter needs a color: subtracting an uncolored shape from a grid-stage (rounded)
    # shape fails with "shape has voxels with no color".
    helm = helm - box((6, 1, 1), color=p.eye.dark).at(5, 18, 9)
    helm = helm.paint(p.eye.dark, where=region((5, 18, 8), (11, 19, 9)))
    # A nose ridge standing proud of the faceplate below the slit.
    helm = helm.union(box((2, 3, 1), color=p.metal.mid).at(7, 15, 10))
    helm = helm.highlight_edges().darken_crevices()
    # Plume: a crest along the top that spills down the back of the helm.
    crest = ellipsoid((1.0, 1.6, 3.8), color=p.red.mid).at(MID, 22.3, 5.0) & box((16, 24, 12))
    tail = box((2, 5, 1), color=p.red.mid).at(7, 17, 1)
    plume = (crest | tail).gradient("y", [p.red.dark, p.red[1], p.red.mid, p.red.light])
    head = helm | plume

    # Legs: plate greaves with knee cops and wide sabatons.
    leg = box((3, 6, 4), color=p.metal.mid).at(4, 0, 4)
    leg = leg.union(
        box((3, 2, 1), color=p.metal.light).at(4, 3, 8),
        box((3, 2, 5), color=p.metal[1]).at(4, 0, 4),
    ).highlight_edges()

    # Arms from boxes (clean edge highlights): pauldron, mail upper arm, a forearm reaching
    # forward, and an oversized gauntlet.
    arm = box((3, 3, 5), color=p.metal.mid).at(1, 13, 3)
    arm = arm.union(
        box((2, 4, 2), color=p.metal[1]).at(2, 10, 5),
        box((2, 1, 2), color=p.metal[1]).at(2, 9, 6),
        box((3, 3, 3), color=p.metal.mid).at(0, 7, 7),
    ).highlight_edges()

    # Sword, held upright in the right hand.
    blade = box((3, 10, 1), color=p.metal.light).at(0, 11, 8)
    fuller = box((1, 9, 1), color=p.metal.mid).at(1, 11, 8)
    tip = cone(r=1.5, h=3, color=p.metal.light).at(1.5, 21, 8.5) & box((3, 3, 1)).at(0, 21, 8)
    guard = box((3, 1, 5), color=p.gold[1]).at(0, 10, 6)
    grip = box((1, 4, 1), color=p.leather.dark).at(1, 6, 8)
    pommel = box((1, 1, 1), color=p.gold.mid).at(1, 5, 8)
    sword = blade | fuller | tip | guard | grip | pommel

    # Heater shield on the left arm: gold rim, blue field, white cross, wooden back.
    outline = [(0, 3.5), (0, 9), (7, 9), (7, 3.5), (3.5, 0)]
    inner = [(1, 3.9), (1, 8), (6, 8), (6, 3.9), (3.5, 1.4)]
    shield = extrude(outline, depth=2, color=p.gold[1]).at(9, 4, 10)
    shield = shield.paint(p.blue.mid, where=inside(extrude(inner, depth=2).at(9, 4, 10)))
    cross = region((12, 5, 11), (13, 12, 12)) | region((10, 9, 11), (15, 10, 12))
    shield = shield.paint(p.cloth.white, where=cross).paint_faces("-z", p.wood.mid)

    return Model(
        parts=[
            Part("torso", torso, pivot=(MID, 8, 6), symmetric="x"),
            Part("head", head, parent="torso", pivot=(MID, 15, 6), symmetric="x"),
            Part("leg_r", leg, parent="torso", pivot=(5.5, 6, 6)).mirrored("leg_l", about_x=MID),
            Part("arm_r", arm, parent="torso", pivot=(2.5, 14, 6)).mirrored("arm_l", about_x=MID),
            Part("sword", sword, parent="arm_r", pivot=(1.5, 8, 8.5)),
            Part("shield", shield, parent="arm_l", pivot=(12.5, 8, 10)),
        ],
        sockets={"hand_r": (1.5, 8, 8.5), "hand_l": (14.5, 8, 8.5)},
    )
