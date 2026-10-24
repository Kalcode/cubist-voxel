from cubist import *


@asset(name="shield", size=(14, 14, 4), tags=["prop", "wood", "metal", "weapon"], symmetric="x")
def shield(p):
    # Round plank face, two voxels thick, facing +z, painted red-wood-red-wood-red.
    face = cylinder(r=6.6, h=2, axis="z", color=p.wood.mid).at(7, 7, 0)
    stripes = band("x", 2, 2) | band("x", 6, 2) | band("x", 10, 2)
    face = face.paint_faces("+z", p.red.mid, where=stripes)
    face = face.dither(p.wood, seed=3, amount=0.35).dither(p.red, seed=4, amount=0.2)

    # Iron rim stands one voxel proud of the planks, with rivets on the diagonals.
    rim = cylinder(r=7, h=3, axis="z", color=p.metal.mid).at(7, 7, 0)
    rim = rim - cylinder(r=6, h=3, axis="z").at(7, 7, 0)
    rivets = region((2, 2, 2), (3, 3, 3)) | region((2, 11, 2), (3, 12, 3))
    rim = rim.paint(p.metal.light, where=rivets)

    # Domed boss on a flat flange.
    flange = cylinder(r=2.6, h=1, axis="z", color=p.metal[1]).at(7, 7, 2)
    boss = sphere(2.4, color=p.metal.mid).at(7, 7, 1.5) & box((14, 14, 2)).at(0, 0, 2)
    boss = boss.highlight_edges()

    # Build the left half and mirror it so the dithered wood is exactly symmetric.
    whole = face | rim | flange | boss
    return (whole & box((7, 14, 4), color=p.metal.mid)).mirror("x", about=7)
