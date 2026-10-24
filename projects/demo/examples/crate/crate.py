from cubist import *


@asset(name="crate", size=(12, 12, 12), tags=["prop", "wood"], grounded=True)
def crate(p):
    panels = box((12, 12, 12), color=p.wood.mid)
    # One-voxel-deep plank seams around the sides.
    for y in (4, 8):
        ring = box((12, 1, 12), color=p.wood.mid).at(0, y, 0) - box((10, 1, 10)).at(1, y, 1)
        panels = panels - ring
    frame = []
    for a in (0, 10):
        for b in (0, 10):
            frame.append(box((12, 2, 2), color=p.wood.dark).at(0, a, b))
            frame.append(box((2, 12, 2), color=p.wood.dark).at(a, 0, b))
            frame.append(box((2, 2, 12), color=p.wood.dark).at(a, b, 0))
    return panels.union(*frame)
