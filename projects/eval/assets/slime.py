from cubist import *

C = 8  # even width: the body is centered between voxel columns 7 and 8


@asset(name="slime", size=(16, 12, 16), tags=["creature", "slime"], grounded=True)
def slime(p):
    # A dome: an ellipsoid whose widest point sits low, cut flat at the ground, with a slight
    # lip where it slumps onto the floor. It leaves the corners free for the puddle.
    body = ellipsoid((6.8, 7.6, 6.8), color=p.slime.mid).at(C, 2.4, C)
    body = body.smooth_union(cylinder(r=6.9, h=1, color=p.slime.mid).at(C, 0, C), k=1.0)
    # Shave the front flat (z < 14) so the face sits on a plane and the eyes read cleanly.
    body = body & box((16, 12, 14))

    # Dark at the bottom to light on top, then a lighter rim at the very base where light
    # shines through the jelly.
    body = body.gradient("y", [p.slime.dark, p.slime[1], p.slime[1], p.slime.mid, p.slime.mid])
    body = body.paint(p.slime[1], where=band("y", 0, 1))

    # Glossy highlights from the top-left: a curved streak on the shoulder, a spot on the crown,
    # and small bubbles seen through the skin.
    streak = ellipsoid((2.0, 2.6, 3.2)).at(3.0, 7.0, 8.5)
    crown = ellipsoid((1.6, 1.0, 1.6)).at(6.0, 9.8, 7.0)
    bubbles = sphere(0.8).at(11.5, 3.0, 13.0) | sphere(0.7).at(2.5, 2.5, 5.0)
    body = body.paint(p.slime.light, where=inside(streak | crown | bubbles) & surface())

    # Two big eyes on the front: every visible voxel in each eye's window, so the steps of the
    # dome read as one patch from any angle. A big glint at each eye's top-left and a small
    # one at its bottom-right, and a smile.
    for x in (4, 9):
        body = body.paint(p.eye.dark, where=region((x, 3, 10), (x + 3, 7, 16)) & surface())
        body = body.paint(p.eye.white, where=region((x, 6, 10), (x + 1, 7, 16)) & surface())
        body = body.paint(p.eye.white, where=region((x + 2, 3, 13), (x + 3, 4, 16)) & surface())
    smile = (
        region((7, 1, 10), (9, 2, 16))
        | region((6, 2, 10), (7, 3, 16))
        | region((9, 2, 10), (10, 3, 16))
    )
    body = body.paint(p.eye.dark, where=smile & surface())

    # A drip running down the front-right flank into a small puddle in the free corner.
    drip = capsule((12.0, 5.0, 12.0), (13.2, 0.8, 13.2), r=0.8, color=p.slime[1])
    puddle = ellipsoid((2.2, 0.8, 2.2), color=p.slime[1]).at(13.5, 0.0, 13.5) & box((16, 1, 16))
    puddle = puddle.paint(p.slime.mid, where=inside(sphere(0.9).at(13.5, 0.5, 13.5)))

    return body | drip | puddle
