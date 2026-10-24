from cubist import *


@asset(name="powder_keg_smoke", size=(4, 4, 4), tags=["particle", "smoke"])
def powder_keg_smoke(p):
    # Black-powder smoke: a pale gray ball, shaded underneath. The emitter grows it and fades it.
    # At 4 wide a sphere loses only its edges and reads round; at 3 it becomes a plus.
    puff = sphere(2, color=p.stone.light).at(2, 2, 2)
    return puff.paint(p.stone.mid, where=faces("bottom"))
