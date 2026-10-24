from cubist import *


@asset(name="cellar_hatch_dust", size=(3, 3, 3), tags=["particle", "smoke"])
def cellar_hatch_dust(p):
    # A pale gray puff kicked up when the hatch slams; emitters grow it and fade it away.
    puff = sphere(1.5, color=p.stone.light).at(1.5, 1.5, 1.5)
    return puff.paint(p.stone.mid, where=faces("bottom"))
