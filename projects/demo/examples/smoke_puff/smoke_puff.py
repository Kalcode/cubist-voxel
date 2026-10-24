from cubist import *


@asset(name="smoke_puff", size=(3, 3, 3), tags=["particle", "smoke"])
def smoke_puff(p):
    # A soft gray ball: emitters scale it up as it rises and shrink it away as it fades.
    puff = sphere(1.5, color=p.stone.light).at(1.5, 1.5, 1.5)
    return puff.paint(p.stone.mid, where=faces("bottom"))
