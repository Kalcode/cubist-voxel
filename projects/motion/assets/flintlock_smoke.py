from cubist import *


@asset(name="flintlock_smoke", size=(3, 3, 3), tags=["particle", "smoke"])
def flintlock_smoke(p):
    # A pale gunsmoke ball, whiter on top: the emitter grows it as it drifts and fades it out.
    puff = sphere(1.5, color=p.stone.light).at(1.5, 1.5, 1.5)
    return puff.paint(p.stone.mid, where=faces("bottom")).paint(p.cloth.white, where=faces("top"))
