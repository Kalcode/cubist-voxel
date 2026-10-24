from cubist import *


@asset(name="forge_smoke", size=(3, 3, 3), tags=["particle", "smoke"])
def forge_smoke(p):
    # A sooty gray ball, lit on top: the emitter grows it as it rises and fades it out.
    puff = sphere(1.5, color=p.stone.mid).at(1.5, 1.5, 1.5)
    return puff.paint(p.stone.light, where=faces("top")).paint(p.stone[1], where=faces("bottom"))
