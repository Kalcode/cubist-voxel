from cubist import *


@asset(name="spark", size=(1, 1, 1), tags=["particle", "fire"])
def spark(p):
    # One glowing voxel; fire colors glow in this style.
    return box((1, 1, 1), color=p.fire.light)
