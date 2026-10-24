from cubist import *


@asset(name="mage_staff_spark", size=(1, 1, 1), tags=["particle", "magic"])
def mage_staff_spark(p):
    # One glowing voxel, the orb's brightest shade.
    return box((1, 1, 1), color=p.glow.light)
