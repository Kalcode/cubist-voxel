import math

from cubist import *


@asset(name="mushroom", size=(14, 12, 14), tags=["prop", "nature"])
def mushroom(p):
    stem = cone(r=3, h=8, r_top=2, color=p.stem.light).at(7, 0, 7)
    dome = ellipsoid((6.5, 4.5, 6.5), color=p.cap.mid).at(7, 7, 7)
    cap = dome - box((16, 5, 16)).at(-1, 2, -1)
    gills = cylinder(r=5.5, h=1, color=p.stem.dark).at(7, 7, 7)
    spots = []
    for theta, phi in [(0, 50), (90, 45), (180, 55), (270, 40), (45, 15), (210, 20), (135, 70)]:
        t, f = math.radians(theta), math.radians(phi)
        x = 7 + 6.5 * math.sin(f) * math.cos(t)
        y = 7 + 4.5 * math.cos(f)
        z = 7 + 6.5 * math.sin(f) * math.sin(t)
        spots.append(sphere(1.4, color=p.white).at(x, y, z))
    # Clip spots to the cap so they read as paint, not bumps.
    spots = spots[0].union(*spots[1:]) & cap
    return stem | cap | gills | spots
