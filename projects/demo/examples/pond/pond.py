from cubist import *


@asset(name="pond", size=(16, 3, 16), tags=["terrain", "water", "stone"], grounded=True)
def pond(p):
    rim = box((16, 3, 16), color=p.stone.mid) - box((12, 2, 12)).at(2, 1, 2)
    rim = rim.masonry(p.stone, block=(3, 1, 3), seed=2).darken_crevices()
    # Water in bands across all four shades: the style's water cycle steps every voxel one
    # shade along the ramp each tick, so the bands seem to flow along x.
    water = box((12, 1, 12), color=p.water.mid).at(2, 1, 2)
    for i, shade in enumerate(p.water):
        water = water.paint(shade, where=stripes("x", period=4, width=1, offset=i))
    return rim | water
