from cubist import *


@asset(name="window", size=(14, 17, 4), tags=["prop", "wood", "glass", "window"], grounded=True)
def window(p):
    sill = box((14, 1, 4), color=p.stone.light).highlight_edges()
    frame = box((12, 16, 2), color=p.wood.dark).at(1, 1, 0) - box((10, 14, 2)).at(2, 2, 0)
    frame = frame.darken_crevices()

    def sash(y, z):
        pane = box((10, 7, 1), color=p.wood.light).at(2, y, z)
        glass = box((8, 5, 1), color=p.glass.dark).at(3, y + 1, z)
        shine = line((4, y + 4, z), (6, y + 2, z), color=p.glass.light, connect=26)
        return pane | glass | shine

    return Model(
        parts=[
            Part("frame", sill | frame | sash(9, 0), pivot=(7, 0, 2)),
            # The lower sash rides in front of the upper one and slides up past it.
            Part("sash", sash(2, 1), parent="frame", pivot=(7, 2, 1.5), joint=slide("+y", 6)),
        ],
        clips=[
            clip("open", 0.8, {"sash": [0, 6]}),
            clip("close", 0.6, {"sash": [6, 0]}),
        ],
    )
