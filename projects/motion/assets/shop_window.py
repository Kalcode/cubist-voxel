from cubist import *


@asset(name="shop_window", size=(18, 14, 4), tags=["building", "wood", "glass", "window"], grounded=True)
def shop_window(p):
    # A sill on an apron and a stepped cornice stand proud of the frame; the panes sit
    # recessed in two tracks.
    sill = box((18, 1, 4), color=p.wood.mid).at(0, 1, 0) | box((14, 1, 3), color=p.wood[1]).at(2, 0, 0)
    cornice = box((18, 1, 4), color=p.wood.mid).at(0, 13, 0) | box((16, 1, 3), color=p.wood[1]).at(1, 12, 0)
    trim = (sill | cornice).highlight_edges(facing=("top", "left"))
    frame = box((16, 10, 3), color=p.wood[1]).at(1, 2, 0) - box((14, 9, 3)).at(2, 3, 0)
    frame = (trim | frame).darken_crevices()

    def sash(x, z):
        # A wooden sash around dark glass, with a bright streak of shine and a fainter one.
        rails = box((8, 9, 1), color=p.wood.light).at(x, 3, z) - box((6, 7, 1)).at(x + 1, 4, z)
        glass = box((6, 7, 1), color=p.blue[1]).at(x + 1, 4, z)
        shine = line((x + 1, 7, z), (x + 4, 10, z), color=p.blue.light, connect=26)
        glint = line((x + 4, 6, z), (x + 5, 7, z), color=p.blue.mid, connect=26)
        return rails | glass | shine | glint

    return Model(
        parts=[
            Part("frame", frame, pivot=(9, 0, 2)),
            # The right pane is fixed in the front track (z = 2).
            Part("pane_r", sash(8, 2), parent="frame", pivot=(12, 3, 2.5)),
            # The left pane rides in the back track (z = 1) and slides right, behind pane_r.
            # A flush pull on its outer stile shows which pane moves (a knob would hit pane_r).
            Part("pane_l", sash(2, 1).paint(p.metal.mid, where=region((2, 7, 1), (3, 9, 2))),
                 parent="frame", pivot=(6, 3, 1.5), joint=slide("+x", 6)),
        ],
        clips=[
            clip("open", 0.8, {"pane_l": [0, 6]}, events=[(0.0, "slide"), (0.8, "thunk")]),
            # Shoved shut: hits the jamb and rebounds a voxel.
            clip(
                "close",
                0.6,
                {"pane_l": [(0, 6), (0.4, 0), (0.48, 1), (0.6, 0)]},
                events=[(0.0, "slide"), (0.4, "thunk")],
            ),
        ],
    )
