from cubist import *

C = 1.5  # odd width, centered on a voxel center so the tip has a middle column


@asset(name="wand", size=(3, 16, 3), tags=["prop", "magic", "wood"], symmetric="x")
def wand(p):
    shaft = cylinder(r=0.8, h=12, color=p.wood.mid).at(C, 0, C).taper(top=0.7)
    grip = cylinder(r=1.2, h=4, color=p.cloth.brown).at(C, 0, C)
    collar = cylinder(r=1.2, h=1, color=p.metal.light).at(C, 11, C)
    gem = sphere(1.3, color=p.potion.mid).at(C, 13.5, C) | cone(
        r=1.0, h=2, color=p.potion.light
    ).at(C, 14, C)
    gem = gem.gradient("y", p.potion)
    return Model(
        parts=[Part("wand", grip | shaft | collar | gem, pivot=(C, 2, C))],
        sockets={"tip": socket((C, 16, C), part="wand", forward="+y", up="+z")},
        clips=[
            # Wind back, flick forward past the target, settle: a flourish about the grip.
            clip(
                "cast",
                0.8,
                [
                    turn("wand", "x", [(0, 0), (0.25, -30), (0.45, 45), (0.8, 0)]),
                    turn("wand", "z", [(0, 0), (0.25, 10), (0.45, -8), (0.8, 0)]),
                ],
                events=[(0.45, "cast")],
            )
        ],
    )
