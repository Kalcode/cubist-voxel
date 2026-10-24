from cubist import *

# Interior runs x 2..13; books stand 4 deep (z 1..4) behind the frame's front edge.
SHELF_Y = (2, 8, 14)
# Per shelf, left to right: (width, height, ramp) for a book, ("gap", width), or
# ("lean", width, ramp) for a book tipped against its right-hand neighbour, or
# ("stack", width, ramp, ramp) for two books lying flat.
SHELVES = [
    [
        (2, 5, "red"),
        (1, 4, "blue"),
        (2, 5, "bone"),
        (1, 4, "leaf"),
        (1, 5, "purple"),
        ("gap", 2),
        (2, 4, "blue"),
        (1, 5, "red"),
    ],
    [
        (1, 4, "blue"),
        (2, 5, "bone"),
        (1, 4, "gold"),
        (2, 5, "red"),
        ("lean", 3, "leaf"),
        (1, 5, "purple"),
        (2, 4, "blue"),
    ],
    [
        (2, 4, "blue"),
        (1, 3, "red"),
        (1, 4, "purple"),
        (2, 3, "bone"),
        ("stack", 3, "red", "blue"),
        (1, 4, "leaf"),
        (2, 4, "red"),
    ],
]


@asset(name="bookshelf", size=(16, 20, 6), tags=["prop", "wood", "furniture"], grounded=True)
def bookshelf(p):
    back = box((16, 20, 1), color=p.wood.dark)
    sides = box((2, 20, 6), color=p.wood.mid).mirror("x", about=8)
    top = box((16, 2, 6), color=p.wood.mid).at(0, 18, 0)
    plinth = box((16, 2, 6), color=p.wood.mid)
    boards = [box((12, 1, 6), color=p.wood.mid).at(2, y - 1, 0) for y in SHELF_Y[1:]]
    frame = (sides | top | plinth).union(*boards)
    # Inset the sides so the top and plinth overhang by one, and cut a kick space between feet.
    inset = box((1, 16, 6)).at(0, 2, 0).mirror("x", about=8)
    kick = box((10, 1, 6)).at(3, 0, 0)
    frame, back = frame - inset - kick, back - inset - kick
    frame = frame.dither(p.wood, seed=3, amount=0.15).highlight_edges()

    books = []
    for y, row in zip(SHELF_Y, SHELVES):
        x = 2
        for item in row:
            if item[0] == "gap":
                x += item[1]
                continue
            if item[0] == "stack":
                w = item[1]
                for i, name in enumerate(item[2:]):
                    ramp = getattr(p, name)
                    flat = box((w - i, 1, 4 - i), color=ramp.mid).at(x, y + i, 1)
                    books.append(flat.paint(ramp.light, where=faces("top")))
                x += w
                continue
            if item[0] == "lean":
                w, ramp = item[1], getattr(p, item[2])
                book = box((1, 4.6, 4), color=ramp.mid).at(x + 1, y, 1)
                books.append(book.rotate(-22, axis="z"))
                x += w
                continue
            w, h, ramp = item[0], item[1], getattr(p, item[2])
            book = box((w, h, 4), color=ramp.mid).at(x, y, 1)
            # A lighter band on the spine near the top.
            book = book.paint(ramp.light, where=band("y", y + h - 2, 1) & faces("front"))
            books.append(book)
            x += w
    shelf = back.union(*books).darken_crevices()
    return shelf | frame
