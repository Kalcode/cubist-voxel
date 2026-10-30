"""The walk lint (a walker the size of Kiln's goblin, plus two voxels, flood-fills a building
from its door) and the storey lint (hideable parts are horizontal bands)."""

import math
import time
from dataclasses import dataclass, field

import goblin
import numpy as np
import pytest

from cubist import Model, Part, asset, box
from cubist.core.grid import VoxelGrid
from cubist.core.model import BuiltAsset, BuiltPart, default_pivot
from cubist.core.palette import Palette
from cubist.core.socket import Socket
from cubist.core.style import StyleBible
from cubist.lint import lint
from cubist.lint.walk import check_storeys, check_walk

P = Palette({"stone": ["#4a4540", "#6b655e", "#8f887f"]})
# Kiln's goblin (demos/Diorama/Goblin.cs): HalfWidth 4, Height 22, StepUp 6.
GOBLIN = {"walker_width": 8, "walker_height": 22, "walker_climb": 6}
STYLE = StyleBible(path=None, name="t", voxel_size_m=0.0625, palette=P, scale=dict(GOBLIN))
WALL = 4  # outer walls and floor slabs
STAIR_X = 16  # where each flight starts; it climbs toward +x
STRIPS = (4, 28)  # flights alternate between these 16-wide strips along the back wall


@dataclass
class Site:
    """A building drawn into a numpy grid (x, y, z), turned into a built asset by ``built()``."""

    solid: np.ndarray
    sockets: dict = field(default_factory=dict)
    ground_y: int = 0
    bands: list = field(default_factory=list)  # (part name, lowest y, occlusion), bottom up
    tags: tuple = ("building",)

    def fill(self, lo, hi, value=True):
        self.solid[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = value

    def built(self) -> BuiltAsset:
        data = self.solid.astype(np.uint8) * P.stone.mid.index
        bands = self.bands or [("house", 0, None)]
        parts = []
        for k, (name, y0, occlusion) in enumerate(bands):
            y1 = bands[k + 1][1] if k + 1 < len(bands) else data.shape[1]
            band = np.zeros_like(data)
            band[:, y0:y1] = data[:, y0:y1]
            grid = VoxelGrid(band).cropped()
            parts.append(BuiltPart(name, grid, default_pivot(grid), None, occlusion=occlusion))
        return BuiltAsset(
            "house",
            self.solid.shape,
            self.tags,
            parts,
            P,
            sockets={k: Socket(tuple(map(float, v))) for k, v in self.sockets.items()},
            ground_y=self.ground_y,
        )


def house(
    *,
    width=100,
    depth=80,
    storeys=2,
    storey=48,
    riser=4,
    door=(16, 34),
    cellar=False,
    cellar_stair=True,
    roof=0,
) -> Site:
    """A walled house, its front (+z) wall holding a door, with a straight flight (5-voxel treads,
    16 wide) up to each floor and a stairwell open above it. With a cellar, ground_y is the ground
    floor's top.

    Sockets: ``door`` on the street outside the door, ``floor_N`` near the front of each floor,
    ``cellar``, and ``tour_NN`` at the foot and head of each flight."""
    g = 48 if cellar else WALL
    top = g + storeys * storey
    levels = ([WALL] if cellar else []) + [g + k * storey for k in range(storeys)]
    site = Site(np.zeros((width, top + roof, depth), bool), ground_y=g)
    site.fill((0, 0, 0), (width, WALL, depth))  # the lowest floor
    for y in [*levels[1:], top]:
        site.fill((0, y - WALL, 0), (width, y, depth))  # the floors above it, and the ceiling
    site.fill((0, 0, 0), (WALL, top, depth))
    site.fill((width - WALL, 0, 0), (width, top, depth))
    site.fill((0, 0, 0), (width, top, WALL))
    site.fill((0, 0, depth - WALL), (width, top, depth))
    if roof:
        site.fill((8, top, 8), (width - 8, top + roof, depth - 8))
    wide, tall = door
    x = width // 2 - wide // 2
    site.fill((x, g, depth - WALL), (x + wide, g + tall, depth), False)

    tour = []
    for k, (a, b) in enumerate(zip(levels, levels[1:], strict=False)):
        if cellar and k == 0 and not cellar_stair:
            continue
        z = STRIPS[k % 2]
        n = math.ceil((b - a) / riser)
        end = STAIR_X + 5 * n
        site.fill((STAIR_X, b - WALL, z), (end, b, z + 16), False)  # the stairwell
        for i in range(1, n + 1):
            site.fill((STAIR_X + 5 * (i - 1), a, z), (end, min(a + riser * i, b), z + 16))
        tour += [(STAIR_X - 6, a, z + 8), (end + 6, b, z + 8)]

    front = depth - WALL - 10
    site.sockets["door"] = (width / 2, g, depth)
    for k, y in enumerate(levels[1:] if cellar else levels):
        site.sockets[f"floor_{k}"] = (width / 2, y, front)
    if cellar:
        site.sockets["cellar"] = (width / 2, WALL, front)
    for k, pos in enumerate(tour, start=1):
        site.sockets[f"tour_{k:02d}"] = pos
    names = (["cellar"] if cellar else []) + ["ground"] + [f"floor_{k}" for k in range(1, storeys)]
    lows = [0] + [y - WALL for y in levels[1:]]
    site.bands = [
        (n, y, None if n == "cellar" else "upper") for n, y in zip(names, lows, strict=True)
    ]
    site.bands.append(("roof", top - WALL, "roof"))
    return site


def walk_errors(a, style=STYLE):
    found = check_walk(a, style)
    assert all(f.check == "walk" and f.severity == "error" for f in found)
    return found


def walk_findings(a, style=STYLE):
    return {(f.severity, f.details["socket"]) for f in check_walk(a, style)}


def widened(site, z):
    """The site with z more voxels of air in front of it (+z), for street-side pieces."""
    site.solid = np.pad(site.solid, ((0, 0), (0, 0), (0, z)))
    return site


def unreached(a, style=STYLE):
    return {f.details["socket"] for f in walk_errors(a, style)}


# --- the walk ------------------------------------------------------------------------------------


def test_a_house_with_a_door_and_a_stair_of_4_voxel_risers_reaches_every_socket():
    a = house().built()
    assert {"door", "floor_0", "floor_1", "tour_01", "tour_02"} <= set(a.sockets)
    assert walk_errors(a) == []
    report = lint(a, STYLE)
    assert not [f for f in report.findings if f.check in ("walk", "storeys")]


def test_lint_runs_the_walk_and_storey_checks():
    a = house(riser=7).built()
    shelf = np.zeros((100, 30, 80), np.uint8)
    shelf[4:8, 20:22, 50:60] = P.stone.mid.index  # an upper part in the middle of the ground floor
    a.parts.append(
        BuiltPart("shelf", VoxelGrid(shelf).cropped(), (6, 20, 55), None, occlusion="upper")
    )
    report = lint(a, STYLE)
    errors = {(f.check, f.details.get("socket") or f.details.get("part")) for f in report.findings}
    assert ("walk", "floor_1") in errors and ("storeys", "ground") in errors
    assert not report.ok


def test_risers_of_6_are_climbed_and_7_are_not():
    assert walk_errors(house(riser=6).built()) == []
    found = walk_errors(house(riser=7).built())
    assert {f.details["socket"] for f in found} == {"floor_1", "tour_02"}  # the flight's head too
    [f] = [f for f in found if f.details["socket"] == "floor_1"]
    assert "'floor_1'" in f.message and "8 × 22" in f.message and "climbs 6" in f.message
    assert "stair or doorway" in f.message
    # The nearest point it reaches is on the floor below, a storey under the socket.
    assert f.details["nearest"][1] == WALL and f.details["distance"] == pytest.approx(48, abs=1)
    assert f.details["walker"] == [8, 22, 6] and f.details["box"] == [10, 24, 10]
    assert f"{f.details['distance']:.0f} voxels away" in f.message


def test_a_doorway_of_10_by_24_passes_and_9_wide_or_23_tall_does_not():
    # The walker is 8 × 22; the lint checks it with two voxels to spare.
    assert walk_errors(house(storeys=1, door=(10, 24)).built()) == []
    assert unreached(house(storeys=1, door=(9, 24)).built()) == {"floor_0"}
    assert unreached(house(storeys=1, door=(10, 23)).built()) == {"floor_0"}


def test_a_step_up_needs_the_box_raised_by_the_climb_to_be_clear():
    # Kiln's goblin steps onto a ledge only when its box, raised by its climb, is clear, so a stair
    # needs climb + height (+ margin) = 30 voxels of air above the tread it steps from.
    def room(ceiling):
        # Air from the floor (y = 4) up to y = ceiling.
        site = house(storeys=1, storey=ceiling, door=(16, 24))
        site.fill((WALL, WALL, WALL), (100 - WALL, WALL + 4, 40))  # a 4-voxel platform at the back
        site.sockets["floor_1"] = (50, WALL + 4, 20)
        return site.built()

    assert unreached(room(34)) == set()  # 30 voxels of air above the floor
    assert unreached(room(33)) == {"floor_1"}  # it could stand on the platform, but can't step up


def test_support_handed_across_the_path_needs_one_footprint_on_both_sides():
    # A corridor exactly the walker's width, its floor gone but for a rail under each edge: the
    # left one ends and the right one starts further on. Across its path the goblin covers 9 of the
    # walker's 10 columns, so it stands toward one rail at a time, and the walker moves over only
    # where one footprint holds both: rail ends 7 apart (along the path) it crosses, 8 apart it
    # doesn't. Kiln's goblin can cross 8 with a diagonal step that crosses both whole-voxel lines
    # in one frame (MoveAxis x, then z, then the ground check); the walker, a move at a time, is
    # the stricter there, the safe way round.
    def rails(apart):
        site = house(storeys=1)
        site.fill((WALL, WALL, 24), (45, 48, 56))  # the corridor's walls, x 45 to 55 open
        site.fill((55, WALL, 24), (100 - WALL, 48, 56))
        site.fill((45, 0, 24), (55, WALL, 56), False)  # its floor
        site.fill((45, 0, 40), (46, WALL, 56))  # the left rail, from the front
        site.fill((54, 0, 24), (55, WALL, 40 - apart))  # the right rail, to the back
        site.sockets["floor_0"] = (50, WALL, 12)  # behind the corridor
        return site.built()

    assert unreached(rails(7)) == set()
    assert unreached(rails(8)) == {"floor_0"}


def test_a_gap_of_8_is_bridged_and_9_is_not():
    # Between two whole voxels the goblin's footprint is 9 wide, so it stays up over a gap of 8 and
    # falls into one of 9: the walker needs something under it halfway through each step, not just
    # under its box.
    def slot(gap):
        site = house(storeys=1)
        site.fill(
            (WALL, 0, 40), (100 - WALL, WALL, 40 + gap), False
        )  # across the floor, through to the void
        site.sockets["floor_0"] = (50, WALL, 20)  # beyond it
        return site.built()

    assert unreached(slot(8)) == set()
    assert unreached(slot(9)) == {"floor_0"}


def test_a_sealed_cellar_is_out_of_reach():
    assert walk_errors(house(cellar=True).built()) == []
    [f] = walk_errors(house(cellar=True, cellar_stair=False).built())
    assert f.details["socket"] == "cellar" and "'cellar'" in f.message
    assert f.details["way"] == "in"


def test_the_walker_must_get_back_out():
    # Going down is a fall, so a flight walked down isn't climbed: the walk back to the street
    # checks it. A cellar stair of 7-voxel risers, and a hatch with no stair, drop the walker
    # into a cellar it can't leave.
    def way_out(a):
        return {(f.details["socket"], f.details["way"]) for f in walk_errors(a)}

    assert way_out(house(storeys=1, cellar=True).built()) == set()  # 4-voxel risers
    found = walk_errors(house(storeys=1, cellar=True, riser=7).built())
    assert {(f.details["socket"], f.details["way"]) for f in found} == {
        ("cellar", "out"),
        ("tour_01", "out"),  # the foot of the flight
    }
    [f] = [f for f in found if f.details["socket"] == "cellar"]
    assert "can't get back from 'cellar'" in f.message and "to the street" in f.message
    assert "stair or doorway" in f.message
    assert f.details["nearest"][1] == 48 and f.details["distance"] == pytest.approx(44, abs=1)
    hatch = house(storeys=1, cellar=True, cellar_stair=False)
    hatch.fill((40, 44, 40), (60, 48, 60), False)  # 20 × 20 through the ground floor
    assert way_out(hatch.built()) == {("cellar", "out")}


def test_a_cellared_footprint_has_no_ground():
    # Kiln carves the x-z bounds of the voxels below ground_y (here the walls and floor, from
    # y = 0 under a ground line at 4), so a building with no floor of its own over that drops the
    # walker.
    site = house(storeys=1)
    site.fill((WALL, 0, WALL), (100 - WALL, WALL, 80 - WALL), False)
    assert unreached(site.built()) == {"floor_0"}


def test_with_nothing_below_the_ground_line_the_ground_runs_under_the_footprint():
    # Nothing to carve: a shell standing on the ground (ground_y 0) with no floor of its own is
    # walked on the ground inside it.
    site = Site(np.zeros((60, 40, 60), bool), ground_y=0)
    site.fill((0, 0, 0), (WALL, 40, 60))
    site.fill((60 - WALL, 0, 0), (60, 40, 60))
    site.fill((0, 0, 0), (60, 40, WALL))
    site.fill((0, 0, 60 - WALL), (60, 40, 60))
    site.fill((22, 0, 60 - WALL), (38, 34, 60), False)  # the doorway
    site.sockets = {"door": (30, 0, 60), "floor_0": (30, 0, 30)}
    assert walk_errors(site.built()) == []


def test_hidden_parts_are_left_out():
    a = house(storeys=1).built()
    plug = np.zeros((100, 60, 80), np.uint8)
    plug[42:58, 4:38, 76:80] = P.stone.mid.index  # fills the doorway
    shut = BuiltPart("shut", VoxelGrid(plug).cropped(), (50, 4, 78), None, hidden=True)
    a.parts.append(shut)
    assert walk_errors(a) == []
    shut.hidden = False
    assert unreached(a) == {"floor_0"}


def test_the_walk_starts_on_the_street_not_in_the_doorway():
    # A wall across the front, 4 voxels out and 46 tall: 'door' sits on the house's face, but the
    # walker comes from the street (outside the asset's x-z bounds), so the wall shuts it out.
    site = widened(house(storeys=1), 8)
    site.fill((0, 0, 84), (100, 50, 88))
    assert unreached(site.built()) == {"door", "floor_0"}
    site.fill((42, WALL, 84), (58, 40, 88), False)  # a gate in it
    assert unreached(site.built()) == set()


def test_the_street_runs_under_eaves_and_jetties():
    # Kiln carves only the cellar's footprint (the x-z bounds of the voxels below ground_y), so the
    # street under a front overhanging it by 1 m is still there.
    site = widened(house(storeys=1, cellar=True), 16)
    site.fill((0, 84, 80), (100, 100, 96))  # a jetty over the street, 1 m deep
    assert site.sockets["door"] == (50, 48, 80)  # on the house's face, under the jetty
    assert unreached(site.built()) == set()


def test_ground_above_the_asset_does_not_break_the_walk():
    # @asset keeps ground_y below the asset's top, but a hand-edited build.json may not.
    site = house(storeys=1)
    site.ground_y = 70  # the house (52 tall) is buried
    site.sockets["door"] = (50, 70, 80)
    assert unreached(site.built()) == {"floor_0"}


def test_a_missing_door_or_floor_0_warns_and_an_unreached_socket_errs():
    # Ruling P-6: buildings older than the kit (Kiln's cottage, smithy, stall and crypt) aren't
    # enterable and must still export, so a missing door or floor_0 is a warning.
    site = house(storeys=1)
    del site.sockets["floor_0"]
    [f] = check_walk(site.built(), STYLE)
    assert (f.severity, f.details["socket"]) == ("warning", "floor_0")
    assert "no 'floor_0' socket" in f.message and "can't check" in f.message
    site.sockets["floor_2"] = (50, 52, 40)  # on the roof: still walked, and out of reach
    assert walk_findings(site.built()) == {("warning", "floor_0"), ("error", "floor_2")}
    del site.sockets["door"]
    found = check_walk(site.built(), STYLE)
    assert {(f.severity, f.details["socket"]) for f in found} == {
        ("warning", "door"),
        ("warning", "floor_0"),
    }
    assert any("no 'door' socket" in f.message for f in found)
    errors = {f.check for f in lint(site.built(), STYLE).findings if f.severity == "error"}
    assert "walk" not in errors


def test_a_door_socket_with_no_street_near_it_is_an_error():
    site = house()
    site.sockets["door"] = (50, 30, 80)  # in the air, above the doorway
    [f] = walk_errors(site.built())
    assert f.details["socket"] == "door" and "within 16 voxels of the street" in f.message
    assert "on the front face, or under the overhang" in f.message
    assert "sockets stay inside the model's bounds" in f.message
    site.sockets["door"] = (50, WALL, 40)  # inside the house, 40 from the street
    [f] = walk_errors(site.built())
    assert f.details["socket"] == "door" and "within 16 voxels of the street" in f.message


def test_only_the_walk_sockets_are_walked():
    site = house(storeys=1)
    site.sockets["chimney"] = (50, 52, 40)  # on the roof
    site.sockets["floor_2"] = (50, 52, 40)
    assert unreached(site.built()) == {"floor_2"}


def test_buildings_and_walkable_assets_are_walked_when_the_style_has_a_walker():
    site = house(storeys=1, door=(9, 24))
    assert unreached(site.built()) == {"floor_0"}
    site.tags = ("walkable",)  # a bridge: door and floor_0 at its deck ends
    assert unreached(site.built()) == {"floor_0"}
    del site.sockets["door"]
    [f] = check_walk(site.built(), STYLE)
    assert f.severity == "warning" and "one end of a bridge's deck" in f.message
    site.sockets["door"] = (50, WALL, 80)
    site.tags = ("prop",)
    assert walk_errors(site.built()) == []
    site.tags = ("building",)
    plain = StyleBible(
        path=None, name="t", voxel_size_m=0.0625, palette=P, scale={"door_height": 34}
    )
    assert walk_errors(site.built(), plain) == []


def test_a_townhouse_sized_asset_walks_in_under_5_seconds():
    # 160 × 368 × 192: a cellar, five storeys and a roof, every floor reached by its stair.
    site = house(width=160, depth=192, storeys=5, cellar=True, roof=80)
    a = site.built()
    assert a.size == (160, 368, 192) and len(a.sockets) == 1 + 5 + 1 + 2 * 5
    start = time.perf_counter()
    found = check_walk(a, STYLE) + check_storeys(a)
    elapsed = time.perf_counter() - start
    assert found == []
    assert elapsed < 5.0, f"the walk and storey lints took {elapsed:.2f} s"
    start = time.perf_counter()
    report = lint(a, STYLE)  # all of it, as an export pays it
    elapsed = time.perf_counter() - start
    assert report.ok and not [f for f in report.findings if f.check in ("walk", "storeys")]
    assert elapsed < 5.0, f"the whole lint took {elapsed:.2f} s"


def strip(kind, n):
    """A walled strip 120 long (x) and 32 wide on a floor 4 thick, walked from 'door' at its -x
    end to 'floor_0' at its +x end, with one obstacle at x = 50: a riser of n, a gap of n across
    it, a doorway n wide (and 30 tall) or n tall (and 16 wide) in a wall, or a step of 4 under a
    ceiling n above the floor. Returns the site and the y the far end stands at."""
    site = Site(np.zeros((120, 84, 40), bool), ground_y=WALL)
    site.fill((0, 0, 0), (120, WALL, 40))
    for lo, hi in [((0, 0, 0), (120, 84, WALL)), ((0, 0, 36), (120, 84, 40))]:
        site.fill(lo, hi)  # walls along it, and across its far end: the street can't go round
    site.fill((116, 0, 0), (120, 84, 40))
    far = WALL
    if kind == "riser":
        site.fill((50, WALL, 0), (120, WALL + n, 40))
        far = WALL + n
    elif kind == "gap":
        site.fill((50, 0, 0), (50 + n, WALL, 40), False)
    elif kind in ("wide", "tall"):
        wide, tall = (n, 30) if kind == "wide" else (16, n)
        site.fill((50, WALL, 0), (54, 84, 40))
        site.fill((50, WALL, 20 - wide // 2), (54, WALL + tall, 20 - wide // 2 + wide), False)
    elif kind == "ceiling":
        site.fill((50, WALL, 0), (120, WALL + 4, 40))
        site.fill((0, WALL + n, 0), (120, WALL + n + 4, 40))
        far = WALL + 4
    site.sockets = {"door": (0, WALL, 20), "floor_0": (110, far, 20)}
    return site, far


def goblin_passes(site, far) -> bool:
    """Kiln's goblin (tests/goblin.py) walking +x down the strip, from a few places across it
    (it fits an 8-wide doorway only centred on a whole voxel), gets past x = 60 at the far end's
    height without falling on the way."""
    for z in (20.0, 20.25, 20.5, 19.5):
        (x, y, _), lowest = goblin.walk(site.solid, (10.3, WALL, z), (1, 0), 4)
        if x > 60 and y == far and lowest >= WALL:
            return True
    return False


def test_the_walker_stays_pinned_to_kiln_s_goblin():
    # The same strips through the goblin's port and through the walk lint. Risers and gaps: the
    # same limit. Doorways and headroom: the walker needs its two voxels to spare.
    for kind, sizes, goblin_limit, walker_limit in [
        ("riser", range(4, 9), 6, 6),  # the most it climbs
        ("gap", range(6, 11), 8, 8),  # the widest it stays up over
        ("wide", range(7, 12), 8, 10),  # the narrowest doorway
        ("tall", range(20, 26), 22, 24),  # the lowest doorway
        ("ceiling", range(26, 33), 28, 30),  # the least air over a step of 4
    ]:
        most = kind in ("riser", "gap")
        for n in sizes:
            site, far = strip(kind, n)
            want_goblin = n <= goblin_limit if most else n >= goblin_limit
            want_walker = n <= walker_limit if most else n >= walker_limit
            assert goblin_passes(site, far) == want_goblin, (kind, n, "goblin")
            assert (not walk_errors(site.built())) == want_walker, (kind, n, "walker")


# --- the storey bands ----------------------------------------------------------------------------


def storeys(*bands):
    """Parts from (name, lowest y, highest y + 1, occlusion, parent)."""

    def fn(p):
        parts = [
            Part(
                name,
                box((8, y1 - y0, 8), color=p.stone.mid).at(0, y0, 0),
                occlusion=occ,
                parent=parent,
            )
            for name, y0, y1, occ, parent in bands
        ]
        return Model(parts=parts)

    return asset(name="h", tags=["building"])(fn).build(P)


def storey_errors(a):
    found = check_storeys(a)
    assert all(f.check == "storeys" and f.severity == "error" for f in found)
    return found


def test_upper_parts_overlapping_by_one_row_are_an_error():
    [f] = storey_errors(
        storeys(
            ("ground", 0, 49, "upper", None),
            ("floor_1", 48, 96, "upper", None),
            ("roof", 96, 120, "roof", None),
        )
    )
    assert f.details == {"part": "ground", "next": "floor_1", "rows": 1, "next_y": 48}
    assert "'ground'" in f.message and "'floor_1'" in f.message and "y = 48" in f.message


def test_abutting_storeys_pass():
    a = storeys(
        ("ground", 0, 48, "upper", None),
        ("floor_1", 48, 96, "upper", None),
        ("roof", 96, 120, "roof", None),
    )
    assert storey_errors(a) == []


def test_the_topmost_part_may_reach_down_past_the_eaves_of_the_storey_under_it():
    a = storeys(
        ("ground", 0, 48, "upper", None),
        ("floor_1", 48, 96, "upper", None),
        ("roof", 88, 120, "roof", None),
    )
    assert storey_errors(a) == []


def test_untagged_and_hidden_parts_are_not_bands_and_inherited_tags_are():
    # The cellar has no class, so it may reach into the ground floor.
    base = [
        ("cellar", 0, 50, None, None),
        ("ground", 44, 96, "upper", None),
        ("roof", 96, 120, "roof", None),
    ]
    assert storey_errors(storeys(*base)) == []
    # A shutter hanging from the ground floor shares its tag, so to an engine it is a band of its
    # own.
    [f] = storey_errors(storeys(*base, ("shutter", 60, 70, None, "ground")))
    assert f.details["part"] == "ground" and f.details["next"] == "shutter"
    assert "'shutter' hangs from 'ground'" in f.message
    a = storeys(*base, ("shutter", 60, 70, None, "ground"))
    a.part("shutter").hidden = True
    assert storey_errors(a) == []


def test_parts_starting_on_the_same_line_share_a_band():
    a = storeys(
        ("ground", 0, 48, "upper", None),
        ("east", 48, 96, "upper", None),
        ("west", 48, 90, "upper", None),
        ("roof", 96, 120, "roof", None),
    )
    assert storey_errors(a) == []
