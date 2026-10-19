import random

import pytest

from vecworld.dream.ideas import parse_places
from vecworld.roam.places import (DRIFT_SPOTS, FALLBACK_PLACES, Journey, Neighbour, assign_bearings,
                                  fallback_neighbours)


def gap(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("n", [1, 2, 3])
def test_bearings_spread_without_came_from(seed, n):
    ns = assign_bearings([f"p{i}" for i in range(n)], None, random.Random(seed))
    assert len(ns) == n and all(0 <= x.bearing < 360 for x in ns)
    for i, a in enumerate(ns):
        for b in ns[i + 1:]:
            assert gap(a.bearing, b.bearing) >= 60


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("n", [1, 2, 3])
def test_came_from_is_behind_arrival(seed, n):
    ns = assign_bearings([f"p{i}" for i in range(n)], ("home", 200.0), random.Random(seed))
    assert len(ns) == n + 1
    home = next(x for x in ns if x.name == "home")
    assert home.bearing == 200.0
    for i, a in enumerate(ns):
        assert 0 <= a.bearing < 360 and len(a.scenery) == 3
        for b in ns[i + 1:]:
            assert gap(a.bearing, b.bearing) >= 60


def test_came_from_bearing_is_normalised():
    ns = assign_bearings(["a"], ("home", 370.0), random.Random(1))
    assert next(x for x in ns if x.name == "home").bearing == 10.0


def make_journey():
    ns = [Neighbour("lake", 90.0, ("reeds", "mud", "boats")), Neighbour("hill", 270.0, ("a", "b", "c"))]
    return Journey("forest", ns)


def test_drift_advances_toward_a_neighbour_then_arrives():
    j = make_journey()
    s = j.step(80.0)
    assert j.drift["lake"] == 1 and s.arrived is None and s.scenery == "reeds"
    assert s.weights[0][0] == "forest" and s.weights[1][0] == "lake"
    assert s.weights[1][1] == pytest.approx(1 / 3) and s.weights[0][1] == pytest.approx(2 / 3)
    s2 = j.step(100.0)
    assert j.drift["lake"] == 2 and s2.weights[1][1] == pytest.approx(2 / 3) and s2.scenery == "mud"
    s3 = j.step(135.0)  # exactly 45 degrees off still counts
    assert s3.arrived == "lake" and j.drift["lake"] == DRIFT_SPOTS


def test_drift_resets_when_turning_away():
    j = make_journey()
    j.step(90.0)
    j.step(90.0)
    s = j.step(0.0)
    assert j.drift["lake"] == 0 and s.arrived is None and s.weights == [("forest", 1.0)] and s.scenery == ""


def test_switching_neighbour_resets_the_other():
    j = make_journey()
    j.step(90.0)
    j.step(270.0)
    assert j.drift == {"lake": 0, "hill": 1}


def test_wraparound_heading():
    j = Journey("a", [Neighbour("n", 350.0, ("x", "y", "z"))])
    j.step(20.0)
    assert j.drift["n"] == 1
    j.step(-10.0)
    assert j.drift["n"] == 2


def test_arrive_puts_origin_behind():
    j = make_journey()
    for _ in range(DRIFT_SPOTS):
        s = j.step(90.0)
    assert s.arrived == "lake"
    j.arrive("lake", ["marsh", "dunes"], random.Random(3))
    assert j.place == "lake" and j.came_from == "forest"
    back = next(x for x in j.neighbours if x.name == "forest")
    assert back.bearing == 270.0
    assert {x.name for x in j.neighbours} == {"forest", "marsh", "dunes"}
    assert all(v == 0 for v in j.drift.values()) and set(j.drift) == {x.name for x in j.neighbours}


def test_fallback_coverage():
    assert len(FALLBACK_PLACES) >= 15
    for key in ("forest", "village", "city", "lake", "desert", "mountain", "coast", "ruins"):
        assert key in FALLBACK_PLACES
    for place, ns in FALLBACK_PLACES.items():
        assert 3 <= len(ns) <= 4 and place not in ns
    generic = fallback_neighbours("a place nobody has heard of")
    assert 3 <= len(generic) <= 4
    assert fallback_neighbours("forest") == FALLBACK_PLACES["forest"]


def test_parse_places_with_chatter_junk_and_long_names():
    raw = ("Here are 3 nearby places:\n"
           "1. Old Mill: a creaking wheel; a mossy weir; a stack of sacks\n"
           "- Reed Marsh: tall green reeds; a rotting jetty; a heron on one leg\n"
           "Nearby places:\n"
           "<script>x</script>\n"
           "no colon here at all\n"
           + "very " * 20 + "long name: a; b; c\n"
           "Pine Ridge: a rocky path; a fallen log; a lone pine\n"
           "Old Mill: duplicate; entry; here\n")
    out = parse_places(raw, 5)
    assert [n for n, _ in out] == ["old mill", "reed marsh", "pine ridge"]
    assert out[0][1] == ("a creaking wheel", "a mossy weir", "a stack of sacks")
    assert parse_places(raw, 2)[-1][0] == "reed marsh"


def test_parse_places_missing_scenery_is_padded_and_bad_phrases_dropped():
    out = parse_places("Salt Flat\nTin Town: a rusty gate; whispers", 4)
    assert [n for n, _ in out] == ["salt flat", "tin town"]
    for _, sc in out:
        assert len(sc) == 3 and all(2 <= len(p.split()) <= 8 for p in sc)
    assert out[1][1][0] == "a rusty gate" and "whispers" not in out[1][1]
    assert parse_places("", 3) == [] and parse_places("Here are some:\n", 3) == []


def test_overlapping_cones_pick_the_nearest_neighbour():
    ns = [Neighbour("origin", 190.0, ("a", "b", "c")), Neighbour("near", 120.0, ("x", "y", "z"))]
    j = Journey("here", ns)
    j.step(160.0)  # origin 30 off, near 40 off: origin nearer
    assert j.drift == {"origin": 1, "near": 0}
    j.step(150.0)  # origin 40 off, near 30 off: near wins, origin resets
    assert j.drift == {"origin": 0, "near": 1}
