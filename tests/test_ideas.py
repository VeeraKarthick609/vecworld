from vecworld.dream.ideas import is_visual, parse_places


def test_non_visual_phrases_are_rejected():
    bad = ["whispering wind through leaves", "hollow laughter echoing among trees", "a faint distant music",
           "the smell of rain", "a field of stars", "drifting fog"]
    assert not any(is_visual(p) for p in bad)
    assert all(is_visual(p) for p in ["a winding stone path", "a mossy stone well", "a misty pond"])



def test_places_request_shape():
    from vecworld.dream.ideas import places_request

    msgs = places_request("old forest", ["village", "lake"], ["foggy"], 3)
    assert [m["role"] for m in msgs] == ["system", "user"]
    u = msgs[1]["content"]
    assert "old forest" in u and "foggy" in u and "3" in u and "Name: phrase; phrase; phrase" in u
    assert "village" in u and "lake" in u  # the journey so far, so names don't repeat


def test_places_request_without_moods_or_journey():
    from vecworld.dream.ideas import places_request

    assert places_request("oasis", [], [], 2)[1]["content"]


def test_qwen_places_filters_and_fills_from_fallbacks_without_a_model():
    from vecworld.dream.ideas import QwenDreamer

    d = object.__new__(QwenDreamer)
    d._generate = lambda messages, max_new_tokens: (
        "Here are places:\n1. Forest: a; b; c\n2. Reed Marsh: tall green reeds; a rotting jetty; a heron on one leg\n")
    out = d.places("forest", ["village"], [], 3)
    assert out[0][0] == "reed marsh" and len(out) == 3 and all(len(sc) == 3 for _, sc in out)
    assert {n for n, _ in out}.isdisjoint({"forest", "village"}) and len({n for n, _ in out}) == 3


def test_qwen_places_excludes_journey_case_insensitively():
    from vecworld.dream.ideas import QwenDreamer

    d = object.__new__(QwenDreamer)
    d._generate = lambda messages, max_new_tokens: "Old Mill: a creaking wheel; a mossy weir; a stack of sacks\n"
    out = d.places("Forest", ["Old Mill"], [], 2)
    assert "old mill" not in [n for n, _ in out] and "forest" not in [n for n, _ in out]


def test_parse_places_drops_the_format_line_echoed_back():
    # Seen live (desert oasis, 2026-10-19): signposts named "name" and "phrase".
    text = "Name: phrase; phrase; phrase\nPhrase: phrase; phrase\nPalm grove: tall date palms; a clay well; reeds"
    assert [p[0] for p in parse_places(text, 3)] == ["palm grove"]
