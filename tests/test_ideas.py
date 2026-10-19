from vecworld.dream.ideas import FALLBACK_IDEAS, ideas_request, parse_ideas


def test_parse_ideas_cleans_and_bounds():
    raw = '1. "An old lantern on a post."\n- a ruined chapel,\n3) A MISTY POND\n\nok\n' + "word " * 40 + "\n<script>x</script>\na ruined chapel"
    assert parse_ideas(raw) == ["an old lantern on a post", "a ruined chapel", "a misty pond"]
    assert parse_ideas("") == [] and len(parse_ideas("\n".join(f"thing number {i}" for i in range(30)))) == 8


def test_parse_ideas_drops_chatter_and_fragments():
    raw = ("Here are 8 objects you might find:\nSure, here you go\nObjects:\nCertainly! A list\nokay so\n"
           "1. A rusted iron gate, half open\n2. An old stone well, covered in moss, by the path\n"
           "3. a moss-covered gravestone\n4. a weathered wooden signpost pointing nowhere in particular at all\n"
           "5. twinkling stars\n6. a single snowflake\n7. pale moonlight\n8. thick fog\n9. an eerie glow\n"
           "10. a misty pond\n11. glowing mushrooms\n12. dense fog obscuring distant peaks\n"
           "13. spectral glow from ancient stone circle\n14. a frosted window")
    assert parse_ideas(raw, 20) == ["a rusted iron gate", "an old stone well", "a moss-covered gravestone",
                                    "a misty pond", "glowing mushrooms", "a frosted window"]


def test_parse_ideas_splits_a_one_line_list_on_commas():
    assert parse_ideas("a lantern, a broken gate, an old well") == ["a lantern", "a broken gate", "an old well"]
    assert parse_ideas("a lantern, a broken gate\nan old well") == ["a lantern", "a broken gate", "an old well"]


def test_fallbacks_are_valid_phrases():
    assert len(FALLBACK_IDEAS) == 12
    assert parse_ideas("\n".join(FALLBACK_IDEAS), 12) == list(FALLBACK_IDEAS)


def test_non_visual_phrases_are_rejected():
    raw = "whispering wind through leaves\nhollow laughter echoing among trees\na faint distant music\nthe smell of rain\na winding stone path\na mossy stone well"
    assert parse_ideas(raw) == ["a winding stone path", "a mossy stone well"]


def test_request_names_moods_only_when_there_are_some():
    assert '"haunted forest" (night, eerie).' in ideas_request("haunted forest", ["night", "eerie"], 8)[1]["content"]
    plain = ideas_request("haunted forest", [], 8)[1]["content"]
    assert '"haunted forest". Only' in plain and "quiet" not in plain


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
