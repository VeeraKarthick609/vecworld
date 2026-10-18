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
