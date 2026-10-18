from vecworld.dream.ideas import FALLBACK_IDEAS, parse_ideas


def test_parse_ideas_cleans_and_bounds():
    raw = '1. "An old lantern on a post."\n- a ruined chapel,\n3) A MISTY POND\n\nok\n' + "word " * 40 + "\n<script>x</script>\na ruined chapel"
    assert parse_ideas(raw) == ["an old lantern on a post", "a ruined chapel", "a misty pond"]
    assert parse_ideas("") == [] and len(parse_ideas("\n".join(f"thing number {i}" for i in range(30)))) == 8


def test_fallbacks_are_valid_phrases():
    assert len(FALLBACK_IDEAS) == 12
    assert parse_ideas("\n".join(FALLBACK_IDEAS), 12) == list(FALLBACK_IDEAS)
