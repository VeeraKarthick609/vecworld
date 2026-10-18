import pytest

from vecworld.config import DATA_DIR
from vecworld.space.index import angle
from vecworld.space.vocab import centre, load_space


@pytest.mark.gpu
def test_llm_neighbourhood_is_sensible():
    from vecworld.space.embedder import LLMEmbedder

    space, mean = load_space(DATA_DIR)
    e = LLMEmbedder()
    ocean = centre(e.embed(["ocean"]), mean)[0]
    words = [w for w, _ in space.neighbours(ocean, k=25)]
    assert {"sea", "water", "waves", "beach", "coast", "marine"} & set(words)
    assert angle(ocean, space.vector("sea")) < angle(ocean, space.vector("keyboard"))
