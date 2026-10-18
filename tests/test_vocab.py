import numpy as np

from vecworld.space.vocab import build_vocab, centre


def test_build_vocab_shape():
    v = build_vocab(500)
    assert len(v) == 500 and len(set(v)) == 500
    assert all(w.isalpha() and w.islower() and 3 <= len(w) <= 15 for w in v)
    assert "ocean" in build_vocab(5000)


def test_vocab_is_picturable_nouns():
    v = set(build_vocab(5000))
    assert not {"are", "have", "will", "may", "while", "might", "yes"} & v
    assert not {"liverpool", "michigan", "olympics"} & v
    assert {"ocean", "ship", "storm", "whale", "loneliness"} <= v


def test_centre_unit_rows():
    raw = np.random.rand(10, 8).astype(np.float32) + 5
    out = centre(raw, raw.mean(0))
    assert np.allclose(np.linalg.norm(out, axis=1), 1, atol=1e-5)
