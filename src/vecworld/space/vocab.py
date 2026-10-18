from functools import lru_cache
from pathlib import Path

import numpy as np

from vecworld.space.index import MeaningSpace


EXTRA_STOPWORDS = {"will", "may", "might", "must", "shall", "yes", "being", "while", "lot", "thing", "things"}


def _is_noun(wordnet, w: str) -> bool:
    synsets = wordnet.synsets(w)
    if not synsets or synsets[0].pos() != "n" or synsets[0].instance_hypernyms():
        return False
    # WordNet capitalises proper nouns ("Olympics"), so only an exact lowercase lemma counts.
    noun = other = 0
    has_common_noun = False
    for s in synsets:
        lemmas = [l for l in s.lemmas() if l.name() == w]
        c = sum(l.count() for l in lemmas)
        if s.pos() == "n":
            noun += c
            has_common_noun |= bool(lemmas)
        else:
            other += c
    return has_common_noun and (noun > other or other == 0)


def _corpora(*names: str) -> None:
    import nltk

    for corpus in names:
        try:
            nltk.data.find(f"corpora/{corpus}")
        except LookupError:
            nltk.download(corpus, quiet=True)


def build_vocab(n: int = 5000) -> list[str]:
    """Most frequent English words whose dominant sense is a common noun, in frequency order."""
    from nltk.corpus import stopwords, wordnet
    from wordfreq import top_n_list

    _corpora("wordnet", "stopwords")
    stop = set(stopwords.words("english")) | EXTRA_STOPWORDS

    out: list[str] = []
    seen: set[str] = set()
    for w in top_n_list("en", 80000):
        if w in seen or w in stop or not (w.isalpha() and w.islower() and 3 <= len(w) <= 15):
            continue
        if not _is_noun(wordnet, w):
            continue
        seen.add(w)
        out.append(w)
        if len(out) == n:
            break
    return out


@lru_cache(maxsize=None)
def is_concrete(word: str) -> bool:
    """Whether the word's dominant noun sense is a physical thing (WordNet), so it can be drawn plainly."""
    from nltk.corpus import wordnet

    _corpora("wordnet")
    senses = wordnet.synsets(word, pos="n")
    if not senses:
        return False
    return wordnet.synset("physical_entity.n.01") in senses[0].closure(lambda s: s.hypernyms())


def centre(raw: np.ndarray, mean: np.ndarray) -> np.ndarray:
    x = np.asarray(raw, np.float32) - mean
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def load_space(data_dir: Path) -> tuple[MeaningSpace, np.ndarray]:
    words = (data_dir / "vocab.txt").read_text().split()
    vectors = np.load(data_dir / "vocab.npy")
    mean = np.load(data_dir / "vocab_mean.npy")
    return MeaningSpace(words, vectors), mean
