import numpy as np

from vecworld.space.index import MeaningSpace


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


TOY_WORDS = ["ocean", "sea", "ship", "desert", "sand"]
TOY_VECS = np.stack([
    unit([1, 0, 0]),
    unit([0.95, 0.1, 0]),
    unit([0.8, 0.6, 0]),
    unit([0, 0, 1]),
    unit([0, 0.1, 0.95]),
])


def make_space() -> MeaningSpace:
    return MeaningSpace(list(TOY_WORDS), TOY_VECS.copy())
