import sys

import numpy as np

from vecworld.config import DATA_DIR, LAYER
from vecworld.space.embedder import LLMEmbedder
from vecworld.space.vocab import centre, load_space

PROBES = ["ocean", "storm", "loneliness", "ship"]

if __name__ == "__main__":
    layer = int(sys.argv[1]) if len(sys.argv) > 1 else LAYER
    words = (DATA_DIR / "vocab.txt").read_text().split()
    raw = LLMEmbedder(layer=layer, device="cuda").embed(words)
    e = LLMEmbedder(layer=layer)  # runtime path: CPU
    mean = raw.mean(axis=0)
    np.save(DATA_DIR / "vocab_mean.npy", mean)
    np.save(DATA_DIR / "vocab.npy", centre(raw, mean))
    space, mean = load_space(DATA_DIR)
    print(f"layer {layer}")
    for w in PROBES:
        p = centre(e.embed([w]), mean)[0]
        print(w, "->", ", ".join(n for n, _ in space.neighbours(p, k=10, exclude={w})))
