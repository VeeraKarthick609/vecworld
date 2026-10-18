from vecworld.config import DATA_DIR
from vecworld.space.vocab import build_vocab

if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    words = build_vocab(5000)
    (DATA_DIR / "vocab.txt").write_text("\n".join(words) + "\n")
    print(f"wrote {len(words)} words to {DATA_DIR / 'vocab.txt'}")
