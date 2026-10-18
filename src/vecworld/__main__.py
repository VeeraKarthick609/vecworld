from pathlib import Path

import uvicorn

from vecworld.api.app import create_app
from vecworld.config import DATA_DIR, SCENE_DIR
from vecworld.game.service import GameService
from vecworld.scenes.generator import TurboGenerator
from vecworld.scenes.hotspots import ClipSegFinder
from vecworld.space.embedder import LLMEmbedder
from vecworld.space.vocab import load_space

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


def main() -> None:
    space, mean = load_space(DATA_DIR)
    service = GameService(space, mean, LLMEmbedder(), TurboGenerator(), ClipSegFinder(), SCENE_DIR)
    uvicorn.run(create_app(service, WEB_DIR), host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
