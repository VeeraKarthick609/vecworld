import os

# Must be set before torch (or anything importing it) touches CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path  # noqa: E402

import uvicorn  # noqa: E402

from vecworld.api.app import create_app  # noqa: E402
from vecworld.config import SCENE_DIR  # noqa: E402
from vecworld.dream.ideas import QwenDreamer  # noqa: E402
from vecworld.dream.labels import LabelFinder  # noqa: E402
from vecworld.dream.painter import TurboPainter  # noqa: E402
from vecworld.dream.service import DreamService  # noqa: E402

WEB_DIR = Path(__file__).resolve().parents[2] / "web"


def main() -> None:
    service = DreamService(TurboPainter(), QwenDreamer(), LabelFinder(), SCENE_DIR)
    uvicorn.run(create_app(service, WEB_DIR), host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
