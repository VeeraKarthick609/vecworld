import os

# Must be set before torch (or anything importing it) touches CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import logging  # noqa: E402
from pathlib import Path  # noqa: E402

import uvicorn  # noqa: E402

from vecworld.api.app import create_app  # noqa: E402
from vecworld.config import SCENE_DIR  # noqa: E402
from vecworld.roam.depth import AnythingDepth  # noqa: E402
from vecworld.roam.service import RoamService  # noqa: E402

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
log = logging.getLogger("vecworld")


class NoDreamer:
    """VECWORLD_DREAMER=off: no LLM; every place keeps its fallback neighbours."""

    def places(self, place, journey, moods, n):
        return []


def make_painter(name: str):
    """VECWORLD_PAINTER: flux (default) or turbo."""
    if name == "flux":
        from vecworld.flux.painter import FluxPainter

        return FluxPainter()
    if name == "turbo":
        from vecworld.dream.painter import TurboPainter

        return TurboPainter()
    raise SystemExit(f"VECWORLD_PAINTER must be flux or turbo, not {name!r}")


def make_dreamer(setting: str, painter_name: str):
    """VECWORLD_DREAMER: on, off, or auto (default): off beside FLUX, whose ~9 GB leaves no room for Qwen's ~6 GB
    in 14 GB of RAM until the engine's SSD loader lands; on beside Turbo."""
    on = {"on": True, "off": False, "auto": painter_name != "flux"}.get(setting)
    if on is None:
        raise SystemExit(f"VECWORLD_DREAMER must be on, off or auto, not {setting!r}")
    if not on:
        log.warning("dreamer off: places come from the fallback table")
        return NoDreamer()
    from vecworld.dream.ideas import QwenDreamer

    return QwenDreamer()


def main() -> None:
    painter_name = os.environ.get("VECWORLD_PAINTER", "flux")
    painter = make_painter(painter_name)
    dreamer = make_dreamer(os.environ.get("VECWORLD_DREAMER", "auto"), painter_name)
    service = RoamService(painter, AnythingDepth(), dreamer, SCENE_DIR)
    uvicorn.run(create_app(service, WEB_DIR), host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
