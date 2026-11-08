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
    """VECWORLD_PAINTER: flux (the default) or turbo (SD-Turbo, opt-in only: the earlier painter, kept to compare
    against; nothing on the FLUX path uses it since Task 10c)."""
    if name == "flux":
        from vecworld.flux.painter import FluxPainter

        return FluxPainter()
    if name == "turbo":
        from vecworld.dream.painter import TurboPainter

        return TurboPainter()
    raise SystemExit(f"VECWORLD_PAINTER must be flux or turbo, not {name!r}")


def make_dreamer(setting: str, painter_name: str):
    """VECWORLD_DREAMER: on, off, or auto (default): on, except beside FLUX with its blocks' weights in RAM
    (FLUX_WEIGHTS_SOURCE=ram, or VECWORLD_KERNEL=off), whose ~9 GB leaves no room for Qwen's ~6 GB in 14 GB of RAM.
    From the SSD (the default since Task 10b) the painter holds ~2 GB and the dreamer fits beside it."""
    from vecworld.engine import engine_kernel, engine_weights_source

    flux_in_ram = painter_name == "flux" and engine_weights_source(engine_kernel()) == "ram"
    on = {"on": True, "off": False, "auto": not flux_in_ram}.get(setting)
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
    # The models are loaded: their files' pages left in the page cache (~4.5 GB) only crowd the painter and the
    # dreamer toward swap (Task 10b, docs/engine/11-io-uring.md); the FLUX blocks are read with O_DIRECT anyway.
    from vecworld.config import _HF_HUB
    from vecworld.engine.uring import drop_file_cache

    log.info("advised the kernel to drop %.1f GB of model files from the page cache", drop_file_cache(_HF_HUB) / 1e9)
    uvicorn.run(create_app(service, WEB_DIR), host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
