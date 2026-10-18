import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vecworld.game.service import GameError, GameService
from vecworld.game.session import Scene


class JumpIn(BaseModel):
    text: str


class StepIn(BaseModel):
    concept: str


def scene_out(sc: Scene) -> dict:
    return {
        "scene_id": sc.scene_id,
        "image_url": f"/scenes/{sc.scene_id}.png",
        "hotspots": [{"concept": h.concept, "x": h.x, "y": h.y, "w": h.w, "h": h.h} for h in sc.hotspots],
        "mindmap": {
            "me": list(sc.me),
            "trail": [list(p) for p in sc.trail],
            "neighbours": [{"concept": w, "x": x, "y": y} for w, x, y in sc.neighbours],
            "reach_radius": sc.reach_radius,
        },
    }


def error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": code, "message": message}, status_code=status)


def create_app(service: GameService, web_dir: Path) -> FastAPI:
    app = FastAPI(title="vecworld")
    # GPU work is serialised inside the service; a few threads let a second request
    # for a busy session get its 409 instead of waiting in line.
    pool = ThreadPoolExecutor(max_workers=8)

    async def run(fn, *args):
        return await asyncio.get_running_loop().run_in_executor(pool, fn, *args)

    @app.exception_handler(GameError)
    async def game_error(_: Request, e: GameError):
        return error(e.status, e.code, str(e))

    @app.post("/api/session")
    async def new_session():
        return {"session_id": service.new_session().id}

    @app.post("/api/session/{sid}/jump")
    async def jump(sid: str, body: JumpIn):
        return scene_out(await run(service.jump, sid, body.text))

    @app.post("/api/session/{sid}/step")
    async def step(sid: str, body: StepIn):
        return scene_out(await run(service.step, sid, body.concept))

    app.mount("/scenes", StaticFiles(directory=service.scene_dir), name="scenes")
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app
