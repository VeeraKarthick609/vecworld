import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vecworld.roam.errors import GameError
from vecworld.roam.service import RoamService, SpotView


class StartIn(BaseModel):
    place: str = Field(max_length=1000)
    moods: list[str] = []


class NextIn(BaseModel):
    exit_heading: float = Field(allow_inf_nan=False)
    # How far from the current spot's centre the player leaves it (its units). Left out: its walkable radius.
    walked: float = Field(default=None, ge=0, allow_inf_nan=False)


class EnterIn(BaseModel):
    spot_id: str = Field(max_length=64)


class HeadingIn(BaseModel):
    heading: float = Field(allow_inf_nan=False)


class ThemeIn(BaseModel):
    place: str = Field(max_length=1000)


class MoodsIn(BaseModel):
    moods: list[str]


def spot_out(v: SpotView) -> dict:
    return {
        "spot_id": v.spot_id,
        "image_url": f"/scenes/{v.image_path.name}",
        "depth_url": f"/scenes/{v.depth_path.name}",
        "heading": v.heading,
        "scale": v.scale,
        "distance": v.distance,
        "limits": v.limits,
        "place": v.place,
        "neighbours": [{"name": n.name, "bearing": n.bearing} for n in v.neighbours],
        "arrived": v.arrived,
        "served": v.served,
    }


def error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": code, "message": message}, status_code=status)


def create_app(service: RoamService, web_dir: Path) -> FastAPI:
    app = FastAPI(title="vecworld")
    # GPU work is serialised inside the service; a few threads let a second request
    # for a busy session get its 409 instead of waiting in line.
    pool = ThreadPoolExecutor(max_workers=8)

    async def run(fn, *args):
        return await asyncio.get_running_loop().run_in_executor(pool, fn, *args)

    @app.exception_handler(GameError)
    async def game_error(_: Request, e: GameError):
        return error(e.status, e.code, str(e))

    @app.exception_handler(RequestValidationError)
    async def bad_body(_: Request, e: RequestValidationError):
        # Report where and what, never the offending value.
        msg = "; ".join(f"{'.'.join(str(p) for p in err['loc'][1:]) or 'body'}: {err['msg']}" for err in e.errors())
        return error(422, "bad_input", msg or "Invalid request.")

    @app.post("/api/session")
    async def new_session():
        return {"session_id": service.new_session().id}

    @app.post("/api/session/{sid}/start")
    async def start(sid: str, body: StartIn):
        return spot_out(await run(service.start, sid, body.place, body.moods))

    @app.post("/api/session/{sid}/next")
    async def next_spot(sid: str, body: NextIn):
        return spot_out(await run(service.next, sid, body.exit_heading, body.walked))

    @app.post("/api/session/{sid}/enter")
    async def enter(sid: str, body: EnterIn):
        # No GPU work and never waits (a move in flight makes it 409 busy), so it runs inline.
        return spot_out(service.enter(sid, body.spot_id))

    @app.post("/api/session/{sid}/heading", status_code=202)
    async def heading(sid: str, body: HeadingIn):
        # Only re-targets queued work and returns; never waits on the GPU, so it runs inline
        # rather than behind possibly blocked pool threads.
        service.heading(sid, body.heading)
        return {"ok": True}

    @app.get("/api/session/{sid}/progress")
    async def progress(sid: str):
        # How far the spot the player waits for has painted: {"done", "total"}, or {} when it isn't painting. The
        # viewer polls it every second in the fog; it only reads state, so it runs inline.
        return service.progress(sid) or {}

    @app.post("/api/session/{sid}/theme")
    async def theme(sid: str, body: ThemeIn):
        await run(service.set_theme, sid, body.place)
        return {"ok": True}

    @app.post("/api/session/{sid}/moods")
    async def moods(sid: str, body: MoodsIn):
        await run(service.set_moods, sid, body.moods)
        return {"ok": True}

    app.mount("/scenes", StaticFiles(directory=service.scene_dir), name="scenes")
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app
