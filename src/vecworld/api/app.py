import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vecworld.dream.errors import GameError
from vecworld.dream.service import DreamService
from vecworld.dream.session import View


class StartIn(BaseModel):
    theme: str = Field(max_length=1000)
    moods: list[str] = []


class YawIn(BaseModel):
    yaw: float = Field(allow_inf_nan=False)


class ApproachIn(BaseModel):
    label: str


class ThemeIn(BaseModel):
    theme: str = Field(max_length=1000)


class MoodsIn(BaseModel):
    moods: list[str]


def view_out(v: View) -> dict:
    return {
        "view_id": v.view_id,
        "image_url": f"/scenes/{v.view_id}.png",
        "labels": [{"id": l.id, "text": l.text, "u": l.u, "y": l.y, "w": l.w, "h": l.h} for l in v.labels],
        "theme": v.theme,
        "moods": list(v.moods),
        "steps": v.steps,
        "walked_yaw": v.walked_yaw,
    }


def error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": code, "message": message}, status_code=status)


def create_app(service: DreamService, web_dir: Path) -> FastAPI:
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
        return view_out(await run(service.start, sid, body.theme, body.moods))

    @app.post("/api/session/{sid}/walk")
    async def walk(sid: str, body: YawIn):
        return view_out(await run(service.walk, sid, body.yaw))

    @app.post("/api/session/{sid}/back")
    async def back(sid: str, body: YawIn):
        return view_out(await run(service.back, sid, body.yaw))

    @app.post("/api/session/{sid}/approach")
    async def approach(sid: str, body: ApproachIn):
        return view_out(await run(service.approach, sid, body.label))

    @app.post("/api/session/{sid}/theme")
    async def theme(sid: str, body: ThemeIn):
        await run(service.set_theme, sid, body.theme)
        return {"ok": True}

    @app.post("/api/session/{sid}/moods")
    async def moods(sid: str, body: MoodsIn):
        await run(service.set_moods, sid, body.moods)
        return {"ok": True}

    @app.post("/api/session/{sid}/prefetch", status_code=202)
    async def prefetch(sid: str, body: YawIn):
        # Only queues a low-priority job and returns; never waits on the GPU, so it runs inline
        # rather than behind possibly blocked pool threads.
        service.prefetch(sid, body.yaw)
        return {"ok": True}

    app.mount("/scenes", StaticFiles(directory=service.scene_dir), name="scenes")
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app
