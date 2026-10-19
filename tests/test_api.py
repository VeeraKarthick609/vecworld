import threading
import time

import pytest
from conftest import FakeDepth, FakeDreamer, FakePainter
from fastapi.testclient import TestClient

from vecworld.api.app import create_app
from vecworld.roam.service import RoamService


class GatedPainter(FakePainter):
    """Once armed, step() signals `entered` and blocks until `gate` is set."""

    def __init__(self):
        super().__init__()
        self.armed = False
        self.entered, self.gate = threading.Event(), threading.Event()

    def step(self, *a, **kw):
        if self.armed:
            self.entered.set()
            assert self.gate.wait(10)
        return super().step(*a, **kw)


def make_client(tmp_path, painter=None):
    web = tmp_path / "web"
    web.mkdir(exist_ok=True)
    (web / "index.html").write_text("<html><canvas></canvas></html>")
    svc = RoamService(painter or FakePainter(), FakeDepth(), FakeDreamer(), tmp_path / "scenes",
                      spawn=lambda fn: None)
    return TestClient(create_app(svc, web))


@pytest.fixture
def client(tmp_path):
    return make_client(tmp_path)


def new(client):
    return client.post("/api/session").json()["session_id"]


def started(client, place="a quiet forest", moods=()):
    sid = new(client)
    r = client.post(f"/api/session/{sid}/start", json={"place": place, "moods": list(moods)})
    assert r.status_code == 200, r.text
    return sid, r.json()


def test_start_returns_spot(client):
    sid, spot = started(client, moods=["night"])
    assert set(spot) == {"spot_id", "image_url", "depth_url", "heading", "scale", "distance", "place",
                         "neighbours", "arrived"}
    assert spot["place"] == "a quiet forest"
    assert spot["scale"] == 1 and spot["distance"] == 0 and spot["arrived"] is None
    assert spot["neighbours"] and set(spot["neighbours"][0]) == {"name", "bearing"}


def test_scenes_serve_jpeg_and_png(client):
    _, spot = started(client)
    img = client.get(spot["image_url"])
    assert img.status_code == 200 and img.content[:3] == b"\xff\xd8\xff"
    depth = client.get(spot["depth_url"])
    assert depth.status_code == 200 and depth.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_next_returns_new_spot(client):
    sid, first = started(client)
    r = client.post(f"/api/session/{sid}/next", json={"exit_heading": 90})
    assert r.status_code == 200, r.text
    nxt = r.json()
    assert nxt["spot_id"] != first["spot_id"]
    assert nxt["image_url"] != first["image_url"]


def test_theme_and_moods_ok(client):
    sid, _ = started(client)
    assert client.post(f"/api/session/{sid}/theme", json={"place": "a harbour"}).json() == {"ok": True}
    assert client.post(f"/api/session/{sid}/moods", json={"moods": ["night"]}).json() == {"ok": True}


def test_unknown_session_404(client):
    r = client.post("/api/session/nope/next", json={"exit_heading": 0})
    assert r.status_code == 404 and r.json()["error"] == "unknown_session"
    r = client.post("/api/session/nope/start", json={"place": "forest", "moods": []})
    assert r.status_code == 404


def test_not_started_409(client):
    sid = new(client)
    r = client.post(f"/api/session/{sid}/next", json={"exit_heading": 0})
    assert r.status_code == 409 and r.json()["error"] == "not_started"
    assert client.post(f"/api/session/{sid}/heading", json={"heading": 0}).status_code == 409


def test_bad_input_422(client):
    sid = new(client)
    r = client.post(f"/api/session/{sid}/start", json={"place": "   ", "moods": []})
    assert r.status_code == 422 and r.json()["error"] == "bad_input"
    r = client.post(f"/api/session/{sid}/start", json={"place": "x" * 1001, "moods": []})
    assert r.status_code == 422
    r = client.post(f"/api/session/{sid}/start", json={"moods": []})
    assert r.status_code == 422 and r.json()["error"] == "bad_input"


def test_validation_error_never_echoes_input(client):
    sid, _ = started(client)
    secret = "hunter2-secret"
    r = client.post(f"/api/session/{sid}/next", json={"exit_heading": secret})
    assert r.status_code == 422 and secret not in r.text


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("route,key", [("next", "exit_heading"), ("heading", "heading")])
def test_non_finite_heading_422(client, route, key, bad):
    sid, _ = started(client)
    r = client.post(f"/api/session/{sid}/{route}", content=f'{{"{key}": {bad}}}',
                    headers={"content-type": "application/json"})
    assert r.status_code == 422 and r.json()["error"] == "bad_input"


def test_gpu_unavailable_503(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    painter = FakePainter()
    c = make_client(tmp_path, painter)
    sid = new(c)
    painter.fail_next = True
    r = c.post(f"/api/session/{sid}/start", json={"place": "forest", "moods": []})
    assert r.status_code == 503 and r.json()["error"] == "gpu_unavailable"
    assert c.post(f"/api/session/{sid}/start", json={"place": "forest", "moods": []}).status_code == 200


def test_busy_409_while_move_in_flight(tmp_path):
    painter = GatedPainter()
    c = make_client(tmp_path, painter)
    sid, _ = started(c)
    painter.armed = True
    out = {}
    t = threading.Thread(target=lambda: out.update(r=c.post(f"/api/session/{sid}/next", json={"exit_heading": 200})))
    t.start()
    try:
        assert painter.entered.wait(5)
        r = c.post(f"/api/session/{sid}/next", json={"exit_heading": 10})
        assert r.status_code == 409 and r.json()["error"] == "busy"
    finally:
        painter.gate.set()
        t.join(10)
    assert out["r"].status_code == 200


def test_heading_returns_202_while_painter_blocks(tmp_path):
    painter = GatedPainter()
    c = make_client(tmp_path, painter)
    sid, _ = started(c)
    painter.armed = True
    out = {}
    t = threading.Thread(target=lambda: out.update(r=c.post(f"/api/session/{sid}/next", json={"exit_heading": 200})))
    t.start()
    try:
        assert painter.entered.wait(5)
        t0 = time.monotonic()
        r = c.post(f"/api/session/{sid}/heading", json={"heading": 45})
        assert r.status_code == 202 and r.json() == {"ok": True}
        assert time.monotonic() - t0 < 1.0
        assert not painter.gate.is_set() and t.is_alive()   # the move is still blocked
    finally:
        painter.gate.set()
        t.join(10)
    assert out["r"].status_code == 200


def test_static_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "canvas" in r.text
