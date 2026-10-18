import threading
import time

import pytest
from conftest import FakeDreamer, FakePainter
from fastapi.testclient import TestClient

from vecworld.api.app import create_app
from vecworld.dream.labels import Label
from vecworld.dream.service import DreamService


class FakeLabels:
    def __init__(self):
        self._n = 0

    def find(self, img, wanted):
        out = []
        for text, u in wanted:
            self._n += 1
            out.append(Label(f"L{self._n}", text, u, 0.5, 0.05, 0.2))
        return out


class GatedPainter(FakePainter):
    """step() blocks until `gate` is set, after signalling `entered`."""

    def __init__(self):
        super().__init__()
        self.entered, self.gate = threading.Event(), threading.Event()

    def step(self, prev, base, focus, focus_at, seed, **kw):
        self.entered.set()
        assert self.gate.wait(10)
        return super().step(prev, base, focus, focus_at, seed, **kw)


def make_client(tmp_path, painter=None):
    web = tmp_path / "web"
    web.mkdir(exist_ok=True)
    (web / "index.html").write_text("<html><canvas></canvas></html>")
    svc = DreamService(painter or FakePainter(), FakeDreamer(), FakeLabels(), tmp_path / "scenes",
                       spawn=lambda fn: fn())
    return TestClient(create_app(svc, web))


@pytest.fixture
def client(tmp_path):
    return make_client(tmp_path)


def new(client):
    return client.post("/api/session").json()["session_id"]


def started(client, theme="a quiet forest", moods=()):
    sid = new(client)
    r = client.post(f"/api/session/{sid}/start", json={"theme": theme, "moods": list(moods)})
    assert r.status_code == 200, r.text
    return sid, r.json()


def test_full_flow(client):
    sid, v = started(client, moods=["night"])
    assert v["steps"] == 0 and v["theme"] == "a quiet forest" and v["moods"] == ["night"]
    assert v["image_url"] == f"/scenes/{v['view_id']}.png"
    assert client.get(v["image_url"]).status_code == 200

    w = client.post(f"/api/session/{sid}/walk", json={"yaw": 10}).json()
    assert w["steps"] == 1 and w["view_id"] != v["view_id"]
    assert {"id", "text", "u", "y", "w", "h"} <= w["labels"][0].keys()
    assert client.get(w["image_url"]).status_code == 200

    b = client.post(f"/api/session/{sid}/back", json={"yaw": 10}).json()
    assert b["steps"] == 2

    label = b["labels"][-1]["id"]
    a = client.post(f"/api/session/{sid}/approach", json={"label": label})
    assert a.status_code == 200 and a.json()["steps"] == 3

    assert client.post(f"/api/session/{sid}/theme", json={"theme": "a desert"}).json() == {"ok": True}
    assert client.post(f"/api/session/{sid}/moods", json={"moods": []}).json() == {"ok": True}
    assert client.post(f"/api/session/{sid}/walk", json={"yaw": 0}).json()["theme"] == "a desert"


def test_prefetch_returns_202_while_painter_is_blocked(tmp_path):
    painter = GatedPainter()
    client = make_client(tmp_path, painter)
    sid, _ = started(client)
    results = {}
    t = threading.Thread(target=lambda: results.update(
        r=client.post(f"/api/session/{sid}/walk", json={"yaw": 0})))
    t.start()
    try:
        assert painter.entered.wait(5)
        t0 = time.monotonic()
        r = client.post(f"/api/session/{sid}/prefetch", json={"yaw": 90})
        assert r.status_code == 202 and r.json() == {"ok": True}
        assert time.monotonic() - t0 < 1
    finally:
        painter.gate.set()
        t.join(10)
    assert results["r"].status_code == 200


def test_unknown_session_404(client):
    r = client.post("/api/session/nope/walk", json={"yaw": 0})
    assert r.status_code == 404 and r.json()["error"] == "unknown_session"
    assert client.post("/api/session/nope/start", json={"theme": "x", "moods": []}).status_code == 404


def test_walk_before_start_409(client):
    sid = new(client)
    r = client.post(f"/api/session/{sid}/walk", json={"yaw": 0})
    assert r.status_code == 409 and r.json()["error"] == "not_started"


def test_unknown_label_400(client):
    sid, _ = started(client)
    r = client.post(f"/api/session/{sid}/approach", json={"label": "zzz"})
    assert r.status_code == 400 and r.json()["error"] == "unknown_label"


def test_empty_theme_422(client):
    sid = new(client)
    r = client.post(f"/api/session/{sid}/start", json={"theme": "   ", "moods": []})
    assert r.status_code == 422 and r.json()["error"] == "bad_input"


@pytest.mark.parametrize("path,body", [
    ("walk", '{"yaw": NaN}'),
    ("walk", '{"yaw": Infinity}'),
    ("walk", '{"yaw": "left"}'),
    ("walk", "{}"),
    ("back", '{"yaw": NaN}'),
    ("prefetch", '{"yaw": NaN}'),
    ("start", '{"moods": []}'),
    ("start", '{"theme": "' + "x" * 1001 + '", "moods": []}'),
    ("moods", '{"moods": "night"}'),
    ("approach", "{}"),
])
def test_validation_errors_use_error_format(client, path, body):
    sid = new(client)
    r = client.post(f"/api/session/{sid}/{path}", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422
    j = r.json()
    assert j["error"] == "bad_input" and isinstance(j["message"], str) and j["message"]


def test_oom_503_and_server_survives(tmp_path):
    painter = FakePainter()
    client = make_client(tmp_path, painter)
    sid, _ = started(client)
    painter.fail_next = True
    r = client.post(f"/api/session/{sid}/walk", json={"yaw": 0})
    assert r.status_code == 503 and r.json()["error"] == "gpu_unavailable"
    assert client.post(f"/api/session/{sid}/walk", json={"yaw": 0}).status_code == 200
    assert client.post("/api/session").status_code == 200


def test_static_index_served(client):
    assert client.get("/").status_code == 200
