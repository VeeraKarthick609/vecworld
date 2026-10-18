import pytest
from fastapi.testclient import TestClient

from conftest import OomGenerator
from vecworld.api.app import create_app


@pytest.fixture
def client(svc, tmp_path):
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "index.html").write_text("<html><canvas></canvas></html>")
    return TestClient(create_app(svc, tmp_path / "web"))


@pytest.fixture
def client_oom(svc_factory, tmp_path):
    (tmp_path / "web").mkdir(exist_ok=True)
    return TestClient(create_app(svc_factory(generator=OomGenerator()), tmp_path / "web"))


def new(client):
    return client.post("/api/session").json()["session_id"]


def test_full_flow(client):
    sid = new(client)
    r = client.post(f"/api/session/{sid}/jump", json={"text": "ocean"})
    assert r.status_code == 200
    body = r.json()
    assert body["image_url"].startswith("/scenes/") and body["mindmap"]["me"]
    assert {"concept", "x", "y", "w", "h"} <= body["hotspots"][0].keys()
    assert {"me", "trail", "neighbours", "reach_radius"} <= body["mindmap"].keys()
    assert client.get(body["image_url"]).status_code == 200
    c = body["hotspots"][-1]["concept"]
    assert client.post(f"/api/session/{sid}/step", json={"concept": c}).status_code == 200


def test_forward_and_back(client):
    sid = new(client)
    jumped = client.post(f"/api/session/{sid}/jump", json={"text": "ocean"}).json()
    assert jumped["centre"] == "ocean" == jumped["hotspots"][0]["concept"]
    fw = client.post(f"/api/session/{sid}/forward")
    assert fw.status_code == 200 and fw.json()["centre"] != "ocean" and len(fw.json()["mindmap"]["trail"]) == 2
    bk = client.post(f"/api/session/{sid}/back")
    assert bk.status_code == 200 and bk.json()["scene_id"] == jumped["scene_id"]
    assert bk.json()["mindmap"]["trail"] == jumped["mindmap"]["trail"]
    r = client.post(f"/api/session/{sid}/back")
    assert (r.status_code, r.json()["error"]) == (409, "no_history")


@pytest.mark.parametrize("path,payload,status,code", [
    ("jump", {"text": " "}, 422, "bad_input"),
    ("step", {"concept": "ship"}, 409, "not_started"),
    ("forward", None, 409, "not_started"),
    ("back", None, 409, "not_started"),
])
def test_errors(client, path, payload, status, code):
    r = client.post(f"/api/session/{new(client)}/{path}", json=payload)
    assert (r.status_code, r.json()["error"]) == (status, code)
    assert r.json()["message"]


def test_unknown_session(client):
    r = client.post("/api/session/nope/jump", json={"text": "x"})
    assert (r.status_code, r.json()["error"]) == (404, "unknown_session")


def test_oom_is_503_and_server_survives(client_oom):
    sid = new(client_oom)
    r = client_oom.post(f"/api/session/{sid}/jump", json={"text": "ocean"})
    assert (r.status_code, r.json()["error"]) == (503, "gpu_unavailable")
    assert client_oom.post("/api/session").status_code == 200


def test_internal_keyerror_is_not_reported_as_unknown_session(svc, tmp_path):
    (tmp_path / "web").mkdir(exist_ok=True)
    client = TestClient(create_app(svc, tmp_path / "web"), raise_server_exceptions=False)
    sid = new(client)

    def boom(*a, **k):
        raise KeyError("internal")

    svc.space.neighbours = boom
    assert client.post(f"/api/session/{sid}/jump", json={"text": "ocean"}).status_code == 500


def test_index_served(svc):
    from vecworld.__main__ import WEB_DIR

    r = TestClient(create_app(svc, WEB_DIR)).get("/")
    assert r.status_code == 200 and "<canvas" in r.text
    assert TestClient(create_app(svc, WEB_DIR)).get("/app.js").status_code == 200
