import pytest

from smartsight.config import Settings
from smartsight.server import create_app


@pytest.fixture
def client():
    app = create_app(Settings())
    app.testing = True
    return app.test_client()


def test_index_renders(client):
    r = client.get("/")
    assert r.status_code == 200 and b"SmartSight" in r.data


def test_status_contract(client):
    data = client.get("/test_status").get_json()
    for key in ("is_active", "test_finished", "max_attempts", "target_direction", "score", "attempts",
                "time_left", "distance_cm", "vision_level", "response_time", "hesitation", "alert",
                "privacy_mode", "head_stability_pct", "ambient_lux", "fps", "inference_ms"):
        assert key in data


def test_start_reset_flow(client):
    r = client.post("/start", json={"max_attempts": 5})
    assert r.get_json()["max_attempts"] == 5
    assert client.get("/test_status").get_json()["is_active"]
    client.post("/reset")
    assert not client.get("/test_status").get_json()["is_active"]


def test_start_rejects_bad_input(client):
    assert client.post("/start", json={"max_attempts": "abc"}).status_code == 400
    assert client.post("/start", data="not json").status_code == 200  # falls back to default


def test_toggle_privacy(client):
    assert client.post("/toggle_privacy").get_json()["privacy_mode"] is True
    assert client.post("/toggle_privacy").get_json()["privacy_mode"] is False


def test_pause_endpoint(client):
    client.post("/start", json={"max_attempts": 5})
    assert client.post("/pause").get_json()["paused"] is True
    assert client.get("/test_status").get_json()["paused"] is True
    assert client.post("/pause").get_json()["paused"] is False


def test_calibrate_distance_validation(client):
    assert client.post("/calibrate_distance", json={"distance_cm": "x"}).status_code == 400
    assert client.post("/calibrate_distance", json={"distance_cm": 5}).status_code == 400
    assert client.post("/calibrate_distance", json={"distance_cm": 60}).status_code == 409  # camera not started


def test_worker_is_built_once_under_concurrency(monkeypatch):
    import threading
    import time

    import smartsight.server as server

    calls = []

    class FakeWorker:
        def frames(self):
            yield b""

    def slow_build(cfg, session):
        calls.append(1)
        time.sleep(0.05)
        return FakeWorker()

    monkeypatch.setattr(server, "build_worker", slow_build)
    app = server.create_app(Settings())
    client = app.test_client()
    threads = [threading.Thread(target=lambda: client.get("/video_feed")) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1
