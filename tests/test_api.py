import io
import threading
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin

from app.main import (
    AttemptState,
    JobState,
    Settings,
    create_app,
    render_thumbnail,
)


def make_png(size=(320, 240), color=(200, 100, 50)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, format="PNG")
    return out.getvalue()


def png_with_key(key: str, size=(320, 240)) -> bytes:
    info = PngImagePlugin.PngInfo()
    info.add_text("key", key)
    out = io.BytesIO()
    Image.new("RGB", size, (1, 2, 3)).save(out, format="PNG", pnginfo=info)
    return out.getvalue()


@pytest.fixture
def client():
    app = create_app(Settings(slot_count=2, retry_after_seconds=30.0, max_edge=256))
    with TestClient(app) as test_client:
        yield test_client


def wait_until(predicate, timeout: float = 5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met before timeout")


# ---------------------------------------------------------------------------
# Upload + status + download
# ---------------------------------------------------------------------------

def test_upload_returns_batch_and_jobs_become_ready(client):
    resp = client.post(
        "/batches",
        files=[
            ("files", ("a.png", make_png((640, 480)), "image/png")),
            ("files", ("b.png", make_png((300, 600)), "image/png")),
        ],
    )
    assert resp.status_code == 201, resp.text
    batch = resp.json()
    assert batch["total"] == 2
    assert len(batch["jobs"]) == 2
    batch_id = batch["batch_id"]

    wait_until(lambda: client.get(f"/batches/{batch_id}").json()["finished"] == 2)
    status = client.get(f"/batches/{batch_id}").json()
    by_name = {j["filename"]: j for j in status["jobs"]}
    assert by_name["a.png"]["result"]["width"] == 256
    assert by_name["a.png"]["result"]["height"] == 192
    assert by_name["b.png"]["result"]["width"] == 128
    assert by_name["b.png"]["result"]["height"] == 256

    job = by_name["a.png"]
    thumb = client.get(job["result"]["download_url"])
    assert thumb.status_code == 200
    assert thumb.headers["content-type"] == "image/png"
    assert thumb.headers["x-thumbnail-origin"] == "first"
    with Image.open(io.BytesIO(thumb.content)) as img:
        assert img.size == (256, 192)

    job_detail = client.get(f"/jobs/{job['job_id']}").json()
    assert job_detail["result"]["origin"] == "first"


def test_batch_and_job_not_found(client):
    assert client.get("/batches/does-not-exist").status_code == 404
    assert client.get("/jobs/does-not-exist").status_code == 404
    assert client.get("/jobs/does-not-exist/thumbnail").status_code == 404


def test_upload_rejects_invalid_file(client):
    resp = client.post(
        "/batches",
        files=[("files", ("bad.png", b"definitely not a png", "image/png"))],
    )
    assert resp.status_code == 400
    assert "bad.png" in resp.json()["detail"]


def test_upload_rejects_empty_batch(client):
    resp = client.post("/batches")
    assert resp.status_code in (400, 422)


# ---------------------------------------------------------------------------
# Slow-path behaviour over HTTP
# ---------------------------------------------------------------------------

class SlowishRenderer:
    """Renderer that stalls the *first* attempt for a chosen key only.

    The make-up attempt then finishes first, which is the case we want to
    exercise: the retry supplies the deliverable while the original is still
    running.
    """

    def __init__(self, slow_key: str, slow_seconds: float):
        self.slow_key = slow_key
        self.slow_seconds = slow_seconds
        self._lock = threading.Lock()
        self.calls = []
        self._slow_calls = 0

    def __call__(self, data: bytes, max_edge: int):
        with Image.open(io.BytesIO(data)) as img:
            key = img.text["key"]
        with self._lock:
            self.calls.append(key)
            if key == self.slow_key:
                self._slow_calls += 1
                should_stall = self._slow_calls == 1
            else:
                should_stall = False
        if should_stall:
            time.sleep(self.slow_seconds)
        return render_thumbnail(data, max_edge)


def test_http_retry_origin_reported_and_downloadable():
    renderer = SlowishRenderer(slow_key="slow", slow_seconds=0.4)
    app = create_app(
        Settings(slot_count=4, retry_after_seconds=0.1, max_edge=256),
        renderer=renderer,
    )
    with TestClient(app) as client:
        resp = client.post(
            "/batches",
            files=[("files", ("slow.png", png_with_key("slow", (640, 480)), "image/png"))],
        )
        assert resp.status_code == 201
        batch_id = resp.json()["batch_id"]
        job_id = resp.json()["jobs"][0]["job_id"]

        # The make-up attempt finishes before the slow original.
        wait_until(lambda: client.get(f"/jobs/{job_id}").json()["result_ready"])
        job = client.get(f"/jobs/{job_id}").json()
        assert job["result"]["origin"] == "retry"
        assert [a["kind"] for a in job["attempts"]] == ["first", "retry"]

        thumb = client.get(f"/jobs/{job_id}/thumbnail")
        assert thumb.status_code == 200
        assert thumb.headers["x-thumbnail-origin"] == "retry"
        with Image.open(io.BytesIO(thumb.content)) as img:
            assert img.size == (256, 192)

        # Both attempts eventually return; the batch settles as a success.
        wait_until(lambda: client.get(f"/batches/{batch_id}").json()["finished"] == 1)
        final = client.get(f"/jobs/{job_id}").json()
        assert final["state"] == JobState.SUCCEEDED.value
        assert final["attempts"][1]["state"] == AttemptState.SUCCEEDED.value


def test_http_stats_endpoint(client):
    stats = client.get("/stats").json()
    assert stats["slot_count"] == 2
    assert stats["retry_after_seconds"] == 30.0
    assert stats["max_edge"] == 256
    assert stats["max_pixels"] == 50_000_000
    assert stats["pixels_used"] == 0
    assert stats["pixels_free"] == stats["max_pixels"]


def test_upload_rejects_batch_when_single_image_exceeds_pixel_limit():
    app = create_app(
        Settings(slot_count=2, retry_after_seconds=30.0, max_edge=256, max_pixels=999)
    )
    with TestClient(app) as limited_client:
        # 40x30 = 1200 pixels > 999.
        resp = limited_client.post(
            "/batches",
            files=[("files", ("huge.png", make_png((40, 30)), "image/png"))],
        )
        assert resp.status_code == 400
        assert "huge.png" in resp.json()["detail"]
        assert "pixel limit" in resp.json()["detail"]

        # One oversized file rejects the whole batch, including the fitting one.
        resp = limited_client.post(
            "/batches",
            files=[
                ("files", ("small.png", make_png((10, 10)), "image/png")),
                ("files", ("huge.png", make_png((40, 30)), "image/png")),
            ],
        )
        assert resp.status_code == 400
        assert limited_client.get("/stats").json()["jobs"] == {}
        assert limited_client.get("/stats").json()["pixels_used"] == 0


def test_queries_and_downloads_continue_while_validation_waits_for_budget():
    # Every image is 10x10 = 100 pixels and the budget holds exactly one.
    renderer = SlowishRenderer(slow_key="block", slow_seconds=1.0)
    app = create_app(
        Settings(slot_count=2, retry_after_seconds=30.0, max_edge=256, max_pixels=100),
        renderer=renderer,
    )
    with TestClient(app) as limited_client:
        resp = limited_client.post(
            "/batches",
            files=[
                ("files", ("ready.png", png_with_key("ready", (10, 10)), "image/png")),
                ("files", ("block.png", png_with_key("block", (10, 10)), "image/png")),
            ],
        )
        assert resp.status_code == 201
        first_batch = resp.json()["batch_id"]
        jobs = {j["filename"]: j for j in resp.json()["jobs"]}
        ready_id = jobs["ready.png"]["job_id"]

        # "ready" finishes first; "block" then holds the whole budget.
        wait_until(
            lambda: limited_client.get(f"/jobs/{ready_id}").json()["result_ready"]
        )
        block_id = jobs["block.png"]["job_id"]
        wait_until(
            lambda: limited_client.get(f"/jobs/{block_id}").json()["state"]
            == JobState.RUNNING.value
        )
        assert limited_client.get("/stats").json()["pixels_used"] == 100

        upload_finished = threading.Event()

        def upload_more() -> None:
            limited_client.post(
                "/batches",
                files=[("files", ("wait.png", png_with_key("wait", (10, 10)), "image/png"))],
            )
            upload_finished.set()

        worker = threading.Thread(target=upload_more)
        worker.start()
        time.sleep(0.3)
        assert not upload_finished.is_set()  # its validation decode is waiting

        # Existing batches stay queryable and produced thumbnails downloadable.
        assert limited_client.get(f"/batches/{first_batch}").status_code == 200
        thumb = limited_client.get(f"/jobs/{ready_id}/thumbnail")
        assert thumb.status_code == 200
        assert thumb.headers["x-thumbnail-origin"] == "first"
        stats = limited_client.get("/stats").json()
        assert stats["max_pixels"] == 100
        assert stats["pixels_used"] == 100
        assert stats["pixels_free"] == 0

        worker.join(timeout=10)
        assert upload_finished.is_set()
        wait_until(lambda: limited_client.get("/stats").json()["pixels_used"] == 0)


def test_http_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}
