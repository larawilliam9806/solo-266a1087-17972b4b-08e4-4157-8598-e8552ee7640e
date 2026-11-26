import io
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List

import pytest
from PIL import Image, PngImagePlugin

from app.main import (
    AttemptKind,
    AttemptState,
    JobRejected,
    JobState,
    Settings,
    ThumbnailService,
    render_thumbnail,
)


def make_png(key: str, size=(40, 30), color=(10, 20, 30)) -> bytes:
    img = Image.new("RGB", size, color)
    info = PngImagePlugin.PngInfo()
    info.add_text("key", key)
    out = io.BytesIO()
    img.save(out, format="PNG", pnginfo=info)
    return out.getvalue()


@dataclass
class CallPlan:
    started: threading.Event = field(default_factory=threading.Event)
    gate: threading.Event = field(default_factory=threading.Event)
    fail: bool = False


class FakeRenderer:
    """Deterministic renderer driven by per-key call plans.

    Each key has a list of plans, one per attempt that reaches the renderer.
    The renderer signals ``started`` and then blocks on ``gate`` until the test
    releases it (or times out so a bug cannot hang the suite).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._plans: Dict[str, List[CallPlan]] = {}
        self._calls: Dict[str, int] = {}
        self.active = 0
        self.max_active = 0
        self.total_calls = 0

    def plan(self, key: str, n: int) -> List[CallPlan]:
        plans = [CallPlan() for _ in range(n)]
        with self._lock:
            self._plans[key] = plans
        return plans

    def call_count(self, key: str) -> int:
        with self._lock:
            return self._calls.get(key, 0)

    def set_plan_count_if_missing(self, key: str, n: int) -> None:
        with self._lock:
            self._plans.setdefault(key, [CallPlan() for _ in range(n)])

    def __call__(self, data: bytes, max_edge: int):
        with Image.open(io.BytesIO(data)) as img:
            key = img.text["key"]
        with self._lock:
            index = self._calls.get(key, 0)
            self._calls[key] = index + 1
            self.total_calls += 1
            plans = self._plans.setdefault(key, [])
            plans.extend(CallPlan() for _ in range(index + 1 - len(plans)))
            plan = plans[index]
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            plan.started.set()
            if not plan.gate.wait(timeout=15):
                raise TimeoutError(f"gate for {key}[{index}] never opened")
            if plan.fail:
                raise RuntimeError(f"render failed for {key}[{index}]")
            return b"thumb-" + key.encode(), (max_edge, max_edge)
        finally:
            with self._lock:
                self.active -= 1


def wait_until(predicate, timeout: float = 5.0, interval: float = 0.01):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    raise AssertionError("condition not met before timeout")


def make_service(renderer, slots=2, retry_after=10.0, max_pixels=50_000_000):
    return ThumbnailService(
        Settings(
            slot_count=slots,
            retry_after_seconds=retry_after,
            max_edge=256,
            max_pixels=max_pixels,
        ),
        renderer=renderer,
    )


def release_all(plans: Dict[str, List[CallPlan]]) -> None:
    for call_plans in plans.values():
        for plan in call_plans:
            plan.gate.set()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_render_scales_long_edge_to_256_preserving_aspect():
    png, size = render_thumbnail(make_png("x", size=(640, 480)), 256)
    assert size == (256, 192)
    with Image.open(io.BytesIO(png)) as img:
        assert img.format == "PNG"
        assert img.size == (256, 192)


def test_render_portrait_and_no_upscale():
    assert render_thumbnail(make_png("p", size=(300, 600)), 256)[1] == (128, 256)
    assert render_thumbnail(make_png("s", size=(100, 50)), 256)[1] == (100, 50)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_rejects_non_png_and_keeps_batch_atomic():
    renderer = FakeRenderer()
    service = make_service(renderer)
    try:
        with pytest.raises(JobRejected):
            service.create_batch([("ok.png", make_png("ok")), ("bad.png", b"not an image")])
        assert service.stats()["jobs"] == {}
    finally:
        service.shutdown()


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

def test_pending_images_are_dispatched_before_retries():
    """A freed slot goes to a never-started image, not to an overdue retry."""
    renderer = FakeRenderer()
    plans = {k: renderer.plan(k, 2) for k in ("slow", "second", "queued")}
    service = make_service(renderer, slots=2, retry_after=0.05)
    try:
        batch = service.create_batch(
            [("slow.png", make_png("slow")), ("second.png", make_png("second"))]
        )
        wait_until(lambda: plans["slow"][0].started.is_set() and plans["second"][0].started.is_set())
        assert service.stats()["in_flight"] == 2

        queued_batch = service.create_batch([("queued.png", make_png("queued"))])
        queued_id = queued_batch.job_ids[0]
        assert service.get_job(queued_id)["state"] == JobState.PENDING.value

        # "slow" becomes overdue, but no slot is free.
        time.sleep(0.2)
        assert renderer.call_count("slow") == 1

        # Freeing a slot must serve "queued" first even though "slow" is overdue.
        plans["second"][0].gate.set()
        wait_until(lambda: plans["queued"][0].started.is_set())
        assert renderer.call_count("slow") == 1
        assert service.get_job(queued_id)["state"] == JobState.RUNNING.value

        # Only once nothing is pending does "slow" get its one make-up run.
        plans["queued"][0].gate.set()
        wait_until(lambda: renderer.call_count("slow") == 2)
        slow_job = service.get_job(batch.job_ids[0])
        assert [a["kind"] for a in slow_job["attempts"]] == ["first", "retry"]
    finally:
        release_all(plans)
        service.shutdown()


def test_retry_result_downloadable_while_first_still_holds_slot():
    renderer = FakeRenderer()
    plans = {"a": renderer.plan("a", 2)}
    service = make_service(renderer, slots=3, retry_after=0.05)
    try:
        job_id = service.create_batch([("a.png", make_png("a"))]).job_ids[0]
        wait_until(lambda: plans["a"][0].started.is_set())
        wait_until(lambda: renderer.call_count("a") == 2)

        plans["a"][1].gate.set()  # make-up attempt succeeds, first still running
        wait_until(lambda: service.get_job(job_id)["state"] == JobState.SUCCEEDED.value)
        job = service.get_job(job_id)
        assert job["result_ready"] is True
        assert job["result"]["origin"] == AttemptKind.RETRY.value
        assert service.get_result(job_id)[1] == b"thumb-a"

        # The finished copy's slot stays occupied until the slow copy returns.
        assert service.stats()["in_flight"] == 1
        assert not plans["a"][0].gate.is_set()

        plans["a"][0].gate.set()
        wait_until(lambda: service.stats()["in_flight"] == 0)
        # A late success from the original must not replace the delivered result.
        assert service.get_job(job_id)["result"]["origin"] == AttemptKind.RETRY.value
    finally:
        release_all(plans)
        service.shutdown()


def test_first_result_wins_and_retry_still_holds_slot():
    renderer = FakeRenderer()
    plans = {"a": renderer.plan("a", 2)}
    service = make_service(renderer, slots=3, retry_after=0.05)
    try:
        job_id = service.create_batch([("a.png", make_png("a"))]).job_ids[0]
        wait_until(lambda: plans["a"][0].started.is_set())
        wait_until(lambda: renderer.call_count("a") == 2)

        plans["a"][0].gate.set()  # original succeeds first
        wait_until(lambda: service.get_job(job_id)["state"] == JobState.SUCCEEDED.value)
        assert service.get_job(job_id)["result"]["origin"] == AttemptKind.FIRST.value
        assert service.stats()["in_flight"] == 1  # retry copy still occupies a slot

        plans["a"][1].gate.set()
        wait_until(lambda: service.stats()["in_flight"] == 0)
        assert service.get_job(job_id)["result"]["origin"] == AttemptKind.FIRST.value
        assert service.get_result(job_id)[1] == b"thumb-a"
    finally:
        release_all(plans)
        service.shutdown()


def test_retried_job_fails_only_after_both_attempts_fail():
    renderer = FakeRenderer()
    plans = {"a": renderer.plan("a", 2)}
    service = make_service(renderer, slots=3, retry_after=0.05)
    try:
        job_id = service.create_batch([("a.png", make_png("a"))]).job_ids[0]
        wait_until(lambda: plans["a"][0].started.is_set())
        wait_until(lambda: renderer.call_count("a") == 2)

        plans["a"][1].fail = True
        plans["a"][1].gate.set()  # make-up fails while the original still runs
        wait_until(
            lambda: service.get_job(job_id)["attempts"][1]["state"] == AttemptState.FAILED.value
        )
        job = service.get_job(job_id)
        assert job["state"] != JobState.FAILED.value
        assert job["result_ready"] is False

        plans["a"][0].fail = True
        plans["a"][0].gate.set()  # original fails too -> terminal failure
        wait_until(lambda: service.get_job(job_id)["state"] == JobState.FAILED.value)
        job = service.get_job(job_id)
        assert "render failed for a[0]" in job["error"]
        assert "render failed for a[1]" in job["error"]
    finally:
        release_all(plans)
        service.shutdown()


def test_retry_is_dispatched_at_most_once():
    renderer = FakeRenderer()
    plans = {"a": renderer.plan("a", 3)}
    service = make_service(renderer, slots=3, retry_after=0.05)
    try:
        job_id = service.create_batch([("a.png", make_png("a"))]).job_ids[0]
        wait_until(lambda: renderer.call_count("a") == 2)

        plans["a"][1].fail = True
        plans["a"][1].gate.set()
        wait_until(
            lambda: service.get_job(job_id)["attempts"][1]["state"] == AttemptState.FAILED.value
        )

        # Original is still running and overdue, yet no third attempt appears.
        time.sleep(0.3)
        assert renderer.call_count("a") == 2
        assert len(service.get_job(job_id)["attempts"]) == 2

        plans["a"][0].gate.set()
        wait_until(lambda: service.stats()["in_flight"] == 0)
    finally:
        release_all(plans)
        service.shutdown()


def test_unretried_failure_reports_immediately():
    renderer = FakeRenderer()
    plans = {"a": renderer.plan("a", 1)}
    service = make_service(renderer, slots=2, retry_after=10.0)
    try:
        job_id = service.create_batch([("a.png", make_png("a"))]).job_ids[0]
        wait_until(lambda: plans["a"][0].started.is_set())
        plans["a"][0].fail = True
        plans["a"][0].gate.set()
        wait_until(lambda: service.get_job(job_id)["state"] == JobState.FAILED.value)
        assert len(service.get_job(job_id)["attempts"]) == 1
    finally:
        release_all(plans)
        service.shutdown()


def test_global_slot_limit_never_exceeded():
    renderer = FakeRenderer()
    keys = [f"k{i}" for i in range(6)]
    plans = {k: renderer.plan(k, 2) for k in keys}
    service = make_service(renderer, slots=2, retry_after=0.05)
    try:
        batch = service.create_batch([(f"{k}.png", make_png(k)) for k in keys])
        wait_until(lambda: renderer.active == 2)
        time.sleep(0.25)  # overdue-retry logic must not over-dispatch
        assert renderer.max_active <= 2
        assert service.stats()["in_flight"] <= 2

        release_all(plans)
        wait_until(
            lambda: service.get_batch(batch.batch_id)["finished"] == len(keys), timeout=10
        )
        assert renderer.max_active <= 2
        assert service.stats()["in_flight"] == 0
    finally:
        release_all(plans)
        service.shutdown()


# ---------------------------------------------------------------------------
# Global pixel budget
# ---------------------------------------------------------------------------

def test_single_image_over_budget_rejects_whole_batch():
    renderer = FakeRenderer()
    # One 10x10 image is 100 pixels; the limit only allows 99.
    service = make_service(renderer, slots=2, max_pixels=99)
    try:
        with pytest.raises(JobRejected) as exc_info:
            service.create_batch([("huge.png", make_png("huge", size=(10, 10)))])
        assert "huge.png" in str(exc_info.value)
        assert "pixel limit" in str(exc_info.value)
        assert service.stats()["jobs"] == {}
        assert service.stats()["pixels_used"] == 0

        # One bad apple rejects the entire batch, including the fitting file.
        with pytest.raises(JobRejected):
            service.create_batch(
                [
                    ("ok.png", make_png("ok", size=(3, 3))),
                    ("huge.png", make_png("huge", size=(10, 10))),
                ]
            )
        assert service.stats()["jobs"] == {}
        assert service.stats()["pixels_used"] == 0
        assert renderer.total_calls == 0
    finally:
        service.shutdown()


def test_validation_waits_for_pixel_budget_then_continues():
    renderer = FakeRenderer()
    plans = {
        "a": renderer.plan("a", 1),
        "b": renderer.plan("b", 1),
    }
    # Budget holds exactly one 100-pixel image at a time.
    service = make_service(renderer, slots=2, max_pixels=100)
    try:
        batch = service.create_batch([("a.png", make_png("a", size=(10, 10)))])
        a_id = batch.job_ids[0]
        wait_until(lambda: plans["a"][0].started.is_set())
        # The running first attempt holds all 100 pixels.
        stats = service.stats()
        assert stats["pixels_used"] == 100
        assert stats["pixels_free"] == 0

        # A second upload's validation decode must wait for the budget.
        upload_done = threading.Event()

        def upload() -> None:
            service.create_batch([("b.png", make_png("b", size=(10, 10)))])
            upload_done.set()

        worker = threading.Thread(target=upload)
        worker.start()
        time.sleep(0.3)
        assert not upload_done.is_set()
        # While validation waits, existing state stays queryable.
        assert service.get_batch(batch.batch_id)["total"] == 1
        assert service.get_job(a_id)["state"] == JobState.RUNNING.value

        # Releasing the first attempt frees the budget and unblocks validation.
        plans["a"][0].gate.set()
        wait_until(upload_done.is_set, timeout=5)
        # The newly registered job renders as soon as its reservation fits.
        wait_until(lambda: plans["b"][0].started.is_set(), timeout=5)
        plans["b"][0].gate.set()
        wait_until(
            lambda: service.stats()["jobs"].get(JobState.SUCCEEDED.value) == 2, timeout=5
        )
        assert service.stats()["pixels_used"] == 0
        worker.join(timeout=5)
    finally:
        release_all(plans)
        service.shutdown()


def test_pixel_budget_reserved_and_returned_on_success_and_failure():
    renderer = FakeRenderer()
    ok_plans = renderer.plan("ok", 1)
    bad_plans = renderer.plan("bad", 1)
    bad_plans[0].fail = True
    service = make_service(renderer, slots=2, max_pixels=10_000)
    try:
        service.create_batch(
            [
                ("ok.png", make_png("ok", size=(40, 30))),   # 1200 pixels
                ("bad.png", make_png("bad", size=(20, 10))),  # 200 pixels
            ]
        )
        wait_until(lambda: renderer.active == 2)
        assert service.stats()["pixels_used"] == 1400  # both attempts reserve

        release_all({"ok": ok_plans, "bad": bad_plans})
        wait_until(lambda: service.stats()["in_flight"] == 0)
        # A failed attempt returns its reservation just like a successful one.
        assert service.stats()["pixels_used"] == 0
        assert service.stats()["pixels_free"] == 10_000
    finally:
        release_all({"ok": ok_plans, "bad": bad_plans})
        service.shutdown()


def test_pending_job_dispatched_before_overdue_retry():
    """When a slot frees, a fitting never-started job beats an overdue retry.

    Budget 300, two slots.  ``big`` (100 px) and ``holder`` (90 px) occupy
    both slots; a 110 px pending job's validation fits (190 + 110 = 300) but
    it has no slot.  ``big`` grows overdue, yet its make-up copy cannot run
    either.  When the holder finishes, the pending job -- not the retry --
    gets the free slot.
    """
    renderer = FakeRenderer()
    plans = {
        "big": renderer.plan("big", 2),
        "holder": renderer.plan("holder", 1),
        "queued": renderer.plan("queued", 1),
    }
    service = make_service(renderer, slots=2, retry_after=0.05, max_pixels=300)
    try:
        first = service.create_batch(
            [
                ("big.png", make_png("big", size=(10, 10))),
                ("holder.png", make_png("holder", size=(9, 10))),
            ]
        )
        big_id = first.job_ids[0]
        wait_until(
            lambda: plans["big"][0].started.is_set()
            and plans["holder"][0].started.is_set()
        )
        assert service.stats()["pixels_used"] == 190

        second = service.create_batch([("queued.png", make_png("queued", size=(11, 10)))])
        queued_id = second.job_ids[0]
        assert service.get_job(queued_id)["pixels"] == 110
        time.sleep(0.25)
        assert service.get_job(queued_id)["state"] == JobState.PENDING.value
        assert renderer.call_count("big") == 1  # overdue, but no slot for retry

        # Holder frees a slot: the pending job wins over the overdue retry.
        plans["holder"][0].gate.set()
        wait_until(lambda: plans["queued"][0].started.is_set())
        assert service.get_job(queued_id)["state"] == JobState.RUNNING.value
        assert renderer.call_count("big") == 1

        # Ending the first attempt successfully terminates the job, so no
        # make-up copy can be dispatched when its slot frees; then the queued
        # job finishes.  Order matters: big first, queued second.
        plans["big"][0].gate.set()
        wait_until(lambda: service.get_job(big_id)["state"] == JobState.SUCCEEDED.value)
        plans["queued"][0].gate.set()
        wait_until(lambda: service.stats()["in_flight"] == 0)
        assert service.stats()["pixels_used"] == 0
        assert renderer.call_count("big") == 1
    finally:
        release_all(plans)
        service.shutdown()


def test_retry_gets_free_slot_when_no_pending_job_fits():
    """With no pending job left, an overdue fitting retry takes the free slot."""
    renderer = FakeRenderer()
    plans = {
        "big": renderer.plan("big", 2),
        "holder": renderer.plan("holder", 1),
    }
    service = make_service(renderer, slots=2, retry_after=0.05, max_pixels=200)
    try:
        batch = service.create_batch(
            [
                ("big.png", make_png("big", size=(10, 10))),
                ("holder.png", make_png("holder", size=(10, 10))),
            ]
        )
        big_id = batch.job_ids[0]
        wait_until(
            lambda: plans["big"][0].started.is_set()
            and plans["holder"][0].started.is_set()
        )
        time.sleep(0.3)  # big is overdue, but both slots are taken
        assert renderer.call_count("big") == 1

        plans["holder"][0].gate.set()
        wait_until(lambda: renderer.call_count("big") == 2)
        job = service.get_job(big_id)
        assert job["state"] == JobState.RETRYING.value
        assert [a["kind"] for a in job["attempts"]] == ["first", "retry"]
        assert service.stats()["pixels_used"] == 200

        plans["big"][0].gate.set()
        plans["big"][1].gate.set()
        wait_until(lambda: service.stats()["in_flight"] == 0)
        assert service.stats()["pixels_used"] == 0
    finally:
        release_all(plans)
        service.shutdown()


def test_retry_blocked_by_pixel_budget_until_room_frees():
    renderer = FakeRenderer()
    plans = {
        "big": renderer.plan("big", 2),
        "holder": renderer.plan("holder", 1),
    }
    # Budget 250: big = 100 px, holder = 150 px (15x10).  Together they fill
    # the budget, so the overdue make-up copy (another 100 px) is blocked
    # until the holder finishes and returns its reservation.
    service = make_service(renderer, slots=3, retry_after=0.05, max_pixels=250)
    try:
        first = service.create_batch([("big.png", make_png("big", size=(10, 10)))])
        big_id = first.job_ids[0]
        wait_until(lambda: plans["big"][0].started.is_set())

        holder_batch = service.create_batch(
            [("holder.png", make_png("holder", size=(15, 10)))]
        )
        holder_id = holder_batch.job_ids[0]
        wait_until(lambda: plans["holder"][0].started.is_set())
        assert service.stats()["pixels_used"] == 250

        # The holder stays blocked on its gate while "big" grows well past the
        # retry threshold: overdue, free slot, but no 100 px available.
        time.sleep(0.3)
        assert renderer.call_count("big") == 1
        assert service.get_job(big_id)["state"] == JobState.RUNNING.value

        # Releasing the holder returns 150 px and the retry starts.
        plans["holder"][0].gate.set()
        wait_until(
            lambda: service.get_job(holder_id)["state"] == JobState.SUCCEEDED.value
        )
        wait_until(lambda: renderer.call_count("big") == 2)
        retry_attempt = service.get_job(big_id)["attempts"][1]
        assert retry_attempt["kind"] == AttemptKind.RETRY.value
        assert retry_attempt["pixels"] == 100

        plans["big"][0].gate.set()
        plans["big"][1].gate.set()
        wait_until(lambda: service.stats()["in_flight"] == 0)
        assert service.stats()["pixels_used"] == 0
    finally:
        release_all(plans)
        service.shutdown()


def test_retry_runs_when_budget_fits_two_copies():
    renderer = FakeRenderer()
    plans = {"a": renderer.plan("a", 2)}
    service = make_service(renderer, slots=3, retry_after=0.05, max_pixels=200)
    try:
        job_id = service.create_batch([("a.png", make_png("a", size=(10, 10)))]).job_ids[0]
        wait_until(lambda: renderer.call_count("a") == 2)
        # Both copies hold reservations at once.
        assert service.stats()["pixels_used"] == 200
        job = service.get_job(job_id)
        assert [a["kind"] for a in job["attempts"]] == ["first", "retry"]
        release_all(plans)
        wait_until(lambda: service.stats()["in_flight"] == 0)
        assert service.stats()["pixels_used"] == 0
    finally:
        release_all(plans)
        service.shutdown()


def test_shutdown_unblocks_validation_waiting_for_budget():
    renderer = FakeRenderer()
    plans = {"a": renderer.plan("a", 1)}
    service = make_service(renderer, slots=2, max_pixels=100)
    try:
        service.create_batch([("a.png", make_png("a", size=(10, 10)))])
        wait_until(lambda: plans["a"][0].started.is_set())
        assert service.stats()["pixels_used"] == 100

        error: list = []

        def upload() -> None:
            try:
                service.create_batch([("b.png", make_png("b", size=(10, 10)))])
            except RuntimeError as exc:
                error.append(exc)

        worker = threading.Thread(target=upload)
        worker.start()
        time.sleep(0.2)
        assert worker.is_alive()

        service.shutdown()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert error and "shutting down" in str(error[0])
        # Prevent the still-gated render from raising a 15 s gate timeout.
        release_all(plans)
    finally:
        release_all(plans)
        service.shutdown()


def test_stats_reports_pixel_budget_limit_and_usage():
    renderer = FakeRenderer()
    plans = {"a": renderer.plan("a", 1)}
    service = make_service(renderer, slots=2, max_pixels=321)
    try:
        stats = service.stats()
        assert stats["max_pixels"] == 321
        assert stats["pixels_used"] == 0
        assert stats["pixels_free"] == 321

        service.create_batch([("a.png", make_png("a", size=(4, 4)))])
        wait_until(lambda: plans["a"][0].started.is_set())
        stats = service.stats()
        assert stats["max_pixels"] == 321
        assert stats["pixels_used"] == 16
        assert stats["pixels_free"] == 305

        plans["a"][0].gate.set()
        wait_until(lambda: service.stats()["pixels_used"] == 0)
    finally:
        release_all(plans)
        service.shutdown()
