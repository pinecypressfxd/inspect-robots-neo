"""D405 camera thread behavior against a scripted capture."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np
import pytest

from inspect_robots_nero._camera import D405Camera


class FakeCapture:
    def __init__(self, frames: list[np.ndarray], *, gate: Callable[[], None] | None = None) -> None:
        self.frames = list(frames)
        self.released = False
        self._gate = gate
        self._lock = threading.Lock()

    def read(self) -> tuple[bool, np.ndarray]:
        if self._gate is not None:
            self._gate()
        with self._lock:
            if self.frames:
                return True, self.frames.pop(0)
        return False, np.zeros((0,))


def test_read_returns_the_latest_frame_in_rgb() -> None:
    first = np.zeros((2, 2, 3), dtype=np.uint8)
    second = np.full((2, 2, 3), 7, dtype=np.uint8)
    camera = D405Camera("left_rgbd", "/dev/null", capture=FakeCapture([first, second]))
    camera.start()
    try:
        frame, stamp = camera.read(max_age_s=5.0)
        assert frame.shape == (2, 2, 3)
        assert stamp > 0.0
        assert bool((frame == 7).all())  # the *latest* frame wins
    finally:
        camera.stop()


def test_read_times_out_without_frames() -> None:
    camera = D405Camera(
        "left_rgbd", "/dev/null", capture=FakeCapture([]), first_frame_timeout_s=0.05
    )
    camera.start()
    try:
        with pytest.raises(TimeoutError, match="no frames"):
            camera.read(max_age_s=0.05)
    finally:
        camera.stop()


def test_first_frame_wait_uses_its_own_budget_beyond_max_age() -> None:
    # Real D405s deliver their first frame after ~0.7 s of USB warm-up, longer
    # than the 0.5 s freshness budget; the warm-up wait must not share it.
    now = [100.0]
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    class SlowWarmCapture:
        def read(self) -> tuple[bool, np.ndarray]:
            if now[0] >= 101.0:
                return True, frame
            now[0] += 0.05  # each poll advances the fake clock
            return False, np.zeros((0,))

    camera = D405Camera(
        "left_rgbd", "/dev/null", capture=SlowWarmCapture(), clock=lambda: now[0], poll_s=0.001
    )
    camera.start()
    try:
        got, _stamp = camera.read(max_age_s=0.5)
        assert got.shape == (2, 2, 3)
    finally:
        camera.stop()


def test_first_frame_timeout_s_must_be_positive() -> None:
    with pytest.raises(ValueError, match="first_frame_timeout_s"):
        D405Camera("left_rgbd", "/dev/null", first_frame_timeout_s=0.0)


def test_read_rejects_stale_frames_with_injected_clock() -> None:
    now = [100.0]
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    class FrozenCapture:
        def read(self) -> tuple[bool, np.ndarray]:
            now[0] += 10.0  # every read lands 10 s in the future-clock past
            return True, frame

    camera = D405Camera("left_rgbd", "/dev/null", capture=FrozenCapture(), clock=lambda: now[0])
    camera.start()
    try:
        now[0] += 2.0
        with pytest.raises(TimeoutError, match="stale"):
            camera.read(max_age_s=0.5)
    finally:
        camera.stop()


def test_stop_keeps_an_injected_capture_open() -> None:
    capture = FakeCapture([np.zeros((2, 2, 3), dtype=np.uint8)])
    camera = D405Camera("left_rgbd", "/dev/null", capture=capture)
    camera.start()
    time.sleep(0.05)
    camera.stop()
    # The camera did not open this capture, so releasing it stays the owner's job.
    assert capture.released is False
