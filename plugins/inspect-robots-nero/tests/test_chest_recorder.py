"""Chest takes are one MP4 each, sampled from the frame the console already has."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from inspect_robots_nero._chest_recorder import ChestRecorder


class _Stdin:
    def __init__(self) -> None:
        self.buffer = io.BytesIO()
        self.closed = False

    def write(self, data: bytes) -> int:
        return self.buffer.write(data)

    def close(self) -> None:
        self.closed = True


class _Process:
    def __init__(self, argv: list[str], **_kwargs: object) -> None:
        self.stdin = _Stdin()
        self.path = Path(argv[-1])

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.path.write_bytes(b"mp4")
        return 0

    def kill(self) -> None:
        return None


def test_start_writes_rgb_frames_and_stop_counts_the_take(tmp_path: Path) -> None:
    frame = np.zeros((2, 4, 3), dtype=np.uint8)
    frame[0, 0] = (1, 2, 3)
    clocks = iter([0.0, 0.0])
    recorder = ChestRecorder(
        tmp_path,
        fps=30.0,
        popen=_Process,  # type: ignore[arg-type]
        clock=lambda: next(clocks, 1.0),
    )
    assert recorder.start(lambda: frame) is None
    saved = recorder.stop()
    assert saved is not None
    assert Path(saved).is_file()
    status = recorder.status()
    assert status["recording"] is False
    assert status["saved"] == 1
    assert status["directory"] == str(tmp_path)


def test_second_start_is_refused_while_a_take_is_open(tmp_path: Path) -> None:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    recorder = ChestRecorder(tmp_path, popen=_Process, clock=lambda: 0.0)  # type: ignore[arg-type]
    assert recorder.start(lambda: frame) is None
    assert recorder.start(lambda: frame) == "chest recording is already running"
    recorder.stop()
