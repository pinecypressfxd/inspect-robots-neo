"""A chest take is a VideoInput the existing video-to-prompt route can consume."""

from __future__ import annotations

import io
from pathlib import Path

from inspect_robots_intake import IntakeRouter, SpeechInput, VideoInput
from inspect_robots_intake.capture import VideoTakeRecorder
from inspect_robots_intake.contract import IntakeError


class _Stdin:
    def __init__(self) -> None:
        self.buffer = io.BytesIO()

    def write(self, data: bytes) -> int:
        return self.buffer.write(data)

    def close(self) -> None:
        return None


class _Process:
    def __init__(self, argv: list[str], **_kwargs: object) -> None:
        self.stdin = _Stdin()
        self.path = Path(argv[-1])
        self.written = self.stdin

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.path.write_bytes(b"mp4")
        return 0

    def kill(self) -> None:
        return None


class _Transcriber:
    def transcribe(self, speech: SpeechInput) -> str:
        del speech
        return "unused"


class _Reasoner:
    def to_vla_task(self, utterance: str) -> str:
        return utterance


class _Frame:
    shape = (2, 4, 3)

    def tobytes(self) -> bytes:
        return bytes(2 * 4 * 3)


class _Video:
    def to_prompt(self, video: VideoInput) -> str:
        return f"goal from {video.path}"


def _recorder(tmp_path: Path) -> VideoTakeRecorder:
    return VideoTakeRecorder(
        tmp_path,
        fps=30.0,
        popen=_Process,  # type: ignore[arg-type]
        clock=lambda: 0.0,
    )


def test_stop_returns_the_video_input_the_router_already_accepts(tmp_path: Path) -> None:
    frame = _Frame()
    recorder = _recorder(tmp_path)
    recorder.start(lambda: frame)
    video = recorder.stop()
    request = IntakeRouter(_Transcriber(), _Reasoner(), _Video()).from_video(video)
    assert video.path.endswith(".mp4")
    assert request.policy == "agent"
    assert request.prompt == f"goal from {video.path}"
    assert recorder.status()["saved"] == 1
    assert recorder.status()["recording"] is False


def test_a_second_take_cannot_start_while_one_is_open(tmp_path: Path) -> None:
    frame = _Frame()
    recorder = _recorder(tmp_path)
    recorder.start(lambda: frame)
    try:
        recorder.start(lambda: frame)
    except IntakeError as exc:
        assert "already running" in str(exc)
    else:
        raise AssertionError("second start should be refused")
    recorder.stop()
