"""Video intake: a clip becomes the goal prompt an agent executes.

:class:`inspect_robots_intake.capture.VideoTakeRecorder` writes that clip from
chest frames. This module starts from the finished file.

ChatGPT-6 sees sampled frames and writes the goal. The agent then executes
that paragraph. A separate catalog path picks one trained VLA sentence. This
module stops at the prompt; it does not plan motion.
"""

from __future__ import annotations

import base64
import subprocess
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from .contract import IntakeError, VideoInput

WATER_BOTTLE_VLA_PROMPT = "place the water bottle on the black pad"
RED_CUP_VLA_PROMPT = "place the red cup on the black pad"
CUP_OR_BOTTLE_VLA_PROMPTS = (WATER_BOTTLE_VLA_PROMPT, RED_CUP_VLA_PROMPT)

VIDEO_SYSTEM = (
    "You watch sampled frames from one chest-camera demonstration, in time "
    "order. The demonstration may contain several actions. Write every action "
    "you can see, in the order the frames show them. Do not stop after the "
    "first action, and do not drop a later action. For each action, name the "
    "object, which robot arm acts, and the destination. Reply with one goal "
    "paragraph and no tool calls. "
    "The chest camera looks at the table from the robot. The robot's left arm "
    "enters from the left side of the image, and the right arm from the right "
    "side. A person may demonstrate with either hand; ignore that hand. "
    "Choose the robot arm from the task space. Objects on the left half of "
    "the table, and the center when it sits with them, use the left arm. "
    "Most of this table is the left arm's workspace. Use the right arm only "
    "when the object and its destination are on the right half, beside the "
    "arm that enters from the right of the image. Write \"left arm\" or "
    "\"right arm\"."
)
#: Frames sent to the open goal writer. They are spread across the whole clip.
GOAL_SAMPLE_COUNT = 6


class VideoPrompter(Protocol):
    """Turn one video into the agent goal."""

    def to_prompt(self, video: VideoInput) -> str:
        """Return the goal prompt."""
        ...


def video_messages(video: VideoInput) -> list[dict[str, str]]:
    """Path-only messages. The agent goal path sends frames instead."""
    return [
        {"role": "system", "content": VIDEO_SYSTEM},
        {"role": "user", "content": video.path},
    ]


def goal_messages(frame_urls: Sequence[str]) -> list[dict[str, Any]]:
    """Chat messages that show a demonstration and ask for one goal paragraph."""
    if not frame_urls:
        raise IntakeError("video goal needs at least one frame")
    parts: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Chest-camera frames in time order, from the start of the "
                "demonstration to the end. Write every action in that order. "
                "Name the robot arm from where the object sits, not from the "
                "demonstrator's hand."
            ),
        }
    ]
    for index, url in enumerate(frame_urls, start=1):
        parts.append({"type": "text", "text": f"frame {index}"})
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return [
        {"role": "system", "content": VIDEO_SYSTEM},
        {"role": "user", "content": parts},
    ]


class MessageVideoPrompter:
    """Ask ChatGPT-6 for the open goal paragraph of one demonstration."""

    def __init__(
        self,
        complete: Callable[[list[dict[str, Any]]], str],
        sample: Callable[[str], list[str]] | None = None,
    ) -> None:
        self._complete = complete
        self._sample = sample_goal_frames if sample is None else sample

    def to_prompt(self, video: VideoInput) -> str:
        """Return the goal paragraph for ``video``."""
        frames = self._sample(video.path)
        prompt = self._complete(goal_messages(frames)).strip()
        if not prompt:
            raise IntakeError("empty video prompt")
        return prompt


CHOICE_SYSTEM = (
    "You see sampled frames from one chest-camera demonstration. The hand "
    "picks up either the water bottle or the red cup and places it on the "
    "black pad. A bottle, including a green bottle, is the water-bottle task. "
    "A cup, including a coffee cup, is the red-cup task. Reply with exactly "
    "one allowed task sentence and no other words."
)


def choice_messages(
    frame_urls: Sequence[str], choices: Sequence[str]
) -> list[dict[str, Any]]:
    """Chat messages that show the frames and the allowed VLA task sentences."""
    if not frame_urls:
        raise IntakeError("video choice needs at least one frame")
    if len(choices) < 2:
        raise IntakeError("video choice needs at least two VLA task sentences")
    listed = "\n".join(choices)
    parts: list[dict[str, Any]] = [
        {"type": "text", "text": f"Allowed VLA tasks:\n{listed}"}
    ]
    for index, url in enumerate(frame_urls, start=1):
        parts.append({"type": "text", "text": f"frame {index}"})
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return [
        {"role": "system", "content": CHOICE_SYSTEM},
        {"role": "user", "content": parts},
    ]


def match_vla_prompt(reply: str, choices: Sequence[str]) -> str:
    """Return the one allowed sentence contained in ``reply``."""
    text = " ".join(reply.strip().split())
    exact = [choice for choice in choices if text == choice]
    if len(exact) == 1:
        return exact[0]
    contained = [choice for choice in choices if choice in text]
    if len(contained) == 1:
        return contained[0]
    raise IntakeError(f"reply did not choose exactly one VLA task: {reply!r}")


def sample_goal_frames(path: str) -> list[str]:
    """Sample the open-goal frames across the whole demonstration."""
    return sample_jpeg_data_urls(path, count=GOAL_SAMPLE_COUNT)


def sample_jpeg_data_urls(
    path: str, *, count: int = 3, ffmpeg: str = "ffmpeg"
) -> list[str]:
    """Sample up to ``count`` JPEGs spread across the clip, as data URLs."""
    if count < 1:
        raise IntakeError("frame count must be positive")
    try:
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                path,
                "-vf",
                "fps=1,scale=320:-2",
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "-",
            ],
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise IntakeError(f"could not run ffmpeg on {path}: {exc}") from exc
    frames = _split_jpegs(completed.stdout)
    if not frames:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise IntakeError(f"no frames decoded from {path}: {detail or 'empty ffmpeg output'}")
    chosen = _spread(frames, count)
    return ["data:image/jpeg;base64," + base64.b64encode(frame).decode("ascii") for frame in chosen]


def _split_jpegs(blob: bytes) -> list[bytes]:
    frames: list[bytes] = []
    start = 0
    while True:
        begin = blob.find(b"\xff\xd8", start)
        if begin < 0:
            return frames
        end = blob.find(b"\xff\xd9", begin + 2)
        if end < 0:
            return frames
        frames.append(blob[begin : end + 2])
        start = end + 2


def _spread(frames: Sequence[bytes], count: int) -> list[bytes]:
    if len(frames) <= count:
        return list(frames)
    if count == 1:
        return [frames[len(frames) // 2]]
    last = len(frames) - 1
    indexes = [round(index * last / (count - 1)) for index in range(count)]
    return [frames[index] for index in indexes]


class CatalogVideoPrompter:
    """Ask ChatGPT-6 to pick one allowed VLA task sentence for a clip."""

    def __init__(
        self,
        complete: Callable[[list[dict[str, Any]]], str],
        choices: Sequence[str] = CUP_OR_BOTTLE_VLA_PROMPTS,
        sample: Callable[[str], list[str]] = sample_jpeg_data_urls,
    ) -> None:
        self._complete = complete
        self._choices = tuple(choices)
        self._sample = sample

    def to_prompt(self, video: VideoInput) -> str:
        """Return the one allowed VLA task sentence for ``video``."""
        frames = self._sample(video.path)
        reply = self._complete(choice_messages(frames, self._choices))
        return match_vla_prompt(reply, self._choices)
