"""Shared objects for the two intake paths.

Speech and video both end as an :class:`ExecutionRequest`. Downstream policies
already exist: ``vla`` runs one task string, and ``agent`` runs the open goal
from a demonstration. This package does not own arm motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SourceKind = Literal["speech", "video"]
PolicyName = Literal["vla", "agent"]


class IntakeError(ValueError):
    """The intake path could not produce an execution request."""


@dataclass(frozen=True)
class SpeechInput:
    """One microphone clip already written to a wav file."""

    path: str
    sample_rate: int = 16_000


@dataclass(frozen=True)
class VideoInput:
    """One demonstration or observation video on disk."""

    path: str


@dataclass(frozen=True)
class ExecutionRequest:
    """What the existing policy registry should run.

    ``policy="vla"`` is the speech path: ChatGPT-6 has already rewritten the
    utterance into one VLA task string. ``policy="agent"`` is the video path:
    the prompt is the goal the agent executes.
    """

    source: SourceKind
    policy: PolicyName
    prompt: str
    model: str


def policy_kwargs(request: ExecutionRequest) -> dict[str, str]:
    """Constructor kwargs for ``resolve("policy", request.policy, **kwargs)``."""
    if request.policy == "vla":
        return {"prompt": request.prompt}
    return {"prompt": request.prompt, "model": request.model}
