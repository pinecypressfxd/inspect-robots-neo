"""Speech and video front doors for an existing VLA or agent run.

Neither source is a new policy. :class:`IntakeRouter` returns an
:class:`ExecutionRequest` whose ``policy`` name is already registered:
``vla`` for speech, ``agent`` for video.
"""

from __future__ import annotations

from .capture import VideoTakeRecorder
from .contract import (
    ExecutionRequest,
    IntakeError,
    SpeechInput,
    VideoInput,
    policy_kwargs,
)
from .router import DEFAULT_MODEL, IntakeRouter
from .speech import FasterWhisperTranscriber, MessageReasoner
from .video import MessageVideoPrompter

__all__ = [
    "DEFAULT_MODEL",
    "ExecutionRequest",
    "FasterWhisperTranscriber",
    "IntakeError",
    "IntakeRouter",
    "MessageReasoner",
    "MessageVideoPrompter",
    "SpeechInput",
    "VideoInput",
    "VideoTakeRecorder",
    "policy_kwargs",
]

__version__ = "0.1.0"
