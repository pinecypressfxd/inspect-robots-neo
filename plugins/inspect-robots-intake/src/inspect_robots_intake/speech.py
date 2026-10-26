"""Speech intake: local transcription, then a ChatGPT-6 rewrite into a VLA task.

The microphone capture itself stays outside this module. ``inspect-robots-voice``
remains the attended operator-feedback channel; this module consumes a finished
wav and returns the sentence the ``vla`` policy should execute.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol

from .contract import IntakeError, SpeechInput
from .video import CUP_OR_BOTTLE_VLA_PROMPTS, match_vla_prompt

SPEECH_SYSTEM = (
    "You rewrite one spoken robot command into a single imperative task sentence "
    "for a vision-language-action policy. Keep the objects, arms, and destinations "
    "the speaker named. Do not add steps, tools, or commentary. Reply with the "
    "task sentence only."
)


class Transcriber(Protocol):
    """Turn one wav into text. Empty audio returns an empty string."""

    def transcribe(self, speech: SpeechInput) -> str:
        """Return the recognized utterance."""
        ...


class TaskReasoner(Protocol):
    """ChatGPT-6 (or a test double) turns an utterance into a VLA task string."""

    def to_vla_task(self, utterance: str) -> str:
        """Return one task sentence."""
        ...


class _Segment(Protocol):
    text: str


class _WhisperModel(Protocol):
    def transcribe(
        self,
        audio: str,
        *,
        language: str,
        vad_filter: bool,
    ) -> tuple[Iterable[_Segment], object]:
        """Transcribe a wav path."""
        ...


ModelFactory = Callable[[], _WhisperModel]


SPEECH_CHOICE_SYSTEM = (
    "You hear one spoken request and choose the VLA task it asks for. "
    "Thirst, water, a mineral-water bottle, or a green bottle selects the "
    "water-bottle task. Examples: 我口渴, 拿一下矿泉水瓶, 拿一下绿色的瓶子. "
    "Coffee, being sleepy, or any drink that is not plain water selects the "
    "red-cup task. Examples: 我想喝咖啡, 我困了. Reply with exactly one "
    "allowed task sentence and no other words."
)


def speech_choice_messages(
    utterance: str, choices: Sequence[str] = CUP_OR_BOTTLE_VLA_PROMPTS
) -> list[dict[str, Any]]:
    """Chat messages that force the reply onto one allowed VLA task."""
    listed = "\n".join(choices)
    return [
        {"role": "system", "content": SPEECH_CHOICE_SYSTEM},
        {"role": "user", "content": f"Allowed VLA tasks:\n{listed}\n\nUtterance:\n{utterance}"},
    ]


def speech_messages(utterance: str) -> list[dict[str, str]]:
    """Chat-completions messages for the speech rewrite."""
    return [
        {"role": "system", "content": SPEECH_SYSTEM},
        {"role": "user", "content": utterance},
    ]


class FasterWhisperTranscriber:
    """faster-whisper adapter. The model is built on first use.

    ``model_factory`` exists so tests never download weights. Production code
    passes nothing and loads ``WhisperModel(model, device=..., compute_type=...)``.
    """

    def __init__(
        self,
        *,
        model: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "zh",
        model_factory: ModelFactory | None = None,
    ) -> None:
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._model_factory = model_factory
        self._model: _WhisperModel | None = None

    def transcribe(self, speech: SpeechInput) -> str:
        """Transcribe ``speech.path`` and drop surrounding whitespace."""
        segments, _info = self._load().transcribe(
            speech.path, language=self._language, vad_filter=True
        )
        return "".join(segment.text for segment in segments).strip()

    def _load(self) -> _WhisperModel:
        if self._model is None:
            self._model = (
                self._model_factory() if self._model_factory is not None else self._default_model()
            )
        return self._model

    def _default_model(self) -> _WhisperModel:
        from faster_whisper import WhisperModel

        return WhisperModel(self._model_name, device=self._device, compute_type=self._compute_type)


class MessageReasoner:
    """Task reasoner over any completer that accepts chat messages.

    The completer is the ChatGPT-6 call. This class only owns the speech prompt
    and rejects a blank rewrite.
    """

    def __init__(self, complete: Callable[[list[dict[str, str]]], str]) -> None:
        self._complete = complete

    def to_vla_task(self, utterance: str) -> str:
        """Rewrite ``utterance`` into one VLA task sentence."""
        text = utterance.strip()
        if not text:
            raise IntakeError("empty transcript")
        task = self._complete(speech_messages(text)).strip()
        if not task:
            raise IntakeError("empty vla task")
        return task


class CatalogSpeechReasoner:
    """ChatGPT-6 picks the water-bottle task or the red-cup task."""

    def __init__(
        self,
        complete: Callable[[list[dict[str, Any]]], str],
        choices: Sequence[str] = CUP_OR_BOTTLE_VLA_PROMPTS,
    ) -> None:
        self._complete = complete
        self._choices = tuple(choices)

    def to_vla_task(self, utterance: str) -> str:
        """Return the one allowed VLA task sentence for ``utterance``."""
        text = utterance.strip()
        if not text:
            raise IntakeError("empty transcript")
        reply = self._complete(speech_choice_messages(text, self._choices))
        return match_vla_prompt(reply, self._choices)
