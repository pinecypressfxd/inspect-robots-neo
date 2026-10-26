"""Route the two intake sources onto the policies that already execute them."""

from __future__ import annotations

from dataclasses import dataclass

from .contract import ExecutionRequest, IntakeError, SpeechInput, VideoInput
from .speech import TaskReasoner, Transcriber
from .video import VideoPrompter

DEFAULT_MODEL = "gpt-6"


@dataclass
class IntakeRouter:
    """Speech goes to ``vla`` after a rewrite. Video goes to ``agent``.

    The agent reads the video goal and acts. Speech does not enter that tool
    loop; ChatGPT-6 has already produced the VLA task string.
    """

    transcriber: Transcriber
    reasoner: TaskReasoner
    video_prompter: VideoPrompter
    model: str = DEFAULT_MODEL

    def from_speech(self, speech: SpeechInput) -> ExecutionRequest:
        """Transcribe, rewrite, and request a direct VLA run."""
        utterance = self.transcriber.transcribe(speech).strip()
        if not utterance:
            raise IntakeError("empty transcript")
        task = self.reasoner.to_vla_task(utterance)
        return ExecutionRequest(
            source="speech",
            policy="vla",
            prompt=task,
            model=self.model,
        )

    def from_video(self, video: VideoInput) -> ExecutionRequest:
        """Turn the clip into a goal and request an agent run."""
        prompt = self.video_prompter.to_prompt(video)
        return ExecutionRequest(
            source="video",
            policy="agent",
            prompt=prompt,
            model=self.model,
        )

    def from_catalog_speech(self, utterance: str, reasoner: TaskReasoner) -> ExecutionRequest:
        """Pick one trained VLA task sentence from speech and request that VLA act."""
        prompt = reasoner.to_vla_task(utterance)
        return ExecutionRequest(
            source="speech",
            policy="vla",
            prompt=prompt,
            model=self.model,
        )

    def from_catalog_video(self, video: VideoInput, prompter: VideoPrompter) -> ExecutionRequest:
        """Pick one trained VLA task sentence and request that VLA act.

        Unlike :meth:`from_video`, the sentence is already the checkpoint's
        task string, so it goes to ``vla`` rather than the agent.
        """
        prompt = prompter.to_prompt(video)
        return ExecutionRequest(
            source="video",
            policy="vla",
            prompt=prompt,
            model=self.model,
        )
