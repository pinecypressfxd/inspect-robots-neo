from pathlib import Path

import pytest

from inspect_robots_intake import (
    ExecutionRequest,
    IntakeError,
    IntakeRouter,
    SpeechInput,
    VideoInput,
    policy_kwargs,
)
from inspect_robots_intake.speech import speech_messages
from inspect_robots_intake.video import video_messages


class _Transcriber:
    def transcribe(self, speech: SpeechInput) -> str:
        return f"heard {speech.path}"


class _Blank:
    def transcribe(self, speech: SpeechInput) -> str:
        return "  "


class _Reasoner:
    def to_vla_task(self, utterance: str) -> str:
        return f"task: {utterance}"


class _Video:
    def to_prompt(self, video: VideoInput) -> str:
        return f"goal from {video.path}"


def _router(**overrides: object) -> IntakeRouter:
    parts: dict[str, object] = {
        "transcriber": _Transcriber(),
        "reasoner": _Reasoner(),
        "video_prompter": _Video(),
    }
    parts.update(overrides)
    return IntakeRouter(**parts)  # type: ignore[arg-type]


def test_speech_rewrites_then_requests_vla() -> None:
    request = _router().from_speech(SpeechInput("clip.wav"))
    assert request == ExecutionRequest(
        source="speech",
        policy="vla",
        prompt="task: heard clip.wav",
        model="gpt-6",
    )
    assert policy_kwargs(request) == {"prompt": "task: heard clip.wav"}


def test_video_requests_the_agent() -> None:
    request = _router().from_video(VideoInput("demo.mp4"))
    assert request.policy == "agent"
    assert request.prompt == "goal from demo.mp4"
    assert policy_kwargs(request) == {"prompt": "goal from demo.mp4", "model": "gpt-6"}


def test_catalog_video_requests_the_chosen_vla_act() -> None:
    request = _router().from_catalog_video(VideoInput("demo.mp4"), _Video())
    assert request.policy == "vla"
    assert request.prompt == "goal from demo.mp4"
    assert policy_kwargs(request) == {"prompt": "goal from demo.mp4"}


def test_speech_choice_maps_onto_one_vla_task() -> None:
    from inspect_robots_intake.speech import CatalogSpeechReasoner
    from inspect_robots_intake.video import WATER_BOTTLE_VLA_PROMPT

    reasoner = CatalogSpeechReasoner(lambda _messages: WATER_BOTTLE_VLA_PROMPT)
    request = _router().from_catalog_speech("我口渴", reasoner)
    assert request.policy == "vla"
    assert request.prompt == WATER_BOTTLE_VLA_PROMPT


def test_choice_reply_must_name_exactly_one_vla_task() -> None:
    from inspect_robots_intake.video import (
        CUP_OR_BOTTLE_VLA_PROMPTS,
        RED_CUP_VLA_PROMPT,
        WATER_BOTTLE_VLA_PROMPT,
        CatalogVideoPrompter,
        match_vla_prompt,
    )

    cup = RED_CUP_VLA_PROMPT
    bottle = WATER_BOTTLE_VLA_PROMPT
    assert match_vla_prompt(f"  {cup}  ", CUP_OR_BOTTLE_VLA_PROMPTS) == cup
    seen: list[object] = []

    def complete(messages: list[object]) -> str:
        seen.append(messages)
        return bottle

    prompter = CatalogVideoPrompter(complete, sample=lambda path: ["data:image/jpeg;base64,aa"])
    assert prompter.to_prompt(VideoInput("clip.mp4")) == bottle
    content = seen[0][1]["content"]  # type: ignore[index]
    assert any(part.get("type") == "image_url" for part in content)
    with pytest.raises(IntakeError, match="exactly one"):
        match_vla_prompt(f"{cup} {bottle}", CUP_OR_BOTTLE_VLA_PROMPTS)


def test_empty_transcript_is_rejected() -> None:
    with pytest.raises(IntakeError, match="empty transcript"):
        _router(transcriber=_Blank()).from_speech(SpeechInput("silence.wav"))


def test_agent_video_goal_shows_the_demonstration_frames() -> None:
    from inspect_robots_intake.video import VIDEO_SYSTEM, MessageVideoPrompter

    seen: list[object] = []

    def complete(messages: list[object]) -> str:
        seen.append(messages)
        return "  用左手把红色方块放进红色咖啡杯。  "

    prompter = MessageVideoPrompter(
        complete, sample=lambda path: ["data:image/jpeg;base64,aa"]
    )
    goal = prompter.to_prompt(VideoInput("clip.mp4"))

    assert goal == "用左手把红色方块放进红色咖啡杯。"
    assert seen[0][0]["content"] == VIDEO_SYSTEM  # type: ignore[index]
    content = seen[0][1]["content"]  # type: ignore[index]
    assert any(part.get("type") == "image_url" for part in content)
    assert "Allowed VLA" not in str(seen[0])


def test_agent_video_goal_rejects_an_empty_paragraph() -> None:
    from inspect_robots_intake.video import MessageVideoPrompter

    prompter = MessageVideoPrompter(
        lambda _messages: "   ", sample=lambda path: ["data:image/jpeg;base64,aa"]
    )
    with pytest.raises(IntakeError, match="empty video prompt"):
        prompter.to_prompt(VideoInput("clip.mp4"))


def test_goal_sampling_keeps_the_last_action(tmp_path: Path) -> None:
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required")
    clip = tmp_path / "demo.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=32x32:r=1:d=9",
            "-f",
            "lavfi",
            "-i",
            "color=c=white:s=32x32:r=1:d=1",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0",
            str(clip),
        ],
        check=True,
    )
    from inspect_robots_intake.video import GOAL_SAMPLE_COUNT, sample_goal_frames

    urls = sample_goal_frames(str(clip))
    assert len(urls) == GOAL_SAMPLE_COUNT
    last = tmp_path / "last.jpg"
    last.write_bytes(__import__("base64").b64decode(urls[-1].split(",", 1)[1]))
    gray = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(last),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    assert sum(gray) / len(gray) > 200


def test_video_system_keeps_later_actions() -> None:
    from inspect_robots_intake.video import VIDEO_SYSTEM

    assert "later action" in VIDEO_SYSTEM
    assert "in the order" in VIDEO_SYSTEM
    assert "ignore that hand" in VIDEO_SYSTEM
    assert "left arm's workspace" in VIDEO_SYSTEM


def test_message_builders_keep_the_source_text() -> None:
    speech = speech_messages("把杯子放到垫子上")
    video = video_messages(VideoInput("demo.mp4"))
    assert speech[1]["content"] == "把杯子放到垫子上"
    assert video[1]["content"] == "demo.mp4"
    assert "task sentence" in speech[0]["content"]
    assert "goal" in video[0]["content"]
