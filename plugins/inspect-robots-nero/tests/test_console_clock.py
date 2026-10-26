"""Tool-timeline clocks stay put when the page asks for the history again."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mission_console.py"
_SPEC = importlib.util.spec_from_file_location("mission_console_clock", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_CONSOLE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONSOLE)


def test_logged_tool_time_is_kept_on_the_feed_entry() -> None:
    message = _CONSOLE._compact_message(
        {
            "role": "assistant",
            "content": None,
            "at": "13:41:02",
            "tool_calls": [{"function": {"name": "move_to", "arguments": '{"left_x": 0.5}'}}],
        }
    )
    assert message is not None
    assert message["tools"][0]["at"] == "13:41:02"


def test_a_second_look_reuses_the_first_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    feed = _CONSOLE.RunFeed()
    ticks = iter(["13:41:02", "13:53:53"])
    monkeypatch.setattr(_CONSOLE.time, "strftime", lambda _fmt: next(ticks))
    assert feed._clock(7, None) == "13:41:02"
    assert feed._clock(7, None) == "13:41:02"
    assert feed._clock(7, "13:40:01") == "13:40:01"


def test_stop_keeps_the_last_frame_until_the_run_stores_one() -> None:
    pool = _CONSOLE.CameraPool({})
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    frame[0, 0] = (1, 2, 3)
    for camera in pool._cameras.values():
        camera.read = lambda max_age_s, kept=frame: (kept, 0.0)  # type: ignore[method-assign]
        camera.stop = lambda: None  # type: ignore[method-assign]
    pool._started = set(pool._cameras)

    pool.stop()
    held = pool.held_frame("chest_rgbd")

    assert held is not None
    assert tuple(held[0, 0]) == (1, 2, 3)
    assert pool.held_frame("missing") is None


def test_status_lamp_sits_left_of_the_cameras() -> None:
    assert 'id="signal"' in _CONSOLE._PAGE_HEAD
    assert 'id="signallabel"' in _CONSOLE._PAGE_HEAD
    assert _CONSOLE._PAGE_TAIL.startswith("</div>\n</div>\n</div>\n")


def test_guide_line_is_one_console_message() -> None:
    assert _CONSOLE.guide_line("  杯子在左边  先抬高  ") == "杯子在左边 先抬高"
    assert _CONSOLE.guide_line("x" * 300) == "x" * 240
    with pytest.raises(ValueError, match="non-empty"):
        _CONSOLE.guide_line("   ")
    with pytest.raises(ValueError, match="cannot start"):
        _CONSOLE.guide_line("/stop now")


def test_video_goal_starts_agent_and_harness_only() -> None:
    assert _CONSOLE.start_reads_video_goal("agent", "video")
    assert _CONSOLE.start_reads_video_goal("harness", "video")
    assert not _CONSOLE.start_reads_video_goal("harness", "speech")
    assert not _CONSOLE.start_reads_video_goal("vla", "video")
    assert not _CONSOLE.start_reads_video_goal("vla-origin", "video")


def test_overlay_replaces_only_the_direct_http_task() -> None:
    text = """\
    modes:
      direct_http:
        task: old task
      umi_replay_http:
        task: keep me
"""
    rendered = _CONSOLE.overlay_direct_http_task(text, 'please pick: the "red" cup')
    assert 'task: "please pick: the \\"red\\" cup"' in rendered
    assert "task: keep me" in rendered
    assert "old task" not in rendered
