"""Wire replay HTML rendering and the ``inspect --replay`` CLI surface."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from inspect_robots._wire_replay import render_wire_replay
from inspect_robots.cli import main
from inspect_robots.log import EvalLog, EvalResults, EvalSpec, EvalStats, SceneResult

_PNG = b"\x89PNG\r\n\x1a\nreplay-frame"
_PNG_SHA = hashlib.sha256(_PNG).hexdigest()
_DATA_URL = f"data:image/png;base64,{base64.b64encode(_PNG).decode('ascii')}"


def _chat_row(**overrides: Any) -> dict[str, Any]:
    """Build one well-formed OpenAI-style captured call."""
    row: dict[str, Any] = {
        "call": 0,
        "attempt": 0,
        "endpoint": "/v1/chat/completions",
        "duration_s": 0.5,
        "status": 200,
        "request": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,$blob:{_PNG_SHA}"},
                        },
                        {"type": "text", "text": "move <fast>"},
                    ],
                }
            ]
        },
        "response": {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "moving",
                        "reasoning_content": "plan: <grip> first",
                        "tool_calls": [
                            {"function": {"name": "move_by", "arguments": '{"d": [0.1, 0, 0]}'}}
                        ],
                    }
                }
            ]
        },
    }
    row.update(overrides)
    return row


def _capture(tmp_path: Path) -> Path:
    """Create one capture directory holding the single deduplicated blob."""
    capture = tmp_path / "wire" / "run-1"
    blobs = capture / "blobs"
    blobs.mkdir(parents=True)
    (blobs / f"{_PNG_SHA}.png").write_bytes(_PNG)
    return capture


def _write_calls(capture: Path, trial: str, lines: list[str]) -> Path:
    """Write one trial's calls.jsonl verbatim from pre-rendered lines."""
    trial_dir = capture / trial
    trial_dir.mkdir(parents=True)
    calls = trial_dir / "calls.jsonl"
    calls.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return calls


def test_render_wire_replay_shows_new_images_calls_and_badges(tmp_path: Path) -> None:
    """Two calls, a duplicate blob, a response-less call, and a torn line render."""
    capture = _capture(tmp_path)
    _write_calls(
        capture,
        "scene-0-e0",
        [
            json.dumps(_chat_row()),
            json.dumps(_chat_row(call=1, duration_s=None, status=None, response=None)),
            "not-json{",
            "[1, 2]",
        ],
    )

    document = render_wire_replay(capture)

    assert document.startswith("<!doctype html>")
    assert document.count(_DATA_URL) == 1  # duplicate reference embeds no second image
    assert 'class="badge ok">200<' in document
    assert ">0.500s<" in document
    assert 'class="badge none">no response<' in document
    assert document.count('<article class="call">') == 2  # malformed lines skipped
    assert document.index("call 0") < document.index("call 1")  # newest last
    assert "scene-0-e0" in document and "2 calls" in document
    assert "move_by" in document
    assert "{\n  &quot;d&quot;: [" in document  # pretty-printed JSON arguments
    assert "plan: &lt;grip&gt; first" in document  # reasoning, escaped once
    assert "<grip>" not in document and "<fast>" not in document
    assert "moving" in document


def test_render_wire_replay_skips_missing_blobs_and_degrades_foreign_rows(
    tmp_path: Path,
) -> None:
    """Missing blobs, hostile response shapes, and unreadable trials never crash."""
    capture = _capture(tmp_path)
    absent = "b" * 64
    anthropic = {
        "call": 0,
        "endpoint": "/v1/messages",
        "duration_s": 1.0,
        "status": 429,
        "request": {"messages": [{"content": [f"$blob:{absent}"]}]},
        "response": {
            "content": [
                42,
                {"type": "thinking", "thinking": "grip <now>"},
                {"type": "thinking", "thinking": None},
                {"type": "tool_use", "name": "grip", "input": {"close": True}},
                {"type": "text", "text": "done"},
                {"type": "text", "text": None},
            ]
        },
    }
    raw_body = {
        "response": "gateway <panic>",
    }
    corrupt_message = {"call": 2, "response": {"choices": [{"message": "corrupt"}]}}
    malformed_calls = {
        "call": 3,
        "response": {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            42,
                            {"function": 5},
                            {"function": {"name": "say", "arguments": "later{"}},
                        ],
                    }
                }
            ]
        },
    }
    text_only = {"call": 4, "response": {"choices": [{"message": {"content": "hi there"}}]}}
    _write_calls(
        capture,
        "scene-1-e0",
        [
            json.dumps(row)
            for row in (anthropic, raw_body, corrupt_message, malformed_calls, text_only)
        ],
    )
    # A calls.jsonl that cannot be read (here: a directory) is skipped, not fatal.
    (capture / "scene-2-e0").mkdir()
    (capture / "scene-2-e0" / "calls.jsonl").mkdir()

    document = render_wire_replay(capture)

    assert 'class="badge error">429<' in document
    assert document.count("<img") == 0  # missing blob file skips the image
    assert "grip" in document
    assert "grip &lt;now&gt;" in document and "<now>" not in document
    assert "{\n  &quot;close&quot;: true\n}" in document
    assert "done" in document
    assert "gateway &lt;panic&gt;" in document and "<panic>" not in document
    assert "later{" in document  # unparsable arguments render raw
    assert "hi there" in document  # a message without tool calls still renders text
    assert document.count('<article class="call">') == 5
    assert "scene-2-e0" not in document


def test_render_wire_replay_escapes_hostile_header_values(tmp_path: Path) -> None:
    """Hostile call/attempt strings render escaped, never as live markup."""
    capture = _capture(tmp_path)
    hostile: dict[str, Any] = {
        "call": "</script><script>alert(1)</script>",
        "attempt": '<img src=x onerror="alert(2)">',
        "request": {},
        "response": None,
    }
    _write_calls(capture, "scene-0-e0", [json.dumps(hostile)])

    document = render_wire_replay(capture)

    assert document.count('<article class="call">') == 1
    assert "</script>" not in document and "<script>" not in document
    assert "<img" not in document
    assert "&lt;/script&gt;&lt;script&gt;alert(1)&lt;/script&gt;" in document
    assert "&lt;img src=x onerror=&quot;alert(2)&quot;&gt;" in document


def test_render_wire_replay_notes_capture_without_readable_calls(tmp_path: Path) -> None:
    """A capture directory with no trials renders a noting page, never a crash."""
    document = render_wire_replay(tmp_path / "wire" / "absent")

    assert document.startswith("<!doctype html>")
    assert "no readable wire calls" in document


def _replay_log(trial_metadata: tuple[dict[str, Any], ...]) -> EvalLog:
    """Build one minimal saved log whose single scene points at wire captures."""
    return EvalLog(
        version=1,
        status="success",
        eval=EvalSpec(
            task="agent-run",
            policy="agent",
            embodiment="e",
            created="x",
            inspect_robots_version="0",
        ),
        results=EvalResults(total_scenes=1, total_trials=len(trial_metadata), metrics={}),
        stats=EvalStats(started_at="a", completed_at="b", duration_s=0.0, total_steps=1),
        samples=(
            SceneResult(
                scene_id="s0",
                status="success",
                epochs=tuple({} for _ in trial_metadata),
                termination_reasons=tuple("success" for _ in trial_metadata),
                trial_metadata=trial_metadata,
            ),
        ),
    )


def _write_log_file(tmp_path: Path, log: EvalLog) -> Path:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(log.to_dict()), encoding="utf-8")
    return path


def _write_cli_capture(tmp_path: Path, trials: tuple[str, ...] = ("s0-e0",)) -> None:
    capture = tmp_path / "wire" / "run"
    blobs = capture / "blobs"
    blobs.mkdir(parents=True)
    (blobs / f"{_PNG_SHA}.png").write_bytes(_PNG)
    for trial in trials:
        calls = capture / trial / "calls.jsonl"
        calls.parent.mkdir(parents=True)
        calls.write_text(f"{json.dumps(_chat_row())}\n", encoding="utf-8")


def test_inspect_replay_writes_one_page_with_a_section_per_trial(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two trials of one run render into one page with two trial sections."""
    _write_cli_capture(tmp_path, ("s0-e0", "s0-e1"))
    path = _write_log_file(
        tmp_path,
        _replay_log(
            (
                {"wire_capture": "wire/run/s0-e0/calls.jsonl"},
                {"wire_capture": "wire/run/s0-e1/calls.jsonl"},
            )
        ),
    )

    assert main(["inspect", str(path), "--replay"]) == 0

    assert f"wrote {tmp_path / 'wire-replay.html'}" in capsys.readouterr().out
    document = (tmp_path / "wire-replay.html").read_text(encoding="utf-8")
    assert document.count('class="capture"') == 1  # shared run dir, one capture section
    assert "s0-e0" in document and "s0-e1" in document
    assert document.count(_DATA_URL) == 2  # dedup is per trial file


def test_inspect_replay_writes_a_noting_page_for_dangling_pointers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A pointer whose capture directory is absent still writes an explaining page."""
    path = _write_log_file(tmp_path, _replay_log(({"wire_capture": "wire/run/s0-e0/calls.jsonl"},)))

    assert main(["inspect", str(path), "--replay"]) == 0

    assert f"wrote {tmp_path / 'wire-replay.html'}" in capsys.readouterr().out
    document = (tmp_path / "wire-replay.html").read_text(encoding="utf-8")
    assert "no readable wire calls" in document


@pytest.mark.parametrize("pointer", [None, "../outside/calls.jsonl", "calls.jsonl"])
def test_inspect_replay_degrades_to_a_note_without_captures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], pointer: str | None
) -> None:
    """Missing, escaping, and too-shallow pointers note and write nothing."""
    metadata = ({} if pointer is None else {"wire_capture": pointer},)
    log = _replay_log(metadata)
    padded = dataclasses.replace(log.samples[0], epochs=({}, {}))
    path = _write_log_file(tmp_path, dataclasses.replace(log, samples=(padded,)))

    assert main(["inspect", str(path), "--replay"]) == 0

    assert "no wire capture recorded" in capsys.readouterr().out
    assert not (tmp_path / "wire-replay.html").exists()


def test_inspect_without_replay_writes_no_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without --replay the command leaves the log directory untouched."""
    _write_cli_capture(tmp_path)
    path = _write_log_file(tmp_path, _replay_log(({"wire_capture": "wire/run/s0-e0/calls.jsonl"},)))

    assert main(["inspect", str(path)]) == 0

    assert "wire-replay" not in capsys.readouterr().out
    assert not (tmp_path / "wire-replay.html").exists()


def test_replay_page_renders_escaped_header_fields(tmp_path: Path) -> None:
    capture = _capture_with(tmp_path)
    from inspect_robots._wire_replay import render_wire_replay_page

    page = render_wire_replay_page(
        [capture],
        "log.json",
        fields={"instruction": "put <b>the</b> cup", "created": "2026-09-17T02:00:00+00:00"},
    )
    assert '<dl class="runmeta">' in page
    assert "put &lt;b&gt;the&lt;/b&gt; cup" in page
    assert "<dt>created</dt><dd>2026-09-17T02:00:00+00:00</dd>" in page


def test_replay_page_omits_the_header_without_fields(tmp_path: Path) -> None:
    capture = _capture_with(tmp_path)
    from inspect_robots._wire_replay import render_wire_replay_page

    assert '<dl class="runmeta">' not in render_wire_replay_page([capture], "log.json")


def test_replay_page_omits_rows_with_empty_values(tmp_path: Path) -> None:
    capture = _capture_with(tmp_path)
    from inspect_robots._wire_replay import render_wire_replay_page

    page = render_wire_replay_page(
        [capture], "log.json", fields={"instruction": "", "status": "completed"}
    )
    assert "instruction" not in page
    assert "<dt>status</dt><dd>completed</dd>" in page


def _capture_with(tmp_path: Path) -> Path:
    """One capture dir with a single well-formed call."""
    capture = _capture(tmp_path)
    _write_calls(capture, "scene-0-e0", [json.dumps(_chat_row())])
    return capture
