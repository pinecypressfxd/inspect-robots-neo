import json
import subprocess
from pathlib import Path

import pytest

from inspect_robots_pai.agents.codex import AgentError, CodexSynthesizer, CodexWriter
from inspect_robots_pai.agents.gemini import GeminiDiagnoser
from inspect_robots_pai.cli import main, trial_command
from inspect_robots_pai.export.lerobot import export_lerobot


def test_label_command_writes_success_false(tmp_path: Path) -> None:
    trial = tmp_path / "trial"
    trial.mkdir()
    (trial / "diagnostics.jsonl").write_text(
        json.dumps({"process_ok": True}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as caught:
        main(["label", str(trial), "--success", "false"])

    assert caught.value.code == 0
    row = json.loads((trial / "diagnostics.jsonl").read_text(encoding="utf-8"))
    assert row["success"] is False


def test_export_lerobot_returns_zero_without_success(tmp_path: Path) -> None:
    run = tmp_path / "run"
    trial = run / "trials" / "0"
    trial.mkdir(parents=True)
    (run / "contract.json").write_text(
        json.dumps({"task": "nothing to export"}) + "\n",
        encoding="utf-8",
    )
    (trial / "diagnostics.jsonl").write_text(
        json.dumps({"process_ok": True, "success": False}) + "\n",
        encoding="utf-8",
    )
    (trial / "steps.jsonl").write_text("{}\n", encoding="utf-8")

    assert export_lerobot(run, tmp_path / "out") == 0


def test_export_lerobot_command_calls_exporter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Path] = {}

    def fake_export(run_dir: Path, out_dir: Path) -> int:
        seen["run"] = Path(run_dir)
        seen["out"] = Path(out_dir)
        return 0

    monkeypatch.setattr("inspect_robots_pai.cli.export_lerobot", fake_export)
    run = tmp_path / "run"
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as caught:
        main(["export-lerobot", str(run), str(out)])

    assert caught.value.code == 0
    assert seen["run"] == run
    assert seen["out"] == out


def test_trial_command_lists_pai_nero_without_running() -> None:
    assert trial_command("put the bottle", "candidate", "trials/0") == [
        "inspect-robots",
        "put the bottle",
        "--policy",
        "pai",
        "--embodiment",
        "nero",
        "-P",
        "program=candidate",
        "-P",
        "trial_dir=trials/0",
    ]


def test_evolve_agent_error_exits_without_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[object] = []
    created: list[str] = []

    class TrackingWriter(CodexWriter):
        def __init__(self) -> None:
            created.append("writer")

        def __call__(self, candidate: Path, instruction: str) -> None:
            raise AgentError("offline")

    class TrackingDiagnoser(GeminiDiagnoser):
        def __init__(self) -> None:
            created.append("diagnoser")

    class TrackingSynthesizer(CodexSynthesizer):
        def __init__(self) -> None:
            created.append("synthesizer")

    monkeypatch.setattr("inspect_robots_pai.cli.CodexWriter", TrackingWriter)
    monkeypatch.setattr("inspect_robots_pai.cli.GeminiDiagnoser", TrackingDiagnoser)
    monkeypatch.setattr("inspect_robots_pai.cli.CodexSynthesizer", TrackingSynthesizer)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: calls.append(("run", args)))
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: calls.append(("popen", args)))

    with pytest.raises(SystemExit) as caught:
        main(
            [
                "evolve",
                "put the water bottle on the black pad.",
                "--run-dir",
                str(tmp_path / "run"),
                "--library-root",
                str(tmp_path / "library"),
                "--memory-root",
                str(tmp_path / "memory"),
            ]
        )

    assert caught.value.code == 2
    assert "候选未晋升" in capsys.readouterr().out
    assert calls == []
    assert created == ["writer", "diagnoser", "synthesizer"]
    assert not (tmp_path / "library").exists()
