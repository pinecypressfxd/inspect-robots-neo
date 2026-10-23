import json
from pathlib import Path

from inspect_robots_pai.agents.codex import AgentError
from inspect_robots_pai.evolution.loop import evolve
from inspect_robots_pai.execution.trial import label_trial


def _noop_diagnoser(run_dir: Path, failed_indices: list[int]) -> str:
    return "diagnosis"


def _noop_synthesizer(candidate: Path, diagnosis: str) -> str:
    return "next instruction"


def _valid_writer(task: str):
    def writer(candidate: Path, instruction: str) -> None:
        (Path(candidate) / "__init__.py").write_text(
            f"TASK_NAME = {task!r}\n\ndef run(port, trial_dir):\n    return None\n",
            encoding="utf-8",
        )

    return writer


def test_preflight_failure_does_not_create_trials(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    library = tmp_path / "library"
    memory = tmp_path / "memory"
    executed = tmp_path / "executed"
    task = "bottle"

    def writer(candidate: Path, instruction: str) -> None:
        (Path(candidate) / "__init__.py").write_text("def run(:", encoding="utf-8")

    def executor(run_dir: Path, trial_index: int) -> None:
        executed.write_text("called", encoding="utf-8")

    result = evolve(
        task,
        run_dir,
        library,
        memory,
        writer=writer,
        diagnoser=_noop_diagnoser,
        synthesizer=_noop_synthesizer,
        executor=executor,
    )

    assert result == "stopped"
    assert not (run_dir / "trials").exists()
    assert not executed.exists()


def test_probe_stops_after_three_failures(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    library = tmp_path / "library"
    memory = tmp_path / "memory"
    task = "grasp"
    calls = {"count": 0}

    def executor(run_dir: Path, trial_index: int) -> None:
        calls["count"] += 1
        trial = Path(run_dir) / "trials" / str(trial_index)
        trial.mkdir(parents=True, exist_ok=True)
        (trial / "diagnostics.jsonl").write_text(
            json.dumps({"process_ok": False, "success": False}) + "\n",
            encoding="utf-8",
        )

    result = evolve(
        task,
        run_dir,
        library,
        memory,
        trial_count=1,
        max_rounds=5,
        probe_max=3,
        writer=_valid_writer(task),
        diagnoser=_noop_diagnoser,
        synthesizer=_noop_synthesizer,
        executor=executor,
    )

    assert calls["count"] == 3
    assert result == "stopped"
    assert not (library / task).exists()


def test_success_promotes_on_first_round(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    library = tmp_path / "library"
    memory = tmp_path / "memory"
    task = "place"
    calls = {"count": 0}

    def executor(run_dir: Path, trial_index: int) -> None:
        calls["count"] += 1
        trial = Path(run_dir) / "trials" / str(trial_index)
        trial.mkdir(parents=True, exist_ok=True)
        (trial / "diagnostics.jsonl").write_text(
            json.dumps({"process_ok": True}) + "\n",
            encoding="utf-8",
        )
        label_trial(trial, True)

    result = evolve(
        task,
        run_dir,
        library,
        memory,
        max_rounds=5,
        writer=_valid_writer(task),
        diagnoser=_noop_diagnoser,
        synthesizer=_noop_synthesizer,
        executor=executor,
    )

    assert result == "promoted"
    assert (library / task).is_dir()
    assert calls["count"] == 1
    contract = json.loads((run_dir / "contract.json").read_text(encoding="utf-8"))
    assert contract == {"task": task, "trial_count": 1}


def test_agent_error_stops(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    library = tmp_path / "library"
    memory = tmp_path / "memory"
    task = "hold"
    (library / "existing").mkdir(parents=True)
    (library / "existing" / "keep.txt").write_text("same", encoding="utf-8")

    def snapshot() -> dict[str, str]:
        return {
            path.relative_to(library).as_posix(): path.read_text(encoding="utf-8")
            for path in library.rglob("*")
            if path.is_file()
        }

    before = snapshot()

    def writer(candidate: Path, instruction: str) -> None:
        raise AgentError("offline")

    def executor(run_dir: Path, trial_index: int) -> None:
        raise AssertionError("executor must not run")

    result = evolve(
        task,
        run_dir,
        library,
        memory,
        writer=writer,
        diagnoser=_noop_diagnoser,
        synthesizer=_noop_synthesizer,
        executor=executor,
    )

    assert result == "stopped"
    assert snapshot() == before
    assert not (library / task).exists()
