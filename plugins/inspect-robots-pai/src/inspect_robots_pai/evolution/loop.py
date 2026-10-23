"""Evolve an isolated candidate between trials, then promote or stop."""

from __future__ import annotations

import importlib.util
import json
import shutil
from collections.abc import Callable
from itertools import count
from pathlib import Path

from inspect_robots_pai.agents.codex import AgentError
from inspect_robots_pai.evolution.contract import passed, promote
from inspect_robots_pai.memory.scaffold import scaffold
from inspect_robots_pai.memory.store import lookup_task

Writer = Callable[[Path, str], None]
Diagnoser = Callable[[Path, list[int]], str]
Synthesizer = Callable[[Path, str], str]
Executor = Callable[[Path, int], None]

_LOAD_IDS = count(1)


def _load_candidate(candidate: Path):
    load_id = next(_LOAD_IDS)
    init_py = Path(candidate) / "__init__.py"
    spec = importlib.util.spec_from_file_location(f"pai_candidate_{load_id}", init_py)
    if spec is None or spec.loader is None:
        raise ImportError(init_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate_ok(candidate: Path, task: str) -> bool:
    try:
        module = _load_candidate(candidate)
    except Exception:
        return False
    return getattr(module, "TASK_NAME", None) == task


def preflight(candidate: Path, task: str, writer: Writer, instruction: str) -> bool:
    """Load the candidate. On failure, ask the writer once and load again."""
    if _candidate_ok(candidate, task):
        return True
    writer(candidate, instruction)
    return _candidate_ok(candidate, task)


def _diagnostics(run_dir: Path, index: int) -> dict | None:
    path = Path(run_dir) / "trials" / str(index) / "diagnostics.jsonl"
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return None
    record = json.loads(lines[0])
    if not isinstance(record, dict):
        return None
    return record


def _has_success_key(run_dir: Path, index: int) -> bool:
    record = _diagnostics(run_dir, index)
    return record is not None and "success" in record


def _trial_recovered(run_dir: Path, index: int) -> bool:
    record = _diagnostics(run_dir, index)
    if record is None:
        return False
    return bool(record.get("process_ok")) and bool(record.get("success"))


def _prepare_candidate(task: str, run_dir: Path, memory_root: Path) -> Path:
    candidate = Path(run_dir) / "candidate"
    if candidate.exists():
        return candidate
    found = lookup_task(memory_root, task)
    if found is not None and found.is_dir():
        candidate.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(found, candidate)
        return candidate
    scaffold(candidate, task)
    return candidate


def _promote_passed(
    run_dir: Path,
    candidate: Path,
    library_root: Path,
    task: str,
    memory_root: Path,
    trial_count: int,
) -> None:
    contract = Path(run_dir) / "contract.json"
    contract.write_text(
        json.dumps({"task": task, "trial_count": trial_count}) + "\n",
        encoding="utf-8",
    )
    promote(
        run_dir,
        candidate,
        library_root,
        task,
        memory_root,
        description=task,
        failure="",
        repair="promoted",
    )


def _revise(
    run_dir: Path,
    candidate: Path,
    still: list[int],
    diagnoser: Diagnoser,
    synthesizer: Synthesizer,
) -> str:
    diagnosis = diagnoser(Path(run_dir), still)
    return synthesizer(candidate, diagnosis)


def _diagnose_and_stop(
    run_dir: Path,
    candidate: Path,
    still: list[int],
    diagnoser: Diagnoser,
    synthesizer: Synthesizer,
) -> str:
    try:
        _revise(run_dir, candidate, still, diagnoser, synthesizer)
    except AgentError:
        return "stopped"
    return "stopped"


def evolve(
    task: str,
    run_dir: Path,
    library_root: Path,
    memory_root: Path,
    trial_count: int = 1,
    max_rounds: int = 5,
    probe_max: int = 3,
    writer: Writer | None = None,
    diagnoser: Diagnoser | None = None,
    synthesizer: Synthesizer | None = None,
    executor: Executor | None = None,
) -> str:
    """Return ``promoted`` or ``stopped`` without calling live agent services."""
    if writer is None or diagnoser is None or synthesizer is None or executor is None:
        raise AgentError("evolve callables are required")

    run_dir = Path(run_dir)
    candidate = _prepare_candidate(task, run_dir, Path(memory_root))
    instruction = task
    failures: list[int] = []

    for _round_index in range(max_rounds):
        try:
            writer(candidate, instruction)
        except AgentError:
            return "stopped"
        try:
            ready = preflight(candidate, task, writer, instruction)
        except AgentError:
            return "stopped"
        if not ready:
            return "stopped"

        if failures:
            still: list[int] = []
            for index in failures:
                recovered = False
                for _attempt in range(probe_max):
                    executor(run_dir, index)
                    if _trial_recovered(run_dir, index):
                        recovered = True
                        break
                if not recovered:
                    still.append(index)
            if still:
                return _diagnose_and_stop(run_dir, candidate, still, diagnoser, synthesizer)
            failures = []
        else:
            for index in range(trial_count):
                executor(run_dir, index)
            if any(not _has_success_key(run_dir, index) for index in range(trial_count)):
                return "stopped"

        if passed(run_dir, trial_count):
            _promote_passed(run_dir, candidate, library_root, task, memory_root, trial_count)
            return "promoted"

        failures = [index for index in range(trial_count) if not _trial_recovered(run_dir, index)]
        still = []
        for index in failures:
            recovered = False
            # The execute pass already used one attempt. Finish the probe budget.
            for _attempt in range(probe_max - 1):
                executor(run_dir, index)
                if _trial_recovered(run_dir, index):
                    recovered = True
                    break
            if not recovered:
                still.append(index)
        if still:
            return _diagnose_and_stop(run_dir, candidate, still, diagnoser, synthesizer)
        if passed(run_dir, trial_count):
            _promote_passed(run_dir, candidate, library_root, task, memory_root, trial_count)
            return "promoted"

    return "stopped"
