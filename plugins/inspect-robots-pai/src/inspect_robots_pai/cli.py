"""Command line for labeling trials, evolving a candidate, and exporting."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from inspect_robots_pai.agents.codex import AgentError, CodexSynthesizer, CodexWriter
from inspect_robots_pai.agents.gemini import GeminiDiagnoser
from inspect_robots_pai.evolution.loop import evolve
from inspect_robots_pai.execution.trial import label_trial
from inspect_robots_pai.export.lerobot import export_lerobot


class _OperatorInterrupt(Exception):
    """The operator interrupted labeling, so the candidate stays unpromoted."""


class _AgentWriter:
    """Call the Codex writer and stop the process when it fails."""

    def __init__(self) -> None:
        self._inner = CodexWriter()

    def __call__(self, candidate: Path, instruction: str) -> None:
        """Raise SystemExit 2 after an agent failure, before any robot process."""
        try:
            self._inner(candidate, instruction)
        except AgentError:
            print("候选未晋升")
            raise SystemExit(2) from None


def trial_command(task: str, program: str | Path, trial_dir: str | Path) -> list[str]:
    """Return the inspect-robots argv for one Pai trial. Callers execute it."""
    return [
        "inspect-robots",
        str(task),
        "--policy",
        "pai",
        "--embodiment",
        "nero",
        "-P",
        f"program={program}",
        "-P",
        f"trial_dir={trial_dir}",
    ]


def wait_for_label(run_dir: str | Path, index: int) -> str:
    """Block until diagnostics contains success.

    A keyboard interrupt returns ``stopped`` and does not promote.
    """
    path = Path(run_dir) / "trials" / str(index) / "diagnostics.jsonl"
    try:
        while True:
            if path.is_file():
                lines = [
                    line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
                ]
                if lines:
                    record = _json_record(lines[0])
                    if "success" in record:
                        return "labeled"
            time.sleep(0.2)
    except KeyboardInterrupt:
        return "stopped"


def execute_recorded_trial(task: str, program: str | Path, run_dir: str | Path, index: int) -> str:
    """Run one trial subprocess, then wait for a label.

    Unit tests do not call this. Interrupt returns ``stopped`` and does not promote.
    """
    trial_dir = Path(run_dir) / "trials" / str(index)
    trial_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(trial_command(task, program, trial_dir), check=False)
    return wait_for_label(run_dir, index)


def _json_record(line: str) -> dict[str, object]:
    record = json.loads(line)
    if not isinstance(record, dict):
        return {}
    return record


def main(argv: list[str] | None = None) -> None:
    """Dispatch label, evolve, and export-lerobot."""
    raise SystemExit(_main(argv))


def _main(argv: list[str] | None) -> int:
    parser = argparse.ArgumentParser(prog="inspect-robots-pai")
    commands = parser.add_subparsers(dest="command", required=True)

    label = commands.add_parser("label")
    label.add_argument("trial_dir")
    label.add_argument("--success", required=True, choices=("true", "false"))

    export = commands.add_parser("export-lerobot")
    export.add_argument("run_dir")
    export.add_argument("out_dir")

    evolve_parser = commands.add_parser("evolve")
    evolve_parser.add_argument("task")
    evolve_parser.add_argument("--run-dir", required=True)
    evolve_parser.add_argument("--library-root")
    evolve_parser.add_argument("--memory-root")
    evolve_parser.add_argument("--trial-count", type=int, default=1)
    evolve_parser.add_argument("--max-rounds", type=int, default=5)
    evolve_parser.add_argument("--probe-max", type=int, default=3)

    args = parser.parse_args(argv)
    if args.command == "label":
        label_trial(args.trial_dir, args.success == "true")
        return 0
    if args.command == "export-lerobot":
        export_lerobot(args.run_dir, args.out_dir)
        return 0
    return _evolve(args)


def _evolve(args: argparse.Namespace) -> int:
    package = Path(__file__).resolve().parent
    library = Path(args.library_root or package / "code_library" / "nero")
    memory = Path(args.memory_root or package / "memory")
    run_dir = Path(args.run_dir)
    writer = _AgentWriter()
    diagnoser = GeminiDiagnoser()
    synthesizer = CodexSynthesizer()

    def executor(current_run: Path, index: int) -> None:
        status = execute_recorded_trial(args.task, current_run / "candidate", current_run, index)
        if status == "stopped":
            raise _OperatorInterrupt

    try:
        evolve(
            args.task,
            run_dir,
            library,
            memory,
            trial_count=args.trial_count,
            max_rounds=args.max_rounds,
            probe_max=args.probe_max,
            writer=writer,
            diagnoser=diagnoser,
            synthesizer=synthesizer,
            executor=executor,
        )
    except _OperatorInterrupt:
        return 0
    return 0
