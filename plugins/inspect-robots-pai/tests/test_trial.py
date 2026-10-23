import json
from pathlib import Path

import numpy as np
import pytest

from inspect_robots_pai.execution.trial import label_trial, run_trial
from inspect_robots_pai.ports import StepView


class FakePort:
    def __init__(self) -> None:
        self.move_count = 0

    def observation(self) -> StepView:
        return StepView(
            instruction="test instruction",
            images={
                "left": np.zeros((2, 2), dtype=np.uint8),
                "right": np.zeros((2, 2), dtype=np.uint8),
                "chest": np.zeros((2, 2), dtype=np.uint8),
            },
            camera_times={"left": 0.0, "right": 0.0, "chest": 0.0},
            body_time=0.0,
            eef_state=np.zeros(20),
            joint_pos=np.zeros(16),
        )

    def move(self, targets: dict) -> dict:
        self.move_count += 1
        return {"held": False, "steps": 1}

    def set_gripper(self, side: str, width: float) -> dict:
        return {}

    def auto_check(self) -> dict:
        return {"ok": False}


def _write_program(program_dir: Path, source: str) -> None:
    program_dir.mkdir(parents=True, exist_ok=True)
    (program_dir / "__init__.py").write_text(source, encoding="utf-8")


def test_run_trial_does_not_set_success(tmp_path: Path) -> None:
    program = tmp_path / "program"
    _write_program(
        program,
        "def run(port, trial_dir):\n    port.move({'left_x': 0.1})\n",
    )
    trial = tmp_path / "trial"
    result = run_trial(program, trial, FakePort())

    assert result.process_ok is True
    assert result.termination == "returned"
    diag = json.loads(result.diagnostics.read_text(encoding="utf-8").splitlines()[0])
    assert "success" not in diag
    assert json.loads(result.auto_check.read_text(encoding="utf-8")) == {"ok": False}

    label_trial(trial, True)
    diag_after = json.loads(result.diagnostics.read_text(encoding="utf-8").splitlines()[0])
    assert diag_after["success"] is True

    step = json.loads((trial / "steps.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert len(step["eef_state"]) == 20
    assert step["targets"]["left_x"] == 0.1
    assert (trial / "frames" / "left" / "000000.png").is_file()


def test_81st_action_fails(tmp_path: Path) -> None:
    program = tmp_path / "program"
    _write_program(
        program,
        "def run(port, trial_dir):\n"
        "    for _ in range(81):\n"
        "        port.move({})\n",
    )
    trial = tmp_path / "trial"
    result = run_trial(program, trial, FakePort(), max_actions=80)

    assert result.process_ok is False
    assert result.termination == "action_budget"


def test_wall_clock(tmp_path: Path) -> None:
    program = tmp_path / "program"
    _write_program(
        program,
        "def run(port, trial_dir):\n    port.move({'left_x': 0.1})\n",
    )
    trial = tmp_path / "trial"
    result = run_trial(program, trial, FakePort(), wall_s=0)

    assert result.process_ok is False
    assert result.termination == "wall_clock"
