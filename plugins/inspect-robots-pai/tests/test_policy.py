import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from inspect_robots.types import Observation
from inspect_robots_pai import pai_policy


def _write_program(program: Path) -> None:
    program.mkdir(parents=True, exist_ok=True)
    (program / "__init__.py").write_text(
        "def run(port, trial_dir):\n    port.move({'left_x': 0.02})\n",
        encoding="utf-8",
    )


def _observation() -> Observation:
    image = np.zeros((2, 2), dtype=np.uint8)
    return Observation(
        images={
            "left_rgbd": image,
            "right_rgbd": image,
            "chest_rgbd": image,
        },
        state={
            "eef_state": np.zeros(20),
            "joint_pos": np.zeros(16),
        },
    )


def test_pai_policy_requires_program_and_trial_dir() -> None:
    with pytest.raises(ValueError):
        pai_policy()
    with pytest.raises(ValueError):
        pai_policy(program="only-program")
    with pytest.raises(ValueError):
        pai_policy(trial_dir="only-trial")


def test_act_interpolates_move_and_then_requests_stop(tmp_path: Path) -> None:
    program = tmp_path / "program"
    trial = tmp_path / "trial"
    _write_program(program)
    policy = pai_policy(program=program, trial_dir=trial)
    observation = _observation()

    chunk = policy.act(observation)
    last = chunk.actions[-1]
    assert last.data.dtype == np.float64
    assert last.data.shape == (20,)
    assert dict(last.meta) == {}
    assert float(last.data[0]) == pytest.approx(0.02)

    deadline = time.monotonic() + 2.0
    step = None
    while time.monotonic() < deadline:
        path = trial / "steps.jsonl"
        if path.is_file() and path.read_text(encoding="utf-8").strip():
            try:
                row = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
            except json.JSONDecodeError:
                row = None
            if row is not None and "target_vector" in row:
                step = row
                break
        time.sleep(0.01)
    assert step is not None
    assert len(step["target_vector"]) == 20
    assert step["target_vector"][0] == pytest.approx(0.02)

    sys.modules.pop("inspect_robots_pai.agents", None)
    stop = None
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        nxt = policy.act(observation)
        if nxt.actions[0].meta.get("request_stop") is True:
            stop = nxt
            break
    assert stop is not None
    assert stop.actions[0].meta["stop_reason"] == "returned"
