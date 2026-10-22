from pathlib import Path

import numpy as np

from inspect_robots.types import Observation
from inspect_robots_evo.episode import ScriptedModel, ToolTurn
from inspect_robots_evo.policy import EvoPolicy
from inspect_robots_evo.workspace import Workspace


def _obs() -> Observation:
    return Observation(state={"eef_state": np.zeros(20)})


def test_move_ee_emits_a_chunk_then_stop(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path, "lift", mode="test", stamp="s")
    source = "def solve():\n    move_ee({'left_z': 0.02})\n    return 'up'\n"
    workspace.write_file("submission/solution.py", source)
    model = ScriptedModel([ToolTurn("submit_code", {}), ToolTurn("done", {})])
    policy = EvoPolicy(model=model, workspace=workspace, outcome_reader=lambda: True)
    chunk = policy.act(_obs())
    assert chunk.meta.get("request_stop") is not True
    assert len(chunk.actions) == 2
    assert float(chunk.actions[-1].data[2]) == 0.02
    stop = policy.act(_obs())
    assert stop.actions[0].meta["request_stop"] is True
    assert stop.actions[0].meta["stop_reason"] == "done"


def test_estop_requests_stop(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path, "lift", mode="test", stamp="s")
    source = "def solve():\n    move_ee({'left_z': 0.02})\n    return 'up'\n"
    workspace.write_file("submission/solution.py", source)
    model = ScriptedModel([ToolTurn("submit_code", {})])
    policy = EvoPolicy(model=model, workspace=workspace)
    policy.abort.set()
    stop = policy.act(_obs())
    assert stop.actions[0].meta["stop_reason"] == "aborted"
