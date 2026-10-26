from pathlib import Path

import numpy as np
import pytest

from inspect_robots.scene import Scene
from inspect_robots.types import Observation
from inspect_robots_evo.episode import ScriptedModel, ToolTurn
from inspect_robots_evo.policy import EvoPolicy, ResponsesModel, coding_model
from inspect_robots_evo.workspace import Workspace


def _obs() -> Observation:
    return Observation(state={"eef_state": np.zeros(20)})


class _Reply:
    def __init__(self) -> None:
        self.tool_calls: list[object] = []


class _Client:
    def complete(self, messages: list[object], tools: list[object]) -> _Reply:
        return _Reply()


def test_responses_model_counts_calls() -> None:
    model = ResponsesModel(_Client())  # type: ignore[arg-type]
    turn = model.complete([], [])
    assert model.calls == 1
    assert turn.name == "give_up"


def test_anthropic_wire_disables_thinking() -> None:
    model = coding_model(
        "glm-5.3",
        base_url="https://example.test/v1",
        api_key="k",
        wire="anthropic",
    )
    seen: dict[str, object] = {}

    def complete(
        messages: list[object],
        tools: list[object],
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> _Reply:
        seen["effort"] = reasoning_effort
        return _Reply()

    model.client._client.complete = complete  # type: ignore[attr-defined]
    model.complete([], [])
    assert seen["effort"] == "none"


def test_reset_writes_the_scene_instruction(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    policy = EvoPolicy(ScriptedModel([]), Workspace(root, "playground"))
    policy.reset(Scene(id="scene-0", instruction="stack the blocks"))
    assert (root / "instruction.md").read_text(encoding="utf-8") == "stack the blocks"


def test_low_diagonal_is_clamped_and_split(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path, "lift", mode="test", stamp="s")
    source = (
        "def solve():\n"
        "    step({'left_x': 0.60, 'left_y': 0.20, 'left_z': 0.82})\n"
        "    return 'moved'\n"
    )
    workspace.write_file("submission/solution.py", source)
    model = ScriptedModel([ToolTurn("step_code", {}), ToolTurn("done", {})])
    policy = EvoPolicy(model=model, workspace=workspace, outcome_reader=lambda: True)
    state = np.zeros(20)
    state[0] = 0.30
    state[1] = 0.27
    state[2] = 0.949
    chunk = policy.act(Observation(state={"eef_state": state}))
    assert float(chunk.actions[-1].data[2]) == pytest.approx(0.899)
    assert float(chunk.actions[0].data[2]) == pytest.approx(0.949)


def test_move_ee_emits_a_chunk_then_stop(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path, "lift", mode="test", stamp="s")
    source = "def solve():\n    move_ee({'left_z': 0.02})\n    return 'up'\n"
    workspace.write_file("submission/solution.py", source)
    model = ScriptedModel([ToolTurn("step_code", {}), ToolTurn("done", {})])
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
    model = ScriptedModel([ToolTurn("step_code", {})])
    policy = EvoPolicy(model=model, workspace=workspace)
    policy.abort.set()
    stop = policy.act(_obs())
    assert stop.actions[0].meta["stop_reason"] == "aborted"
