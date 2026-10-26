import json
import threading
from pathlib import Path

from inspect_robots_agent._responses import _translate_tools

from inspect_robots_evo.episode import Episode, ScriptedModel, ToolTurn, _tools
from inspect_robots_evo.workspace import Workspace


class Port:
    def __init__(self) -> None:
        self.calls = 0

    def move(self, targets, abort):
        self.calls += 1
        return {"held": False, "steps": 1}


def _ws(tmp_path: Path, mode: str = "playground") -> Workspace:
    return Workspace.create(tmp_path, "move the cup", mode=mode, stamp="s")


def test_acting_tools_include_descriptions() -> None:
    translated = _translate_tools(
        _tools(("read_file", "write_file", "step_code", "done", "give_up"))
    )
    assert [item["name"] for item in translated] == [
        "read_file",
        "write_file",
        "step_code",
        "done",
        "give_up",
    ]
    assert all(item["description"] for item in translated)


def test_done_does_not_move(tmp_path: Path) -> None:
    port = Port()
    workspace = _ws(tmp_path)
    model = ScriptedModel([ToolTurn("done", {})])
    episode = Episode(workspace, port)
    result = episode.run(model, threading.Event())
    assert result.reason == "done"
    assert port.calls == 0
    assert "move the cup" in str(episode.messages[0]["content"])


def test_bad_tool_arguments_stay_in_the_loop(tmp_path: Path) -> None:
    model = ScriptedModel([ToolTurn("read_file", {}), ToolTurn("done", {})])
    result = Episode(_ws(tmp_path), Port()).run(model, threading.Event())
    assert result.reason == "done"


def test_submit_then_operator_success_freezes_the_new_skill(tmp_path: Path) -> None:
    workspace = _ws(tmp_path)
    solution = "def solve():\n    return 'ok'\n"
    skill = "def place():\n    return 'p'\n"
    model = ScriptedModel(
        [
            ToolTurn("write_file", {"path": "submission/solution.py", "text": solution}),
            ToolTurn("step_code", {}),
            ToolTurn("done", {}),
            ToolTurn("write_file", {"path": "skills/place.py", "text": skill}),
        ]
    )
    result = Episode(workspace, Port(), outcome_reader=lambda: True).run(model, threading.Event())
    assert result.reason == "done"
    outcome = json.loads((workspace.root / "outcome.json").read_text(encoding="utf-8"))
    assert outcome["success"] is True
    frozen = workspace.root / "freeze" / "skills" / "place.py"
    assert frozen.is_file()


def test_reflection_cannot_submit(tmp_path: Path) -> None:
    workspace = _ws(tmp_path)
    model = ScriptedModel([ToolTurn("done", {}), ToolTurn("step_code", {})])
    Episode(workspace, Port(), outcome_reader=lambda: False).run(model, threading.Event())
    assert model.rejected[-1] == "step_code"


def test_test_mode_has_no_reflection(tmp_path: Path) -> None:
    workspace = _ws(tmp_path, mode="test")
    skill = "def x():\n    return 1\n"
    model = ScriptedModel(
        [
            ToolTurn("done", {}),
            ToolTurn("write_file", {"path": "skills/x.py", "text": skill}),
        ]
    )
    Episode(workspace, Port(), outcome_reader=lambda: True).run(model, threading.Event())
    assert model.calls == 1
    assert not (workspace.root / "freeze").exists()


def test_twenty_submits_force_give_up(tmp_path: Path) -> None:
    workspace = _ws(tmp_path)
    workspace.write_file("submission/solution.py", "def solve():\n    return 'ok'\n")
    model = ScriptedModel([ToolTurn("step_code", {}) for _ in range(21)])
    result = Episode(workspace, Port(), max_submits=20).run(model, threading.Event())
    assert result.reason == "give_up"
    assert result.submits == 20
