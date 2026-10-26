import threading
from pathlib import Path

import pytest

from inspect_robots_evo.executor import Executor
from inspect_robots_evo.motion import DIM_LABELS, applied_targets, interpolate, reached
from inspect_robots_evo.workspace import Workspace


class FakePort:
    def __init__(self, held: bool = False) -> None:
        self.calls: list[dict[str, float]] = []
        self.held = held

    def move(self, targets, abort):
        self.calls.append(dict(targets))
        steps = 0 if self.held else 4
        return {"held": self.held, "steps": steps}


def _workspace(tmp_path: Path, source: str) -> Workspace:
    workspace = Workspace.create(tmp_path, "put the cup down", mode="playground", stamp="s")
    workspace.write_file("submission/solution.py", source)
    return workspace


def test_step_action_moves_the_arm(tmp_path: Path) -> None:
    port = FakePort()
    source = "def solve():\n    step({'left_z': 0.2})\n    return 'placed'\n"
    result = Executor(_workspace(tmp_path, source), port).run(threading.Event())
    assert result.status == "returned"
    assert port.calls == [{"left_z": 0.2}]


def test_solve_can_move_and_return_a_string(tmp_path: Path) -> None:
    port = FakePort()
    source = "def solve():\n    move_ee({'left_z': 0.2})\n    return 'placed'\n"
    result = Executor(_workspace(tmp_path, source), port).run(threading.Event())
    assert result.status == "returned"
    assert result.value == "placed"
    assert port.calls == [{"left_z": 0.2}]


def test_move_limit_stops_at_80(tmp_path: Path) -> None:
    port = FakePort()
    source = (
        "def solve():\n"
        "    n = 0\n"
        "    while n < 200:\n"
        "        move_ee({'left_z': 0.2})\n"
        "        n += 1\n"
        "    return 'done'\n"
    )
    result = Executor(_workspace(tmp_path, source), port, max_moves=80).run(threading.Event())
    assert result.status == "limit"
    assert len(port.calls) == 80


def test_os_import_never_calls_the_port(tmp_path: Path) -> None:
    port = FakePort()
    source = "import os\ndef solve():\n    return 'no'\n"
    result = Executor(_workspace(tmp_path, source), port).run(threading.Event())
    assert result.status == "error"
    assert port.calls == []


def test_held_result_is_visible_to_solve(tmp_path: Path) -> None:
    port = FakePort(held=True)
    source = (
        "def solve():\n"
        "    result = move_ee({'left_z': 0.2})\n"
        "    return 'held' if result['held'] else 'moved'\n"
    )
    result = Executor(_workspace(tmp_path, source), port).run(threading.Event())
    assert result.value == "held"


def test_non_string_return_is_an_error(tmp_path: Path) -> None:
    source = "def solve():\n    return 1\n"
    result = Executor(_workspace(tmp_path, source), FakePort()).run(threading.Event())
    assert result.status == "error"


def test_duplicate_skill_name_is_rejected(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, "def solve():\n    return from_a()\n")
    workspace.write_file("skills/a.py", "def from_a():\n    return 'a'\n")
    workspace.write_file("skills/b.py", "def from_a():\n    return 'b'\n")
    port = FakePort()
    result = Executor(workspace, port).run(threading.Event())
    assert result.status == "error"
    assert "from_a" in (result.traceback or "")
    assert port.calls == []


def test_abort_during_move_is_aborted(tmp_path: Path) -> None:
    class AbortingPort:
        def move(self, targets, abort_flag):
            abort_flag.set()
            return {"held": False, "steps": 1}

    source = "def solve():\n    move_ee({'left_z': 0.2})\n    return 'after'\n"
    result = Executor(_workspace(tmp_path, source), AbortingPort()).run(threading.Event())
    assert result.status == "aborted"


def test_timeout_stops_before_a_second_move(tmp_path: Path) -> None:
    clock = {"now": 0.0}

    def tick() -> float:
        clock["now"] += 601.0
        return clock["now"]

    port = FakePort()
    source = "def solve():\n    move_ee({'left_z': 0.2})\n    return 'late'\n"
    result = Executor(_workspace(tmp_path, source), port, clock=tick).run(threading.Event())
    assert result.status == "timeout"
    assert port.calls == []


def test_horizontal_move_stays_at_cruise_height_until_xy_arrives() -> None:
    current = dict.fromkeys(DIM_LABELS, 0.0)
    current["left_x"] = 0.30
    current["left_y"] = 0.27
    current["left_z"] = 0.949
    home = dict(current)
    applied, limited = applied_targets(
        current, {"left_x": 0.60, "left_y": 0.20, "left_z": 0.82}, home
    )
    assert limited == ("left_z",)
    assert applied["left_z"] == pytest.approx(0.899)
    rows, _steps = interpolate(current, applied, home=home)
    z_index = DIM_LABELS.index("left_z")
    x_index = DIM_LABELS.index("left_x")
    descending = [row for row in rows if row[z_index] < 0.949 - 1e-6]
    assert descending
    assert descending[0][x_index] == pytest.approx(0.60)
    assert all(row[z_index] == pytest.approx(0.949) for row in rows if row[x_index] < 0.60 - 1e-6)


def test_interpolate_holds_unnamed_dimensions() -> None:
    current = dict.fromkeys(DIM_LABELS, 0.0)
    current["left_z"] = 0.10
    rows, steps = interpolate(current, {"left_z": 0.12})
    assert steps == 2
    assert rows[-1][DIM_LABELS.index("left_z")] == pytest.approx(0.12)
    assert rows[-1][DIM_LABELS.index("left_x")] == 0.0
    assert reached(rows[-1], {"left_z": 0.12})
