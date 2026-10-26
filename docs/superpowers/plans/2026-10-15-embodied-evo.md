# EmbodiedEvo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 inspect-robots 里增加策略 evo，让 Astra 提交受限 Python，经 Nero 现有末端路径执行，并在 Playground 结束后只冻结技能和教训。

**Architecture:** 新插件 plugins/inspect-robots-evo 拥有工作区、AST 沙箱和提交执行器。EvoPolicy.act() 在 rollout 线程上发动作；solve() 在另一线程里调用 move_ee，通过桥接等待这段动作执行完。插件不操作 CAN，不实现 IK。mission console 增加 evo 启动项和操作员成败标签。

**Tech Stack:** Python 3.11、pytest、inspect_robots.PolicyBase、inspect_robots_agent.ResponsesClient（mog-10-a，wire=responses）。

**Spec:** docs/superpowers/specs/2026-10-15-embodied-evo-design.md

测试都在仓库根 /home/dell/fengxuedong/projects/opensource/inspect-robots 执行，并设置 PYTHONPATH=plugins/inspect-robots-evo/src。

## 文件地图

- plugins/inspect-robots-evo/pyproject.toml：包和策略入口 evo。
- src/inspect_robots_evo/workspace.py：一个任务一个目录，读写白名单，冻结。
- src/inspect_robots_evo/sandbox.py：执行前拒绝非法 import 和危险调用。
- src/inspect_robots_evo/motion.py：20 维标签、绝对目标插补、是否到达。
- src/inspect_robots_evo/executor.py：载入 skills，调用 solve()，执行三个原语。
- src/inspect_robots_evo/episode.py：工具循环、提交预算、操作员标签、反思回合。
- src/inspect_robots_evo/policy.py：把 rollout 和执行线程接起来，并提供 evo_policy。
- plugins/inspect-robots-nero/scripts/mission_console.py：策略选项、启动参数、POST /api/evo-outcome。
- 测试在 plugins/inspect-robots-evo/tests/。真机不进自动测试。

## Task 1: 工作区

**Files:**
- Create: plugins/inspect-robots-evo/pyproject.toml
- Create: plugins/inspect-robots-evo/src/inspect_robots_evo/__init__.py
- Create: plugins/inspect-robots-evo/src/inspect_robots_evo/workspace.py
- Test: plugins/inspect-robots-evo/tests/test_workspace.py

- [ ] **Step 1: 写失败测试**

```python
from pathlib import Path

import pytest

from inspect_robots_evo.workspace import Workspace, WorkspaceError


def test_create_lays_out_the_task_directory(tmp_path: Path) -> None:
    workspace = Workspace.create(
        tmp_path, "pick up the cup", mode="playground", stamp="20261015-170000_aa"
    )
    assert (workspace.root / "task.md").read_text(encoding="utf-8") == "pick up the cup"
    assert "move_ee" in (workspace.root / "primitives.md").read_text(encoding="utf-8")
    assert (workspace.root / "submission" / "solution.py").read_text(encoding="utf-8") == ""
    assert (workspace.root / "skills").is_dir()
    assert (workspace.root / "lessons").is_dir()
    assert (workspace.root / "scratch").is_dir()


def test_playground_write_accepts_skills_and_rejects_task(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path, "task", mode="playground", stamp="s")
    workspace.write_file("skills/pick.py", "def pick():\n    return 1\n")
    workspace.write_file("lessons/grip.md", "close before lift\n")
    with pytest.raises(WorkspaceError, match="task.md"):
        workspace.write_file("task.md", "changed")


def test_test_mode_rejects_skill_writes(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path, "task", mode="test", stamp="s")
    with pytest.raises(WorkspaceError, match="skills"):
        workspace.write_file("skills/pick.py", "def pick():\n    return 1\n")


def test_freeze_copies_only_skills_and_lessons(tmp_path: Path) -> None:
    workspace = Workspace.create(tmp_path, "task", mode="playground", stamp="s")
    workspace.write_file("skills/pick.py", "def pick():\n    return 1\n")
    workspace.write_file("lessons/grip.md", "close before lift\n")
    workspace.write_file("scratch/note.txt", "draft\n")
    workspace.freeze()
    freeze = workspace.root / "freeze"
    assert (freeze / "skills" / "pick.py").is_file()
    assert (freeze / "lessons" / "grip.md").is_file()
    assert not (freeze / "scratch").exists()


def test_seed_copies_freeze_then_locks_those_dirs(tmp_path: Path) -> None:
    source = Workspace.create(tmp_path / "src", "task", mode="playground", stamp="s")
    source.write_file("skills/pick.py", "def pick():\n    return 1\n")
    source.freeze()
    target = Workspace.create(tmp_path / "dst", "task", mode="test", stamp="t")
    target.seed_from(source.root / "freeze")
    assert (target.root / "skills" / "pick.py").is_file()
    with pytest.raises(WorkspaceError):
        target.write_file("skills/pick.py", "def pick():\n    return 2\n")
```

- [ ] **Step 2: 运行测试，确认失败**

Run: PYTHONPATH=plugins/inspect-robots-evo/src .venv/bin/python -m pytest plugins/inspect-robots-evo/tests/test_workspace.py -q

Expected: FAIL，inspect_robots_evo 无法导入。

- [ ] **Step 3: 写最小实现**

pyproject.toml 按 plugins/inspect-robots-harness/pyproject.toml 的 hatchling 布局。包名 inspect-robots-evo，版本 0.1.0，依赖 inspect-robots>=0.44、inspect-robots-agent>=0.26、numpy>=1.24。入口：

```toml
[project.entry-points."inspect_robots.policies"]
evo = "inspect_robots_evo:evo_policy"

[tool.uv.sources]
inspect-robots = { workspace = true }
inspect-robots-agent = { workspace = true }
```

__init__.py 先写 __all__ = []。Task 5 再导出 evo_policy。

workspace.py 使用下面的完整内容：

```python
"""一个任务一个目录。write_file 是唯一写入口。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


class WorkspaceError(ValueError):
    """调用方写到了这个模式不允许的路径。"""


_PRIMITIVES = """\
get_instruction() -> str
get_observation() -> dict with left_rgbd, right_rgbd, chest_rgbd, eef_state
move_ee(targets) -> dict with held and steps
No reset().
Labels: left_x, left_y, left_z, left_r1..left_r6, left_gripper, and the right arm.
Unlisted dimensions hold. Meters. Gripper 0 closed, 0.09 open.
"""


class Workspace:
    def __init__(self, root: Path, mode: str) -> None:
        if mode not in ("playground", "test"):
            raise WorkspaceError(f"unknown mode {mode}")
        self.root = root
        self.mode = mode

    @classmethod
    def create(
        cls, parent: Path, instruction: str, *, mode: str, stamp: str
    ) -> Workspace:
        root = parent / stamp
        root.mkdir(parents=True)
        workspace = cls(root, mode)
        (root / "task.md").write_text(instruction, encoding="utf-8")
        (root / "primitives.md").write_text(_PRIMITIVES, encoding="utf-8")
        (root / "submission").mkdir()
        (root / "submission" / "solution.py").write_text("", encoding="utf-8")
        (root / "skills").mkdir()
        (root / "lessons").mkdir()
        (root / "scratch").mkdir()
        (root / "submits").mkdir()
        return workspace

    def write_file(self, relative: str, text: str) -> None:
        path = self._writable(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def read_file(self, relative: str) -> str:
        path = self._inside(relative)
        if not path.is_file():
            raise WorkspaceError(f"no file {relative}")
        return path.read_text(encoding="utf-8")

    def freeze(self) -> None:
        dest = self.root / "freeze"
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir()
        shutil.copytree(self.root / "skills", dest / "skills")
        shutil.copytree(self.root / "lessons", dest / "lessons")

    def seed_from(self, freeze_dir: Path) -> None:
        for name in ("skills", "lessons"):
            target = self.root / name
            shutil.rmtree(target)
            shutil.copytree(freeze_dir / name, target)

    def record_outcome(self, success: bool) -> None:
        payload = json.dumps({"success": success})
        (self.root / "outcome.json").write_text(payload, encoding="utf-8")

    def _writable(self, relative: str) -> Path:
        path = self._inside(relative)
        rel = path.relative_to(self.root).as_posix()
        locked = rel in ("task.md", "primitives.md")
        locked = locked or rel.startswith(("submits/", "freeze/"))
        if locked:
            raise WorkspaceError(f"cannot write {rel}")
        allowed = rel == "submission/solution.py" or rel.startswith("scratch/")
        if self.mode == "playground":
            allowed = allowed or rel.startswith(("skills/", "lessons/"))
        if not allowed:
            raise WorkspaceError(f"cannot write {rel}")
        return path

    def _inside(self, relative: str) -> Path:
        root = self.root.resolve()
        path = (self.root / relative).resolve()
        if path != root and root not in path.parents:
            raise WorkspaceError(f"path escapes workspace: {relative}")
        return path
```

- [ ] **Step 4: 测试通过**

Run: PYTHONPATH=plugins/inspect-robots-evo/src .venv/bin/python -m pytest plugins/inspect-robots-evo/tests/test_workspace.py -q

Expected: 5 passed。

- [ ] **Step 5: Commit**

```bash
git add plugins/inspect-robots-evo
git commit -m "feat: add the evo task workspace"
```

## Task 2: 执行前沙箱

**Files:**
- Create: plugins/inspect-robots-evo/src/inspect_robots_evo/sandbox.py
- Test: plugins/inspect-robots-evo/tests/test_sandbox.py

- [ ] **Step 1: 写失败测试**

```python
import pytest

from inspect_robots_evo.sandbox import SandboxError, check_source, limited_builtins


def test_math_and_json_are_allowed() -> None:
    check_source("import math\nimport json\n")


@pytest.mark.parametrize(
    "source",
    ["import os", "import subprocess", "from socket import socket"],
)
def test_other_imports_are_rejected(source: str) -> None:
    with pytest.raises(SandboxError):
        check_source(source)


def test_open_call_is_rejected() -> None:
    with pytest.raises(SandboxError):
        check_source("open('x')\n")


def test_runtime_import_of_os_is_rejected() -> None:
    namespace = {"__builtins__": limited_builtins()}
    with pytest.raises(SandboxError):
        exec("import os\n", namespace, namespace)
```

- [ ] **Step 2: 运行测试，确认失败**

Run: PYTHONPATH=plugins/inspect-robots-evo/src .venv/bin/python -m pytest plugins/inspect-robots-evo/tests/test_sandbox.py -q

Expected: FAIL，sandbox 无法导入。

- [ ] **Step 3: 写最小实现**

```python
"""提交代码只能 import math 和 json，也不能调用 open。"""

from __future__ import annotations

import ast


class SandboxError(Exception):
    """代码在执行前或 import 时被拒绝。"""


_ALLOWED = {"math", "json"}
_BANNED_CALLS = {"open", "__import__", "exec", "eval", "compile"}


def check_source(source: str) -> None:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _allow(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                raise SandboxError("relative import is not allowed")
            _allow(node.module)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BANNED_CALLS:
                raise SandboxError(f"call {node.func.id} is not allowed")


def limited_builtins() -> dict[str, object]:
    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if level != 0:
            raise SandboxError("relative import is not allowed")
        _allow(name)
        return __import__(name, globals, locals, fromlist, level)

    return {
        "False": False,
        "None": None,
        "True": True,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "print": print,
        "range": range,
        "round": round,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "__import__": _import,
    }


def _allow(name: str) -> None:
    if name.split(".", 1)[0] not in _ALLOWED:
        raise SandboxError(f"import {name} is not allowed")
```

- [ ] **Step 4: 测试通过**

Run: PYTHONPATH=plugins/inspect-robots-evo/src .venv/bin/python -m pytest plugins/inspect-robots-evo/tests/test_sandbox.py -q

Expected: 6 passed。

- [ ] **Step 5: Commit**

```bash
git add plugins/inspect-robots-evo/src/inspect_robots_evo/sandbox.py plugins/inspect-robots-evo/tests/test_sandbox.py
git commit -m "feat: reject unsafe imports before evo code runs"
```


## Task 3: 原语与一次提交

**Files:**
- Create: plugins/inspect-robots-evo/src/inspect_robots_evo/motion.py
- Create: plugins/inspect-robots-evo/src/inspect_robots_evo/executor.py
- Test: plugins/inspect-robots-evo/tests/test_executor.py

- [ ] **Step 1: 写失败测试**

```python
import threading
from pathlib import Path

import pytest

from inspect_robots_evo.executor import Executor
from inspect_robots_evo.motion import DIM_LABELS, interpolate, reached
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


def test_interpolate_holds_unnamed_dimensions() -> None:
    current = {label: 0.0 for label in DIM_LABELS}
    current["left_z"] = 0.10
    rows, steps = interpolate(current, {"left_z": 0.12})
    assert steps == 2
    assert rows[-1][DIM_LABELS.index("left_z")] == pytest.approx(0.12)
    assert rows[-1][DIM_LABELS.index("left_x")] == 0.0
    assert reached(rows[-1], {"left_z": 0.12})
```

- [ ] **Step 2: 运行测试，确认失败**

Run: PYTHONPATH=plugins/inspect-robots-evo/src .venv/bin/python -m pytest plugins/inspect-robots-evo/tests/test_executor.py -q

Expected: FAIL，executor 无法导入。

- [ ] **Step 3: 写最小实现**

motion.py：

```python
"""绝对末端目标的插补。步长与 Nero 每个控制拍的限幅相同。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

DIM_LABELS: tuple[str, ...] = (
    "left_x", "left_y", "left_z",
    "left_r1", "left_r2", "left_r3", "left_r4", "left_r5", "left_r6",
    "left_gripper",
    "right_x", "right_y", "right_z",
    "right_r1", "right_r2", "right_r3", "right_r4", "right_r5", "right_r6",
    "right_gripper",
)


def _step(label: str) -> float:
    if label.endswith("_gripper") or label[-2:] in ("_x", "_y", "_z"):
        return 0.01
    return 0.05


def interpolate(
    current: Mapping[str, float], targets: Mapping[str, float]
) -> tuple[list[list[float]], int]:
    unknown = set(targets) - set(DIM_LABELS)
    if unknown:
        raise ValueError(f"unknown dimensions: {sorted(unknown)}")
    steps = 1
    for label, goal in targets.items():
        delta = abs(float(goal) - float(current[label]))
        steps = max(steps, math.ceil(delta / _step(label) - 1e-12))
    rows: list[list[float]] = []
    for index in range(1, steps + 1):
        row: list[float] = []
        for label in DIM_LABELS:
            start = float(current[label])
            if label in targets:
                goal = float(targets[label])
                row.append(start + (goal - start) * index / steps)
            else:
                row.append(start)
        rows.append(row)
    return rows, steps


def reached(row: Sequence[float], targets: Mapping[str, float]) -> bool:
    for label, goal in targets.items():
        actual = float(row[DIM_LABELS.index(label)])
        if abs(actual - float(goal)) > _step(label) + 1e-9:
            return False
    return True
```

executor.py：

```python
"""在受限命名空间里执行 solution.py。"""

from __future__ import annotations

import ast
import io
import time
import traceback
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

from inspect_robots_evo.motion import DIM_LABELS
from inspect_robots_evo.sandbox import SandboxError, check_source, limited_builtins
from inspect_robots_evo.workspace import Workspace


class MoveLimit(Exception):
    pass


class TimedOut(Exception):
    pass


class Aborted(Exception):
    pass


@dataclass
class SubmitResult:
    status: str
    value: str | None
    traceback: str | None
    stdout: str
    moves: int


class Executor:
    def __init__(
        self,
        workspace: Workspace,
        port: object,
        *,
        max_moves: int = 80,
        timeout_s: float = 600.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.workspace = workspace
        self.port = port
        self.max_moves = max_moves
        self.timeout_s = timeout_s
        self.clock = clock or time.monotonic

    def run(self, abort) -> SubmitResult:
        buffer = io.StringIO()
        moves = 0
        start = self.clock()
        try:
            namespace = self._load()
        except Exception:
            return SubmitResult("error", None, traceback.format_exc(), "", 0)
        self._bind(namespace, abort, start)

        def move_ee(targets):
            nonlocal moves
            self._guard(abort, start)
            if moves >= self.max_moves:
                raise MoveLimit()
            result = self.port.move(dict(targets), abort)  # type: ignore[attr-defined]
            moves += 1
            if abort.is_set():
                raise Aborted()
            return result

        namespace["move_ee"] = move_ee
        try:
            with redirect_stdout(buffer):
                value = namespace["solve"]()
            if not isinstance(value, str):
                raise TypeError("solve() must return str")
        except MoveLimit:
            return SubmitResult("limit", None, None, buffer.getvalue(), moves)
        except TimedOut:
            return SubmitResult("timeout", None, None, buffer.getvalue(), moves)
        except Aborted:
            return SubmitResult("aborted", None, None, buffer.getvalue(), moves)
        except Exception:
            return SubmitResult("error", None, traceback.format_exc(), buffer.getvalue(), moves)
        return SubmitResult("returned", value, None, buffer.getvalue(), moves)

    def _bind(self, namespace, abort, start) -> None:
        def get_instruction() -> str:
            self._guard(abort, start)
            return (self.workspace.root / "task.md").read_text(encoding="utf-8")

        def get_observation() -> dict[str, object]:
            self._guard(abort, start)
            zeros = {label: 0.0 for label in DIM_LABELS}
            return {
                "left_rgbd": None,
                "right_rgbd": None,
                "chest_rgbd": None,
                "eef_state": zeros,
            }

        namespace["get_instruction"] = get_instruction
        namespace["get_observation"] = get_observation

    def _guard(self, abort, start: float) -> None:
        if abort.is_set():
            raise Aborted()
        if self.clock() - start >= self.timeout_s:
            raise TimedOut()

    def _load(self) -> dict[str, object]:
        namespace: dict[str, object] = {"__builtins__": limited_builtins()}
        defined: set[str] = set()
        skill_dir = self.workspace.root / "skills"
        for path in sorted(skill_dir.glob("*.py")):
            self._exec_skill(path, namespace, defined)
        solution = (self.workspace.root / "submission" / "solution.py").read_text(encoding="utf-8")
        check_source(solution)
        exec(solution, namespace, namespace)  # noqa: S102
        if "solve" not in namespace:
            raise SandboxError("solve() is missing")
        return namespace

    def _exec_skill(self, path: Path, namespace: dict[str, object], defined: set[str]) -> None:
        source = path.read_text(encoding="utf-8")
        check_source(source)
        tree = ast.parse(source)
        names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
        for name in names:
            if name in defined:
                raise SandboxError(f"duplicate skill {name}")
        exec(source, namespace, namespace)  # noqa: S102
        defined.update(names)
```

超时测试里的假时钟第一次就被加到 601，所以 get_instruction 不会先被调用；move_ee 入口的 _guard 会在 port.move 之前抛出 TimedOut。

- [ ] **Step 4: 测试通过**

Run: PYTHONPATH=plugins/inspect-robots-evo/src .venv/bin/python -m pytest plugins/inspect-robots-evo/tests/test_executor.py -q

Expected: 9 passed。

- [ ] **Step 5: Commit**

```bash
git add plugins/inspect-robots-evo/src/inspect_robots_evo/motion.py plugins/inspect-robots-evo/src/inspect_robots_evo/executor.py plugins/inspect-robots-evo/tests/test_executor.py
git commit -m "feat: run one evo submission against a motion port"
```



## Task 4: 模型循环、标签和冻结

**Files:**
- Create: plugins/inspect-robots-evo/src/inspect_robots_evo/episode.py
- Test: plugins/inspect-robots-evo/tests/test_episode.py

- [ ] **Step 1: 写失败测试**

```python
import json
import threading
from pathlib import Path

from inspect_robots_evo.episode import Episode, ScriptedModel, ToolTurn
from inspect_robots_evo.workspace import Workspace


class Port:
    def __init__(self) -> None:
        self.calls = 0

    def move(self, targets, abort):
        self.calls += 1
        return {"held": False, "steps": 1}


def _ws(tmp_path: Path, mode: str = "playground") -> Workspace:
    return Workspace.create(tmp_path, "move the cup", mode=mode, stamp="s")


def test_done_does_not_move(tmp_path: Path) -> None:
    port = Port()
    model = ScriptedModel([ToolTurn("done", {})])
    result = Episode(_ws(tmp_path), port).run(model, threading.Event())
    assert result.reason == "done"
    assert port.calls == 0


def test_submit_then_operator_success_freezes_the_new_skill(tmp_path: Path) -> None:
    workspace = _ws(tmp_path)
    model = ScriptedModel([
        ToolTurn("write_file", {
            "path": "submission/solution.py",
            "text": "def solve():\n    return 'ok'\n",
        }),
        ToolTurn("submit_code", {}),
        ToolTurn("done", {}),
        ToolTurn("write_file", {
            "path": "skills/place.py",
            "text": "def place():\n    return 'p'\n",
        }),
    ])
    result = Episode(workspace, Port(), outcome_reader=lambda: True).run(
        model, threading.Event()
    )
    assert result.reason == "done"
    outcome = json.loads((workspace.root / "outcome.json").read_text(encoding="utf-8"))
    assert outcome["success"] is True
    assert (workspace.root / "freeze" / "skills" / "place.py").is_file()


def test_reflection_cannot_submit(tmp_path: Path) -> None:
    workspace = _ws(tmp_path)
    model = ScriptedModel([ToolTurn("done", {}), ToolTurn("submit_code", {})])
    Episode(workspace, Port(), outcome_reader=lambda: False).run(model, threading.Event())
    assert model.rejected[-1] == "submit_code"


def test_test_mode_has_no_reflection(tmp_path: Path) -> None:
    workspace = _ws(tmp_path, mode="test")
    model = ScriptedModel([
        ToolTurn("done", {}),
        ToolTurn("write_file", {"path": "skills/x.py", "text": "def x():\n    return 1\n"}),
    ])
    Episode(workspace, Port(), outcome_reader=lambda: True).run(model, threading.Event())
    assert model.calls == 1
    assert not (workspace.root / "freeze").exists()


def test_twenty_submits_force_give_up(tmp_path: Path) -> None:
    workspace = _ws(tmp_path)
    workspace.write_file("submission/solution.py", "def solve():\n    return 'ok'\n")
    model = ScriptedModel([ToolTurn("submit_code", {}) for _ in range(21)])
    result = Episode(workspace, Port(), max_submits=20).run(model, threading.Event())
    assert result.reason == "give_up"
    assert result.submits == 20
```

- [ ] **Step 2: 运行测试，确认失败**

Run: PYTHONPATH=plugins/inspect-robots-evo/src .venv/bin/python -m pytest plugins/inspect-robots-evo/tests/test_episode.py -q

Expected: FAIL，episode 无法导入。

- [ ] **Step 3: 写最小实现**

实现文件按下面的行为写完，类名和函数名必须一致：ToolTurn、ScriptedModel、Episode、EpisodeResult、reflect_dir。

ScriptedModel.complete(messages, tools) 把 calls 加 1。队列空时返回 ToolTurn("give_up", {})。工具名不在 tools 里时把名字追加到 rejected 并返回 None。

Episode.run(model, abort)：

- 工具名 read_file、write_file、submit_code、done、give_up。
- abort 置位时 reason 为 aborted。
- submits 达到 max_submits（默认 20）或 model.calls 达到 max_model_calls（默认 80）时 reason 为 give_up，并且不再调用 complete。
- done 的 reason 为 done。give_up 或 complete 返回 None 时 reason 为 give_up。
- submit_code 调用 Executor(workspace, port).run(abort)，把 SubmitResult 写成 submits/000/result.json 这种三位编号，字段为 status、value、traceback、stdout、moves。
- 提交结果 status 为 aborted 时结束循环，reason 为 aborted。
- 循环结束后，outcome_reader 不是 None 时调用它，并把布尔值写入 outcome.json。
- mode 为 playground 且写过标签时，最多 max_reflection_calls（默认 10）次，工具只剩 read_file 和 write_file。complete 返回 None 就停止。然后 workspace.freeze()。
- mode 为 test 不进入反思，也不 freeze。

消息记录用两条：assistant 的 tool_calls，以及 role 为 tool 的结果。两条使用同一个 tool_call_id。先生成 id，再追加，不要用追加之后的列表长度当 id。

reflect_dir(workspace, model) 只在 playground 下跑反思循环并 freeze，不改 outcome.json，也不调用 port。E-STOP 之后的控制台用它。正常结束不要再调一次。

完整的 episode.py：

```python
"""Astra 的读、写、提交、结束，以及 Playground 的反思回合。"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from inspect_robots_evo.executor import Executor, SubmitResult
from inspect_robots_evo.workspace import Workspace

_ACTING = ("read_file", "write_file", "submit_code", "done", "give_up")
_REFLECT = ("read_file", "write_file")


class ToolTurn:
    def __init__(self, name: str, arguments: dict[str, object]) -> None:
        self.name = name
        self.arguments = arguments


class ScriptedModel:
    def __init__(self, turns: list[ToolTurn]) -> None:
        self._turns = list(turns)
        self.calls = 0
        self.rejected: list[str] = []

    def complete(self, messages, tools):
        allowed = {tool["function"]["name"] for tool in tools}
        self.calls += 1
        turn = self._turns.pop(0) if self._turns else ToolTurn("give_up", {})
        if turn.name not in allowed:
            self.rejected.append(turn.name)
            return None
        return turn


def _tools(names: tuple[str, ...]) -> list[dict[str, object]]:
    return [{"type": "function", "function": {"name": name}} for name in names]


@dataclass
class EpisodeResult:
    reason: str
    submits: int


@dataclass
class Episode:
    workspace: Workspace
    port: object
    outcome_reader: Callable[[], bool] | None = None
    max_submits: int = 20
    max_model_calls: int = 80
    max_reflection_calls: int = 10
    messages: list[dict[str, object]] = field(default_factory=list)

    def run(self, model: ScriptedModel, abort) -> EpisodeResult:
        submits = 0
        reason = "give_up"
        while True:
            if abort.is_set():
                reason = "aborted"
                break
            budget = submits >= self.max_submits or model.calls >= self.max_model_calls
            if budget:
                reason = "give_up"
                break
            turn = model.complete(self.messages, _tools(_ACTING))
            if turn is None or turn.name == "give_up":
                reason = "give_up"
                break
            if turn.name == "done":
                reason = "done"
                break
            if turn.name == "read_file":
                text = self.workspace.read_file(str(turn.arguments["path"]))
                self._note(turn, text)
            elif turn.name == "write_file":
                path = str(turn.arguments["path"])
                self.workspace.write_file(path, str(turn.arguments["text"]))
                self._note(turn, "wrote")
            elif turn.name == "submit_code":
                result = Executor(self.workspace, self.port).run(abort)
                self._store(submits, result)
                submits += 1
                detail = f"{result.status} {result.value or ''} {result.traceback or ''}"
                self._note(turn, detail)
                if result.status == "aborted":
                    reason = "aborted"
                    break
        self._close(model)
        return EpisodeResult(reason, submits)

    def _close(self, model: ScriptedModel) -> None:
        if self.outcome_reader is None:
            return
        self.workspace.record_outcome(bool(self.outcome_reader()))
        if self.workspace.mode != "playground":
            return
        for _ in range(self.max_reflection_calls):
            turn = model.complete(self.messages, _tools(_REFLECT))
            if turn is None:
                break
            if turn.name == "read_file":
                text = self.workspace.read_file(str(turn.arguments["path"]))
                self._note(turn, text)
            else:
                path = str(turn.arguments["path"])
                self.workspace.write_file(path, str(turn.arguments["text"]))
                self._note(turn, "wrote")
        self.workspace.freeze()

    def _store(self, index: int, result: SubmitResult) -> None:
        dest = self.workspace.root / "submits" / f"{index:03d}"
        dest.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": result.status,
            "value": result.value,
            "traceback": result.traceback,
            "stdout": result.stdout,
            "moves": result.moves,
        }
        (dest / "result.json").write_text(json.dumps(payload), encoding="utf-8")

    def _note(self, turn: ToolTurn, text: str) -> None:
        call_id = f"call-{len(self.messages)}"
        self.messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": turn.name,
                    "arguments": json.dumps(turn.arguments),
                },
            }],
        })
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})


def reflect_dir(workspace: Workspace, model: ScriptedModel) -> None:
    if workspace.mode != "playground":
        return
    episode = Episode(workspace, port=object())
    for _ in range(episode.max_reflection_calls):
        turn = model.complete(episode.messages, _tools(_REFLECT))
        if turn is None:
            break
        if turn.name == "read_file":
            body = workspace.read_file(str(turn.arguments["path"]))
            episode._note(turn, body)
        else:
            workspace.write_file(str(turn.arguments["path"]), str(turn.arguments["text"]))
            episode._note(turn, "wrote")
    workspace.freeze()
```

- [ ] **Step 4: 测试通过**

Run: PYTHONPATH=plugins/inspect-robots-evo/src .venv/bin/python -m pytest plugins/inspect-robots-evo/tests/test_episode.py -q

Expected: 5 passed。

- [ ] **Step 5: Commit**

```bash
git add plugins/inspect-robots-evo/src/inspect_robots_evo/episode.py plugins/inspect-robots-evo/tests/test_episode.py
git commit -m "feat: add the evo submit and reflection loop"
```


## Task 5: 策略接入 rollout

**Files:**
- Create: plugins/inspect-robots-evo/src/inspect_robots_evo/policy.py
- Modify: plugins/inspect-robots-evo/src/inspect_robots_evo/__init__.py
- Modify: plugins/inspect-robots-evo/src/inspect_robots_evo/executor.py
- Test: plugins/inspect-robots-evo/tests/test_policy.py

- [ ] **Step 1: 写失败测试**

```python
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
```

- [ ] **Step 2: 运行测试，确认失败**

Run: PYTHONPATH=plugins/inspect-robots-evo/src .venv/bin/python -m pytest plugins/inspect-robots-evo/tests/test_policy.py -q

Expected: FAIL，policy 无法导入。

- [ ] **Step 3: 写最小实现**

Executor 增加参数 observation_provider。get_observation 在 provider 不是 None 时返回 provider()，否则返回三路 None 和全零 eef_state。

policy.py 的 RolloutPort.move 把 targets 放到 pending，置位 need，然后每 0.01 秒等待 done。abort 置位时抛出 executor.Aborted。

EvoPolicy.act：

1. 保存 observation。
2. abort 已置位时返回 request_stop，stop_reason 为 aborted，不启动线程。
3. 第一次调用启动守护线程，线程执行 Episode.run，结束时记下 reason 并置位 finished。
4. 若上一次 act 已经发出一块插补，这次先用 reached 比较当前 eef_state 和上次目标。到达则 result 为 held false 和步数；否则 held true 和 steps 0。然后置位 done。
5. 最多等待 2 秒。need 置位时用 interpolate 生成 ActionChunk 并返回。finished 置位时返回 request_stop。否则返回当前姿态的一块保持动作。

eef_state 按 DIM_LABELS 顺序解释。bind 把 embodiment 的 action_space、observation_space、control_hz 抄进 PolicyInfo，name 为 evo。

evo_policy 在 model 为字符串时构造 Provider 和 ResponsesClient。包装器的 complete 调用 client.complete，取第一个 tool call，json.loads 参数，返回 ToolTurn。没有 tool call 时返回 give_up。缺少 model 抛 ValueError。workspace 参数是目录字符串时，用该目录里已有的 task.md 构造 Workspace；mode 来自参数，默认 playground。

__init__.py 从 policy 导出 evo_policy。

- [ ] **Step 4: 测试通过**

Run: PYTHONPATH=plugins/inspect-robots-evo/src .venv/bin/python -m pytest plugins/inspect-robots-evo/tests -q

Expected: 全部 passed。

- [ ] **Step 5: Commit**

```bash
git add plugins/inspect-robots-evo
git commit -m "feat: register the evo policy"
```

## Task 6: 控制台

**Files:**
- Modify: plugins/inspect-robots-nero/scripts/mission_console.py
- Test: plugins/inspect-robots-evo/tests/test_console_command.py

- [ ] **Step 1: 写失败测试**

```python
import argparse
from importlib.machinery import SourceFileLoader
from pathlib import Path


def _console():
    path = Path("plugins/inspect-robots-nero/scripts/mission_console.py")
    return SourceFileLoader("mission_console_evo", str(path)).load_module()


def test_evo_command_uses_unlimited_steps_and_the_workspace() -> None:
    console = _console()
    args = argparse.Namespace(
        policy="evo",
        model="mog-10-a",
        base_url="https://kspmas.ksyun.com/v1",
        api_key_env="JINSHANYUN_CHATGPT_KEY",
        max_speed_frac=0.05,
        evo_mode="playground",
        evo_workspace="/tmp/evo-task",
    )
    _prefix, suffix = console.build_command(args)
    assert suffix[suffix.index("--policy") + 1] == "evo"
    assert suffix[suffix.index("--max-steps") + 1] == "0"
    assert "mode=playground" in suffix
    assert "workspace=/tmp/evo-task" in suffix
```

- [ ] **Step 2: 运行测试，确认失败**

Run: PYTHONPATH=plugins/inspect-robots-evo/src .venv/bin/python -m pytest plugins/inspect-robots-evo/tests/test_console_command.py -q

Expected: FAIL，build_command 仍拼出 agent，或缺少 mode 和 workspace。

- [ ] **Step 3: 写最小实现**

三处白名单加上 evo：build_parser 的 choices、_api_start 的策略元组、页面 select 增加 option value="evo"，文字为 evo (EmbodiedEvo)。

build_parser 增加 --evo-mode，choices 为 playground 和 test，默认 playground。

build_command 在 vla 分支之前处理 policy 为 evo 的 suffix：--policy evo，-P model，-P wire=responses，-P base_url，-P api_key_env，-P mode，-P workspace，--max-steps 0。不要加 max_speed_frac。

_api_start 在 policy 为 evo 时创建 .inspect_traces/evo/YYYYmmdd-HHMMSS_4位hex 目录，记在 server 上，并放进传给 build_suffix 的 namespace 的 evo_workspace。请求体里的 evo_mode 覆盖默认值。

POST /api/evo-outcome 的 body 是 {"success": true} 或 {"success": false}。没有 workspace 时返回 400。有则写入 outcome.json。GET /api/status 在策略是 evo、状态是 ended、且 outcome.json 不存在时增加 needs_outcome true。页面在该字段为真时显示两个按钮，分别 POST true 和 false。

E-STOP 仍走现有 force_end 和阻尼。只有 E-STOP 把进程杀掉、标签随后才写入时，控制台调用 reflect_dir，不启动机器人。正常结束的进程自己在 Episode._close 里反思，控制台不要再调用一次。

- [ ] **Step 4: 测试通过**

Run: PYTHONPATH=plugins/inspect-robots-evo/src .venv/bin/python -m pytest plugins/inspect-robots-evo/tests -q

Expected: 全部 passed。

Run: .venv/bin/ruff check plugins/inspect-robots-evo plugins/inspect-robots-nero/scripts/mission_console.py

Expected: 无输出。

- [ ] **Step 5: Commit**

```bash
git add plugins/inspect-robots-evo plugins/inspect-robots-nero/scripts/mission_console.py
git commit -m "feat: start evo runs from the mission console"
```

## 规格对照

- 新插件和策略名 evo：Task 1、Task 5。
- 不改 agent、harness、vla 的策略实现：Task 6 只改控制台。
- 不操作 CAN、不实现 IK：动作只通过 MotionPort 和 ActionChunk 交给现有 embodiment。
- 工作区、冻结、Test 只读：Task 1。
- 三个原语、80 次、600 秒、非字符串返回、技能重名：Task 3。
- done、give_up、20 次提交、操作员标签、反思回合不能提交：Task 4。
- 沙箱：Task 2。
- 控制台启动和标签：Task 6。
- 真机 Playground 一次加 Test 一次留在实现之后，由操作员做，不写进自动测试。
