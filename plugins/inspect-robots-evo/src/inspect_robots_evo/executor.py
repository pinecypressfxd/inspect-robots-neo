"""在受限命名空间里执行 solution.py."""

from __future__ import annotations

import ast
import io
import time
import traceback
from collections.abc import Callable, Mapping
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from inspect_robots_evo.motion import DIM_LABELS
from inspect_robots_evo.sandbox import SandboxError, check_source, limited_builtins
from inspect_robots_evo.workspace import Workspace


class MoveLimit(Exception):
    """一次提交的 move_ee 次数已经用完."""


class TimedOut(Exception):
    """一次提交的墙钟已经用完."""


class Aborted(Exception):
    """操作员中断了这次提交."""


class MotionPort(Protocol):
    """一次末端动作. 返回 held 和 steps."""

    def move(self, targets: Mapping[str, float], abort: object) -> Mapping[str, object]:
        """执行 targets, 并在 abort 置位时尽快返回."""


@dataclass
class SubmitResult:
    """一次 solve() 的状态, 供 Astra 阅读."""

    status: str
    value: str | None
    traceback: str | None
    stdout: str
    moves: int


class Executor:
    """载入 skills 和 solution.py, 然后调用 solve()."""

    def __init__(
        self,
        workspace: Workspace,
        port: MotionPort,
        *,
        max_moves: int = 80,
        timeout_s: float = 600.0,
        clock: Callable[[], float] | None = None,
        observation_provider: Callable[[], Mapping[str, object]] | None = None,
    ) -> None:
        self.workspace = workspace
        self.port = port
        self.max_moves = max_moves
        self.timeout_s = timeout_s
        self.clock = clock or time.monotonic
        self.observation_provider = observation_provider

    def run(self, abort: object) -> SubmitResult:
        """执行 solve(), 直到返回, 超限, 超时或中断."""
        buffer = io.StringIO()
        moves = 0
        start = self.clock()
        try:
            namespace = self._load()
        except Exception:
            return SubmitResult("error", None, traceback.format_exc(), "", 0)
        self._bind(namespace, abort, start)

        def move_ee(targets: Mapping[str, float]) -> Mapping[str, object]:
            nonlocal moves
            self._guard(abort, start)
            if moves >= self.max_moves:
                raise MoveLimit()
            result = self.port.move(dict(targets), abort)
            moves += 1
            if getattr(abort, "is_set", lambda: False)():
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

    def _bind(self, namespace: dict[str, object], abort: object, start: float) -> None:
        def get_instruction() -> str:
            self._guard(abort, start)
            return (self.workspace.root / "task.md").read_text(encoding="utf-8")

        def get_observation() -> dict[str, object]:
            self._guard(abort, start)
            if self.observation_provider is not None:
                return dict(self.observation_provider())
            zeros = dict.fromkeys(DIM_LABELS, 0.0)
            return {
                "left_rgbd": None,
                "right_rgbd": None,
                "chest_rgbd": None,
                "eef_state": zeros,
            }

        namespace["get_instruction"] = get_instruction
        namespace["get_observation"] = get_observation

    def _guard(self, abort: object, start: float) -> None:
        if getattr(abort, "is_set", lambda: False)():
            raise Aborted()
        if self.clock() - start >= self.timeout_s:
            raise TimedOut()

    def _load(self) -> dict[str, object]:
        namespace: dict[str, object] = {"__builtins__": limited_builtins()}
        defined: set[str] = set()
        skill_dir = self.workspace.root / "skills"
        for path in sorted(skill_dir.glob("*.py")):
            self._exec_skill(path, namespace, defined)
        solution_path = self.workspace.root / "submission" / "solution.py"
        solution = solution_path.read_text(encoding="utf-8")
        check_source(solution)
        exec(solution, namespace, namespace)
        if "solve" not in namespace:
            raise SandboxError("solve() is missing")
        return namespace

    def _exec_skill(
        self, path: Path, namespace: dict[str, object], defined: set[str]
    ) -> None:
        source = path.read_text(encoding="utf-8")
        check_source(source)
        tree = ast.parse(source)
        names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
        for name in names:
            if name in defined:
                raise SandboxError(f"duplicate skill {name}")
        exec(source, namespace, namespace)
        defined.update(names)
