"""Astra 的读, 写, 提交, 结束, 以及 Playground 的反思回合."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from inspect_robots_evo.executor import Executor, MotionPort, SubmitResult
from inspect_robots_evo.workspace import Workspace

_ACTING = ("read_file", "write_file", "submit_code", "done", "give_up")
_REFLECT = ("read_file", "write_file")


class ToolTurn:
    """模型请求的一次工具调用."""

    def __init__(self, name: str, arguments: dict[str, object]) -> None:
        self.name = name
        self.arguments = arguments


class ScriptedModel:
    """按给定顺序返回工具调用, 供测试替换 Astra."""

    def __init__(self, turns: list[ToolTurn]) -> None:
        self._turns = list(turns)
        self.calls = 0
        self.rejected: list[str] = []

    def complete(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> ToolTurn | None:
        """返回下一轮工具. 不允许的工具记入 rejected 并返回 None."""
        allowed = {tool["function"]["name"] for tool in tools}
        self.calls += 1
        turn = self._turns.pop(0) if self._turns else ToolTurn("give_up", {})
        if turn.name not in allowed:
            self.rejected.append(str(turn.name))
            return None
        return turn


def _tools(names: tuple[str, ...]) -> list[dict[str, object]]:
    return [{"type": "function", "function": {"name": name}} for name in names]


def _stopped(abort: object) -> bool:
    is_set = getattr(abort, "is_set", None)
    return bool(is_set()) if callable(is_set) else False


@dataclass
class EpisodeResult:
    """一次任务执行阶段的结束原因和提交次数."""

    reason: str
    submits: int


@dataclass
class Episode:
    """执行阶段的工具循环, 以及 Playground 结束后的改文件回合."""

    workspace: Workspace
    port: MotionPort
    outcome_reader: Callable[[], bool] | None = None
    observation_provider: Callable[[], dict[str, object]] | None = None
    max_submits: int = 20
    max_model_calls: int = 80
    max_reflection_calls: int = 10
    messages: list[dict[str, object]] = field(default_factory=list)

    def run(self, model: ScriptedModel, abort: object) -> EpisodeResult:
        """跑到 done, give_up, 预算用尽或中断, 然后按模式收尾."""
        submits = 0
        reason = "give_up"
        while True:
            if _stopped(abort):
                reason = "aborted"
                break
            over_budget = submits >= self.max_submits or model.calls >= self.max_model_calls
            if over_budget:
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
                result = Executor(
                    self.workspace,
                    self.port,
                    observation_provider=self.observation_provider,
                ).run(abort)
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
        self._reflect(model)
        self.workspace.freeze()

    def _reflect(self, model: ScriptedModel) -> None:
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
        self.messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": turn.name,
                            "arguments": json.dumps(turn.arguments),
                        },
                    }
                ],
            }
        )
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})


def reflect_dir(workspace: Workspace, model: ScriptedModel) -> None:
    """只更新技能和教训并冻结. 不改成败标签, 也不驱动机器人."""
    if workspace.mode != "playground":
        return
    episode = Episode(workspace, port=_IdlePort())
    episode._reflect(model)
    workspace.freeze()


class _IdlePort:
    def move(self, targets: dict[str, float], abort: object) -> dict[str, object]:
        raise RuntimeError("reflection does not move the robot")
