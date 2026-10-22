"""把受限程序的 move_ee 接到 rollout 的动作块上."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np
from inspect_robots_agent._llm import Provider
from inspect_robots_agent._responses import ResponsesClient

from inspect_robots.policy import PolicyBase, PolicyConfig, PolicyInfo
from inspect_robots.spaces import Box
from inspect_robots.types import Action, ActionChunk, Observation
from inspect_robots_evo.episode import Episode, ScriptedModel, ToolTurn
from inspect_robots_evo.executor import Aborted
from inspect_robots_evo.motion import DIM_LABELS, interpolate, reached
from inspect_robots_evo.workspace import Workspace


class RolloutPort:
    """把一次 move_ee 交给下一次 act(), 直到这段动作执行完."""

    def __init__(self, abort: threading.Event) -> None:
        self.abort = abort
        self.pending: dict[str, float] | None = None
        self.result: dict[str, object] | None = None
        self.need = threading.Event()
        self.done = threading.Event()

    def move(self, targets: dict[str, float], abort: object) -> dict[str, object]:
        """阻塞到 act() 交回 held 和 steps."""
        if _flag(abort) or self.abort.is_set():
            raise Aborted()
        self.result = None
        self.pending = dict(targets)
        self.done.clear()
        self.need.set()
        while not self.done.wait(0.01):
            if _flag(abort) or self.abort.is_set():
                raise Aborted()
        return dict(self.result or {})


class ResponsesModel:
    """把 ResponsesClient 的一次工具调用转成 ToolTurn."""

    def __init__(self, client: ResponsesClient) -> None:
        self.client = client

    def complete(
        self, messages: list[dict[str, object]], tools: list[dict[str, object]]
    ) -> ToolTurn:
        """取第一个工具调用. 没有工具调用时视为 give_up."""
        reply = self.client.complete(messages, tools)
        if not reply.tool_calls:
            return ToolTurn("give_up", {})
        call = reply.tool_calls[0]
        arguments = json.loads(call.arguments or "{}")
        return ToolTurn(call.name, arguments)


class EvoPolicy(PolicyBase):
    """一次任务里由 Astra 提交代码, 由 rollout 执行末端动作."""

    def __init__(
        self,
        model: ScriptedModel | ResponsesModel,
        workspace: Workspace,
        outcome_reader=None,
    ) -> None:
        self.model = model
        self.workspace = workspace
        self.outcome_reader = outcome_reader
        self.abort = threading.Event()
        self.port = RolloutPort(self.abort)
        self.info = PolicyInfo(name="evo", action_space=Box(shape=(20,)))
        self.config = PolicyConfig()
        self._observation: Observation | None = None
        self._started = False
        self._awaiting = False
        self._targets: dict[str, float] = {}
        self._steps = 0
        self._reason = "give_up"
        self._finished = threading.Event()

    def bind(self, embodiment_info) -> None:
        """采用本体自己的动作空间和观察空间."""
        self.info = PolicyInfo(
            name="evo",
            action_space=embodiment_info.action_space,
            observation_space=embodiment_info.observation_space,
            control_hz=embodiment_info.control_hz,
        )

    def act(self, observation: Observation) -> ActionChunk:
        """发下一段插补, 或在任务结束时请求停止."""
        self._observation = observation
        if self.abort.is_set():
            self.port.done.set()
            return self._stop("aborted")
        if not self._started:
            self._started = True
            threading.Thread(target=self._run, daemon=True).start()
        if self._awaiting:
            self._finish_motion()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if self.abort.is_set():
                self.port.done.set()
                return self._stop("aborted")
            if self.port.need.is_set() and self.port.pending is not None:
                return self._emit(self.port.pending)
            if self._finished.is_set() and not self._awaiting:
                return self._stop(self._reason)
            time.sleep(0.01)
        return self._hold()

    def _run(self) -> None:
        try:
            episode = Episode(
                self.workspace,
                self.port,
                outcome_reader=self.outcome_reader,
                observation_provider=self._provider,
            )
            self._reason = episode.run(self.model, self.abort).reason
        finally:
            self._finished.set()

    def _finish_motion(self) -> None:
        row = [self._eef_map()[label] for label in DIM_LABELS]
        held = not reached(row, self._targets)
        self.port.result = {"held": held, "steps": 0 if held else self._steps}
        self._awaiting = False
        self.port.need.clear()
        self.port.done.set()

    def _emit(self, targets: dict[str, float]) -> ActionChunk:
        self.port.need.clear()
        rows, steps = interpolate(self._eef_map(), targets)
        self._steps = steps
        self._targets = dict(targets)
        self._awaiting = True
        actions = [Action(np.asarray(row, dtype=np.float64), {}) for row in rows]
        return ActionChunk(actions)

    def _provider(self) -> dict[str, object]:
        images = self._observation.images if self._observation is not None else {}
        return {
            "left_rgbd": images.get("left_rgbd"),
            "right_rgbd": images.get("right_rgbd"),
            "chest_rgbd": images.get("chest_rgbd"),
            "eef_state": self._eef_map(),
        }

    def _eef_map(self) -> dict[str, float]:
        if self._observation is None or "eef_state" not in self._observation.state:
            return dict.fromkeys(DIM_LABELS, 0.0)
        values = self._observation.state["eef_state"]
        return {label: float(values[index]) for index, label in enumerate(DIM_LABELS)}

    def _stop(self, reason: str) -> ActionChunk:
        action = Action(self._pose(), {"request_stop": True, "stop_reason": reason})
        return ActionChunk([action])

    def _hold(self) -> ActionChunk:
        return ActionChunk([Action(self._pose(), {})])

    def _pose(self) -> np.ndarray:
        return np.asarray([self._eef_map()[label] for label in DIM_LABELS], dtype=np.float64)


def evo_policy(**kwargs: object) -> EvoPolicy:
    """按命令行参数构造策略. 没有 model 时拒绝启动."""
    model = kwargs.get("model")
    if model is None:
        raise ValueError("evo requires a model")
    mode = str(kwargs.get("mode", "playground"))
    workspace = _workspace(kwargs.get("workspace"), mode)
    if isinstance(model, str):
        key_env = str(kwargs.get("api_key_env", ""))
        client = ResponsesClient(
            Provider(
                base_url=str(kwargs.get("base_url", "")),
                api_key=os.environ.get(key_env, ""),
                model=model,
                wire="responses",
            )
        )
        return EvoPolicy(ResponsesModel(client), workspace)
    if not hasattr(model, "complete"):
        raise ValueError("evo model must be a name or a complete() client")
    return EvoPolicy(model, workspace)


def _workspace(value: object, mode: str) -> Workspace:
    if isinstance(value, Workspace):
        return value
    if not isinstance(value, str):
        raise ValueError("evo requires a workspace path or Workspace")
    root = Path(value)
    return Workspace(root, mode)


def _flag(abort: object) -> bool:
    is_set = getattr(abort, "is_set", None)
    return bool(is_set()) if callable(is_set) else False
