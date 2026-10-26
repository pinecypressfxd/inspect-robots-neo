"""HarnessVLA: analytic primitives, a trial memory, and restaged VLA retries.

The planner (Astra, through the agent plugin's chat wire) may call only the
fixed library below. ``move_to``, ``set_gripper``, and ``release`` are
analytic. ``vla_act`` is the one VLA primitive and only grasps a red cup: the
runtime fills the trained sentence, and the attempt runs until its time bound
or a tracking interrupt. A later ``vla_act`` is refused until an analytic
primitive has restaged the arm. ``remember`` appends a fact to the trial
memory, which is attached to every later observation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from inspect_robots import ActionChunk, Observation
from inspect_robots.errors import ConfigError
from inspect_robots_agent._llm import ToolCall
from inspect_robots_vla.executor import ExecutorPolicy, ExecutorPolicyConfig, _LibraryTurn

_OPEN_WIDTH_M = 0.09
_MEMORY_LIMIT = 16
_HOME_Z_SLACK_M = 0.02
_LEFT_Z_INDEX = 2
# The 1008 checkpoint's task string. The 3 s cap keeps the grasp and cuts
# the place that follows it.
_VLA_GRASP_PROMPT = "please pick the red cup and place it on the black mat"
# The 1008 skill places the cup inside an 8 s window. Keep one attempt
# inside the grasp and cut it before that place.
_VLA_ATTEMPT_LIMIT_S = 3.0

_HARNESS_TEMPLATE = """You are the HarnessVLA planner for a real embodiment named \
{name!r}. Carry out the Goal in the order it states. You have a fixed \
primitive library and a trial memory. Respond with exactly one tool call per \
turn. Read the trial memory before every call. Lines that start with \
operator: are the operator's live guidance; follow them on the next motion. \
The first tool call must \
move an arm (move_to, set_gripper, release, or vla_act). remember does not \
move the arm; store a fact with it only after a motion, about where an \
object is or why an attempt failed. give_up only when the cameras show the \
Goal is impossible.

Analytic primitives, for every object:
- move_to: absolute end-effector targets. Unnamed dimensions hold their \
current value. Near a container, raise z first, translate in xy at constant \
z, then lower z. Standing objects already in the workspace (cups, blocks, \
and objects just placed) are obstacles, the same as a container rim. Before \
any horizontal move, raise z alone until the point 0.03 m below the TCP and \
any held object clear the tallest object on the path, then translate at that \
constant height. Lower vertically only after the hand is centered on an \
empty landing spot. When the destination already holds an object, do not \
descend past its side; if the Goal asks to stack, move directly above that \
object first, then lower vertically. This overrides the embodiment note that \
a diagonal low carry is fine.
- set_gripper: absolute width in meters, 0 closed to 0.09 open. Grasp blocks, \
jars, and every object other than the red cup this way.
- release: open one gripper to 0.09 while holding position. Use it to put an \
object down at the destination the Goal names.

VLA primitive, the red cup only:
- vla_act(max_seconds): one grasp attempt, and only when that hand is already \
close to the red cup and the Goal names a red cup. The runtime sends the \
trained sentence. The attempt runs until max_seconds or a tracking \
interrupt. The runtime caps every attempt at {limit:.0f} s, even when \
max_seconds is longer. Do not raise the arm before vla_act. The trained \
grasp starts at this trial's home height. If the left hand is more than \
0.02 m above that height, move_to left_z back to it before vla_act. The \
first vla_act may use the current pose only at that height. Every later \
vla_act must be preceded by move_to, set_gripper, or release. After {attempts} \
attempts, grasp the red cup with set_gripper instead.

When the cameras show a grasp, set_gripper locks the current closing width \
and move_to carries the object to the destination the Goal names. Call done \
when every object in the Goal is in its destination. You have a budget of \
{budget} LLM calls for the whole trial."""

_HARNESS_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "move_to",
            "description": (
                "Analytic free-space motion to absolute Cartesian end-effector "
                "targets. Unnamed dimensions hold their current value. Raise "
                "above standing objects before any horizontal move, then "
                "lower only when centered. Use it to approach, carry, and "
                "restage before another vla_act."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "targets": {
                        "type": "object",
                        "description": "Map of dimension name to absolute value.",
                    },
                    "note": {"type": "string"},
                },
                "required": ["targets", "note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_gripper",
            "description": (
                "Set one gripper's absolute opening width in meters "
                "(0 closed to 0.09 open) while holding position. This is the "
                "grasp for every object other than the red cup, and the lock "
                "after a red-cup grasp."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "arm": {"type": "string", "description": "left or right"},
                    "width_m": {"type": "number"},
                    "note": {"type": "string"},
                },
                "required": ["arm", "width_m", "note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "release",
            "description": (
                "Analytic primitive: open one gripper to 0.09 while holding "
                "position, to put an object down."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "arm": {"type": "string", "description": "left or right"},
                    "note": {"type": "string"},
                },
                "required": ["arm", "note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "vla_act",
            "description": (
                "One VLA grasp attempt for the red cup, only when the Goal "
                "names a red cup and this hand is already close. The runtime "
                "sends the trained sentence. Runs until max_seconds or a "
                "tracking interrupt. The runtime caps the attempt at 3 s. "
                "Refused when the left hand is more than 0.02 m above this "
                "trial's home height; lower left_z to home first. "
                "A repeat requires an analytic restage "
                "(move_to, set_gripper, or release) since the previous attempt."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_seconds": {
                        "type": "number",
                        "description": "Upper bound for this attempt, in seconds.",
                    },
                },
                "required": ["max_seconds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Append one concrete fact to the trial memory. The arm does "
                "not move. Read this memory before retrying a primitive."
            ),
            "parameters": {
                "type": "object",
                "properties": {"fact": {"type": "string"}},
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Declare the Goal finished. The trial ends.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "hindsight": {"type": "string"},
                },
                "required": ["summary", "hindsight"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "give_up",
            "description": "Stop because the cameras show the Goal is impossible.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "hindsight": {"type": "string"},
                },
                "required": ["reason", "hindsight"],
            },
        },
    },
]


@dataclass(frozen=True)
class HarnessVlaConfig(ExecutorPolicyConfig):
    """Hybrid config plus the VLA attempt cap recorded in the eval log."""

    max_vla_attempts: int = 3
    name: str = "harness"


class HarnessVlaPolicy(ExecutorPolicy):
    """HarnessVLA planner: analytic primitives, memory, and restaged retries.

    One ``vla_act`` is one red-cup grasp attempt. Healthy tracking does not
    hand control back at the checkpoint; the attempt ends at its time bound
    or a tracking interrupt. The next ``vla_act`` is refused until
    ``move_to``, ``set_gripper``, or ``release`` has run, and after
    ``max_vla_attempts`` the red cup must be grasped analytically.
    """

    accepts_operator_messages = True
    _memory: list[str]
    _operator_seen: set[tuple[int | None, str]]
    _home_left_z: float | None
    _vla_attempts: int
    _staged_since_vla: bool
    _max_vla_attempts: int

    def __init__(self, *args: Any, max_vla_attempts: int = 3, **kwargs: Any) -> None:
        if (
            isinstance(max_vla_attempts, bool)
            or not isinstance(max_vla_attempts, int)
            or max_vla_attempts < 1
        ):
            raise ConfigError(f"max_vla_attempts must be an int >= 1, got {max_vla_attempts!r}")
        self._max_vla_attempts = max_vla_attempts
        kwargs["name"] = "harness"
        super().__init__(*args, **kwargs)
        recorded = asdict(self.config)
        recorded["max_vla_attempts"] = max_vla_attempts
        recorded["name"] = "harness"
        self.config = HarnessVlaConfig(**recorded)

    def _rearm_trial_state(self) -> None:
        super()._rearm_trial_state()
        self._memory = []
        self._operator_seen = set()
        self._home_left_z = None
        self._vla_attempts = 0
        self._staged_since_vla = True

    def act(self, observation: Observation) -> ActionChunk:
        """Store live operator guidance before the executor step.

        Guidance typed during an analytic motion is read on the next planner
        turn. A running VLA segment keeps going.
        """
        self._absorb_operator(observation)
        self._remember_home(observation)
        return super().act(observation)

    def _absorb_operator(self, observation: Observation) -> bool:
        messages = observation.extra.get("operator_messages")
        if not isinstance(messages, list):
            return False
        fresh = False
        for message in messages:
            if not isinstance(message, dict):
                continue
            text = message.get("text")
            if not isinstance(text, str):
                continue
            cleaned = " ".join(text.split())
            if not cleaned:
                continue
            if len(cleaned) > 240:
                cleaned = cleaned[:240]
            step = message.get("t")
            key = (step if isinstance(step, int) else None, cleaned)
            if key in self._operator_seen:
                continue
            self._operator_seen.add(key)
            self._remember(f"operator: {cleaned}")
            fresh = True
        return fresh

    def _remember_home(self, observation: Observation) -> None:
        """Keep the left-hand height of the first pose in this trial."""
        if self._home_left_z is not None:
            return
        state = observation.state.get("eef_state")
        if state is None:
            return
        array = np.asarray(state, dtype=np.float64)
        if array.shape[-1] <= _LEFT_Z_INDEX:
            return
        self._home_left_z = float(array[_LEFT_Z_INDEX])

    def _home_height_block(self, observation: Observation) -> str | None:
        """Refuse a grasp whose start is above the height the skill was trained from."""
        if self._home_left_z is None:
            return None
        height = float(np.asarray(self._require_eef_state(observation))[_LEFT_Z_INDEX])
        if height <= self._home_left_z + _HOME_Z_SLACK_M:
            return None
        return (
            f"left_z is {height:.3f}, more than 0.02 m above this trial's "
            f"home {self._home_left_z:.3f}. Lower left_z to {self._home_left_z:.3f} "
            "with move_to before vla_act. Do not raise the arm to leave "
            "clearance before a grasp."
        )

    def _segment_outcome(self, observation: Observation) -> str | None:
        """End on tracking abort or the time bound, not on a healthy checkpoint."""
        outcome = super()._segment_outcome(observation)
        if outcome == "checkpoint":
            return None
        return outcome

    def _planner_tools(self) -> list[dict[str, Any]]:
        """The fixed library. Goals without a red cup omit ``vla_act``."""
        if self._goal_allows_vla():
            return _HARNESS_TOOLS
        hidden = {"vla_act"}
        return [tool for tool in _HARNESS_TOOLS if tool["function"]["name"] not in hidden]

    def _system_prompt(self) -> str:
        """Harness role text, the trained sentence when relevant, then embodiment notes."""
        formatted = _HARNESS_TEMPLATE.format(
            name=self._embodiment_name,
            budget=self._budget_llm_calls,
            attempts=self._max_vla_attempts,
            limit=_VLA_ATTEMPT_LIMIT_S,
        )
        if self._goal_allows_vla():
            formatted += (
                "\n\nTrained skill: pass no subgoal. vla_act always sends "
                f"{_VLA_GRASP_PROMPT!r}. Any other wording is not available."
            )
        else:
            formatted += (
                "\n\nThis goal does not name a red cup, so vla_act is unavailable. "
                "Grasp and place with move_to, set_gripper, and release."
            )
        docs = self._embodiment_docs
        if docs is not None and docs.strip():
            formatted = formatted + "\n\nEmbodiment notes:\n" + docs.strip()
        return formatted

    def _observation_content(self, observation: Observation) -> list[dict[str, Any]]:
        """The executor view plus the trial memory the planner must read."""
        parts = super()._observation_content(observation)
        first = parts[0]
        if isinstance(first, dict) and first.get("type") == "text":
            body = "(empty)" if not self._memory else "\n".join(self._memory)
            first["text"] = str(first["text"]) + "\n\nTrial memory:\n" + body
        return parts

    def _on_analytic(self, call: ToolCall) -> None:
        self._staged_since_vla = True
        self._remember(f"{call.name} accepted; the arm may vla_act again")

    def _on_segment_end(self, outcome: str) -> None:
        self._staged_since_vla = False
        self._remember(
            f"vla_act attempt {self._vla_attempts} ended: {outcome}; "
            f"steps={self._segment_steps}. Restage before another vla_act."
        )

    def _begin_segment(self, call: ToolCall, subgoal: str, granted_s: float) -> None:
        super()._begin_segment(call, subgoal, granted_s)
        self._messages[-1]["content"] = (
            f"vla_act attempt {self._vla_attempts} of {self._max_vla_attempts} "
            f"for up to {granted_s:.3g}s. The VLA prompt is the trained sentence. "
            "This attempt runs until the time bound or a tracking interrupt. "
            "Then read the trial memory and restage with move_to, set_gripper, "
            "or release before another vla_act."
        )

    def _library_turn(self, call: ToolCall, observation: Observation) -> _LibraryTurn | None:
        if call.name == "remember":
            return self._remember_turn(call)
        if call.name == "release":
            return self._release_turn(call, observation)
        if call.name == "vla_act":
            return self._vla_turn(call, observation)
        return None

    def _remember(self, fact: str) -> None:
        self._memory.append(fact)
        overflow = len(self._memory) - _MEMORY_LIMIT
        if overflow > 0:
            self._memory = self._memory[overflow:]
            self._memory[0] = f"(earlier entries omitted) {self._memory[0]}"

    def _arguments(self, call: ToolCall) -> dict[str, Any] | None:
        try:
            arguments = json.loads(call.arguments)
        except (TypeError, ValueError):
            return None
        if not isinstance(arguments, dict):
            return None
        return arguments

    def _remember_turn(self, call: ToolCall) -> _LibraryTurn:
        arguments = self._arguments(call)
        fact = "" if arguments is None else arguments.get("fact")
        if not isinstance(fact, str) or not fact.strip():
            return _LibraryTurn(reply="remember fact must be a non-empty string", failed=True)
        text = " ".join(fact.split())
        if len(text) > 240:
            text = text[:240]
        self._remember(f"note: {text}")
        return _LibraryTurn(reply=f"remembered: {text}")

    def _release_turn(self, call: ToolCall, observation: Observation) -> _LibraryTurn:
        arguments = self._arguments(call)
        if arguments is None:
            return _LibraryTurn(reply="release arguments must be a JSON object", failed=True)
        note = arguments.get("note")
        if not isinstance(note, str) or not note.strip():
            return _LibraryTurn(reply="release note is required", failed=True)
        arm = str(arguments.get("arm", "")).strip().lower()
        synthetic = ToolCall(
            id=call.id,
            name="set_gripper",
            arguments=json.dumps({"arm": arm, "width_m": _OPEN_WIDTH_M, "note": note}),
        )
        moved, error = self._analytic_move(synthetic, observation)
        if moved is None or error is not None:
            return _LibraryTurn(reply=error or "release produced no motion", failed=True)
        self._on_analytic(call)
        return _LibraryTurn(reply="gripper released to 0.09", chunk=moved)

    def _vla_turn(self, call: ToolCall, observation: Observation) -> _LibraryTurn:
        if not self._goal_allows_vla():
            return _LibraryTurn(
                reply=(
                    "this goal does not name a red cup; grasp with move_to and "
                    "set_gripper. vla_act is only for a red cup"
                ),
                failed=True,
            )
        blocked = self._home_height_block(observation)
        if blocked is not None:
            return _LibraryTurn(reply=blocked, failed=True)
        if self._vla_attempts >= self._max_vla_attempts:
            return _LibraryTurn(
                reply=(
                    f"vla_act attempt cap {self._max_vla_attempts} is used; "
                    "grasp the red cup with set_gripper"
                ),
                failed=True,
            )
        if not self._staged_since_vla:
            return _LibraryTurn(
                reply="restage with move_to, set_gripper, or release before another vla_act",
                failed=True,
            )
        arguments = self._arguments(call)
        raw = None if arguments is None else arguments.get("max_seconds")
        if (
            isinstance(raw, bool)
            or not isinstance(raw, int | float)
            or not np.isfinite(raw)
            or raw <= 0
        ):
            return _LibraryTurn(
                reply=f"max_seconds must be a finite number > 0, got {raw!r}",
                failed=True,
            )
        delegated = ToolCall(
            id=call.id,
            name="delegate_skill",
            arguments=json.dumps(
                {
                    "subgoal": _VLA_GRASP_PROMPT,
                    "max_seconds": min(float(raw), _VLA_ATTEMPT_LIMIT_S),
                }
            ),
        )
        subgoal, granted = self._parse_delegation(delegated)
        if subgoal is None:
            return _LibraryTurn(reply=self._delegation_error, failed=True)
        self._vla_attempts += 1
        self._staged_since_vla = False
        self._begin_segment(delegated, subgoal, granted)
        self._remember(f"vla_act attempt {self._vla_attempts} started for {granted:.3g}s")
        return _LibraryTurn(chunk=self._execute_chunk(observation))


def harness_policy(
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    **kwargs: Any,
) -> HarnessVlaPolicy:
    """Registry factory for the ``harness`` policy (entry point ``harness``).

    Accepts the same arguments as
    [`HarnessVlaPolicy`][inspect_robots_harness.policy.HarnessVlaPolicy]; the CLI
    forwards each ``-P key=value`` pair here.
    """
    return HarnessVlaPolicy(model, base_url, api_key_env, **kwargs)
