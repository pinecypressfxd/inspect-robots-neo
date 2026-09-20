"""HybridPolicy: an LLM planner delegating skill segments to the VLA.

Helix-style layering (plan 0084): the planner (Astra through the agent
plugin's chat wire) decomposes the goal and hands each short skill to the
``umi-replay`` VLA execution loop; at every checkpoint, tracking abort, or
time cap it sees the latest observation plus a compact progress note and
decides again. The VLA is the hands, the LLM is the brain, and ``done`` /
``give_up`` stay with the LLM. As with the pure adapter, this module adds no
safety logic of its own: every emitted action still passes the rollout's
approver chain.
"""

from __future__ import annotations

import contextlib
import copy
import enum
import json
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from inspect_robots_agent._llm import (
    ENV_MODEL,
    AssistantMessage,
    ChatClient,
    ToolCall,
    resolve_provider,
)
from inspect_robots_agent._png import png_data_url

from inspect_robots import (
    Action,
    ActionChunk,
    ActionSemantics,
    Box,
    EmbodimentInfo,
    Observation,
    ObservationSpace,
    PolicyBase,
    PolicyConfig,
    PolicyInfo,
    Scene,
)
from inspect_robots.errors import ConfigError, PolicyError

from ._anchor import (
    EEF_STATE_DIM,
    anchor_chunk,
    eef_state_to_umi_rpy_state,
    tracking_error,
    umi_last_action_state,
)
from ._client import VlaClient, VlaServiceError
from ._config import (
    CHECKPOINT_INTERVAL_S,
    CHUNK_STEPS,
    CONTROL_HZ,
    MAX_SKILL_SECONDS,
    STATE_KEY,
    SUBMIT_IMAGES,
    TRACKING_ABORT_POS_M,
    TRACKING_ABORT_ROT_DEG,
    VLA_BASE_URL,
    VLA_POLL_INTERVAL_S,
    VLA_POLL_TIMEOUT_S,
    VLA_SUBMIT_TIMEOUT_S,
)
from .policy import VlaWire, _parse_submit_images

#: Same default as the agent plugin's ``max_llm_calls`` (duplicated there and
#: in capx; keep the three in sync).
BUDGET_LLM_CALLS = 100
#: Values within this of an abort threshold count as at-threshold, matching
#: the anchor module's boundary convention (only a strict excess aborts).
_ABORT_EPS = 1e-9
#: Consecutive planner turns with no usable tool call before giving up on the
#: model (mirrors the agent plugin's ``_MAX_CONSECUTIVE_FAILURES``).
_MAX_TURNS_WITHOUT_CALL = 3

_NUDGE = "Respond with exactly one tool call."


class LlmWire(Protocol):
    """The planner seam HybridPolicy needs; the agent plugin's client satisfies it.

    The Protocol (not the concrete client) is the constructor's type so tests
    can inject a scripted fake without inheritance.
    """

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        reasoning_effort: str | float | None = None,
    ) -> AssistantMessage:
        """One chat completion over the running conversation."""
        ...


class _EffortLlm(LlmWire):
    """Chat adapter that pins ``reasoning_effort`` on every completion.

    Gateways that reject tool calls combined with a reasoning-effort field
    (HTTP 400 "Function tools with reasoning_effort are not supported")
    need the field either absent or ``"none"``; ``effort=None`` passes the
    request through untouched.
    """

    def __init__(self, inner: LlmWire, effort: str | None) -> None:
        self._inner = inner
        self._effort = effort

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        reasoning_effort: str | float | None = None,
    ) -> AssistantMessage:
        """One completion with the configured effort (None leaves it unset)."""
        effort = self._effort if reasoning_effort is None else reasoning_effort
        if effort is None:
            return self._inner.complete(messages, tools)
        return self._inner.complete(messages, tools, reasoning_effort=effort)


@dataclass(frozen=True)
class HybridPolicyConfig(PolicyConfig):
    """Inference-time configuration recorded in the eval log.

    Extends the core ``PolicyConfig``; ``eval()`` serializes configs with
    ``dataclasses.asdict``, so these fields land in ``EvalSpec.policy_config``.
    """

    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    vla_base_url: str = VLA_BASE_URL
    prompt: str | None = None
    submit_images: tuple[str, ...] = SUBMIT_IMAGES
    state_key: str = STATE_KEY
    control_hz: float = CONTROL_HZ
    checkpoint_interval_s: float = CHECKPOINT_INTERVAL_S
    max_skill_seconds: float = MAX_SKILL_SECONDS
    tracking_abort_pos_m: float = TRACKING_ABORT_POS_M
    tracking_abort_rot_deg: float = TRACKING_ABORT_ROT_DEG
    budget_llm_calls: int = BUDGET_LLM_CALLS
    name: str = "hybrid"


class _Phase(enum.Enum):
    """Where the policy stands between two ``act()`` calls."""

    PLANNING = "planning"
    EXECUTING = "executing"
    DECIDING = "deciding"
    TERMINAL = "terminal"


_SYSTEM_TEMPLATE = """You are the planner of a hybrid robot policy controlling a \
real embodiment named {name!r}. A vision-language-action skill policy is your \
hands: it executes short trained skills, but it cannot reason and only \
generalizes to skills close to its training tasks. Division of labor: use \
move_to and set_gripper for free-space transport, staging, retreat, and any \
motion the skill policy was not trained on; reserve delegate_skill for \
contact-rich phases close to a trained skill (grasping, constrained \
placement). Phrase each delegate_skill subgoal as a near-verbatim excerpt of \
the user's goal or of a trained task phrasing; do not invent novel wording \
for the skill policy. At every checkpoint you receive the latest observation \
and a skill progress note; decide whether to continue with analytic moves, \
delegate the next skill, retry with a different staging pose, or correct \
course. A skill_interrupted report means the hands did not track the \
commanded motion; restage with move_to and reconsider rather than repeating \
the same skill unchanged. Respond with exactly one tool call per turn. When \
the goal is achieved call done; if it cannot be achieved call give_up. Note \
what you are learning about this rig and task as you go. You have a budget \
of {budget} LLM calls for the whole trial."""

_HINDSIGHT_DESCRIPTION = (
    "What you wish you had known at the start of this trial. Concrete, "
    "transferable facts about this rig, task, or embodiment; say 'none' if "
    "nothing qualifies."
)

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "move_to",
            "description": (
                "Analytic free-space motion: move to absolute Cartesian "
                "end-effector targets (a straight line at a fixed safe "
                "speed). Unnamed dimensions hold their current value. Use "
                "this for transport, staging, retreat, and re-approach "
                "before or between VLA skills; reserve delegate_skill for "
                "contact-rich phases."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "targets": {
                        "type": "object",
                        "description": (
                            "Map of dimension name to absolute value. Valid "
                            "names come from the embodiment docs."
                        ),
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
                "(0 closed to 0.09 open) while holding position."
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
            "name": "delegate_skill",
            "description": (
                "Hand one contact-rich skill segment (grasping, constrained placement, "
                "fixture actuation) to the VLA skill policy for open-loop "
                "execution. Stage with move_to first when the approach pose "
                "matters; you regain control at the next checkpoint, on a "
                "tracking interrupt, or when the time bound elapses."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subgoal": {
                        "type": "string",
                        "description": (
                            "One short, concrete skill the VLA was trained to perform, "
                            "phrased the way its training prompts read."
                        ),
                    },
                    "max_seconds": {
                        "type": "number",
                        "description": (
                            "Upper bound on this delegation in seconds; the policy caps "
                            "it at its own MAX_SKILL_SECONDS."
                        ),
                    },
                },
                "required": ["subgoal", "max_seconds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": ("Declare the task finished. The trial ends; a scorer judges success."),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "hindsight": {"type": "string", "description": _HINDSIGHT_DESCRIPTION},
                },
                "required": ["summary", "hindsight"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "give_up",
            "description": "Stop trying; the task cannot be completed. The trial ends.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "hindsight": {"type": "string", "description": _HINDSIGHT_DESCRIPTION},
                },
                "required": ["reason", "hindsight"],
            },
        },
    },
]


def _positive(name: str, value: float, *, zero_ok: bool = False) -> float:
    """Validate one finite numeric tunable, returning it as ``float``."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not np.isfinite(value)
        or value < 0
        or (value == 0 and not zero_ok)
    ):
        bound = ">= 0" if zero_ok else "> 0"
        raise ConfigError(f"{name} must be a finite number {bound}, got {value!r}")
    return float(value)


def _sanitize(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return an image-free deep copy suitable for persistence or visualization.

    Same contract as the agent and capx plugins' helpers: the planner
    conversation carries streamed camera frames as ``image_url`` parts, which
    must not land in the eval log or the live stream.
    """
    sanitized = copy.deepcopy(messages)
    for message in sanitized:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for index, part in enumerate(content):
            if isinstance(part, dict) and part.get("type") == "image_url":
                content[index] = {
                    "type": "text",
                    "text": "[image omitted: streamed camera frame]",
                }
    return sanitized


class HybridPolicy(PolicyBase):
    """The ``hybrid`` policy: LLM planning, VLA execution, checkpoints between.

    One ``act()`` returns exactly one chunk: either a VLA chunk executing the
    current skill segment (task = the delegated subgoal, anchored on the
    current observed EE state, with no fallback anchor), or the one-action
    stop chunk a ``done`` / ``give_up`` / exhausted call budget produces. The
    state machine advances across ``act()`` calls with all mutable state on
    ``self``; ``reset()`` re-arms it per trial (including the VLA retry flag).
    A segment ends, handing control back to the planner, at every
    ``checkpoint_interval_s`` of wall time, when the delegation's time bound
    (capped by ``max_skill_seconds``) elapses, or when the last commanded step
    tracks beyond the abort thresholds.

    Controller assumptions: the tracking check and the segment step counts
    assume the default controller plays each returned chunk to its end. A
    programmatic ``replan_interval`` (re-inferring mid-chunk) or chunk
    ensembling is unsupported here: rows the controller never played would be
    measured as tracking error, and the progress note would overcount steps.

    Ownership: the framework has no close hook for policies (``eval()`` closes
    only the embodiments it resolves), so the chat and VLA clients this
    constructor builds are closed on ``close()`` and again, guarded, at
    finalization (``__del__``). Injected backends stay the caller's.
    """

    # Per-trial mutable state; armed in __init__ and re-armed in reset() by
    # _rearm_trial_state(). Declared here (not inferred) because reset() and
    # the segment paths assign narrower values than the later uses read.
    _phase: _Phase
    _messages: list[dict[str, Any]]
    _delta_cursor: int
    _calls_used: int
    _previous_attempt_failed: bool
    _subgoal: str | None
    _segment_granted: float
    _segment_deadline: float
    _next_checkpoint: float
    _segment_steps: int
    _segment_chunks: int
    _last_commanded: np.ndarray | None
    _last_tracking: tuple[float, float] | None
    _interrupt_reason: str | None
    _delegation_error: str

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        *,
        vla_base_url: str = VLA_BASE_URL,
        wire: str = "chat",
        effort: str | None = None,
        prompt: str | None = None,
        submit_images: Sequence[str] | str = SUBMIT_IMAGES,
        state_key: str = STATE_KEY,
        control_hz: float = CONTROL_HZ,
        timeout_s: float = VLA_SUBMIT_TIMEOUT_S,
        poll_interval_s: float = VLA_POLL_INTERVAL_S,
        poll_timeout_s: float = VLA_POLL_TIMEOUT_S,
        checkpoint_interval_s: float = CHECKPOINT_INTERVAL_S,
        max_skill_seconds: float = MAX_SKILL_SECONDS,
        tracking_abort_pos_m: float = TRACKING_ABORT_POS_M,
        tracking_abort_rot_deg: float = TRACKING_ABORT_ROT_DEG,
        budget_llm_calls: int = BUDGET_LLM_CALLS,
        name: str = "hybrid",
        llm: LlmWire | None = None,
        vla: VlaWire | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        for arg_name, value in (
            ("model", model),
            ("base_url", base_url),
            ("api_key_env", api_key_env),
        ):
            if value is not None and not isinstance(value, str):
                raise ConfigError(
                    f"{arg_name} must be a string or None, got {value!r}.\n"
                    f"fix: the -P parser coerces unquoted values; pass "
                    f"-P '{arg_name}=\"value\"'"
                )
        if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
            raise ConfigError(
                f"prompt must be a non-empty string or None, got {prompt!r}.\n"
                'fix: pass -P prompt="the task instruction", or omit -P prompt= '
                "to use the scene instruction"
            )
        if not isinstance(state_key, str) or not state_key:
            raise ConfigError(f"state_key must be a non-empty string, got {state_key!r}")
        if not isinstance(vla_base_url, str) or not vla_base_url.startswith(
            ("http://", "https://")
        ):
            raise ConfigError(
                f"vla_base_url must be an http:// or https:// URL, got {vla_base_url!r}.\n"
                "fix: pass -P vla_base_url=http://127.0.0.1:10055"
            )
        if (
            isinstance(budget_llm_calls, bool)
            or not isinstance(budget_llm_calls, int)
            or budget_llm_calls < 1
        ):
            raise ConfigError(f"budget_llm_calls must be an int >= 1, got {budget_llm_calls!r}")
        self._control_hz = _positive("control_hz", control_hz)
        self._timeout_s = _positive("timeout_s", timeout_s)
        self._poll_interval_s = _positive("poll_interval_s", poll_interval_s)
        self._poll_timeout_s = _positive("poll_timeout_s", poll_timeout_s)
        # checkpoint_interval_s == 0 is meaningful: decide after every chunk.
        self._checkpoint_interval_s = _positive(
            "checkpoint_interval_s", checkpoint_interval_s, zero_ok=True
        )
        self._max_skill_seconds = _positive("max_skill_seconds", max_skill_seconds)
        self._tracking_abort_pos_m = _positive("tracking_abort_pos_m", tracking_abort_pos_m)
        self._tracking_abort_rot_deg = _positive("tracking_abort_rot_deg", tracking_abort_rot_deg)
        self._submit_images = _parse_submit_images(submit_images)
        self._state_key = state_key
        self._prompt = prompt
        self._budget_llm_calls = budget_llm_calls

        environ = dict(os.environ) if env is None else env
        owned_llm: LlmWire | None = None
        if llm is None:
            provider = resolve_provider(
                model=model or environ.get(ENV_MODEL),
                base_url=base_url,
                api_key_env=api_key_env,
                env=environ,
            )
            if wire == "responses":
                from inspect_robots_agent._responses import ResponsesClient

                owned_llm = ResponsesClient(provider)
            elif wire != "chat":
                raise ConfigError(f"hybrid wire must be 'chat' or 'responses', got {wire!r}")
            else:
                owned_llm = ChatClient(provider)
            self._llm: LlmWire = _EffortLlm(owned_llm, effort)
            self._effort = effort
        else:
            self._llm = llm
            self._effort = None
        self._owned_llm = owned_llm  # LlmWire; close() uses duck-typed .close()
        owned_vla: VlaClient | None = None
        if vla is None:
            owned_vla = VlaClient(
                vla_base_url,
                timeout_s=self._timeout_s,
                poll_interval_s=self._poll_interval_s,
                poll_timeout_s=self._poll_timeout_s,
            )
            self._vla: VlaWire = owned_vla
        else:
            self._vla = vla
        self._owned_vla = owned_vla

        self.info = PolicyInfo(
            name=name,
            action_space=Box(
                shape=(EEF_STATE_DIM,),
                semantics=ActionSemantics(
                    control_mode="eef_abs_pose",
                    rotation_repr="rot6d",
                    gripper="continuous",
                    frame="base",
                ),
            ),
            observation_space=ObservationSpace(state_keys=frozenset({state_key})),
            control_hz=self._control_hz,
        )
        self.config = HybridPolicyConfig(
            action_horizon=CHUNK_STEPS,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            vla_base_url=vla_base_url,
            prompt=prompt,
            submit_images=self._submit_images,
            state_key=state_key,
            control_hz=self._control_hz,
            checkpoint_interval_s=self._checkpoint_interval_s,
            max_skill_seconds=self._max_skill_seconds,
            tracking_abort_pos_m=self._tracking_abort_pos_m,
            tracking_abort_rot_deg=self._tracking_abort_rot_deg,
            budget_llm_calls=budget_llm_calls,
            name=name,
        )
        self._embodiment_name = "(unbound)"
        self._embodiment_docs: str | None = None
        self._action_semantics: Any = None
        self._request_id = time.time_ns()
        self._rearm_trial_state()

    # -- lifecycle ---------------------------------------------------------------

    def bind(self, embodiment_info: EmbodimentInfo) -> None:
        """Fail fast on camera mismatch, then record the embodiment for prompts.

        ``submit_images`` must name cameras the embodiment declares: a typo
        here would otherwise surface only as a mid-rollout failure on the
        first chunk. An embodiment that declares no cameras cannot be checked
        in advance; the act-time missing-camera error still guards it.
        """
        cameras = embodiment_info.observation_space.camera_names
        if cameras:
            missing = [n for n in self._submit_images if n not in cameras]
            if missing:
                raise ConfigError(
                    f"hybrid submit_images name camera(s) {missing} that embodiment "
                    f"{embodiment_info.name!r} does not declare; declared cameras: "
                    f"{sorted(cameras)}.\n"
                    "fix: point -P submit_images= at cameras the embodiment provides"
                )
        self._embodiment_name = embodiment_info.name
        self._embodiment_docs = embodiment_info.docs
        self._action_semantics = embodiment_info.action_space.semantics

    def reset(self, scene: Scene) -> None:
        """Start a fresh trial: goal, conversation, budget, and segment state.

        The VLA retry flag is re-armed here (a previous trial's failure must
        not drain this one's retry). ``_request_id`` is deliberately not
        reset: ids must never repeat for a long-lived service.
        """
        goal = self._prompt or scene.instruction
        if not goal:
            raise PolicyError(
                "hybrid has no goal for the planner: pass -P prompt=... or run "
                "scenes that carry instructions"
            )
        self._rearm_trial_state()
        self._messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": f"Goal: {goal}"},
        ]

    def transcript(self) -> list[dict[str, Any]] | None:
        """Return an image-free deep copy of the planner conversation.

        The rollout's duck-typed end-of-trial hook: the sanitized conversation
        lands in ``TrialRecord.policy_transcript`` (and so the eval log) the
        same way the agent policy's does.
        """
        return _sanitize(self._messages) if self._messages else None

    def transcript_delta(self) -> list[dict[str, Any]] | None:
        """Sanitized messages appended since the previous call (live-stream hook)."""
        new = self._messages[self._delta_cursor :]
        self._delta_cursor = len(self._messages)
        return _sanitize(new) if new else None

    def close(self) -> None:
        """Close the clients this policy built; injected backends stay the caller's."""
        if self._owned_vla is not None:
            self._owned_vla.close()
            self._owned_vla = None
        if self._owned_llm is not None:
            close = getattr(self._owned_llm, "close", None)
            if callable(close):
                close()
            self._owned_llm = None

    def __del__(self) -> None:
        # Guarded: finalization may run during interpreter teardown, where even
        # attribute access can fail; an unclosed socket is the lesser evil.
        with contextlib.suppress(Exception):
            self.close()

    # -- the loop ------------------------------------------------------------------

    def act(self, observation: Observation) -> ActionChunk:
        """Advance one step: execute the segment, or spend a planner turn."""
        if self._phase is _Phase.TERMINAL:
            raise PolicyError(
                "hybrid already ended this trial (done/give_up); the rollout should "
                "have stopped on request_stop"
            )
        if self._phase is _Phase.EXECUTING:
            outcome = self._segment_outcome(observation)
            if outcome is None:
                return self._execute_chunk(observation)
            self._interrupt_reason = outcome
            self._phase = _Phase.DECIDING
        return self._decide(observation)

    def _segment_outcome(self, observation: Observation) -> str | None:
        """Why the segment should end now, or ``None`` to keep executing.

        The tracking check compares the last commanded step's target against
        the observed EE state: the default controller plays the returned
        chunk open-loop, so this is the first observation after those steps.
        """
        eef_state = self._require_eef_state(observation)
        commanded = self._last_commanded
        if commanded is not None:
            try:
                pos_m, rot_deg = tracking_error(commanded, eef_state)
            except VlaServiceError as exc:
                raise PolicyError(
                    f"hybrid could not measure tracking error against "
                    f"state[{self._state_key!r}]: {exc}"
                ) from exc
            self._last_tracking = (pos_m, rot_deg)
            beyond = pos_m > self._tracking_abort_pos_m + _ABORT_EPS or (
                rot_deg > self._tracking_abort_rot_deg + _ABORT_EPS
            )
            if beyond:
                return (
                    f"skill_interrupted: pos_err={pos_m:.4f} m, rot_err={rot_deg:.1f} "
                    f"deg (limits {self._tracking_abort_pos_m} m, "
                    f"{self._tracking_abort_rot_deg} deg)"
                )
        now = time.monotonic()
        if now >= self._segment_deadline:
            return f"max_skill_seconds cap reached ({self._segment_granted:.3g} s granted)"
        if now >= self._next_checkpoint:
            return "checkpoint"
        return None

    def _execute_chunk(self, observation: Observation) -> ActionChunk:
        """One VLA round trip for the current subgoal (the VlaPolicy path)."""
        subgoal = self._subgoal
        assert subgoal is not None  # EXECUTING implies a delegation armed one
        images = self._submit_frames(observation)
        eef_state = self._require_eef_state(observation)
        self._request_id += 1
        request_id = self._request_id
        started = time.perf_counter()
        # A transient service failure earns exactly one retry, unless the
        # previous attempt also failed, when the failure surfaces immediately.
        attempts = 1 if self._previous_attempt_failed else 2
        chunk = None
        failure: VlaServiceError | None = None
        for _ in range(attempts):
            try:
                current14 = eef_state_to_umi_rpy_state(eef_state)
                state28 = umi_last_action_state(current14, self._prev_umi_state)
                chunk = self._vla.infer(images, state28, subgoal, request_id=request_id)
                break
            except VlaServiceError as exc:
                failure = exc
        if chunk is None:
            self._previous_attempt_failed = True
            assert failure is not None
            outcome = (
                "failed twice (one retry)"
                if attempts == 2
                else "failed; the previous attempt also failed, so no retry"
            )
            raise PolicyError(
                f"hybrid VLA request {request_id} for subgoal {subgoal!r} {outcome}: {failure}"
            ) from failure
        self._previous_attempt_failed = False
        try:
            targets = anchor_chunk(eef_state, chunk)
        except VlaServiceError as exc:
            # A malformed chunk is bad data, not a dead service: the retry stays armed.
            raise PolicyError(
                f"hybrid VLA request {request_id} returned a chunk that cannot be anchored: {exc}"
            ) from exc
        # Per-step commanded targets: the final row is the next tracking check.
        self._last_commanded = targets[-1]
        self._segment_steps += len(targets)
        self._segment_chunks += 1
        return ActionChunk(
            actions=[Action(data=row) for row in targets],
            control_hz=self._control_hz,
            inference_latency_s=time.perf_counter() - started,
            meta={"request_id": request_id, "subgoal": subgoal},
        )

    def _decide(self, observation: Observation) -> ActionChunk:
        """One planner turn: observation (plus progress note) in, one tool call out."""
        self._messages.append({"role": "user", "content": self._observation_content(observation)})
        invalid_turns = 0
        while True:
            if self._calls_used >= self._budget_llm_calls:
                return self._forced_give_up("LLM call budget exhausted", observation)
            message = self._llm.complete(self._messages, _TOOLS)
            self._calls_used += 1
            self._messages.append(message.raw())
            calls = message.tool_calls
            if not calls:
                self._messages.append({"role": "user", "content": _NUDGE})
                invalid_turns += 1
                if invalid_turns >= _MAX_TURNS_WITHOUT_CALL:
                    raise PolicyError(
                        f"hybrid planner produced no tool call in {invalid_turns} consecutive turns"
                    )
                continue
            call, extras = calls[0], calls[1:]
            if call.name in ("move_to", "set_gripper"):
                moved, motion_error = self._analytic_move(call, observation)
                if moved is None or motion_error is not None:
                    self._reply((call,), motion_error or "no motion produced")
                    self._reply(extras, "ignored: one tool call per turn")
                    invalid_turns += 1
                    if invalid_turns >= _MAX_TURNS_WITHOUT_CALL:
                        raise PolicyError(
                            "hybrid planner tool calls kept failing; last error: "
                            f"{motion_error or 'no motion produced'}"
                        )
                    continue
                self._reply((call,), "analytic motion accepted")
                self._reply(extras, "ignored: one tool call per turn")
                self._last_tool_note = str(call.arguments)
                return moved
            if call.name == "delegate_skill":
                subgoal, granted = self._parse_delegation(call)
                if subgoal is None:
                    self._reply((call,), self._delegation_error)
                    self._reply(extras, "ignored: one tool call per turn")
                    invalid_turns += 1
                    if invalid_turns >= _MAX_TURNS_WITHOUT_CALL:
                        raise PolicyError(
                            "hybrid planner tool calls kept failing; last error: "
                            f"{self._delegation_error}"
                        )
                    continue
                self._begin_segment(call, subgoal, granted)
                self._reply(extras, "ignored: one tool call per turn")
                return self._execute_chunk(observation)
            if call.name in ("done", "give_up"):
                chunk = self._stop(call, observation)
                self._reply(extras, "ignored: one tool call per turn")
                return chunk
            error = (
                f"unknown tool {call.name!r}; available: "
                "move_to, set_gripper, delegate_skill, done, give_up"
            )
            self._reply((call,), error)
            self._reply(extras, "ignored: one tool call per turn")
            invalid_turns += 1
            if invalid_turns >= _MAX_TURNS_WITHOUT_CALL:
                raise PolicyError(f"hybrid planner tool calls kept failing; last error: {error}")

    # -- planner helpers ------------------------------------------------------------

    def _parse_delegation(self, call: ToolCall) -> tuple[str | None, float]:
        """Validate one delegate_skill call; the error text lands on ``self``.

        Returns ``(subgoal, granted_seconds)``, or ``(None, 0.0)`` with
        ``self._delegation_error`` set when the arguments are unusable.
        """
        try:
            arguments = json.loads(call.arguments)
        except (TypeError, ValueError):
            arguments = None
        if not isinstance(arguments, dict):
            self._delegation_error = f"arguments for {call.name} are not a JSON object"
            return None, 0.0
        subgoal = arguments.get("subgoal")
        if not isinstance(subgoal, str) or not subgoal.strip():
            self._delegation_error = "subgoal must be a non-empty string"
            return None, 0.0
        max_seconds = arguments.get("max_seconds")
        if (
            isinstance(max_seconds, bool)
            or not isinstance(max_seconds, int | float)
            or not np.isfinite(max_seconds)
            or max_seconds <= 0
        ):
            self._delegation_error = f"max_seconds must be a finite number > 0, got {max_seconds!r}"
            return None, 0.0
        return subgoal, min(float(max_seconds), self._max_skill_seconds)

    def _analytic_move(
        self, call: ToolCall, observation: Observation
    ) -> tuple[ActionChunk | None, str | None]:
        """Build one analytic interpolated chunk from a move_to/set_gripper call.

        Free-space motions run through the same interpolation the agent
        policy uses: partial absolute targets against the current EE state,
        split into per-dim step-limited increments. Returns (chunk, None) on
        success or (None, error) for a structured planner retry.
        """
        import json as _json

        try:
            arguments = _json.loads(call.arguments)
        except ValueError:
            return None, "move_to/set_gripper arguments are not valid JSON"
        if not isinstance(arguments, dict):
            return None, "move_to/set_gripper arguments must be a JSON object"
        note = arguments.get("note")
        if not isinstance(note, str) or not note.strip():
            return None, "note is required: describe the motion and why"
        semantics = self._action_semantics
        labels = list(getattr(semantics, "dim_labels", None) or ())
        if not labels:
            return None, "the embodiment declares no dimension labels for move_to"
        index = {label: position for position, label in enumerate(labels)}
        try:
            eef_state = self._require_eef_state(observation)
        except PolicyError as exc:
            return None, str(exc)
        target = eef_state.astype(np.float64).copy()
        if call.name == "set_gripper":
            arm = str(arguments.get("arm", "")).strip().lower()
            width = arguments.get("width_m")
            dim = f"{arm}_gripper" if arm in ("left", "right") else None
            if dim is None or dim not in index:
                return None, "set_gripper arm must be 'left' or 'right'"
            if not isinstance(width, (int, float)) or isinstance(width, bool):
                return None, "set_gripper width_m must be a number (0 to 0.09)"
            arguments = {"targets": {dim: float(width)}}
        values = arguments.get("targets")
        if not isinstance(values, dict) or not values:
            return None, "move_to targets must be a non-empty object of name: value"
        for label, raw in values.items():
            if label not in index:
                return None, f"unknown dimension {label!r}; valid names: {', '.join(labels)}"
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                return None, f"value for {label!r} must be a finite number"
            target[index[label]] = float(raw)
        if not np.isfinite(target).all():
            return None, "targets hold non-finite values"
        step_limits = getattr(semantics, "max_step", None)
        ratios = []
        for label in values:
            position = index[label]
            distance = abs(target[position] - eef_state[position])
            limit = float(step_limits[position]) if step_limits else 0.01
            if limit > 0 and distance > 0:
                ratios.append(distance / limit)
        steps = max(1, math.ceil(max(ratios, default=0.0)))
        fractions = np.linspace(1.0 / steps, 1.0, steps)
        rows = [
            (eef_state + (target - eef_state) * fraction).astype(np.float32)
            for fraction in fractions
        ]
        rows[-1] = target.astype(np.float32)
        return (
            ActionChunk(
                actions=[Action(data=row) for row in rows],
                control_hz=self._control_hz,
                meta={"analytic": call.name},
            ),
            None,
        )

    def _begin_segment(self, call: ToolCall, subgoal: str, granted_s: float) -> None:
        """Arm a fresh skill segment and tell the planner what was granted."""
        now = time.monotonic()
        self._subgoal = subgoal
        self._segment_granted = granted_s
        self._segment_deadline = now + granted_s
        self._next_checkpoint = now + self._checkpoint_interval_s
        self._segment_steps = 0
        self._segment_chunks = 0
        self._last_commanded = None
        self._last_tracking = None
        self._interrupt_reason = None
        self._phase = _Phase.EXECUTING
        capped = (
            f" (capped by max_skill_seconds={self._max_skill_seconds})"
            if granted_s < self._max_skill_seconds
            else ""
        )
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": (
                    f"delegated to the VLA skill policy: {subgoal!r} for up to "
                    f"{granted_s:.3g}s{capped}; you regain control at the next "
                    "checkpoint, on a tracking interrupt, or when the time bound "
                    "elapses"
                ),
            }
        )

    def _stop(self, call: ToolCall, observation: Observation) -> ActionChunk:
        """Turn done/give_up into the one-action stop chunk (the agent's _stop shape)."""
        try:
            arguments = json.loads(call.arguments)
        except (TypeError, ValueError):
            arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        detail = str(arguments.get("summary") or arguments.get("reason") or "")
        meta: dict[str, Any] = {
            "request_stop": True,
            "stop_reason": call.name,
            "stop_detail": detail,
        }
        hindsight = arguments.get("hindsight")
        if isinstance(hindsight, str) and hindsight.strip() and hindsight.strip().lower() != "none":
            meta["stop_hindsight"] = hindsight.strip()
        self._reply((call,), f"{call.name}: {detail}")
        self._phase = _Phase.TERMINAL
        observed = observation.state.get(self._state_key)
        data = np.zeros(EEF_STATE_DIM) if observed is None else np.asarray(observed)
        return ActionChunk(actions=[Action(data=data, meta=meta)], control_hz=self._control_hz)

    def _forced_give_up(self, why: str, observation: Observation) -> ActionChunk:
        """A synthetic give_up when the call budget runs dry (never a model turn).

        The synthetic call is not appended to the conversation: it is not
        model output, so it stays out of any transcript.
        """
        meta: dict[str, Any] = {
            "request_stop": True,
            "stop_reason": "give_up",
            "stop_detail": why,
        }
        self._phase = _Phase.TERMINAL
        observed = observation.state.get(self._state_key)
        data = np.zeros(EEF_STATE_DIM) if observed is None else np.asarray(observed)
        return ActionChunk(actions=[Action(data=data, meta=meta)], control_hz=self._control_hz)

    def _reply(self, calls: Sequence[ToolCall], result: str) -> None:
        """Append the same tool-result message for each call id given."""
        for call in calls:
            self._messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    def _system_prompt(self) -> str:
        """The hybrid role text plus whatever embodiment notes bind() recorded."""
        formatted = _SYSTEM_TEMPLATE.format(
            name=self._embodiment_name, budget=self._budget_llm_calls
        )
        docs = self._embodiment_docs
        if docs is not None and docs.strip():
            formatted = formatted + "\n\nEmbodiment notes:\n" + docs.strip()
        return formatted

    def _observation_content(self, observation: Observation) -> list[dict[str, Any]]:
        """The planner's view: state text, progress note, and submit-camera frames."""
        lines = ["Current observation."]
        if observation.instruction:
            lines.append(f"Instruction: {observation.instruction}")
        for key, value in observation.state.items():
            array = np.asarray(value, dtype=np.float64)
            lines.append(f"state[{key}]: {np.round(array, 4).tolist()}")
        if self._phase is _Phase.DECIDING:
            lines.append(self._progress_note())
        parts: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(lines)}]
        for name in sorted(self._submit_images):
            frame = observation.images.get(name)
            if frame is None:
                continue
            parts.append({"type": "text", "text": f"camera {name!r}:"})
            parts.append({"type": "image_url", "image_url": {"url": png_data_url(frame)}})
        return parts

    def _progress_note(self) -> str:
        """The compact segment report the planner decides on."""
        tracking = self._last_tracking
        error = (
            "unmeasured"
            if tracking is None
            else f"pos {tracking[0]:.4f} m, rot {tracking[1]:.1f} deg"
        )
        return (
            f"skill progress for {self._subgoal!r}: "
            f"steps_executed={self._segment_steps}, chunks={self._segment_chunks}, "
            f"last tracking error: {error}. "
            f"segment ended: {self._interrupt_reason or 'checkpoint'}"
        )

    # -- execution helpers ----------------------------------------------------------

    def _require_eef_state(self, observation: Observation) -> np.ndarray:
        """The observed EE anchor; executing has no fallback target (plan 0084)."""
        observed = observation.state.get(self._state_key)
        if observed is None:
            raise PolicyError(
                f"observation has no state[{self._state_key!r}] to anchor on or "
                "measure tracking against; the hybrid always anchors on the current "
                "observed state.\n"
                "fix: pair this policy with an embodiment that provides the key, or "
                "pass -P state_key=<its 20-dim EE state key>"
            )
        return np.asarray(observed)

    def _submit_frames(self, observation: Observation) -> dict[str, np.ndarray]:
        """The submit_images frames in sorted key order (the wire's image order)."""
        missing = [name for name in self._submit_images if name not in observation.images]
        if missing:
            raise PolicyError(
                f"observation lacks camera(s) {missing} named by submit_images; "
                f"available: {sorted(observation.images)}.\n"
                "fix: point -P submit_images= at cameras the embodiment provides"
            )
        return {name: observation.images[name] for name in sorted(self._submit_images)}

    def _rearm_trial_state(self) -> None:
        self._prev_umi_state = None
        """Reset every per-trial mutable field; called from __init__ and reset."""
        self._phase = _Phase.PLANNING
        self._messages = []
        self._delta_cursor = 0
        self._calls_used = 0
        self._previous_attempt_failed = False
        self._subgoal = None
        self._segment_granted = 0.0
        self._segment_deadline = 0.0
        self._next_checkpoint = 0.0
        self._segment_steps = 0
        self._segment_chunks = 0
        self._last_commanded = None
        self._last_tracking = None
        self._interrupt_reason = None
        self._delegation_error = ""


def hybrid_policy(
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    **kwargs: Any,
) -> HybridPolicy:
    """Registry factory for the ``hybrid`` policy (entry point ``hybrid``).

    Accepts the same arguments as
    [`HybridPolicy`][inspect_robots_vla.hybrid.HybridPolicy]; the CLI forwards
    each ``-P key=value`` pair here.
    """
    return HybridPolicy(model, base_url, api_key_env, **kwargs)
