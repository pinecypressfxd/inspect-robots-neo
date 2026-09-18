"""HybridPolicy tests: the planner state machine over injected fakes.

The fake LLM speaks only the ``.complete(messages, tools)`` seam (returning
real ``AssistantMessage`` objects with scripted tool calls) and the fake VLA
the ``infer`` seam, so these tests pin the state machine: PLANNING to
EXECUTING to DECIDING and back, the subgoal handed verbatim to the VLA,
tracking aborts, time caps, the call budget, and the stop chunk's shape.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest
from inspect_robots_agent._llm import AssistantMessage, ToolCall
from scipy.spatial.transform import Rotation

from inspect_robots import (
    ActionSemantics,
    Box,
    CameraSpec,
    EmbodimentInfo,
    Observation,
    ObservationSpace,
    Policy,
    Scene,
)
from inspect_robots.errors import ConfigError, PolicyError
from inspect_robots_vla._client import VlaChunk, VlaServiceError
from inspect_robots_vla._config import ACTION_DIM_VLA
from inspect_robots_vla.hybrid import HybridPolicy, HybridPolicyConfig, hybrid_policy

_LEFT_RPY = (0.10, -0.20, 0.30)
_RIGHT_RPY = (-0.40, 0.25, -0.15)


@dataclasses.dataclass
class _VlaCall:
    """One infer() round trip exactly as the policy issued it."""

    images: dict[str, np.ndarray]
    state: np.ndarray
    task: str
    request_id: int


class _FakeVla:
    """Scripted stand-in for VlaClient: each infer() pops the next scripted step."""

    def __init__(self, script: Sequence[VlaChunk | Exception]) -> None:
        self.script = list(script)
        self.calls: list[_VlaCall] = []
        self.closed = False

    def infer(
        self,
        images: Mapping[str, np.ndarray] | None,
        state: np.ndarray,
        task: str,
        *,
        request_id: int,
    ) -> VlaChunk:
        self.calls.append(
            _VlaCall(
                images={} if images is None else dict(images),
                state=np.array(state),
                task=task,
                request_id=request_id,
            )
        )
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    def close(self) -> None:
        self.closed = True


@dataclasses.dataclass
class _LlmCall:
    """One complete() call: a snapshot of the conversation plus the tool schemas."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]


class _FakeLlm:
    """Scripted stand-in for the chat client: each complete() pops one turn."""

    def __init__(self, script: Sequence[AssistantMessage]) -> None:
        self.script = list(script)
        self.calls: list[_LlmCall] = []

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AssistantMessage:
        self.calls.append(_LlmCall(messages=copy.deepcopy(messages), tools=tools))
        return self.script.pop(0)


def _columns6(matrix: np.ndarray) -> np.ndarray:
    """Stack the first two rotation-matrix columns: the Zhou 6D form."""
    return np.concatenate([matrix[:, 0], matrix[:, 1]])


def _eef_state() -> np.ndarray:
    """A plausible dual-arm EE state in the nero 20-dim layout (float64)."""
    state = np.zeros(20)
    state[0:3] = (0.30, -0.10, 0.20)
    state[3:9] = _columns6(Rotation.from_euler("xyz", _LEFT_RPY).as_matrix())
    state[9] = 0.05
    state[10:13] = (0.28, 0.12, 0.19)
    state[13:19] = _columns6(Rotation.from_euler("xyz", _RIGHT_RPY).as_matrix())
    state[19] = 0.04
    return state


def _hwc(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(4, 8, 3), dtype=np.uint8)


_FRAMES = {"left_rgbd": _hwc(1), "right_rgbd": _hwc(2), "chest_rgbd": _hwc(3)}


def _observation(
    *,
    eef: np.ndarray | None = None,
    frames: Mapping[str, np.ndarray] | None = None,
    with_state: bool = True,
) -> Observation:
    return Observation(
        images=dict(_FRAMES if frames is None else frames),
        state={"eef_state": _eef_state() if eef is None else eef} if with_state else {},
    )


def _chunk(deltas: np.ndarray, request_id: int = 1) -> VlaChunk:
    return VlaChunk(request_id=request_id, deltas=np.asarray(deltas, dtype=np.float32))


def _zero_chunk() -> VlaChunk:
    return _chunk(np.zeros((2, ACTION_DIM_VLA)))


def _left_x_chunk(total: float) -> VlaChunk:
    """A (3, 14) chunk whose left-arm x deltas sum to ``total`` (other dims zero)."""
    deltas = np.zeros((3, ACTION_DIM_VLA), dtype=np.float32)
    deltas[:, 0] = total / 3
    return _chunk(deltas)


def _text_of(messages: Sequence[dict[str, Any]]) -> str:
    """All text parts of a conversation snapshot, joined (image parts skipped)."""
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text")))
    return "\n".join(parts)


def _last_user(call: _LlmCall) -> dict[str, Any]:
    """The newest user message of a snapshot (the observation the planner saw)."""
    return next(m for m in reversed(call.messages) if m.get("role") == "user")


def _delegate(subgoal: str, max_seconds: float | None = 30.0) -> AssistantMessage:
    arguments: dict[str, Any] = {"subgoal": subgoal}
    if max_seconds is not None:
        arguments["max_seconds"] = max_seconds
    return AssistantMessage(
        content=None,
        tool_calls=(ToolCall(id="c", name="delegate_skill", arguments=json.dumps(arguments)),),
    )


def _done(summary: str = "placed the cup", hindsight: str = "none") -> AssistantMessage:
    return AssistantMessage(
        content=None,
        tool_calls=(
            ToolCall(
                id="c",
                name="done",
                arguments=json.dumps({"summary": summary, "hindsight": hindsight}),
            ),
        ),
    )


def _give_up(reason: str = "cannot reach") -> AssistantMessage:
    return AssistantMessage(
        content=None,
        tool_calls=(ToolCall(id="c", name="give_up", arguments=json.dumps({"reason": reason})),),
    )


def _no_call(text: str = "let me think") -> AssistantMessage:
    return AssistantMessage(content=text, tool_calls=())


_PROMPT = "put the cup on the saucer"


def _policy(
    llm: _FakeLlm,
    vla: _FakeVla,
    **kwargs: Any,
) -> HybridPolicy:
    """A hybrid with both backends injected and the prompt pinned."""
    return hybrid_policy("test/model", llm=llm, vla=vla, prompt=_PROMPT, **kwargs)


def _reset(policy: HybridPolicy, instruction: str = _PROMPT) -> None:
    policy.reset(Scene(id="s", instruction=instruction))


def test_planning_executing_deciding_done_full_path() -> None:
    llm = _FakeLlm([_delegate("lift the cup"), _done("cup on saucer", "grip lower")])
    vla = _FakeVla([_zero_chunk()])
    policy = _policy(llm, vla, checkpoint_interval_s=0.0)
    assert isinstance(policy, Policy)
    assert policy.info.name == "hybrid"
    assert isinstance(policy.config, HybridPolicyConfig)
    assert policy.config.budget_llm_calls == 100

    _reset(policy)
    chunk = policy.act(_observation())

    # PLANNING: one LLM call, system prompt + goal, tool schemas handed over.
    assert len(llm.calls) == 1
    (first,) = llm.calls
    assert first.messages[0]["role"] == "system"
    system = str(first.messages[0]["content"])
    assert "delegate_skill" in system
    assert _PROMPT in _text_of([first.messages[1]])
    assert {tool["function"]["name"] for tool in first.tools} == {
        "delegate_skill",
        "done",
        "give_up",
    }
    # The observation message carries the state line and the submit cameras.
    user = _last_user(first)
    assert "state[eef_state]" in _text_of([user])
    image_parts = [p for p in user["content"] if p.get("type") == "image_url"]
    assert len(image_parts) == 3

    # EXECUTING: the subgoal goes to the VLA verbatim, anchored on the state.
    assert vla.calls[0].task == "lift the cup"
    from inspect_robots_vla._anchor import eef_state_to_umi_rpy_state, umi_last_action_state

    np.testing.assert_allclose(
        vla.calls[0].state,
        umi_last_action_state(eef_state_to_umi_rpy_state(_eef_state())),
        atol=1e-6,
    )
    assert list(vla.calls[0].images) == ["chest_rgbd", "left_rgbd", "right_rgbd"]
    assert len(chunk) == 2
    assert chunk.meta["subgoal"] == "lift the cup"
    assert "request_stop" not in chunk.actions[0].meta

    # The chunk played; checkpoint (interval 0) returns control to the planner.
    stop = policy.act(_observation())

    assert len(llm.calls) == 2
    note = _text_of([_last_user(llm.calls[1])])
    assert "steps_executed=2" in note
    assert "chunks=1" in note
    assert "checkpoint" in note
    assert "lift the cup" in note
    # done: a one-action stop chunk mirroring the agent's _stop shape.
    assert len(stop) == 1
    meta = stop.actions[0].meta
    assert meta["request_stop"] is True
    assert meta["stop_reason"] == "done"
    assert meta["stop_detail"] == "cup on saucer"
    assert meta["stop_hindsight"] == "grip lower"
    np.testing.assert_allclose(np.asarray(stop.actions[0].data), _eef_state(), atol=1e-9)


def test_delegate_twice_passes_each_subgoal_verbatim() -> None:
    subgoals = ("pick up the 'red' cup, gently", "place it on the saucer")
    llm = _FakeLlm([_delegate(subgoals[0]), _delegate(subgoals[1]), _done()])
    vla = _FakeVla([_zero_chunk(), _zero_chunk()])
    policy = _policy(llm, vla, checkpoint_interval_s=0.0)
    _reset(policy)

    first = policy.act(_observation())
    second = policy.act(_observation())
    stop = policy.act(_observation())

    assert [call.task for call in vla.calls] == list(subgoals)
    assert first.meta["subgoal"] == subgoals[0]
    assert second.meta["subgoal"] == subgoals[1]
    assert second.actions[0].meta.get("request_stop") is None
    assert vla.calls[1].request_id > vla.calls[0].request_id
    assert stop.actions[0].meta["request_stop"] is True
    # Each decision reported the segment that just ended, per segment.
    assert subgoals[0] in _text_of([_last_user(llm.calls[1])])
    assert "steps_executed=2" in _text_of([_last_user(llm.calls[1])])
    assert subgoals[1] in _text_of([_last_user(llm.calls[2])])
    assert "steps_executed=2" in _text_of([_last_user(llm.calls[2])])


def test_tracking_abort_mid_segment_reports_skill_interrupted() -> None:
    llm = _FakeLlm([_delegate("reach for the cup"), _done()])
    vla = _FakeVla([_left_x_chunk(0.06)])
    policy = _policy(llm, vla)  # default checkpoint 5 s: only the abort decides
    _reset(policy)

    policy.act(_observation())  # commands +6 cm on the left arm's x
    stop = policy.act(_observation())  # robot never moved: 6 cm > 3 cm threshold

    note = _text_of([_last_user(llm.calls[1])])
    assert "skill_interrupted" in note
    assert "pos_err=0.0600" in note
    assert len(vla.calls) == 1  # the aborted segment issued no further chunk
    assert stop.actions[0].meta["request_stop"] is True


def test_exactly_at_threshold_does_not_abort_the_segment() -> None:
    llm = _FakeLlm([_delegate("slide the cup"), _done()])
    vla = _FakeVla([_zero_chunk(), _zero_chunk()])
    policy = _policy(llm, vla)  # checkpoint 5 s, well past a two-act test
    _reset(policy)

    policy.act(_observation())  # commands zero motion
    observed = _eef_state()
    observed[0] += 0.03  # exactly the position threshold (to float precision)
    chunk = policy.act(_observation(eef=observed))

    assert len(vla.calls) == 2  # segment continued: a fresh chunk, same subgoal
    assert vla.calls[1].task == "slide the cup"
    assert chunk.actions[0].meta.get("request_stop") is None


def test_max_skill_seconds_caps_a_delegation() -> None:
    llm = _FakeLlm([_delegate("keep stirring", 999.0), _done()])
    vla = _FakeVla([_zero_chunk()])
    policy = _policy(llm, vla, max_skill_seconds=1e-6, checkpoint_interval_s=5.0)
    _reset(policy)

    policy.act(_observation())
    policy.act(_observation())  # deadline elapsed: back to the planner

    # The delegation's tool result named the cap, and the progress note reports it.
    assert "max_skill_seconds" in _text_of(llm.calls[1].messages)
    assert "max_skill_seconds cap reached" in _text_of([_last_user(llm.calls[1])])
    assert len(vla.calls) == 1


def test_budget_exhaustion_forces_give_up() -> None:
    llm = _FakeLlm([_delegate("try once"), _done()])
    vla = _FakeVla([_zero_chunk()])
    policy = _policy(llm, vla, budget_llm_calls=1, checkpoint_interval_s=0.0)
    _reset(policy)

    policy.act(_observation())  # spends the single call on the delegation
    stop = policy.act(_observation())

    assert len(llm.calls) == 1  # no second call: the give_up is synthetic
    meta = stop.actions[0].meta
    assert meta["request_stop"] is True
    assert meta["stop_reason"] == "give_up"
    assert meta["stop_detail"] == "LLM call budget exhausted"
    assert "stop_hindsight" not in meta


def test_give_up_tool_ends_the_trial() -> None:
    llm = _FakeLlm([_give_up("the cup is out of reach")])
    vla = _FakeVla([])
    policy = _policy(llm, vla)
    _reset(policy)

    stop = policy.act(_observation())

    assert vla.calls == []
    meta = stop.actions[0].meta
    assert meta["stop_reason"] == "give_up"
    assert meta["stop_detail"] == "the cup is out of reach"
    with pytest.raises(PolicyError, match="already ended"):
        policy.act(_observation())


def test_invalid_delegate_arguments_are_corrected_within_the_turn() -> None:
    llm = _FakeLlm(
        [_delegate("missing bound", max_seconds=None), _delegate("valid subgoal", 10.0), _done()]
    )
    vla = _FakeVla([_zero_chunk()])
    policy = _policy(llm, vla, checkpoint_interval_s=0.0)
    _reset(policy)

    chunk = policy.act(_observation())

    assert len(llm.calls) == 2  # the invalid turn re-asked inside the same act
    assert "max_seconds" in _text_of(llm.calls[1].messages)
    assert vla.calls[0].task == "valid subgoal"
    assert chunk.meta["subgoal"] == "valid subgoal"


def test_three_turns_without_a_tool_call_raise_policy_error() -> None:
    llm = _FakeLlm([_no_call(), _no_call(), _no_call()])
    policy = _policy(llm, vla=_FakeVla([]))
    _reset(policy)

    with pytest.raises(PolicyError, match="tool call"):
        policy.act(_observation())

    assert len(llm.calls) == 3


def test_reset_rearms_the_retry_and_clears_segment_state() -> None:
    error = VlaServiceError("dropped")
    llm = _FakeLlm([_delegate("first"), _delegate("second")])
    vla = _FakeVla([error, error, error, _zero_chunk()])
    policy = _policy(llm, vla)
    _reset(policy, instruction="a scene instruction the prompt overrides")

    with pytest.raises(PolicyError, match="dropped"):
        policy.act(_observation())  # infer fails twice: the retry is spent
    assert len(vla.calls) == 2

    _reset(policy, instruction="ignored scene text")
    chunk = policy.act(_observation())

    # The retry re-armed: two fresh infer attempts, the second succeeds.
    assert len(vla.calls) == 4
    assert len(chunk) == 2
    # The conversation restarted, and the configured prompt is the goal.
    second = llm.calls[1]
    assert len(second.messages) == 3  # system, goal, current observation
    assert _PROMPT in _text_of([second.messages[1]])


def test_bind_fails_fast_on_camera_mismatch_and_records_docs() -> None:
    docs = "Left arm mounts lower; gripper widths are meters."
    cameras = (
        CameraSpec("left_rgbd", 4, 8),
        CameraSpec("right_rgbd", 4, 8),
        CameraSpec("chest_rgbd", 4, 8),
    )
    space = ObservationSpace(cameras=cameras, state_keys=frozenset({"eef_state"}))
    info = EmbodimentInfo(
        name="nero",
        action_space=Box(
            shape=(20,),
            semantics=ActionSemantics(
                control_mode="eef_abs_pose", rotation_repr="rot6d", gripper="continuous"
            ),
        ),
        observation_space=space,
        docs=docs,
    )
    llm = _FakeLlm([_delegate("look around"), _done()])
    vla = _FakeVla([_zero_chunk()])
    policy = _policy(llm, vla, checkpoint_interval_s=0.0)
    policy.bind(info)  # all submit cameras declared: binds cleanly
    _reset(policy)
    policy.act(_observation())
    assert docs in str(llm.calls[0].messages[0]["content"])

    mismatch = EmbodimentInfo(
        name="two-cam",
        action_space=Box(shape=(20,)),
        observation_space=ObservationSpace(
            cameras=cameras[:2], state_keys=frozenset({"eef_state"})
        ),
    )
    with pytest.raises(ConfigError, match="chest_rgbd"):
        policy.bind(mismatch)

    # An embodiment that declares no cameras cannot be checked here; the
    # act-time missing-camera error still guards it.
    policy.bind(
        EmbodimentInfo(
            name="bare", action_space=Box(shape=(20,)), observation_space=ObservationSpace()
        )
    )


def test_executing_requires_the_state_key_with_no_fallback() -> None:
    llm = _FakeLlm([_delegate("move"), _done()])
    vla = _FakeVla([_zero_chunk()])
    policy = _policy(llm, vla, checkpoint_interval_s=5.0)
    _reset(policy)

    policy.act(_observation())
    with pytest.raises(PolicyError, match="eef_state"):
        policy.act(_observation(with_state=False))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"prompt": 42}, "prompt"),
        ({"vla_base_url": "127.0.0.1:10055"}, "vla_base_url"),
        ({"budget_llm_calls": 0}, "budget_llm_calls"),
        ({"max_skill_seconds": 0}, "max_skill_seconds"),
        ({"checkpoint_interval_s": -1.0}, "checkpoint_interval_s"),
        ({"tracking_abort_pos_m": 0.0}, "tracking_abort_pos_m"),
        ({"tracking_abort_rot_deg": -5.0}, "tracking_abort_rot_deg"),
        ({"control_hz": 0}, "control_hz"),
        ({"state_key": ""}, "state_key"),
        ({"submit_images": "left_rgbd"}, "2..3"),
        ({"model": 42}, "model"),
    ],
)
def test_construction_rejects_bad_kwargs(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        hybrid_policy(llm=_FakeLlm([]), vla=_FakeVla([]), **kwargs)


def test_reset_without_a_goal_raises_policy_error() -> None:
    policy = hybrid_policy("m", llm=_FakeLlm([]), vla=_FakeVla([]))
    with pytest.raises(PolicyError, match="prompt"):
        policy.reset(Scene(id="s", instruction=""))


def test_default_llm_and_vla_clients_are_built_and_closed() -> None:
    policy = hybrid_policy(
        "openrouter/qwen",
        base_url=None,
        api_key_env=None,
        env={"OPENROUTER_API_KEY": "k"},
        vla_base_url="http://127.0.0.1:10055",
    )
    assert policy._owned_llm is not None
    assert policy._owned_vla is not None
    policy.close()
    assert policy._owned_llm is None
    assert policy._owned_vla is None
    policy.close()  # idempotent
    policy.__del__()  # guarded: closing twice must not raise

    with pytest.raises(ConfigError, match="no API key found"):
        hybrid_policy("m", env={}, vla=_FakeVla([]))


def test_close_leaves_injected_backends_alone() -> None:
    llm = _FakeLlm([])
    vla = _FakeVla([])
    policy = _policy(llm, vla)
    policy.close()
    assert not vla.closed


def test_transcript_hooks_expose_the_planner_conversation() -> None:
    llm = _FakeLlm([_delegate("lift the cup"), _done("cup on saucer")])
    vla = _FakeVla([_zero_chunk()])
    policy = _policy(llm, vla, checkpoint_interval_s=0.0)
    assert policy.transcript() is None  # no trial yet: no conversation

    _reset(policy)
    policy.act(_observation())  # delegate + one executed chunk
    first_delta = policy.transcript_delta()
    assert first_delta is not None
    dumped = json.dumps(first_delta)
    assert "image omitted" in dumped  # frames are stubbed, never serialized
    assert "image_url" not in dumped
    assert "lift the cup" in dumped
    assert first_delta[0]["role"] == "system"

    policy.act(_observation())  # checkpoint -> done
    second_delta = policy.transcript_delta()
    full = policy.transcript()
    assert second_delta is not None and full is not None
    # The delta carries only what was appended since the previous read; the
    # full transcript carries the whole conversation including both turns.
    assert len(second_delta) < len(full)
    assert "cup on saucer" in json.dumps(second_delta)
    assert "Goal: " + _PROMPT in json.dumps(full)
    # A deep copy: callers cannot corrupt the live conversation through it.
    full.append({"role": "user", "content": "forged"})
    assert len(policy.transcript() or []) == len(full) - 1
    # A fresh trial re-arms the cursor: the next delta starts from system, and
    # the goal line is the configured prompt (it overrides scene instructions).
    _reset(policy, instruction="a scene instruction the prompt overrides")
    fresh = policy.transcript_delta()
    assert fresh is not None
    assert [m["role"] for m in fresh] == ["system", "user"]
    assert "Goal: " + _PROMPT in json.dumps(fresh)
