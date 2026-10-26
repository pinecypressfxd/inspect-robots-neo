"""HarnessVLA: fixed primitives, trial memory, and repeated VLA grasps."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from inspect_robots import Scene
from inspect_robots.errors import ConfigError
from inspect_robots_agent._llm import AssistantMessage, ToolCall
from inspect_robots_harness import HarnessVlaPolicy, harness_policy

_spec = importlib.util.spec_from_file_location(
    "_hybrid_policy_tests",
    Path(__file__).resolve().parents[2] / "inspect-robots-vla" / "tests" / "test_executor.py",
)
assert _spec is not None and _spec.loader is not None
_hybrid_tests = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _hybrid_tests
_spec.loader.exec_module(_hybrid_tests)
_bind_semantics = _hybrid_tests._bind_semantics
_done = _hybrid_tests._done
_eef_state = _hybrid_tests._eef_state
_FakeLlm = _hybrid_tests._FakeLlm
_FakeVla = _hybrid_tests._FakeVla
_move_to = _hybrid_tests._move_to
_observation = _hybrid_tests._observation
_text_of = _hybrid_tests._text_of
_zero_chunk = _hybrid_tests._zero_chunk


def _vla_act(max_seconds: float = 30.0) -> AssistantMessage:
    return AssistantMessage(
        content=None,
        tool_calls=(
            ToolCall(
                id="c",
                name="vla_act",
                arguments=json.dumps({"max_seconds": max_seconds}),
            ),
        ),
    )


def _release(arm: str = "left") -> AssistantMessage:
    return AssistantMessage(
        content=None,
        tool_calls=(
            ToolCall(
                id="c",
                name="release",
                arguments=json.dumps({"arm": arm, "note": "put the block down"}),
            ),
        ),
    )


def _remember(fact: str) -> AssistantMessage:
    return AssistantMessage(
        content=None,
        tool_calls=(ToolCall(id="c", name="remember", arguments=json.dumps({"fact": fact})),),
    )


def _policy(llm: _FakeLlm, vla: _FakeVla, **kwargs: Any) -> HarnessVlaPolicy:
    policy = harness_policy(
        "test/model",
        llm=llm,
        vla=vla,
        prompt=kwargs.pop("prompt", "put the red cup in the basket"),
        **kwargs,
    )
    assert isinstance(policy, HarnessVlaPolicy)
    _bind_semantics(policy)
    policy.reset(Scene(id="s", instruction="put the red cup in the basket"))
    return policy


def test_vla_act_sends_trained_sentence_and_ignores_checkpoint() -> None:
    llm = _FakeLlm([_vla_act(), _done()])
    vla = _FakeVla([_zero_chunk(), _zero_chunk()])
    policy = _policy(llm, vla, checkpoint_interval_s=0.0)
    assert policy.info.name == "harness"
    assert policy.config.name == "harness"
    assert policy.config.max_vla_attempts == 3

    first = policy.act(_observation())
    second = policy.act(_observation())

    assert [call.task for call in vla.calls] == [
        "please pick the red cup and place it on the black mat",
        "please pick the red cup and place it on the black mat",
    ]
    assert first.meta["subgoal"] == "please pick the red cup and place it on the black mat"
    assert second.meta["subgoal"] == "please pick the red cup and place it on the black mat"
    assert len(llm.calls) == 1
    system = str(llm.calls[0].messages[0]["content"])
    assert "Trial memory" in _text_of([llm.calls[0].messages[-1]]) or "memory" in system
    assert "Standing objects already in the workspace" in system
    assert "diagonal low carry is fine" in system
    names = {tool["function"]["name"] for tool in llm.calls[0].tools}
    assert names == {
        "move_to",
        "set_gripper",
        "release",
        "vla_act",
        "remember",
        "done",
        "give_up",
    }


def test_vla_attempt_is_capped_before_the_trained_place() -> None:
    llm = _FakeLlm([_vla_act(15)])
    policy = _policy(llm, _FakeVla([_zero_chunk()]))

    policy.act(_observation())

    assert "for up to 3s" in _text_of(policy._messages)
    assert "caps every attempt at 3 s" in str(llm.calls[0].messages[0]["content"])


def test_vla_act_waits_until_the_left_hand_is_back_at_home() -> None:
    home = _eef_state()
    raised = home.copy()
    raised[2] = home[2] + 0.03
    llm = _FakeLlm([_move_to({"left_x": 0.40}), _vla_act(), _done()])
    vla = _FakeVla([_zero_chunk()])
    policy = _policy(llm, vla)

    policy.act(_observation(eef=home))
    stopped = policy.act(_observation(eef=raised))

    assert vla.calls == []
    assert stopped.actions[0].meta["stop_reason"] == "done"
    text = _text_of(policy._messages)
    assert "above this trial's home" in text
    assert f"{home[2]:.3f}" in text


def test_vla_act_allows_a_left_hand_within_two_centimetres_of_home() -> None:
    home = _eef_state()
    near = home.copy()
    near[2] = home[2] + 0.02
    llm = _FakeLlm([_move_to({"left_x": 0.40}), _vla_act()])
    vla = _FakeVla([_zero_chunk()])
    policy = _policy(llm, vla)

    policy.act(_observation(eef=home))
    policy.act(_observation(eef=near))

    assert len(vla.calls) == 1


def test_repeat_vla_act_waits_for_an_analytic_restage() -> None:
    llm = _FakeLlm([_vla_act(), _vla_act(), _done()])
    vla = _FakeVla([_zero_chunk()])
    policy = _policy(llm, vla, max_vla_attempts=2, max_skill_seconds=1e-6)

    policy.act(_observation())
    stopped = policy.act(_observation())

    assert len(vla.calls) == 1
    assert stopped.actions[0].meta["stop_reason"] == "done"
    transcript = _text_of(llm.calls[-1].messages)
    assert "restage with move_to, set_gripper, or release" in transcript
    assert "Restage before another vla_act." in transcript


def test_attempt_cap_stops_another_vla_act() -> None:
    llm = _FakeLlm([_vla_act(), _vla_act(), _move_to({"left_x": 0.40}), _done()])
    vla = _FakeVla([_zero_chunk()])
    policy = _policy(llm, vla, max_vla_attempts=1, max_skill_seconds=1e-6)

    policy.act(_observation())
    carried = policy.act(_observation())

    assert len(vla.calls) == 1
    assert carried.meta.get("analytic") == "move_to"
    transcript = _text_of(llm.calls[-1].messages)
    assert "attempt cap" in transcript
    assert "vla_act attempt 1 ended" in _text_of([llm.calls[1].messages[-1]])


def test_operator_guidance_waits_for_the_next_analytic_decision() -> None:
    llm = _FakeLlm([_move_to({"left_z": 0.99}), _done()])
    vla = _FakeVla([])
    policy = _policy(llm, vla)
    policy.act(_observation())
    guided = _observation()
    guided = type(guided)(
        images=guided.images,
        state=guided.state,
        extra={"operator_messages": [{"t": 4, "text": "杯子在左边 先抬高"}]},
    )

    stopped = policy.act(guided)

    assert vla.calls == []
    assert stopped.actions[0].meta["stop_reason"] == "done"
    assert "operator: 杯子在左边 先抬高" in _text_of(llm.calls[-1].messages)


def test_chinese_red_cup_goal_keeps_vla_act() -> None:
    goal = "请将红色的杯子放到二的位置。"
    llm = _FakeLlm([_done()])
    vla = _FakeVla([])
    policy = harness_policy("test/model", llm=llm, vla=vla, prompt=goal)
    _bind_semantics(policy)
    policy.reset(Scene(id="s", instruction=goal))

    policy.act(_observation())

    names = {tool["function"]["name"] for tool in llm.calls[0].tools}
    assert "vla_act" in names
    system = str(llm.calls[0].messages[0]["content"])
    assert "does not name a red cup" not in system
    assert "already close to the red cup" in system


def test_goal_without_red_cup_hides_vla_act() -> None:
    llm = _FakeLlm([_vla_act(), _release("right"), _done()])
    vla = _FakeVla([_zero_chunk()])
    policy = harness_policy("test/model", llm=llm, vla=vla, prompt="slide the box left")
    _bind_semantics(policy)
    policy.reset(Scene(id="s", instruction="slide the box left"))

    chunk = policy.act(_observation())

    names = {tool["function"]["name"] for tool in llm.calls[0].tools}
    assert "vla_act" not in names
    assert "release" in names
    assert vla.calls == []
    assert chunk.meta.get("analytic") == "set_gripper"
    assert chunk.actions[-1].data[19] == pytest.approx(0.09, abs=1e-6)
    assert "does not name a red cup" in _text_of(llm.calls[-1].messages)


def test_remember_stays_in_the_planner_loop() -> None:
    llm = _FakeLlm([_remember("red block is left of the basket"), _done()])
    vla = _FakeVla([])
    policy = _policy(llm, vla)

    stop = policy.act(_observation())

    assert vla.calls == []
    assert stop.actions[0].meta["stop_reason"] == "done"
    assert "remembered: red block is left of the basket" in _text_of(llm.calls[-1].messages)
    assert len(policy._memory) == 1


def test_release_opens_the_named_gripper() -> None:
    llm = _FakeLlm([_release("left"), _done()])
    policy = _policy(llm, _FakeVla([]))

    chunk = policy.act(_observation())
    row = chunk.actions[-1].data
    assert row[9] == pytest.approx(0.09, abs=1e-6)
    assert row[19] == pytest.approx(_eef_state()[19], abs=1e-6)
    assert row[0:3] == pytest.approx(_eef_state()[0:3], abs=1e-6)


def test_max_vla_attempts_must_be_positive() -> None:
    with pytest.raises(ConfigError, match="max_vla_attempts"):
        harness_policy("m", llm=_FakeLlm([]), vla=_FakeVla([]), max_vla_attempts=0)
