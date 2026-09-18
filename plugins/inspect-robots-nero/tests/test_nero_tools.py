"""Operator tool logic (home/disable) against fake arms."""

from __future__ import annotations

import pytest

from inspect_robots_nero._arm import NeroArm
from inspect_robots_nero._config import HOME_LEFT, HOME_RIGHT
from inspect_robots_nero._tools import build_arms, connect_all, disable_arms, home_arms

from .fakes import FakeAgxRobot


def _connected_arms() -> tuple[dict[str, NeroArm], dict[str, FakeAgxRobot]]:
    robots = {"left": FakeAgxRobot(), "right": FakeAgxRobot()}
    arms = {
        side: NeroArm(side, f"can_{side}", robot=robot, sleep=lambda _seconds: None)
        for side, robot in robots.items()
    }
    for arm in arms.values():
        arm.connect()
    return arms, robots


def test_build_arms_uses_config_channels() -> None:
    arms = build_arms(["left"])
    assert arms["left"].channel == "can_left"


def test_home_arms_moves_both_arms_to_config_home() -> None:
    arms, robots = _connected_arms()
    home_arms(arms, wait_s=0.0, sleep=lambda _seconds: None)
    assert robots["left"].move_j_calls == [tuple(HOME_LEFT)]
    assert robots["right"].move_j_calls == [tuple(HOME_RIGHT)]
    for robot in robots.values():
        assert robot.enabled
        assert robot.mode == "normal"


def test_home_arms_raises_when_an_arm_fails_to_enable() -> None:
    arms, robots = _connected_arms()
    robots["right"].enable_result = False
    with pytest.raises(RuntimeError, match="enable"):
        home_arms(arms, wait_s=0.0, sleep=lambda _seconds: None)
    assert robots["left"].move_j_calls == []


def test_disable_arms_polls_until_all_joints_report_limp() -> None:
    arms, robots = _connected_arms()
    robots["left"].joints_enable = [True] * 7  # left lags one poll interval
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        robots["left"].joints_enable = [False] * 7

    now = [0.0]

    def fake_clock() -> float:
        now[0] += 0.05
        return now[0]

    assert disable_arms(arms, timeout_s=5.0, poll_s=0.1, clock=fake_clock, sleep=fake_sleep)
    assert sleeps
    for robot in robots.values():
        assert not robot.enabled
        assert robot.damped_stops >= 1  # damping precedes every disable attempt


def test_disable_arms_times_out_when_a_joint_stays_enabled() -> None:
    arms, robots = _connected_arms()
    robots["right"].joints_enable = [True] * 7  # stuck: never reports limp
    now = [0.0]

    def fake_clock() -> float:
        now[0] += 1.0
        return now[0]

    assert not disable_arms(
        arms, timeout_s=0.5, poll_s=0.01, clock=fake_clock, sleep=lambda _s: None
    )


def test_connect_all_disconnects_successful_sides_when_one_fails() -> None:
    robots = {"left": FakeAgxRobot(), "right": FakeAgxRobot()}

    def boom() -> None:
        raise OSError("can interface down")

    robots["right"].connect = boom  # type: ignore[method-assign]
    arms = {
        "left": NeroArm("left", "can_left", robot=robots["left"], sleep=lambda _s: None),
        "right": NeroArm("right", "can_right", robot=robots["right"], sleep=lambda _s: None),
    }
    with pytest.raises(OSError, match="can interface down"):
        connect_all(arms)
    assert robots["left"].disconnected
    assert not robots["right"].disconnected


def test_home_accepts_an_already_enabled_arm() -> None:
    arms, robots = _connected_arms()
    for robot in robots.values():
        robot.enabled = True  # an earlier process enabled and never closed
        robot.enable_result = False  # firmware returns falsy when already enabled
    home_arms(arms, wait_s=0.0, sleep=lambda _seconds: None)
    for robot in robots.values():
        assert robot.move_j_calls  # homing proceeded despite the falsy enable
