"""Operator tool logic (home/disable) against fake arms."""

from __future__ import annotations

import pytest

from inspect_robots_nero._arm import NeroArm
from inspect_robots_nero._config import (
    GRIPPER_FORCE,
    GRIPPER_INIT_WIDTH_M,
    HOME_LEFT,
    HOME_RIGHT,
)
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


_FAKE_INITIAL = (0.1, 0.2, 0.0, 0.5, 0.0, 0.0, 0.3)  # FakeAgxRobot's start pose


def test_home_arms_primes_then_move_js_straight_home() -> None:
    arms, robots = _connected_arms()
    home_arms(arms, wait_s=0.0, sleep=lambda _seconds: None)
    for robot in robots.values():
        # The fake starts disabled, so Home clears the e-stop latch first.
        assert robot.motion_resets == 1
    for side, robot in robots.items():
        # The reference reset: one move_j commanding the CURRENT joints
        # (priming the trajectory planner onto the measured pose), then one
        # move_j straight to the config home -- no follower streaming.
        home = HOME_LEFT if side == "left" else HOME_RIGHT
        assert robot.move_js_calls == []
        # Seat, leave follower and reseat, then one trajectory to home.
        assert robot.move_j_calls == [_FAKE_INITIAL, _FAKE_INITIAL, tuple(home)]
        assert robot.mode == "normal"


def test_home_seats_a_holding_arm_before_any_mode_packet() -> None:
    """A live MIT hold must not see a bare mode frame before the measured joints.

    Home opens a fresh driver whose mit_mode default is position-velocity.
    set_follower_mode / set_speed_percent / enable emit that default with no
    joint target and the arm sags. The reference reset seats move_j(current)
    first; speed comes after, and an already-enabled arm is left alone.
    """
    arms, robots = _connected_arms()
    for robot in robots.values():
        robot.enabled = True
    home_arms(arms, wait_s=0.0, sleep=lambda _seconds: None)
    for robot in robots.values():
        assert robot.motion_resets == 0
        kinds = [str(event[0]) for event in robot.events]
        first_move = kinds.index("move_j")
        assert kinds[:first_move] == []
        assert "set_follower_mode" not in kinds
        assert robot.events[first_move] == ("move_j", _FAKE_INITIAL)
        assert kinds.index("set_normal_mode") > first_move
        assert "set_speed_percent" in kinds[first_move:]


def test_home_after_estop_resets_before_enable() -> None:
    arms, robots = _connected_arms()
    for robot in robots.values():
        robot.joints_enable = [False] * 7
        robot.enable_result = False

        def enable_once_reset(bound: FakeAgxRobot = robot) -> bool:
            bound.events.append(("enable",))
            if bound.motion_resets < 1:
                return False
            bound.enabled = True
            bound.joints_enable = [True] * 7
            bound.enable_result = True
            return True

        robot.enable = enable_once_reset  # type: ignore[method-assign]

    home_arms(arms, wait_s=0.0, sleep=lambda _seconds: None)
    for robot in robots.values():
        kinds = [str(event[0]) for event in robot.events]
        assert kinds.index("reset") < kinds.index("enable") < kinds.index("move_j")


def test_home_arms_opens_both_grippers_to_init_width() -> None:
    arms, robots = _connected_arms()
    home_arms(arms, wait_s=0.0, sleep=lambda _seconds: None)
    for robot in robots.values():
        assert robot.effector_kind == "AGX_GRIPPER"
        # Home is a clean state: each gripper is commanded fully open once,
        # releasing whatever width the previous run left (e.g. a grasp).
        assert robot.effectors[0].moves == [(GRIPPER_INIT_WIDTH_M, GRIPPER_FORCE)]


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
