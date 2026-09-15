"""NeroArm/NeroGripper wrapper behavior against fakes."""

from __future__ import annotations

import pytest

from inspect_robots_nero._arm import NeroArm
from inspect_robots_nero._gripper import NeroGripper

from .fakes import FakeAgxRobot, FakeEffector, LegacyEffector


def _arm(robot: FakeAgxRobot) -> NeroArm:
    return NeroArm("left", "can_left", robot=robot, sleep=lambda _seconds: None)


def test_connect_with_injected_robot_skips_the_sdk() -> None:
    robot = FakeAgxRobot()
    arm = _arm(robot)
    arm.connect()
    assert robot.connected
    arm.set_speed_percent(20)
    assert robot.speed == 20
    arm.set_normal_mode()
    assert robot.mode == "follower"


def test_move_js_records_floats() -> None:
    robot = FakeAgxRobot()
    arm = _arm(robot)
    arm.connect()
    arm.move_js([1, 2, 3, 4, 5, 6, 7])
    assert robot.move_js_calls == [(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)]


def test_move_js_adopts_the_commanded_joints_as_readback() -> None:
    robot = FakeAgxRobot()
    arm = _arm(robot)
    arm.connect()
    arm.move_js([1, 2, 3, 4, 5, 6, 7])
    assert arm.read_state() == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)


def test_read_state_coerces_and_reports_missing() -> None:
    robot = FakeAgxRobot()
    arm = _arm(robot)
    assert arm.read_state() is None  # not connected yet
    arm.connect()
    assert arm.read_state() == robot.angles
    robot.angles = None  # type: ignore[assignment]
    assert arm.read_state() is None


def test_disable_and_close() -> None:
    robot = FakeAgxRobot()
    arm = _arm(robot)
    arm.connect()
    arm.disable()
    assert not robot.enabled
    arm.close()
    assert robot.disconnected


def test_unknown_firmware_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="firmware"):
        NeroArm("left", "can_left", firmware="v999", robot=FakeAgxRobot())


def test_gripper_move_and_readback() -> None:
    robot = FakeAgxRobot()
    arm = _arm(robot)
    arm.connect()
    effector = FakeEffector()
    gripper = NeroGripper(arm, effector=effector)
    gripper.start()  # no-op: the effector was injected
    gripper.move(0.04, 0.3)
    assert effector.moves == [(0.04, 0.3)]
    assert gripper.read_state() == 0.04


def test_gripper_start_attaches_the_vendor_effector() -> None:
    robot = FakeAgxRobot()
    arm = _arm(robot)
    arm.connect()
    gripper = NeroGripper(arm, sleep=lambda _seconds: None)
    gripper.start()
    assert robot.effector_kind == "AGX_GRIPPER"
    assert gripper.read_state() == 0.09


def test_gripper_legacy_api_fallback() -> None:
    robot = FakeAgxRobot()
    arm = _arm(robot)
    arm.connect()
    gripper = NeroGripper(arm, effector=LegacyEffector())
    gripper.move(0.05, 0.2)
    assert gripper.read_state() == 0.05


def test_gripper_stop_detaches_the_effector() -> None:
    gripper = NeroGripper(_arm(FakeAgxRobot()), effector=FakeEffector())
    gripper.stop()
    assert gripper.read_state() is None


def test_gripper_start_requires_connected_arm() -> None:
    gripper = NeroGripper(_arm(FakeAgxRobot()))
    with pytest.raises(RuntimeError, match="connected"):
        gripper.start()
