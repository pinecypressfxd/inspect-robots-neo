"""NeroArm/NeroGripper wrapper behavior against fakes."""

from __future__ import annotations

import pytest

from inspect_robots_nero._arm import _FIRMWARE_NAMES, NeroArm
from inspect_robots_nero._gripper import NeroGripper

from .fakes import FakeAgxRobot, FakeEffector, LegacyEffector

# Keys AgxArmFactory registers for robot="nero", comm="can" (lowercase,
# verbatim from the vendor SDK's registry block and register_arm docstring).
_REGISTRY_KEYS = {"default", "v111", "v112", "v120", "v121"}


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


@pytest.mark.parametrize("alias", sorted(_FIRMWARE_NAMES))
def test_firmware_aliases_resolve_to_registry_keys(alias: str) -> None:
    arm = NeroArm("left", "can_left", firmware=alias, robot=FakeAgxRobot(), sleep=lambda _s: None)
    assert arm._firmware in _REGISTRY_KEYS, alias


def test_default_firmware_is_the_config_version() -> None:
    arm = NeroArm("left", "can_left", robot=FakeAgxRobot(), sleep=lambda _s: None)
    assert arm._firmware == "v112"


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


def test_move_j_and_position_mode_wrap_the_vendor_calls() -> None:
    robot = FakeAgxRobot()
    arm = _arm(robot)
    arm.connect()
    arm.set_position_mode()
    arm.move_j([1, 2, 3, 4, 5, 6, 7])
    assert robot.mode == "normal"
    assert robot.move_j_calls == [(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)]


def test_enable_status_reports_seven_bools_or_none() -> None:
    robot = FakeAgxRobot()
    arm = _arm(robot)
    assert arm.enable_status() is None  # not connected yet
    arm.connect()
    arm.enable()
    assert arm.enable_status() == [True] * 7
    arm.disable()
    assert arm.enable_status() == [False] * 7


def test_enable_status_rejects_malformed_vendor_feedback() -> None:
    robot = FakeAgxRobot()
    arm = _arm(robot)
    arm.connect()
    robot.joints_enable = [True, True]  # wrong length
    with pytest.raises(ValueError, match="7 bools"):
        arm.enable_status()
