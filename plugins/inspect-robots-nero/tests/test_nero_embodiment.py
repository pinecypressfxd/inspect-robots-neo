"""Full reset/step/close loop for the nero embodiment over fakes."""

from __future__ import annotations

import importlib.resources
from typing import cast

import numpy as np
import pytest

from inspect_robots import Action, Scene
from inspect_robots.conformance import assert_embodiment_conformant
from inspect_robots.errors import EmbodimentFault
from inspect_robots_nero import nero_embodiment
from inspect_robots_nero._arm import NeroArm
from inspect_robots_nero._config import GRIPPER_FORCE, GRIPPER_INIT_WIDTH_M, HOME_LEFT, HOME_RIGHT
from inspect_robots_nero._kinematics import NeroKinematics, matrix_to_rot6d
from inspect_robots_nero.embodiment import CameraFactory

from .fakes import FakeAgxRobot

_FRAME = np.full((480, 640, 3), 9, dtype=np.uint8)

_URDF = str(importlib.resources.files("inspect_robots_nero") / "assets" / "dual_nero_pika.urdf")
_KINEMATICS = NeroKinematics(_URDF)
_HOME_POSES = _KINEMATICS.fk(_KINEMATICS.q_from(HOME_LEFT, HOME_RIGHT))


def _home_action(left_width: float, right_width: float) -> np.ndarray:
    """A 20-dim action holding both arms at their home TCP poses (IK-reachable)."""
    return np.concatenate(
        [
            _HOME_POSES["left"][:3, 3],
            matrix_to_rot6d(_HOME_POSES["left"][:3, :3]),
            [left_width],
            _HOME_POSES["right"][:3, 3],
            matrix_to_rot6d(_HOME_POSES["right"][:3, :3]),
            [right_width],
        ]
    )


class FakeCamera:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.frame = _FRAME

    def start(self) -> None:
        self.started = True

    def read(self, *, max_age_s: float) -> tuple[np.ndarray, float]:
        del max_age_s
        return self.frame, 1.0

    def stop(self) -> None:
        self.stopped = True


class _ScriptedClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 1.0 / 30.0
        return self.now


class Harness:
    def __init__(self, *, operator_reset_confirm: bool = False) -> None:
        self.robots = {"left": FakeAgxRobot(), "right": FakeAgxRobot()}
        self.cameras = {name: FakeCamera() for name in ("left_rgbd", "right_rgbd", "chest_rgbd")}
        self.sleeps: list[float] = []

        def make_arm(side: str) -> NeroArm:
            arm = NeroArm(side, f"can_{side}", robot=self.robots[side], sleep=lambda _s: None)
            arm.connect()  # factories own construct+connect; the embodiment enables/configures
            return arm

        def make_camera(name: str) -> FakeCamera:
            camera = self.cameras[name]
            camera.start()  # factories own construct+start, like _default_camera_factory
            return camera

        embodiment = nero_embodiment(
            operator_reset_confirm=operator_reset_confirm,
            arm_factory=make_arm,
            camera_factory=make_camera,
            clock=_ScriptedClock(),
            sleep=self.sleeps.append,
        )
        self.embodiment = embodiment


def test_reset_enables_homes_and_returns_first_observation() -> None:
    harness = Harness()
    observation = harness.embodiment.reset(Scene(id="s0", instruction="put the cup on the pad"))
    for robot in harness.robots.values():
        assert robot.enabled and robot.speed == 20 and robot.mode == "follower"
        assert robot.move_js_calls[-1] == tuple(
            HOME_LEFT if robot is harness.robots["left"] else HOME_RIGHT
        )
    for camera in harness.cameras.values():
        assert camera.started and not camera.stopped
    assert observation.instruction == "put the cup on the pad"
    assert set(observation.images) == {"left_rgbd", "right_rgbd", "chest_rgbd"}
    assert observation.state["eef_state"].shape == (20,)
    assert observation.state["joint_pos"].shape == (16,)
    gripper_dims = (observation.state["joint_pos"][14], observation.state["joint_pos"][15])
    assert gripper_dims == (0.09, 0.09)  # the fake effector reports its 0.09 init width


def test_reset_confirm_prompts_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return ""

    monkeypatch.setattr("builtins.input", fake_input)
    harness = Harness(operator_reset_confirm=True)
    harness.embodiment.reset(Scene(id="s0", instruction="x"))
    assert prompts and "Arrange the scene" in prompts[0]


def test_step_commands_bounded_joint_deltas_and_gripper_widths() -> None:
    harness = Harness()
    harness.embodiment.reset(Scene(id="s0", instruction="x"))
    action = _home_action(GRIPPER_INIT_WIDTH_M, 0.02)
    result = harness.embodiment.step(Action(data=action))
    for robot in harness.robots.values():
        assert len(robot.move_js_calls) == 2  # home, then one step command
    assert result.observation.images["left_rgbd"].shape == (480, 640, 3)


def test_step_paces_itself_to_control_hz() -> None:
    harness = Harness()
    harness.embodiment.reset(Scene(id="s0", instruction="x"))
    action = _home_action(0.04, 0.04)
    baseline = len(harness.sleeps)
    harness.embodiment.step(Action(data=action))
    first = len(harness.sleeps)
    harness.embodiment.step(Action(data=action))
    assert first > baseline  # the first step slept to honor the cadence after reset
    assert len(harness.sleeps) > first  # the second step slept again


def test_action_shape_mismatch_is_rejected() -> None:
    harness = Harness()
    harness.embodiment.reset(Scene(id="s0", instruction="x"))
    with pytest.raises(ValueError, match="shape"):
        harness.embodiment.step(Action(data=np.zeros(6)))


def test_unreachable_action_raises_embodiment_fault() -> None:
    harness = Harness()
    harness.embodiment.reset(Scene(id="s0", instruction="x"))
    far = _home_action(GRIPPER_INIT_WIDTH_M, GRIPPER_INIT_WIDTH_M)
    far[0] = 50.0  # five meters past any reachable pose
    with pytest.raises(EmbodimentFault):
        harness.embodiment.step(Action(data=far))


def test_gripper_force_is_the_configured_constant() -> None:
    harness = Harness()
    harness.embodiment.reset(Scene(id="s0", instruction="x"))
    action = _home_action(0.03, GRIPPER_INIT_WIDTH_M)
    harness.embodiment.step(Action(data=action))
    # The left gripper effector recorded (width, force); the fake stores moves.
    left_arm = harness.embodiment._arms["left"]
    assert left_arm.raw_robot.effector_kind == "AGX_GRIPPER"
    left_gripper = harness.embodiment._grippers["left"]
    effector = left_gripper._effector
    assert effector is not None
    assert effector.moves[-1] == (0.03, GRIPPER_FORCE)


def test_close_disables_arms_and_stops_cameras() -> None:
    harness = Harness()
    harness.embodiment.reset(Scene(id="s0", instruction="x"))
    harness.embodiment.close()
    for robot in harness.robots.values():
        assert not robot.enabled
    for camera in harness.cameras.values():
        assert camera.stopped


def test_step_clamps_per_tick_joint_delta() -> None:
    harness = Harness()
    harness.embodiment.reset(Scene(id="s0", instruction="x"))
    pre = {side: np.asarray(robot.angles) for side, robot in harness.robots.items()}
    # +0.15 m along +x stays inside the sampled workspace and converges from the
    # home seed; the raw IK solution moves the left arm 0.54 rad, well past the
    # 0.1 rad per-tick clamp, so the clipped path is genuinely exercised.
    action = _home_action(GRIPPER_INIT_WIDTH_M, GRIPPER_INIT_WIDTH_M)
    action[0] += 0.15
    harness.embodiment.step(Action(data=action))
    for side in ("left", "right"):
        commanded = np.asarray(harness.robots[side].move_js_calls[-1])
        assert float(np.max(np.abs(commanded - pre[side]))) <= 0.1 + 1e-9
    left_commanded = np.asarray(harness.robots["left"].move_js_calls[-1])
    assert float(np.max(np.abs(left_commanded - pre["left"]))) > 0.02


def _failing_camera_harness() -> tuple[Harness, CameraFactory]:
    """A harness whose camera factory blows up on the second camera (right_rgbd)."""
    harness = Harness()
    working = harness.embodiment._camera_factory

    def failing(name: str) -> FakeCamera:
        if name == "right_rgbd":
            raise RuntimeError(f"camera {name} is busy")
        return cast(FakeCamera, working(name))

    harness.embodiment._camera_factory = failing
    return harness, working


def test_partial_bring_up_failure_tears_down_and_raises_fault() -> None:
    harness, _working = _failing_camera_harness()
    with pytest.raises(EmbodimentFault, match="bring-up failed"):
        harness.embodiment.reset(Scene(id="s0", instruction="x"))
    for robot in harness.robots.values():
        assert not robot.enabled
        assert robot.disconnected
    assert harness.cameras["left_rgbd"].started and harness.cameras["left_rgbd"].stopped
    # Bring-up is incremental: the third camera was never built or started.
    assert not harness.cameras["chest_rgbd"].started


def test_bring_up_retry_after_failure_reenables_the_robots() -> None:
    harness, working = _failing_camera_harness()
    with pytest.raises(EmbodimentFault, match="bring-up failed"):
        harness.embodiment.reset(Scene(id="s0", instruction="x"))
    harness.embodiment._camera_factory = working  # restore the working factory
    observation = harness.embodiment.reset(Scene(id="s0", instruction="x"))
    for robot in harness.robots.values():
        assert robot.enabled and robot.connected
    assert observation.state["joint_pos"].shape == (16,)


def test_conformance_still_passes_when_wired() -> None:
    assert_embodiment_conformant(Harness().embodiment.info)
