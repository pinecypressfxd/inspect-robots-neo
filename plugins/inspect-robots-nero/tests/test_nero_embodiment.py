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
from inspect_robots_nero.embodiment import (
    CameraFactory,
    _older_joint_sample,
    _past_joint_sample,
    sensor_alignment_fault,
)

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


class _ScriptedClock:
    def __init__(self) -> None:
        self.now = 0.0

    def peek(self) -> float:
        return self.now

    def __call__(self) -> float:
        self.now += 1.0 / 30.0
        return self.now


class FakeCamera:
    def __init__(self, clock: _ScriptedClock) -> None:
        self.started = False
        self.stopped = False
        self.frame = _FRAME
        self._clock = clock
        self.stamp_offset_s = 0.0

    def start(self) -> None:
        self.started = True

    def read(self, *, max_age_s: float) -> tuple[np.ndarray, float]:
        del max_age_s
        return self.frame, self._clock.peek() + self.stamp_offset_s

    def stop(self) -> None:
        self.stopped = True


class Harness:
    def __init__(
        self, *, operator_reset_confirm: bool = False, trace_dir: str | None = None
    ) -> None:
        self.robots = {"left": FakeAgxRobot(), "right": FakeAgxRobot()}
        self.clock = _ScriptedClock()
        self.cameras = {
            name: FakeCamera(self.clock) for name in ("left_rgbd", "right_rgbd", "chest_rgbd")
        }
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
            clock=self.clock,
            sleep=self.sleeps.append,
            trace_dir=trace_dir,
        )
        self.embodiment = embodiment


def test_reset_enables_homes_and_returns_first_observation() -> None:
    harness = Harness()
    observation = harness.embodiment.reset(Scene(id="s0", instruction="put the cup on the pad"))
    for robot in harness.robots.values():
        assert robot.enabled and robot.speed == 20 and robot.mode == "follower"
        home = HOME_LEFT if robot is harness.robots["left"] else HOME_RIGHT
        # Seat with move_j, enter follower, then the home trajectory.
        # No move_js here: that API has no smoothing and twitches a settled arm.
        assert robot.move_j_calls[0] == (0.1, 0.2, 0.0, 0.5, 0.0, 0.0, 0.3)
        assert robot.move_j_calls[-1] == tuple(home)
        assert robot.move_js_calls == []
        events = [event[0] for event in robot.events]
        move_j_at = [index for index, kind in enumerate(events) if kind == "move_j"]
        assert "set_follower_mode" not in events[: move_j_at[0]]
        assert "set_speed_percent" not in events[: move_j_at[0]]
        assert move_j_at[0] < events.index("set_follower_mode") < move_j_at[-1]
    for camera in harness.cameras.values():
        assert camera.started and not camera.stopped
    assert observation.instruction == "put the cup on the pad"
    assert set(observation.images) == {"left_rgbd", "right_rgbd", "chest_rgbd"}
    assert observation.state["eef_state"].shape == (20,)
    assert observation.state["joint_pos"].shape == (16,)
    gripper_dims = (observation.state["joint_pos"][14], observation.state["joint_pos"][15])
    assert gripper_dims == (0.09, 0.09)  # the fake effector reports its 0.09 init width


def test_observation_carries_measured_proprio_history() -> None:
    """The aligned pose is the history anchor, and 200 Hz reads append before it."""
    harness = Harness()
    observation = harness.embodiment.reset(Scene(id="s0", instruction="x"))
    history = observation.extra["proprio_history"]
    anchor = observation.extra["proprio_time_ns"]
    assert history[-1][0] == anchor
    assert np.allclose(history[-1][1], observation.state["eef_state"])

    stored = len(harness.embodiment._proprio)
    harness.embodiment.control_hz = 200.0
    harness.embodiment._control_pump = harness.embodiment._pump_control_sync
    stay = np.asarray(observation.state["eef_state"], dtype=np.float64).copy()
    stepped = harness.embodiment.step(Action(data=stay))
    later = stepped.observation.extra["proprio_history"]
    assert len(harness.embodiment._proprio) > stored
    assert later[-1][0] == stepped.observation.extra["proprio_time_ns"]
    assert np.allclose(later[-1][1], stepped.observation.state["eef_state"])
    stamps = [stamp for stamp, _pose in later]
    assert stamps == sorted(set(stamps))


def test_past_joint_sample_stays_at_or_before_the_image() -> None:
    """A reading newer than the picture is not the pose sent with that picture."""
    joints = np.zeros(14)
    image_ns = 1_000_000_000
    samples = (
        (image_ns - 20_000_000, joints, 0.09, 0.09),
        (image_ns + 5_000_000, joints + 1.0, 0.05, 0.09),
    )
    chosen = _past_joint_sample(samples, image_ns)
    assert chosen is not None
    assert chosen[0] == image_ns - 20_000_000
    stale = ((image_ns - 33_333_334, joints, 0.09, 0.09),)
    assert _past_joint_sample(stale, image_ns) is None


def test_older_joint_sample_matches_the_ros_window() -> None:
    """100 ms earlier, within 33 ms. A larger gap is refused."""
    joints = np.zeros(14)
    anchor = 300_000_000
    older = anchor - 99_999_999
    samples = (
        (older - 20_000_000, joints, 0.09, 0.09),
        (older + 10_000_000, joints, 0.08, 0.09),
        (anchor, joints, 0.07, 0.09),
    )
    chosen = _older_joint_sample(samples, anchor)
    assert chosen is not None
    assert chosen[0] == older - 20_000_000
    gapped = (
        (anchor - 99_999_999 - 33_333_334, joints, 0.09, 0.09),
        (anchor, joints, 0.09, 0.09),
    )
    assert _older_joint_sample(gapped, anchor) is None


def test_reset_confirm_prompts_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return ""

    monkeypatch.setattr("builtins.input", fake_input)
    harness = Harness(operator_reset_confirm=True)
    harness.embodiment.reset(Scene(id="s0", instruction="x"))
    assert prompts and "Arrange the scene" in prompts[0]


def test_connected_session_owns_the_reset_confirm(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    inputs: list[str] = []

    def no_input(prompt: str) -> str:
        inputs.append(prompt)
        return ""

    monkeypatch.setattr("builtins.input", no_input)

    class FakeSession:
        def __init__(self) -> None:
            self.gates: list[str] = []
            self.lines: list[str] = []

        def status(self, line: str | None) -> None:
            pass

        def write_line(self, text: str) -> None:
            self.lines.append(text)

        def gate(self, prompt: str, *, hint: str | None = None) -> None:
            self.gates.append(prompt)

    session = FakeSession()
    harness = Harness(operator_reset_confirm=True)
    harness.embodiment.connect_operator_session(session)
    harness.embodiment.reset(Scene(id="s0", instruction="put the cup down"))
    assert not inputs  # the session owns stdin; the embodiment never reads it
    assert session.gates == ["Arrange the scene, instruction: put the cup down"]
    assert any("put the cup down" in line for line in session.lines)
    out = capsys.readouterr().out
    assert "Arrange the scene" not in out
    assert out.startswith("[timing] connect+home ")


def test_step_commands_bounded_joint_deltas_and_gripper_widths() -> None:
    harness = Harness()
    harness.embodiment.reset(Scene(id="s0", instruction="x"))
    action = _home_action(GRIPPER_INIT_WIDTH_M, 0.02)
    result = harness.embodiment.step(Action(data=action))
    for robot in harness.robots.values():
        assert len(robot.move_js_calls) == 1
    assert result.observation.images["left_rgbd"].shape == (480, 640, 3)


def test_trace_dir_writes_the_commanded_sample(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))
    harness = Harness(trace_dir=str(root))
    harness.embodiment.reset(Scene(id="s0", instruction="x"))
    harness.embodiment.step(Action(data=_home_action(GRIPPER_INIT_WIDTH_M, GRIPPER_INIT_WIDTH_M)))
    harness.embodiment.close()
    episode = next(path for path in root.iterdir() if path.is_dir())
    assert (root / "current").read_text(encoding="utf-8").strip() == episode.name
    line = (episode / "motion.jsonl").read_text(encoding="utf-8").strip()
    assert '"held":false' in line
    assert len(line) > 50


def test_sensor_alignment_matches_the_ros_windows() -> None:
    times = {"left_rgbd": 1.00, "right_rgbd": 1.02, "chest_rgbd": 1.03}
    assert (
        sensor_alignment_fault(
            now=1.05,
            camera_times=times,
            joint_time=1.01,
            skew_s=0.040,
            lag_s=0.033,
            stale_s=0.100,
        )
        is None
    )
    assert (
        sensor_alignment_fault(
            now=1.05,
            camera_times={**times, "chest_rgbd": 1.05},
            joint_time=1.01,
            skew_s=0.040,
            lag_s=0.033,
            stale_s=0.100,
        )
        == "triplet_skew"
    )
    assert (
        sensor_alignment_fault(
            now=1.20,
            camera_times=times,
            joint_time=1.01,
            skew_s=0.040,
            lag_s=0.033,
            stale_s=0.100,
        )
        == "stale"
    )
    assert (
        sensor_alignment_fault(
            now=1.05,
            camera_times=times,
            joint_time=0.90,
            skew_s=0.040,
            lag_s=0.033,
            stale_s=0.100,
        )
        == "arm_lag"
    )


def test_skewed_cameras_fault_the_observation() -> None:
    harness = Harness()
    harness.cameras["chest_rgbd"].stamp_offset_s = 0.050

    def advancing(delay_s: float) -> None:
        harness.sleeps.append(delay_s)
        harness.clock.now += delay_s

    harness.embodiment._sleep = advancing
    with pytest.raises(EmbodimentFault, match="triplet_skew"):
        harness.embodiment.reset(Scene(id="s0", instruction="x"))


def test_unreachable_target_holds_and_keeps_the_rollout_alive(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An IK-impossible pose holds the arms instead of faulting the eval.

    The planner commands over-ambitious targets as a matter of course; the
    failure is caught before anything is commanded, so the step holds the
    joints, warns once per streak, and the next reachable action executes
    normally -- the planner sees the unmoved state in between and replans.
    """
    harness = Harness()
    harness.embodiment.reset(Scene(id="s0", instruction="x"))
    before = {side: list(robot.move_js_calls) for side, robot in harness.robots.items()}

    unreachable = _home_action(0.09, 0.09)
    unreachable[0] += 5.0  # 5 m out of the workspace: no seed can converge
    result = harness.embodiment.step(Action(data=unreachable))
    unreachable_again = _home_action(0.09, 0.09)
    unreachable_again[0] += 5.0
    harness.embodiment.step(Action(data=unreachable_again))

    assert result.observation.state["eef_state"].shape == (20,)
    for side, robot in harness.robots.items():
        # Rejected before any move_js: the move_j hold stays, nothing is streamed.
        assert robot.move_js_calls == before[side]
    assert capsys.readouterr().err.count("holding position") == 1  # once per streak

    harness.embodiment.step(Action(data=_home_action(0.05, 0.05)))
    for robot in harness.robots.values():
        assert len(robot.move_js_calls) == len(before["left"]) + 1


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
    harness.embodiment._disable_on_close = True
    harness.embodiment.reset(Scene(id="s0", instruction="x"))
    harness.embodiment.close()
    for robot in harness.robots.values():
        assert not robot.enabled
    for camera in harness.cameras.values():
        assert camera.stopped


def test_step_holds_when_the_joint_jump_exceeds_the_policy_limit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A 0.15 m shift needs ~0.54 rad. Agent (0.1) holds; vla (1.0) commands it."""
    harness = Harness()
    harness.embodiment.reset(Scene(id="s0", instruction="x"))
    before = {side: list(robot.move_js_calls) for side, robot in harness.robots.items()}
    action = _home_action(GRIPPER_INIT_WIDTH_M, GRIPPER_INIT_WIDTH_M)
    action[0] += 0.15
    harness.embodiment.step(Action(data=action))
    for side, robot in harness.robots.items():
        assert robot.move_js_calls == before[side]
    assert "holding position" in capsys.readouterr().err

    wide = Harness()
    wide.embodiment.max_joint_step_rad = 1.0
    wide.embodiment.reset(Scene(id="s0", instruction="x"))
    pre = {side: np.asarray(robot.angles) for side, robot in wide.robots.items()}
    wide.embodiment.step(Action(data=action))
    left_commanded = np.asarray(wide.robots["left"].move_js_calls[-1])
    delta = float(np.max(np.abs(left_commanded - pre["left"])))
    assert 0.1 < delta <= 1.0 + 1e-9


def test_one_centimeter_lift_is_commanded_at_the_agent_joint_limit() -> None:
    """A 1 cm raise needs slightly more than one 0.1 rad IK iteration.

    Harness move_to splits motion into 1 cm ticks. Those solutions were
    success=True at about 0.11-0.13 rad, and the old 0.1 hold froze the arm
    for the rest of the chunk.
    """
    harness = Harness()
    harness.embodiment.reset(Scene(id="s0", instruction="x"))
    action = np.asarray(harness.embodiment._last_ee, dtype=np.float64).copy()
    action[2] += 0.01
    harness.embodiment.step(Action(data=action))
    assert harness.robots["left"].move_js_calls


def test_small_orientation_offset_stays_near_home() -> None:
    """A few milliradians of orientation must not drag the arms toward URDF neutral."""
    from scipy.spatial.transform import Rotation

    from inspect_robots_nero._kinematics import rot6d_to_matrix

    harness = Harness()
    harness.embodiment.reset(Scene(id="s0", instruction="x"))
    action = np.asarray(harness.embodiment._last_ee, dtype=np.float64).copy()
    rotation = rot6d_to_matrix(action[3:9])
    turned = Rotation.from_euler("xyz", [0.0, 0.002, 0.0]).as_matrix() @ rotation
    action[3:9] = matrix_to_rot6d(turned)
    home = np.asarray(harness.robots["left"].angles)
    harness.embodiment.step(Action(data=action))
    assert harness.robots["left"].move_js_calls
    delta = float(np.max(np.abs(np.asarray(harness.robots["left"].move_js_calls[-1]) - home)))
    assert delta < 0.05


def test_vla_control_hz_splits_one_30hz_target_across_200hz() -> None:
    """vla passes control_hz=200: one policy tick solves seven times, 5 ms apart."""
    harness = Harness()
    harness.embodiment.control_hz = 200.0
    harness.embodiment.max_joint_step_rad = 1.0
    # Wire the synchronous pump so the 200 Hz substeps run inline (the
    # background thread stays off in tests).
    harness.embodiment._control_pump = harness.embodiment._pump_control_sync
    harness.embodiment.reset(Scene(id="s0", instruction="x"))
    home = np.asarray(harness.robots["left"].angles)
    stay = np.asarray(harness.embodiment._last_ee, dtype=np.float64).copy()
    slept = len(harness.sleeps)
    harness.embodiment.step(Action(data=stay))
    calls = list(harness.robots["left"].move_js_calls)
    assert len(calls) == 7  # round(200 / 30)
    assert harness.sleeps[slept:].count(1.0 / 200.0) == 6
    assert float(np.max(np.abs(np.asarray(calls[-1]) - home))) < 0.05

    seen: list[float] = []

    def sample(dt: float) -> np.ndarray:
        assert dt == pytest.approx(1.0 / 200.0)
        pose = stay.copy()
        pose[0] += 0.04 * (len(seen) + 1)
        seen.append(float(pose[0]))
        return pose

    # Each 5 ms tick reads the trajectory sampler. It does not chord two poses.
    harness.embodiment.step(Action(data=stay, meta={"trajectory_sample": sample}))
    assert len(seen) == 7
    moved = harness.robots["left"].move_js_calls[len(calls) :]
    assert len(moved) == 7
    first = float(np.max(np.abs(np.asarray(moved[0]) - home)))
    last = float(np.max(np.abs(np.asarray(moved[-1]) - home)))
    assert first > 0.02
    assert first < last


def test_non_finite_joint_feedback_raises_embodiment_fault() -> None:
    harness = Harness()
    harness.embodiment.reset(Scene(id="s0", instruction="x"))
    harness.robots["left"].angles = (float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    action = _home_action(GRIPPER_INIT_WIDTH_M, GRIPPER_INIT_WIDTH_M)
    with pytest.raises(EmbodimentFault, match="non-finite"):
        harness.embodiment.step(Action(data=action))


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
        assert robot.enabled  # a failed bring-up must not add a gravity fall
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


def test_close_leaves_arms_enabled_by_default() -> None:
    harness = Harness()
    harness.embodiment.reset(Scene(id="s0", instruction="x"))
    harness.embodiment.close()
    for robot in harness.robots.values():
        assert robot.enabled  # no brakes: automatic disable is a gravity fall
        assert robot.disconnected
    for camera in harness.cameras.values():
        assert camera.stopped


def test_close_disables_only_when_requested() -> None:
    harness = Harness()
    harness.embodiment._disable_on_close = True
    harness.embodiment.reset(Scene(id="s0", instruction="x"))
    harness.embodiment.close()
    for robot in harness.robots.values():
        assert not robot.enabled
