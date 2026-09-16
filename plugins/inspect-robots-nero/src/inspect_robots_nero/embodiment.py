"""Expose the dual Nero CAN arms and D405 cameras through the embodiment contract.

Construction builds only static spaces; arms, grippers, and cameras connect
lazily on the first reset. Actions are 20-dim absolute end-effector poses
(position + rot6d + gripper width per arm); each step solves IK and commands
one bounded joint increment per arm at ``control_hz``.
"""

from __future__ import annotations

import contextlib
import importlib.resources
import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np

from inspect_robots import (
    Action,
    ActionSemantics,
    Box,
    CameraSpec,
    EmbodimentBase,
    EmbodimentInfo,
    Observation,
    ObservationSpace,
    Scene,
    StateField,
    StateSpec,
    StepResult,
)
from inspect_robots.embodiment import RESETTABLE, SELF_PACED
from inspect_robots.errors import EmbodimentFault
from inspect_robots.spaces import CANONICAL_STATE_UNITS
from inspect_robots_nero._config import (
    ACTION_DIM,
    CAMERA_DEFAULTS,
    CAMERA_MAX_AGE_S,
    CONTROL_HZ,
    DEFAULT_MAX_STEP,
    DIM_LABELS,
    DUAL_NERO_DOCS,
    GRIPPER_WIDTH_MAX_M,
    GRIPPER_WIDTH_MIN_M,
    RESET_SETTLE_TIMEOUT_S,
    ROT6D_BOUNDS,
)
from inspect_robots_nero._gripper import NeroGripper
from inspect_robots_nero._kinematics import NeroKinematicsError, matrix_to_rot6d, rot6d_to_matrix

ArmFactory = Callable[[str], Any]
CameraFactory = Callable[[str], Any]


def _parse_camera_overrides(value: str) -> dict[str, str]:
    """Parse the ``-E cameras=`` string form: ``name=device`` comma entries.

    Mirrors the ros adapter's ``_parse_cameras`` string handling: entries split
    on the first ``=`` so device paths need no quoting, blank entries are
    skipped, and a non-blank string that yields no entries is an error. Device
    emptiness is validated by the caller so both input forms share one error.
    """
    overrides: dict[str, str] = {}
    for raw_entry in value.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        name, separator, device = entry.partition("=")
        name = name.strip()
        if not separator or not name:
            raise ValueError(
                f"cameras entry {entry!r} must be name=device, for example left_rgbd=/dev/video0"
            )
        if name in overrides:
            raise ValueError(
                f"cameras contains duplicate name {name!r}; camera names must be unique"
            )
        overrides[name] = device.strip()
    if value.strip() and not overrides:
        raise ValueError(f"cameras string form parsed to no overrides: {value!r}")
    return overrides


def _bounds_for(
    workspace_low: Sequence[float], workspace_high: Sequence[float]
) -> tuple[list[float], list[float]]:
    rot_low, rot_high = ROT6D_BOUNDS
    return (
        [*workspace_low, *[rot_low] * 6, GRIPPER_WIDTH_MIN_M],
        [*workspace_high, *[rot_high] * 6, GRIPPER_WIDTH_MAX_M],
    )


class NeroEmbodiment(EmbodimentBase):
    """Drive the dual Nero arms with absolute end-effector pose actions.

    ``cameras`` accepts a mapping of camera name to device path (programmatic
    use) or the ``-E`` string form ``name=device`` with comma-separated
    entries; both override only the named devices and leave the other
    ``CAMERA_DEFAULTS`` entries untouched.
    """

    def __init__(
        self,
        *,
        control_hz: float = CONTROL_HZ,
        workspace_low: tuple[float, float, float] | None = None,
        workspace_high: tuple[float, float, float] | None = None,
        max_step: tuple[float | None, ...] | None = None,
        cameras: Mapping[str, str] | str | None = None,
        operator_reset_confirm: bool = True,
        reset_settle_timeout_s: float = RESET_SETTLE_TIMEOUT_S,
        camera_max_age_s: float = CAMERA_MAX_AGE_S,
        arm_factory: ArmFactory | None = None,
        camera_factory: CameraFactory | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        name: str = "nero",
    ) -> None:
        if not math.isfinite(control_hz) or control_hz <= 0:
            raise ValueError(
                f"control_hz must be positive and finite, got {control_hz!r}; pass -E control_hz=30"
            )
        if reset_settle_timeout_s <= 0 or not math.isfinite(reset_settle_timeout_s):
            raise ValueError(
                f"reset_settle_timeout_s must be positive and finite, "
                f"got {reset_settle_timeout_s!r}"
            )
        if camera_max_age_s <= 0 or not math.isfinite(camera_max_age_s):
            raise ValueError(
                f"camera_max_age_s must be positive and finite, got {camera_max_age_s!r}"
            )
        resolved_max_step = DEFAULT_MAX_STEP if max_step is None else tuple(max_step)
        if len(resolved_max_step) != ACTION_DIM or any(
            entry is not None and (not math.isfinite(entry) or entry <= 0)
            for entry in resolved_max_step
        ):
            raise ValueError(f"max_step must hold {ACTION_DIM} finite positive entries (or None)")

        # Default workspace bounds come from FK sampling over the packaged URDF;
        # explicit workspace_low/workspace_high replace them for both arms alike.
        sampled: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {}
        if workspace_low is None or workspace_high is None:
            from inspect_robots_nero._kinematics import NeroKinematics

            urdf_path = (
                importlib.resources.files("inspect_robots_nero") / "assets" / "dual_nero_pika.urdf"
            )
            sampled = NeroKinematics(str(urdf_path)).sample_workspace_bounds()
        workspace: dict[str, tuple[list[float], list[float]]] = {}
        for side in ("left", "right"):
            low = list(workspace_low) if workspace_low is not None else list(sampled[side][0])
            high = list(workspace_high) if workspace_high is not None else list(sampled[side][1])
            if len(low) != 3 or len(high) != 3 or not all(math.isfinite(v) for v in (*low, *high)):
                raise ValueError(
                    f"workspace_low/workspace_high must hold three finite values per axis; "
                    f"got {low} / {high}"
                )
            if any(lo > hi for lo, hi in zip(low, high, strict=True)):
                raise ValueError(
                    f"workspace bounds must be elementwise low <= high for the {side} arm"
                )
            workspace[side] = (low, high)

        resolved_cameras = dict(CAMERA_DEFAULTS)
        if cameras is not None:
            overrides: Mapping[str, str] = (
                _parse_camera_overrides(cameras) if isinstance(cameras, str) else cameras
            )
            unknown = sorted(set(overrides) - set(CAMERA_DEFAULTS))
            if unknown:
                raise ValueError(
                    f"cameras keys {unknown} are not declared cameras; "
                    f"valid names: {sorted(CAMERA_DEFAULTS)}"
                )
            for camera_name, device in overrides.items():
                if not isinstance(device, str) or not device:
                    raise ValueError(
                        f"camera {camera_name!r} override must be a non-empty device path"
                    )
                resolved_cameras[camera_name] = {**CAMERA_DEFAULTS[camera_name], "device": device}

        low_array = np.asarray(
            _bounds_for(*workspace["left"])[0] + _bounds_for(*workspace["right"])[0],
            dtype=np.float64,
        )
        high_array = np.asarray(
            _bounds_for(*workspace["left"])[1] + _bounds_for(*workspace["right"])[1],
            dtype=np.float64,
        )
        semantics = ActionSemantics(
            control_mode="eef_abs_pose",
            rotation_repr="rot6d",
            gripper="continuous",
            frame="base",
            dim_labels=DIM_LABELS,
            max_step=resolved_max_step,
        )
        self.info = EmbodimentInfo(
            name=name,
            action_space=Box(
                shape=(ACTION_DIM,), low=low_array, high=high_array, semantics=semantics
            ),
            observation_space=ObservationSpace(
                cameras=tuple(
                    CameraSpec(camera_name, cast(int, spec["height"]), cast(int, spec["width"]))
                    for camera_name, spec in resolved_cameras.items()
                ),
                state=StateSpec(
                    fields=(
                        StateField("eef_state", (ACTION_DIM,), CANONICAL_STATE_UNITS["eef_pose"]),
                        StateField("joint_pos", (16,), CANONICAL_STATE_UNITS["joint_pos"]),
                    )
                ),
            ),
            control_hz=control_hz,
            is_simulated=False,
            capabilities=frozenset({SELF_PACED, RESETTABLE}),
            supported_setups=frozenset(),
            supported_target_kinds=frozenset(),
            docs=DUAL_NERO_DOCS,
        )
        self.control_hz = float(control_hz)
        self.cameras = resolved_cameras
        self.operator_reset_confirm = operator_reset_confirm
        self.reset_settle_timeout_s = float(reset_settle_timeout_s)
        self.camera_max_age_s = float(camera_max_age_s)
        self._clock = clock
        self._sleep = sleep
        self._instruction: str | None = None
        self._operator_session: Any | None = None
        self._arm_factory = arm_factory or self._default_arm_factory
        self._camera_factory = camera_factory or self._default_camera_factory
        self._arms: dict[str, Any] = {}
        self._grippers: dict[str, Any] = {}
        self._cameras: dict[str, Any] = {}
        self._kinematics: Any | None = None
        self._last_commanded: dict[str, np.ndarray] = {}
        self._last_step_time: float | None = None
        self._max_joint_step = 0.1
        self._connected = False

    def _default_arm_factory(self, side: str) -> Any:
        from inspect_robots_nero._arm import NeroArm
        from inspect_robots_nero._config import CAN_CHANNELS, FIRMWARE_VERSION

        arm = NeroArm(side, CAN_CHANNELS[side], firmware=FIRMWARE_VERSION)
        arm.connect()
        return arm

    def _default_camera_factory(self, name: str) -> Any:
        from inspect_robots_nero._camera import D405Camera

        spec = self.cameras[name]
        camera = D405Camera(
            name,
            str(spec["device"]),
            width=cast(int, spec["width"]),
            height=cast(int, spec["height"]),
            fps=cast(int, spec["fps"]),
            pixel_format=str(spec["pixel_format"]),
            clock=self._clock,
        )
        camera.start()
        return camera

    def _ensure_connected(self) -> None:
        if self._connected:
            return
        from inspect_robots_nero._config import (
            GRIPPER_FORCE,
            GRIPPER_INIT_WIDTH_M,
            HOME_LEFT,
            HOME_RIGHT,
            SPEED_PERCENT,
        )
        from inspect_robots_nero._kinematics import NeroKinematics

        # Build incrementally, storing each object as it is created: a factory
        # that aborts mid-bring-up still leaves the built half reachable for
        # teardown, so a retry never stacks a second arm pair on the CAN bus.
        try:
            for side in ("left", "right"):
                self._arms[side] = self._arm_factory(side)
            for arm in self._arms.values():
                arm.enable()
                arm.set_speed_percent(SPEED_PERCENT)
                arm.set_normal_mode()
            for side in ("left", "right"):
                gripper = NeroGripper(self._arms[side], sleep=self._sleep)
                self._grippers[side] = gripper
                gripper.start()
                gripper.move(GRIPPER_INIT_WIDTH_M, GRIPPER_FORCE)
            for name in self.cameras:
                self._cameras[name] = self._camera_factory(name)
            urdf = (
                importlib.resources.files("inspect_robots_nero") / "assets" / "dual_nero_pika.urdf"
            )
            self._kinematics = NeroKinematics(str(urdf))
            self._last_commanded = {
                "left": np.asarray(HOME_LEFT),
                "right": np.asarray(HOME_RIGHT),
            }
        except Exception as exc:
            self._abort_bring_up()
            raise EmbodimentFault(f"nero bring-up failed: {exc}") from exc
        self._connected = True

    def _abort_bring_up(self) -> None:
        """Tear down a partially built hardware set after a failed bring-up."""
        for camera in self._cameras.values():
            with contextlib.suppress(Exception):
                camera.stop()
        for gripper in self._grippers.values():
            with contextlib.suppress(Exception):
                gripper.stop()
        for arm in self._arms.values():
            with contextlib.suppress(Exception):
                arm.disable()
                arm.close()
        self._arms = {}
        self._grippers = {}
        self._cameras = {}

    def _drive_home(self) -> None:
        from inspect_robots_nero._config import HOME_LEFT, HOME_RIGHT, RESET_SETTLE_TOL_RAD

        homes = {"left": np.asarray(HOME_LEFT), "right": np.asarray(HOME_RIGHT)}
        for side, home in homes.items():
            self._arms[side].move_js(home.tolist())
        deadline = self._clock() + self.reset_settle_timeout_s
        while self._clock() < deadline:
            settled = True
            for side, home in homes.items():
                reading = self._arms[side].read_state()
                if reading is None or len(reading) != 7:
                    settled = False
                    break
                if float(np.max(np.abs(np.asarray(reading) - home))) > RESET_SETTLE_TOL_RAD:
                    settled = False
                    break
            if settled:
                self._sleep(1.0 / self.control_hz)
                return
            self._sleep(0.02)
        raise EmbodimentFault(
            "arms did not settle at home within "
            f"reset_settle_timeout_s={self.reset_settle_timeout_s:g}s"
        )

    @staticmethod
    def _target_pose(xyz: np.ndarray, rot6d: np.ndarray) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = rot6d_to_matrix(rot6d)
        pose[:3, 3] = xyz
        return pose

    def _read_joints(self) -> np.ndarray:
        readings: list[float] = []
        for side in ("left", "right"):
            reading = self._arms[side].read_state()
            if reading is None or len(reading) != 7:
                raise EmbodimentFault(f"no joint feedback from the {side} arm; check the CAN link")
            if not all(math.isfinite(value) for value in reading):
                raise EmbodimentFault(
                    f"non-finite joint feedback from the {side} arm; check the CAN link"
                )
            readings.extend(float(value) for value in reading)
        return np.asarray(readings)

    def _pace(self) -> None:
        if self._last_step_time is not None:
            # ">=" not ">": an exact zero sleep honors the tick when the clock lands
            # exactly on the boundary (the scripted test clock always does).
            remaining = self._last_step_time + (1.0 / self.control_hz) - self._clock()
            if remaining >= 0:
                self._sleep(remaining)
        self._last_step_time = self._clock()

    def connect_operator_session(self, session: Any) -> None:
        """Stand down from stdin ownership: the framework console owns it for this run.

        After this call the embodiment never reads stdin or prints its own
        output; the reset confirmation routes through the session's gate. The
        hook never fires under --no-prompt, without a TTY, or from direct
        rollout()/eval() calls, so reset keeps the input() fallback.
        """
        self._operator_session = session

    def _confirm_operator_reset(self, instruction: str) -> None:
        """Block until the operator confirms the arranged scene."""
        if self._operator_session is not None:
            self._operator_session.write_line(
                f"Operator reset required for instruction: {instruction}"
            )
            self._operator_session.gate(f"Arrange the scene, instruction: {instruction}")
            return
        print(f"Operator reset required for instruction: {instruction}")
        input("Arrange the scene, then press Enter to continue: ")

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Connect lazily, settle both arms at home, and return the first observation."""
        del seed
        self._instruction = scene.instruction
        self._ensure_connected()
        if self.operator_reset_confirm:
            self._confirm_operator_reset(scene.instruction)
        self._drive_home()
        # Reset counts as the previous control tick so the first step paces itself.
        self._last_step_time = self._clock()
        return self._assemble_observation()

    def step(self, action: Action) -> StepResult:
        """Solve IK for one 20-dim EE target and command one bounded joint increment."""
        data = np.asarray(action.data, dtype=np.float64).ravel()
        if data.shape != (20,):
            raise ValueError(
                f"action has shape {data.shape}, expected (20,); the nero action vector is "
                "[left xyz, left rot6d, left width, right xyz, right rot6d, right width]"
            )
        from inspect_robots_nero._config import GRIPPER_FORCE

        self._pace()
        assert self._kinematics is not None
        left_target = self._target_pose(data[0:3], data[3:9])
        right_target = self._target_pose(data[10:13], data[13:19])
        joints = self._read_joints()
        q_seed = self._kinematics.q_from(joints[:7], joints[7:14])
        try:
            q_solution = self._kinematics.solve(left_target, right_target, q_seed)
        except NeroKinematicsError as exc:
            raise EmbodimentFault(f"nero IK failed: {exc}") from exc
        left_current, right_current = joints[:7], joints[7:14]
        left_command = left_current + np.clip(
            q_solution[:7] - left_current, -self._max_joint_step, self._max_joint_step
        )
        right_command = right_current + np.clip(
            q_solution[7:] - right_current, -self._max_joint_step, self._max_joint_step
        )
        self._arms["left"].move_js(left_command.tolist())
        self._arms["right"].move_js(right_command.tolist())
        self._last_commanded = {"left": left_command, "right": right_command}
        self._grippers["left"].move(float(data[9]), GRIPPER_FORCE)
        self._grippers["right"].move(float(data[19]), GRIPPER_FORCE)
        return StepResult(observation=self._assemble_observation())

    def _assemble_observation(self) -> Observation:
        assert self._kinematics is not None
        joints = self._read_joints()
        poses = self._kinematics.fk(self._kinematics.q_from(joints[:7], joints[7:14]))
        widths = [self._grippers[side].read_state() for side in ("left", "right")]
        left_width = widths[0] if widths[0] is not None else 0.0
        right_width = widths[1] if widths[1] is not None else 0.0
        eef_state = np.concatenate(
            [
                poses["left"][:3, 3],
                matrix_to_rot6d(poses["left"][:3, :3]),
                [left_width],
                poses["right"][:3, 3],
                matrix_to_rot6d(poses["right"][:3, :3]),
                [right_width],
            ]
        )
        joint_pos = np.concatenate([joints, [left_width, right_width]])
        images: dict[str, np.ndarray] = {}
        image_times: dict[str, float] = {}
        for name, camera in self._cameras.items():
            frame, stamp = camera.read(max_age_s=self.camera_max_age_s)
            images[name] = frame
            image_times[name] = stamp
        return Observation(
            images=images,
            state={"eef_state": eef_state, "joint_pos": joint_pos},
            instruction=self._instruction,
            image_times=image_times,
            state_time=min(image_times.values()),
        )

    def close(self) -> None:
        """Stop cameras, detach grippers, and disable both arms (best effort each)."""
        for camera in self._cameras.values():
            with contextlib.suppress(Exception):
                camera.stop()
        for gripper in self._grippers.values():
            with contextlib.suppress(Exception):
                gripper.stop()
        for arm in self._arms.values():
            with contextlib.suppress(Exception):
                arm.disable()
                arm.close()
        self._connected = False
