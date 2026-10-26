"""Expose the dual Nero CAN arms and D405 cameras through the embodiment contract.

Construction builds only static spaces; arms, grippers, and cameras connect
lazily on the first reset. Actions are 20-dim absolute end-effector poses
(position + rot6d + gripper width per arm). Each policy tick is split across
``control_hz`` (200 Hz for vla) and ProxQP IK commands the joints when the jump
stays inside ``max_joint_step_rad``.
"""

from __future__ import annotations

import contextlib
import importlib.resources
import math
import sys
import threading
import time
from bisect import bisect_left
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

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
from inspect_robots_nero._arm import clamp_to_joint_envelope
from inspect_robots_nero._config import (
    ACTION_DIM,
    ACTION_HZ,
    ALIGNMENT_FAILURE_TIMEOUT_S,
    CAMERA_DEFAULTS,
    CAMERA_MAX_AGE_S,
    CONTROL_HZ,
    DEFAULT_MAX_STEP,
    DIM_LABELS,
    DISABLE_ON_CLOSE,
    DUAL_NERO_DOCS,
    GRIPPER_WIDTH_MAX_M,
    GRIPPER_WIDTH_MIN_M,
    IK_DAMPING_COST,
    IK_FRAME_LM_DAMPING,
    IK_FRAME_ORIENTATION_COST,
    IK_FRAME_POSITION_COST,
    IK_MAX_ITERATIONS,
    IK_MAX_JOINT_STEP_RAD,
    IK_POSTURE_COST,
    IK_TCP_ROTATION_RPY_RAD,
    IK_TCP_TRANSLATION_M,
    MAX_ARM_STATE_LAG_S,
    MAX_IMAGE_TRIPLET_SKEW_S,
    MAX_PROPRIOCEPTION_PAIR_SKEW_S,
    RESET_SETTLE_TIMEOUT_S,
    ROT6D_BOUNDS,
    WARMUP_MIN_ALIGNED_GROUPS,
    WARMUP_MIN_FRAMES,
)
from inspect_robots_nero._gripper import NeroGripper
from inspect_robots_nero._kinematics import matrix_to_rot6d, rot6d_to_matrix

ArmFactory = Callable[[str], Any]
CameraFactory = Callable[[str], Any]
# 3 action steps at 30 Hz, and the one-step tolerance, in nanoseconds.
_HISTORY_INTERVAL_NS = 99_999_999
_HISTORY_TOLERANCE_NS = 33_333_333
_JointSample = tuple[int, np.ndarray, float, float]


def _image_observation_ns(image_times: Mapping[str, float]) -> int:
    """ROS camera-group time: the midpoint of the oldest and newest stamp."""
    stamps = tuple(round(float(stamp) * 1e9) for stamp in image_times.values())
    return (min(stamps) + max(stamps)) // 2


def _past_joint_sample(
    samples: Sequence[_JointSample],
    observation_ns: int,
    *,
    tolerance_ns: int = _HISTORY_TOLERANCE_NS,
) -> _JointSample | None:
    """Latest joint reading at or before the image time, within one action step.

    ROS rejects the frame when the arm sample is newer than the picture or
    more than ``max_arm_state_lag_ms`` behind it. Pairing a fresh pose with
    an older wrist image makes the grasp overshoot.
    """
    covered = tuple(sample for sample in samples if sample[0] <= observation_ns)
    if not covered:
        return None
    chosen = covered[-1]
    if observation_ns - chosen[0] > tolerance_ns:
        return None
    return chosen


def _older_joint_sample(
    samples: Sequence[_JointSample],
    anchor_ns: int,
) -> _JointSample | None:
    """The joint reading 100 ms before ``anchor_ns``, or None when ROS would skip."""
    covered = tuple(sample for sample in samples if sample[0] <= anchor_ns)
    if not covered or covered[-1][0] != anchor_ns:
        return None
    target_ns = anchor_ns - _HISTORY_INTERVAL_NS
    if target_ns < covered[0][0]:
        return None
    timestamps = tuple(sample[0] for sample in covered)
    index = bisect_left(timestamps, target_ns + 1) - 1
    chosen = covered[index]
    if target_ns - chosen[0] > _HISTORY_TOLERANCE_NS or chosen[0] == anchor_ns:
        return None
    return chosen


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


def _interpolate_action(origin: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    """Pose at ``alpha`` along the segment from ``origin`` to ``target``."""
    weight = float(alpha)
    if weight >= 1.0:
        return np.asarray(target, dtype=np.float64).copy()
    blended = np.empty(20, dtype=np.float64)
    for xyz, rot, grip in (
        (slice(0, 3), slice(3, 9), 9),
        (slice(10, 13), slice(13, 19), 19),
    ):
        blended[xyz] = (1.0 - weight) * origin[xyz] + weight * target[xyz]
        start = Rotation.from_matrix(rot6d_to_matrix(origin[rot]))
        end = Rotation.from_matrix(rot6d_to_matrix(target[rot]))
        mixed = Slerp([0.0, 1.0], Rotation.concatenate([start, end]))([weight])[0]
        blended[rot] = matrix_to_rot6d(np.asarray(mixed.as_matrix(), dtype=np.float64))
        blended[grip] = (1.0 - weight) * float(origin[grip]) + weight * float(target[grip])
    return blended


def sensor_alignment_fault(
    *,
    now: float,
    camera_times: Mapping[str, float],
    joint_time: float,
    skew_s: float,
    lag_s: float,
    stale_s: float,
) -> str | None:
    """Return why a camera triplet and arm sample are not usable together.

    A negative age (a stamp slightly ahead of ``now``) counts as fresh. The
    arm lags when the oldest image is newer than the joint sample by more
    than ``lag_s``.
    """
    if not camera_times:
        return "no_cameras"
    newest = max(camera_times.values())
    oldest = min(camera_times.values())
    if newest - oldest > skew_s:
        return "triplet_skew"
    if now - newest > stale_s:
        return "stale"
    if oldest - joint_time > lag_s:
        return "arm_lag"
    return None


def _bounds_for(
    workspace_low: Sequence[float], workspace_high: Sequence[float]
) -> tuple[list[float], list[float]]:
    rot_low, rot_high = ROT6D_BOUNDS
    return (
        [*workspace_low, *[rot_low] * 6, GRIPPER_WIDTH_MIN_M],
        [*workspace_high, *[rot_high] * 6, GRIPPER_WIDTH_MAX_M],
    )


def _ik_sample(result: Any, *, held: bool, jump: float, max_step: float) -> dict[str, Any]:
    """Scalar IK diagnostics the reference report plots, without joint vectors."""
    sample: dict[str, Any] = {
        "joint_jump_rejected": held,
        "joint_jump_max_delta_rad": jump,
        "joint_jump_threshold_rad": max_step,
    }
    for name, value in vars(result).items():
        if isinstance(value, bool):
            sample[name] = value
        elif isinstance(value, int | float) and not isinstance(value, bool):
            sample[name] = float(value)
    return sample


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
        disable_on_close: bool = DISABLE_ON_CLOSE,
        reset_settle_timeout_s: float = RESET_SETTLE_TIMEOUT_S,
        camera_max_age_s: float = CAMERA_MAX_AGE_S,
        max_image_triplet_skew_s: float = MAX_IMAGE_TRIPLET_SKEW_S,
        max_arm_state_lag_s: float = MAX_ARM_STATE_LAG_S,
        max_joint_step_rad: float = IK_MAX_JOINT_STEP_RAD,
        alignment_failure_timeout_s: float = ALIGNMENT_FAILURE_TIMEOUT_S,
        max_proprioception_pair_skew_s: float = MAX_PROPRIOCEPTION_PAIR_SKEW_S,
        arm_factory: ArmFactory | None = None,
        camera_factory: CameraFactory | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        trace_dir: str | None = None,
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
        if max_image_triplet_skew_s <= 0 or not math.isfinite(max_image_triplet_skew_s):
            raise ValueError(
                "max_image_triplet_skew_s must be positive and finite, "
                f"got {max_image_triplet_skew_s!r}"
            )
        if max_arm_state_lag_s <= 0 or not math.isfinite(max_arm_state_lag_s):
            raise ValueError(
                f"max_arm_state_lag_s must be positive and finite, got {max_arm_state_lag_s!r}"
            )
        if max_joint_step_rad <= 0 or not math.isfinite(max_joint_step_rad):
            raise ValueError(
                f"max_joint_step_rad must be positive and finite, got {max_joint_step_rad!r}"
            )
        if alignment_failure_timeout_s <= 0 or not math.isfinite(alignment_failure_timeout_s):
            raise ValueError(
                f"alignment_failure_timeout_s must be positive and finite, "
                f"got {alignment_failure_timeout_s!r}"
            )
        if max_proprioception_pair_skew_s <= 0 or not math.isfinite(max_proprioception_pair_skew_s):
            raise ValueError(
                f"max_proprioception_pair_skew_s must be positive and finite, "
                f"got {max_proprioception_pair_skew_s!r}"
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
        self._disable_on_close = bool(disable_on_close)
        self.reset_settle_timeout_s = float(reset_settle_timeout_s)
        self.camera_max_age_s = float(camera_max_age_s)
        self.max_image_triplet_skew_s = float(max_image_triplet_skew_s)
        self.max_arm_state_lag_s = float(max_arm_state_lag_s)
        self.max_joint_step_rad = float(max_joint_step_rad)
        self.alignment_failure_timeout_s = float(alignment_failure_timeout_s)
        self.max_proprioception_pair_skew_s = float(max_proprioception_pair_skew_s)
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
        self._ik: Any | None = None
        # Warns once per streak of held steps (an unreachable target's chunk
        # can hold for many steps); re-arms on the next solved step.
        self._ik_hold_warned = False
        self._trace_dir = trace_dir if isinstance(trace_dir, str) and trace_dir.strip() else None
        self._trace: Any | None = None
        self._last_commanded: dict[str, np.ndarray] = {}
        self._js_streaming = False
        self._last_ee: np.ndarray | None = None
        self._last_step_time: float | None = None
        self._connected = False
        # Async 200 Hz control loop: when control_hz > ACTION_HZ the joint
        # commands run in a daemon thread so the 30 Hz policy tick is not
        # blocked by 7 x 5 ms of IK.  The thread uses real wall time for
        # time-based interpolation; the injected clock/sleep stay with the
        # policy thread for deterministic pacing and observation assembly.
        self._control_thread: threading.Thread | None = None
        self._control_stop = threading.Event()
        self._control_lock = threading.Lock()
        self._control_target: np.ndarray | None = None
        self._control_target_time: float = 0.0
        self._control_prev: np.ndarray | None = None
        # vla only: the 200 Hz loop reads the policy trajectory itself.
        # None keeps the 30 Hz chord used by agent and hybrid.
        self._trajectory_sample: Callable[[float], np.ndarray] | None = None
        self._trajectory_clock: float | None = None
        self._arm_lock = threading.Lock()
        # Test hook: when non-None, step() runs substeps synchronously via
        # this pump instead of the background thread.
        self._control_pump: Callable[[], None] | None = None
        # 200 Hz joint readings, oldest first. Poses are computed on the
        # policy thread when an observation is assembled.
        self._proprio_lock = threading.Lock()
        self._proprio: deque[_JointSample] = deque()
        self._gripper_widths = (0.0, 0.0)

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
        )
        from inspect_robots_nero._kinematics import NeroKinematics

        # Build incrementally, storing each object as it is created: a factory
        # that aborts mid-bring-up still leaves the built half reachable for
        # teardown, so a retry never stacks a second arm pair on the CAN bus.
        try:
            for side in ("left", "right"):
                self._arms[side] = self._arm_factory(side)
            for arm in self._arms.values():
                # A holding arm is already in the previous task's MIT stream.
                # enable / speed / follower each emit a mode frame; on a fresh
                # driver that frame's mit field is the position-velocity
                # default and the arm sags before any joint target arrives.
                flags = arm.enable_status() or []
                if flags and all(flags):
                    continue
                arm.enable()
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
            from inspect_robots_nero._proxqp_ik import PinocchioProxQPIKSolver

            self._ik = PinocchioProxQPIKSolver(
                str(urdf),
                ee_frame_names=["left_gripper_flange", "right_gripper_flange"],
                frame_position_cost=IK_FRAME_POSITION_COST,
                frame_orientation_cost=IK_FRAME_ORIENTATION_COST,
                posture_cost=IK_POSTURE_COST,
                damping_cost=IK_DAMPING_COST,
                frame_lm_damping=IK_FRAME_LM_DAMPING,
                tcp_translation_m=IK_TCP_TRANSLATION_M,
                tcp_rotation_rpy_rad=IK_TCP_ROTATION_RPY_RAD,
                dt=1.0 / self.control_hz,
                max_iterations=IK_MAX_ITERATIONS,
                max_joint_step_rad=self.max_joint_step_rad,
            )
            # Posture pulls toward this pose. ROS sets it to home; the
            # constructor default is the URDF neutral pose, and from home that
            # pull is a ~0.45 rad step on both arms.
            home_joints = {
                f"left_joint{index}": float(value) for index, value in enumerate(HOME_LEFT, start=1)
            }
            home_joints.update(
                {
                    f"right_joint{index}": float(value)
                    for index, value in enumerate(HOME_RIGHT, start=1)
                }
            )
            self._ik.build_initial_q(home_joints)
            self._last_commanded = {
                "left": np.asarray(HOME_LEFT),
                "right": np.asarray(HOME_RIGHT),
            }
        except Exception as exc:
            self._abort_bring_up()
            raise EmbodimentFault(f"nero bring-up failed: {exc}") from exc
        self._connected = True

    def _abort_bring_up(self) -> None:
        """Tear down a partially built hardware set after a failed bring-up.

        Arms stay enabled (holding) per ``disable_on_close``; a failed bring-up
        must not add a gravity fall to the failure.
        """
        for camera in self._cameras.values():
            with contextlib.suppress(Exception):
                camera.stop()
        for gripper in self._grippers.values():
            with contextlib.suppress(Exception):
                gripper.stop()
        for arm in self._arms.values():
            with contextlib.suppress(Exception):
                if self._disable_on_close:
                    arm.damped_disable()
                arm.close()
        self._arms = {}
        self._grippers = {}
        self._cameras = {}

    def _drive_home(self) -> None:
        from inspect_robots_nero._config import (
            HOME_LEFT,
            HOME_RIGHT,
            RESET_SETTLE_TOL_RAD,
            SPEED_PERCENT,
        )

        homes = {"left": np.asarray(HOME_LEFT), "right": np.asarray(HOME_RIGHT)}
        self._js_streaming = False
        for side in homes:
            reading = self._arms[side].read_state()
            if reading is None or len(reading) != 7:
                raise EmbodimentFault(f"no joint feedback from the {side} arm while homing")
            # Seat the planner on the measured pose before any speed or
            # follower packet, then one firmware move_j to the config home.
            # move_js would jump with no trajectory; that is the sag.
            self._arms[side].move_j(list(reading))
            self._arms[side].set_speed_percent(SPEED_PERCENT)
        self._sleep(0.5)
        # ROS turns follower on before the home trajectory, then move_j
        # arrives on a planned path. Switching follower on after that
        # trajectory has settled, and catching it with move_js, is an
        # unsmoothed mode change: the arm twitches as inference starts.
        for arm in self._arms.values():
            arm.set_normal_mode()
        for side, home in homes.items():
            commanded = clamp_to_joint_envelope(home)
            self._arms[side].move_j(commanded.tolist())
            self._last_commanded[side] = commanded
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

    def _now(self) -> float:
        """Sample time without advancing a test clock that exposes ``peek``."""
        peek = getattr(self._clock, "peek", None)
        if peek is not None:
            return float(peek())
        return float(self._clock())

    def _read_joints(self) -> np.ndarray:
        readings: list[float] = []
        for side in ("left", "right"):
            with self._arm_lock:
                reading = self._arms[side].read_state()
            if reading is None or len(reading) != 7:
                raise EmbodimentFault(f"no joint feedback from the {side} arm; check the CAN link")
            if not all(math.isfinite(value) for value in reading):
                raise EmbodimentFault(
                    f"non-finite joint feedback from the {side} arm; check the CAN link"
                )
            readings.extend(float(value) for value in reading)
        return np.asarray(readings)

    def _read_joints_timed(self) -> tuple[np.ndarray, float]:
        """Read both arms, returning joints and the wall-clock pair skew.

        ``read_state`` does not expose firmware sample timestamps, so the
        wall-clock gap between the two reads is the proxy for the
        ``max_proprioception_pair_skew_ms`` check in ros2_nero_inference.yaml.
        """
        readings: list[float] = []
        first_read_time = time.monotonic()
        for side in ("left", "right"):
            with self._arm_lock:
                reading = self._arms[side].read_state()
            if reading is None or len(reading) != 7:
                raise EmbodimentFault(f"no joint feedback from the {side} arm; check the CAN link")
            if not all(math.isfinite(value) for value in reading):
                raise EmbodimentFault(
                    f"non-finite joint feedback from the {side} arm; check the CAN link"
                )
            readings.extend(float(value) for value in reading)
        pair_skew = time.monotonic() - first_read_time
        return np.asarray(readings), pair_skew

    def _set_control_target(self, start: np.ndarray, end: np.ndarray) -> None:
        """Publish the pose to command now and the pose due one period later."""
        with self._control_lock:
            self._control_prev = np.asarray(start, dtype=np.float64).copy()
            self._control_target = np.asarray(end, dtype=np.float64).copy()
            self._control_target_time = time.monotonic()

    def _start_control_thread(self) -> None:
        """Launch the daemon 200 Hz IK loop (vla path only)."""
        if self._control_thread is not None:
            return
        self._control_stop.clear()
        self._control_thread = threading.Thread(
            target=self._control_loop, name="nero-control", daemon=True
        )
        self._control_thread.start()

    def _stop_control_thread(self) -> None:
        if self._control_thread is None:
            return
        self._control_stop.set()
        self._control_thread.join(timeout=2.0)
        self._control_thread = None

    def _control_loop(self) -> None:
        """Background 200 Hz loop. vla samples the trajectory; other policies chord.

        Uses ``time.monotonic`` / ``time.sleep`` so a stall on the policy
        thread, including camera alignment, does not stop this clock.
        """
        tick = 1.0 / self.control_hz
        last_tick = time.monotonic()
        while not self._control_stop.is_set():
            now = time.monotonic()
            elapsed = now - last_tick
            if elapsed < tick:
                time.sleep(max(0.0, tick - elapsed))
                continue
            last_tick = now
            self._control_tick(now)

    def _control_tick(self, now: float, *, dt: float | None = None) -> None:
        """One 200 Hz command: a trajectory sample, or the 30 Hz chord."""
        with self._control_lock:
            sampler = self._trajectory_sample
            if sampler is not None:
                if dt is None:
                    clock = self._trajectory_clock
                    dt = 1.0 / self.control_hz if clock is None else max(0.0, now - clock)
                self._trajectory_clock = now
            else:
                target = (
                    self._control_target.copy() if self._control_target is not None else None
                )
                prev = self._control_prev.copy() if self._control_prev is not None else None
                target_time = self._control_target_time
        if sampler is not None:
            if dt <= 0:
                return
            pose = np.asarray(sampler(dt), dtype=np.float64).ravel()
            if pose.shape != (20,):
                return
            if self._command_pose(pose):
                with self._control_lock:
                    self._last_ee = pose.copy()
            return
        if target is None:
            if self._last_ee is None:
                return
            target = self._last_ee.copy()
        if prev is None:
            prev = target
        period = 1.0 / ACTION_HZ
        alpha = min(1.0, (now - target_time) / period) if period > 0 else 1.0
        blended = _interpolate_action(prev, target, float(alpha))
        if self._command_pose(blended):
            with self._control_lock:
                self._last_ee = blended.copy()

    def _pump_control_sync(self) -> None:
        """Run one batch of substeps synchronously (test hook).

        Replaces the background thread when ``_control_pump`` is wired: the
        same 7x IK + move_js loop the old inline ``step`` ran, so tests keep
        their deterministic clock and sleep recording.
        """
        with self._control_lock:
            target = self._control_target
            prev = self._control_prev
        substeps = max(1, round(self.control_hz / ACTION_HZ))
        if self._trajectory_sample is not None:
            for index in range(substeps):
                self._control_tick(0.0, dt=1.0 / self.control_hz)
                if index + 1 < substeps:
                    self._sleep(1.0 / self.control_hz)
            return
        if target is None or prev is None:
            return
        tracked = prev.copy()
        for index in range(1, substeps + 1):
            pose = target if substeps == 1 else _interpolate_action(prev, target, index / substeps)
            if self._command_pose(pose):
                tracked = pose
            if index < substeps:
                self._sleep(1.0 / self.control_hz)
        with self._control_lock:
            self._last_ee = np.asarray(tracked, dtype=np.float64).copy()

    def _pace(self) -> None:
        if self._last_step_time is not None:
            # Pace on the policy clock (ACTION_HZ, 30 Hz), not control_hz: the
            # 200 Hz joint loop runs in a background thread and must not slow
            # the policy below its trained rate.
            remaining = self._last_step_time + (1.0 / ACTION_HZ) - self._clock()
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
        with self._control_lock:
            self._trajectory_sample = None
            self._trajectory_clock = None
        with self._proprio_lock:
            self._proprio.clear()
        started = time.perf_counter()
        self._ensure_connected()
        if self.operator_reset_confirm:
            self._confirm_operator_reset(scene.instruction)
        self._drive_home()
        print(f"[timing] connect+home {time.perf_counter() - started:.1f}s", flush=True)
        self._warmup_sensors()
        print(f"[timing] warmup {time.perf_counter() - started:.1f}s", flush=True)
        # Reset counts as the previous control tick so the first step paces itself.
        self._last_step_time = self._clock()
        observation = self._assemble_observation()
        joints = np.asarray(observation.state["joint_pos"], dtype=np.float64)
        self._last_ee = self._tcp_action(joints[:14], float(joints[14]), float(joints[15]))
        if self.control_hz > ACTION_HZ and self._control_pump is None:
            self._start_control_thread()
        self._open_trace()
        return observation

    def step(self, action: Action) -> StepResult:
        """Track one 30 Hz EE target, commanding joints at ``control_hz``.

        When ``control_hz`` exceeds ``ACTION_HZ`` (vla: 200 Hz) the 7x IK loop
        runs in a background thread; ``step`` just publishes the target and
        returns immediately so the policy keeps its 30 Hz cadence.  When
        ``control_hz == ACTION_HZ`` (agent/tests: 30 Hz) the single IK solve
        runs inline as before.
        """
        data = np.asarray(action.data, dtype=np.float64).ravel()
        if data.shape != (20,):
            raise ValueError(
                f"action has shape {data.shape}, expected (20,); the nero action vector is "
                "[left xyz, left rot6d, left width, right xyz, right rot6d, right width]"
            )
        self._pace()
        if self._control_thread is not None or self._control_pump is not None:
            sample = action.meta.get("trajectory_sample")
            if callable(sample):
                # The sampler owns the clock. A blocked observation below does
                # not stop the 200 Hz thread from reading the trajectory.
                with self._control_lock:
                    self._trajectory_sample = sample
                    self._trajectory_clock = None
            else:
                nxt = action.meta.get("servo_next")
                end = data if nxt is None else np.asarray(nxt, dtype=np.float64).ravel()
                if end.shape != (20,):
                    end = data
                with self._control_lock:
                    self._trajectory_sample = None
                self._set_control_target(data, end)
            if self._control_pump is not None:
                self._control_pump()
            return StepResult(observation=self._assemble_observation())
        origin = data if self._last_ee is None else self._last_ee
        substeps = max(1, round(self.control_hz / ACTION_HZ))
        tracked = origin
        for index in range(1, substeps + 1):
            pose = data if substeps == 1 else _interpolate_action(origin, data, index / substeps)
            if self._command_pose(pose):
                tracked = pose
            if index < substeps:
                self._sleep(1.0 / self.control_hz)
        self._last_ee = np.asarray(tracked, dtype=np.float64).copy()
        return StepResult(observation=self._assemble_observation())

    def _tcp_action(self, joints: np.ndarray, left_width: float, right_width: float) -> np.ndarray:
        """Current TCP pose in the same 20-dim layout the policy commands."""
        assert self._kinematics is not None and self._ik is not None
        q = self._kinematics.q_from(joints[:7], joints[7:14])
        chunks: list[np.ndarray] = []
        for side, width in (("left", left_width), ("right", right_width)):
            tcp = self._ik.forward_kinematics(q, self._ik.tcp_frame_name(f"{side}_gripper_flange"))
            chunks.append(tcp[:3, 3])
            chunks.append(matrix_to_rot6d(tcp[:3, :3]))
            chunks.append(np.asarray([width], dtype=np.float64))
        return np.concatenate(chunks)

    def _command_pose(self, data: np.ndarray) -> bool:
        """One IK solve and, when the jump is acceptable, one ``move_js``.

        Does not read cameras. The policy thread assembles the observation
        once per 30 Hz tick; this method also runs on the 200 Hz control thread.
        """
        from inspect_robots_nero._config import GRIPPER_FORCE

        assert self._kinematics is not None and self._ik is not None
        left_target = self._target_pose(data[0:3], data[3:9])
        right_target = self._target_pose(data[10:13], data[13:19])
        joints = self._read_joints()
        self._record_proprio(joints, self._gripper_widths)
        q_seed = self._kinematics.q_from(joints[:7], joints[7:14])
        result = self._ik.solve(
            {
                self._ik.tcp_frame_name("left_gripper_flange"): left_target,
                self._ik.tcp_frame_name("right_gripper_flange"): right_target,
            },
            q_seed,
        )
        left_solution, right_solution = self._kinematics.q_split(result.joint_positions)
        jump = float(
            max(
                np.max(np.abs(left_solution - joints[:7])),
                np.max(np.abs(right_solution - joints[7:14])),
            )
        )
        # Each ProxQP iteration is capped at max_joint_step_rad, and the solver
        # runs IK_MAX_ITERATIONS of them. A 1 cm analytic step at 30 Hz often
        # needs a little more than one iteration (the harness logs were
        # success=True at 0.11-0.13 rad against a 0.10 cap). Hold only when
        # the solution exceeds that whole budget, or the QP itself fails.
        jump_limit = self.max_joint_step_rad * IK_MAX_ITERATIONS
        held = not result.success or jump > jump_limit
        solution = np.concatenate([left_solution, right_solution])
        self._record_motion(
            data,
            joints,
            solution,
            bool(result.success),
            jump,
            held,
            ik=_ik_sample(result, held=held, jump=jump, max_step=self.max_joint_step_rad),
        )
        if held:
            if not self._ik_hold_warned:
                print(
                    "warning: nero IK cannot reach the commanded pose; holding "
                    "position until a reachable target arrives "
                    f"(success={result.success}, joint_jump_rad={jump:.4f}, "
                    f"limit={jump_limit:.4f})",
                    file=sys.stderr,
                )
                self._ik_hold_warned = True
            # Before the first move_js the arm is still on the move_j hold.
            # Republishing through move_js would engage MIT with no smoothing
            # and twitch the arm. After streaming has started, keep publishing
            # or that hold drops.
            if self._js_streaming:
                self._stream_hold()
            return False
        self._ik_hold_warned = False
        left_command = clamp_to_joint_envelope(left_solution)
        right_command = clamp_to_joint_envelope(right_solution)
        self._js_streaming = True
        with self._arm_lock:
            self._arms["left"].move_js(left_command.tolist())
            self._arms["right"].move_js(right_command.tolist())
            self._last_commanded = {"left": left_command, "right": right_command}
            self._grippers["left"].move(float(data[9]), GRIPPER_FORCE)
            self._grippers["right"].move(float(data[19]), GRIPPER_FORCE)
        return True

    def _stream_hold(self) -> None:
        """Keep the follower stream on the last commanded joints."""
        for side, commanded in self._last_commanded.items():
            hold = clamp_to_joint_envelope(commanded)
            with self._arm_lock:
                self._arms[side].move_js(hold.tolist())
            self._last_commanded[side] = hold

    def _warmup_sensors(self) -> None:
        """Wait for camera frames to settle and alignment to stabilize.

        Mirrors ``ros2_nero_inference.yaml``: each camera stream must
        deliver ``WARMUP_MIN_FRAMES`` distinct frames, then
        ``WARMUP_MIN_ALIGNED_GROUPS`` consecutive aligned sensor
        readings must pass before the first inference.  An alignment
        fault (cameras genuinely skewed, CAN link down) propagates
        immediately instead of looping forever.
        """
        seen_times: dict[str, set[float]] = {name: set() for name in self._cameras}
        consecutive_aligned = 0
        while True:
            # Advance the clock so camera timestamps change (the scripted
            # test clock only advances on __call__, not on peek).
            self._clock()
            for name, camera in self._cameras.items():
                _frame, stamp = camera.read(max_age_s=self.camera_max_age_s)
                seen_times[name].add(float(stamp))
            if not all(len(ts) >= WARMUP_MIN_FRAMES for ts in seen_times.values()):
                consecutive_aligned = 0
                self._sleep(1.0 / 30.0)
                continue
            # Once frames are flowing, let alignment faults
            # propagate: a genuine skew or CAN fault must surface,
            # not loop inside the warmup.
            self._read_aligned_sensors()
            consecutive_aligned += 1
            if consecutive_aligned >= WARMUP_MIN_ALIGNED_GROUPS:
                return

    def _stamp_ns(self) -> int:
        """Nanoseconds on the same clock the cameras stamp their frames with."""
        return round(self._now() * 1e9)

    def _read_gripper_widths(self) -> tuple[float, float]:
        with self._arm_lock:
            widths = [self._grippers[side].read_state() for side in ("left", "right")]
        return (
            float(widths[0]) if widths[0] is not None else 0.0,
            float(widths[1]) if widths[1] is not None else 0.0,
        )

    def _read_aligned_sensors(
        self,
    ) -> tuple[int, np.ndarray, dict[str, np.ndarray], dict[str, float], tuple[float, float]]:
        """Re-read until the picture and the arm sample are the same moment.

        The returned joints are the latest reading at or before the camera
        group's midpoint, not the read that happened to land after it.
        """
        deadline = self._now() + self.alignment_failure_timeout_s
        reason = "unaligned"
        while True:
            now = self._now()
            images: dict[str, np.ndarray] = {}
            image_times: dict[str, float] = {}
            for name, camera in self._cameras.items():
                frame, stamp = camera.read(max_age_s=self.camera_max_age_s)
                images[name] = frame
                image_times[name] = float(stamp)
            joints, pair_skew = self._read_joints_timed()
            joint_time = self._now()
            reason = sensor_alignment_fault(
                now=now,
                camera_times=image_times,
                joint_time=joint_time,
                skew_s=self.max_image_triplet_skew_s,
                lag_s=self.max_arm_state_lag_s,
                stale_s=self.camera_max_age_s,
            )
            if reason is None and pair_skew > self.max_proprioception_pair_skew_s:
                reason = "proprioception_pair_skew"
            if reason is None:
                widths = self._read_gripper_widths()
                self._gripper_widths = widths
                self._record_proprio(joints, widths)
                with self._proprio_lock:
                    recorded = tuple(self._proprio)
                chosen = _past_joint_sample(recorded, _image_observation_ns(image_times))
                if chosen is None:
                    reason = "robot_state_stale"
                else:
                    return chosen[0], chosen[1], images, image_times, (chosen[2], chosen[3])
            if self._now() >= deadline:
                raise EmbodimentFault(
                    "cameras and arm state did not align "
                    f"({reason}); triplet skew <= {self.max_image_triplet_skew_s:.3f}s, "
                    f"arm lag <= {self.max_arm_state_lag_s:.3f}s, "
                    f"staleness <= {self.camera_max_age_s:.3f}s, "
                    f"proprio pair skew <= {self.max_proprioception_pair_skew_s:.3f}s"
                )
            self._sleep(0.005)

    def _record_proprio(self, joints: np.ndarray, widths: tuple[float, float]) -> int:
        """Append one joint reading and return its monotonic timestamp.

        The 200 Hz loop only stores joints. Forward kinematics stays on the
        policy thread, which already owns ``_kinematics``; running it here as
        well dropped that loop from 5 ms to about 9 ms and raced the model.
        """
        stamp = self._stamp_ns()
        sample = (
            stamp,
            np.asarray(joints[:14], dtype=np.float64).copy(),
            float(widths[0]),
            float(widths[1]),
        )
        with self._proprio_lock:
            if not self._proprio or stamp > self._proprio[-1][0]:
                self._proprio.append(sample)
            elif stamp == self._proprio[-1][0]:
                self._proprio[-1] = sample
            else:
                stamps = [item[0] for item in self._proprio]
                index = bisect_left(stamps, stamp)
                self._proprio.insert(index, sample)
            cutoff = stamp - 5_000_000_000
            while self._proprio and self._proprio[0][0] < cutoff:
                self._proprio.popleft()
            while len(self._proprio) > 4096:
                self._proprio.popleft()
        return stamp

    def _history_poses(
        self,
        anchor_ns: int,
        anchor_pose: np.ndarray,
    ) -> tuple[tuple[int, np.ndarray], ...]:
        """Anchor pose plus the measured pose 100 ms earlier, when it exists.

        Same refusal rules as the ROS history manager: the older sample has to
        fall within one 30 Hz step of that target, and it has to be a different
        reading. One extra forward-kinematics call covers that older reading.
        """
        with self._proprio_lock:
            samples = tuple(self._proprio)
        older = _older_joint_sample(samples, anchor_ns)
        anchor = (anchor_ns, np.asarray(anchor_pose, dtype=np.float64).copy())
        if older is None:
            return (anchor,)
        stamp, joints, left_width, right_width = older
        return ((stamp, self._measured_eef(joints, (left_width, right_width))), anchor)

    def _measured_eef(self, joints: np.ndarray, widths: tuple[float, float]) -> np.ndarray:
        """TCP pose for a joint reading at the given gripper widths."""
        assert self._kinematics is not None
        poses = self._kinematics.fk(self._kinematics.q_from(joints[:7], joints[7:14]))
        left_width, right_width = widths
        return np.concatenate(
            [
                poses["left"][:3, 3],
                matrix_to_rot6d(poses["left"][:3, :3]),
                [left_width],
                poses["right"][:3, 3],
                matrix_to_rot6d(poses["right"][:3, :3]),
                [right_width],
            ]
        )

    def _assemble_observation(self) -> Observation:
        assert self._kinematics is not None
        anchor_ns, joints, images, image_times, (left_width, right_width) = (
            self._read_aligned_sensors()
        )
        poses = self._kinematics.fk(self._kinematics.q_from(joints[:7], joints[7:14]))
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
        history = self._history_poses(anchor_ns, eef_state)
        joint_pos = np.concatenate([joints, [left_width, right_width]])
        return Observation(
            images=images,
            state={"eef_state": eef_state, "joint_pos": joint_pos},
            instruction=self._instruction,
            image_times=image_times,
            state_time=min(image_times.values()),
            extra={"proprio_time_ns": anchor_ns, "proprio_history": history},
        )

    def _open_trace(self) -> None:
        """Start a new episode directory when ``trace_dir`` was requested."""
        if self._trace is not None:
            self._trace.close()
            self._trace = None
        if self._trace_dir is None:
            return
        from inspect_robots_nero._trace import EpisodeTrace

        self._trace = EpisodeTrace(self._trace_dir)
        print(f"[trace] recording {self._trace.episode}", flush=True)

    def _record_motion(
        self,
        commanded: np.ndarray,
        joints: np.ndarray,
        solution: np.ndarray,
        success: bool,
        jump: float,
        held: bool,
        ik: dict[str, Any] | None = None,
    ) -> None:
        if self._trace is None:
            return
        widths = (float(commanded[9]), float(commanded[19]))
        measured = self._tcp_action(joints[:14], *widths)
        issued = self._tcp_action(solution[:14], *widths)
        self._trace.record_motion(
            commanded=commanded,
            measured=measured,
            issued=issued,
            joints=joints[:14],
            solution=solution,
            success=success,
            jump=jump,
            held=held,
            ik=ik,
        )

    def close(self) -> None:
        """Stop cameras, detach grippers, and close the CAN links.

        The arms stay ENABLED and holding unless ``disable_on_close`` is set:
        this rig has no brakes, so an automatic disable is a gravity fall; the
        operator disables deliberately (console E-STOP or arm_tools disable).
        """
        self._stop_control_thread()
        if self._trace is not None:
            self._trace.close()
            self._trace = None
        for camera in self._cameras.values():
            with contextlib.suppress(Exception):
                camera.stop()
        for gripper in self._grippers.values():
            with contextlib.suppress(Exception):
                gripper.stop()
        for arm in self._arms.values():
            with contextlib.suppress(Exception):
                if self._disable_on_close:
                    arm.damped_disable()
                arm.close()
        self._connected = False
