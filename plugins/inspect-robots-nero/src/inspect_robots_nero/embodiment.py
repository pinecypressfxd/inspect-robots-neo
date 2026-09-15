"""Expose the dual Nero CAN arms and D405 cameras through the embodiment contract.

Construction builds only static spaces; arms, grippers, and cameras connect
lazily on the first reset. Actions are 20-dim absolute end-effector poses
(position + rot6d + gripper width per arm); each step solves IK and commands
one bounded joint increment per arm at ``control_hz``.
"""

from __future__ import annotations

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

ArmFactory = Callable[[str], Any]
CameraFactory = Callable[[str], Any]


def _bounds_for(
    workspace_low: Sequence[float], workspace_high: Sequence[float]
) -> tuple[list[float], list[float]]:
    rot_low, rot_high = ROT6D_BOUNDS
    return (
        [*workspace_low, *[rot_low] * 6, GRIPPER_WIDTH_MIN_M],
        [*workspace_high, *[rot_high] * 6, GRIPPER_WIDTH_MAX_M],
    )


class NeroEmbodiment(EmbodimentBase):
    """Drive the dual Nero arms with absolute end-effector pose actions."""

    def __init__(
        self,
        *,
        control_hz: float = CONTROL_HZ,
        workspace_low: tuple[float, float, float] | None = None,
        workspace_high: tuple[float, float, float] | None = None,
        max_step: tuple[float | None, ...] | None = None,
        cameras: Mapping[str, str] | None = None,
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
            unknown = sorted(set(cameras) - set(CAMERA_DEFAULTS))
            if unknown:
                raise ValueError(
                    f"cameras keys {unknown} are not declared cameras; "
                    f"valid names: {sorted(CAMERA_DEFAULTS)}"
                )
            for camera_name, device in cameras.items():
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
        self._arm_factory = arm_factory
        self._camera_factory = camera_factory
        self._clock = clock
        self._sleep = sleep
        self._instruction: str | None = None

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Prepare a scene and return its initial observation (wired in a later task)."""
        del scene, seed
        raise NotImplementedError("hardware bring-up lands with the reset/step task")

    def step(self, action: Action) -> StepResult:
        """Issue one action (wired in a later task)."""
        del action
        raise NotImplementedError("hardware bring-up lands with the reset/step task")
