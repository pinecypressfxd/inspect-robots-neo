"""Operator bring-up tools: home the arms or drop them limp on demand.

Homing streams follower-mode ``move_js`` increments, the pairing the
embodiment's reset uses and the only one this firmware honors: on Nero
firmware >= 1.12 there is no position mode (``set_normal_mode()`` is a
no-op the vendor warns about), so a ``move_j`` trajectory executes wrongly
in the default follower mode. The disable path polls per-joint enable
flags until every joint reports limp or a timeout expires.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from inspect_robots_nero._arm import NeroArm, clamp_to_joint_envelope
from inspect_robots_nero._config import (
    CAN_CHANNELS,
    CONTROL_HZ,
    GRIPPER_FORCE,
    GRIPPER_INIT_WIDTH_M,
    HOME_LEFT,
    HOME_RIGHT,
    RESET_SETTLE_TIMEOUT_S,
    RESET_SETTLE_TOL_RAD,
    SPEED_PERCENT,
)
from inspect_robots_nero._gripper import NeroGripper

HOMES: Mapping[str, Sequence[float]] = {"left": HOME_LEFT, "right": HOME_RIGHT}

Clock = Callable[[], float]
Sleeper = Callable[[float], None]


def build_arms(sides: Sequence[str]) -> dict[str, NeroArm]:
    """Construct wrappers for the given sides from the config channels."""
    unknown = sorted(set(sides) - set(HOMES))
    if unknown:
        raise ValueError(f"unknown sides {unknown}; valid: {sorted(HOMES)}")
    return {side: NeroArm(side, CAN_CHANNELS[side]) for side in sides}


def connect_all(arms: Mapping[str, NeroArm]) -> None:
    """Connect every arm; disconnect the ones already up if a later one fails."""
    connected: list[str] = []
    try:
        for side, arm in arms.items():
            arm.connect()
            connected.append(side)
    except BaseException:
        for side in reversed(connected):
            with contextlib.suppress(Exception):
                arms[side].close()
        raise


def home_arms(
    arms: Mapping[str, NeroArm],
    *,
    wait_s: float,
    sleep: Sleeper = time.sleep,
    clock: Clock = time.monotonic,
    enable_retries: int = 3,
    retry_pause_s: float = 0.5,
    control_hz: float = CONTROL_HZ,
    max_joint_step_rad: float = 0.1,
    settle_tol_rad: float = RESET_SETTLE_TOL_RAD,
    settle_timeout_s: float = RESET_SETTLE_TIMEOUT_S,
) -> None:
    """Enable every arm, open both grippers, then stream follower joints to home.

    ``enable()`` returns falsy both for a refused enable and for an arm that
    is already enabled (firmware semantics), so a falsy return retries a few
    times and then accepts the state when the per-joint flags say enabled.
    The grippers open to ``GRIPPER_INIT_WIDTH_M`` before any joint motion so
    home lands in a known clean state without interleaving the effector's
    CAN traffic with the joint stream. Both arms then step toward the config
    home poses in bounded ``move_js`` increments (follower mode has no
    firmware smoothing, so a direct jump would command a violent velocity)
    until settled, mirroring the embodiment's reset drive.
    """
    for arm in arms.values():
        # A latched e-stop (from damped_disable) makes the firmware ignore
        # motion commands until the controller is reset; reset unconditionally
        # so homing always doubles as the recovery path.
        with contextlib.suppress(Exception):
            arm.reset_motion_controller()
        if arm.enable():
            continue
        for _ in range(enable_retries):
            enabled = arm.enable_status() or []
            if enabled and all(enabled):
                break
            sleep(retry_pause_s)
            if arm.enable():
                break
        else:
            raise RuntimeError(f"failed to enable the {arm.side} arm before homing")
    for arm in arms.values():
        arm.set_speed_percent(SPEED_PERCENT)
        # The wrapper prefers follower mode: the only mode this firmware
        # honors (>= 1.12 has no position mode).
        arm.set_normal_mode()
    for arm in arms.values():
        NeroGripper(arm, sleep=sleep).move(GRIPPER_INIT_WIDTH_M, GRIPPER_FORCE)
    targets = {side: clamp_to_joint_envelope(list(HOMES[side])) for side in arms}
    deadline = clock() + settle_timeout_s
    while True:
        currents: dict[str, np.ndarray] = {}
        settled = True
        for side, arm in arms.items():
            reading = arm.read_state()
            if reading is None or len(reading) != 7:
                raise RuntimeError(f"no joint feedback from the {arm.side} arm while homing")
            current = np.asarray(reading, dtype=np.float64)
            currents[side] = current
            if float(np.max(np.abs(current - targets[side]))) > settle_tol_rad:
                settled = False
        if settled:
            for side, arm in arms.items():
                arm.move_js(targets[side].tolist())
            break
        if clock() >= deadline:
            raise RuntimeError(
                f"arms did not settle at home within {settle_timeout_s:g}s; "
                "check the joint feedback and for physical obstruction"
            )
        for side, arm in arms.items():
            step = np.clip(targets[side] - currents[side], -max_joint_step_rad, max_joint_step_rad)
            arm.move_js((currents[side] + step).tolist())
        sleep(1.0 / control_hz)
    sleep(wait_s)


def disable_arms(
    arms: Mapping[str, NeroArm],
    *,
    timeout_s: float,
    poll_s: float = 0.1,
    clock: Clock = time.monotonic,
    sleep: Sleeper = time.sleep,
) -> bool:
    """Disable every arm; True once all joints report limp, False on timeout.

    Arms without enable feedback (``enable_status() is None``) stay pending
    and can only leave the loop through the timeout.
    """
    deadline = clock() + timeout_s
    pending = dict(arms)
    while pending and clock() < deadline:
        for side, arm in list(pending.items()):
            arm.damped_disable()
            status = arm.enable_status()
            if status is not None and not any(status):
                del pending[side]
        if pending:
            sleep(poll_s)
    return not pending
