"""Operator bring-up tools: home the arms or drop them limp on demand.

Homing is the reference stack's reset recipe (nero_env ``_dispatch_reset``):
per arm, set the speed, command the CURRENT joint angles first -- the
trajectory planner re-seats onto the measured pose, and skipping this
prime is what made a plain ``move_j`` home execute wrongly on this
firmware -- pause half a second, then one firmware ``move_j`` straight to
the config home. Grippers open after the arms. The disable path polls
per-joint enable flags until every joint reports limp or a timeout
expires.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Mapping, Sequence

from inspect_robots_nero._arm import NeroArm, clamp_to_joint_envelope
from inspect_robots_nero._config import (
    CAN_CHANNELS,
    GRIPPER_FORCE,
    GRIPPER_INIT_WIDTH_M,
    HOME_LEFT,
    HOME_RIGHT,
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
    prime_s: float = 0.5,
) -> None:
    """Firmware-reset every arm to home: the reference recipe, nothing else.

    Mirrors nero_env's start + _dispatch_reset pair and deliberately carries
    no latch-clearing or enable-retry machinery: set the follower mode the
    reference runs under, one plain enable, then per arm command the CURRENT
    joint angles (seating the trajectory planner onto the measured pose),
    pause, and one firmware ``move_j`` straight to the config home. Grippers
    open after the arms. Recovery from a latched e-stop is the operator's
    explicit business (arm power cycle), not Home's.
    """
    for arm in arms.values():
        arm.set_normal_mode()
        arm.enable()
    for side, arm in arms.items():
        arm.set_speed_percent(SPEED_PERCENT)
        reading = arm.read_state()
        if reading is None or len(reading) != 7:
            raise RuntimeError(f"no joint feedback from the {arm.side} arm while homing")
        arm.move_j(list(reading))
        sleep(prime_s)
        arm.move_j(clamp_to_joint_envelope(list(HOMES[side])).tolist())
    for arm in arms.values():
        NeroGripper(arm, sleep=sleep).move(GRIPPER_INIT_WIDTH_M, GRIPPER_FORCE)
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
