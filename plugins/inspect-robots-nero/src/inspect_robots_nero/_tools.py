"""Operator bring-up tools: home the arms or drop them limp on demand.

Homing is the reference stack's reset (nero_env ``_dispatch_reset``): one
firmware ``move_j`` of the measured joints, so the trajectory planner seats
on the live pose, then one ``move_j`` to the config home. Speed is set only
after that seat. A fresh driver defaults ``mit_mode`` to position-velocity;
emitting follower mode or speed before any joint target drops the MIT hold
the previous task left, and the arm sags. The disable path polls per-joint
enable flags until every joint reports limp or a timeout expires.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable, Mapping, Sequence

from inspect_robots_nero._arm import NeroArm, clamp_to_joint_envelope
from inspect_robots_nero._config import (
    CAN_CHANNELS,
    GRIPPER_FORCE,
    GRIPPER_INIT_WIDTH_M,
    HOME_LEFT,
    HOME_RIGHT,
    RESET_SETTLE_TOL_RAD,
    SPEED_PERCENT,
)
from inspect_robots_nero._gripper import NeroGripper

HOMES: Mapping[str, Sequence[float]] = {"left": HOME_LEFT, "right": HOME_RIGHT}

Clock = Callable[[], float]
Sleeper = Callable[[float], None]
logger = logging.getLogger(__name__)


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
    settle_timeout_s: float = 20.0,
    clock: Clock = time.monotonic,
) -> None:
    """Return every arm to the config home with one firmware joint trajectory.

    An arm that already reports every joint enabled is not re-enabled and is
    not switched into follower mode: both of those emit a mode frame, and on
    a fresh driver that frame carries the default position-velocity MIT field
    with no joint target, which releases the hold left by the previous task.
    An arm that is not fully enabled has usually just come out of E-stop.
    That stop latches until ``reset``; enable and ``move_j`` are ignored
    until then, so recovery resets the motion controller before enabling.
    Each arm is seated with ``move_j`` of its measured joints, then leaves
    follower mode and reseats, so a wrist still holding the last ``move_js``
    target joins the trajectory. Speed is set only after that seat, then one
    ``move_j`` goes to the config home.
    Grippers open after the arms. Returns once every arm is within
    ``RESET_SETTLE_TOL_RAD`` or raises when ``settle_timeout_s`` expires.
    """
    for arm in arms.values():
        flags = arm.enable_status() or []
        if flags and all(flags):
            continue
        arm.reset_motion_controller()
        sleep(0.2)
        if not arm.enable():
            raise RuntimeError(f"failed to enable the {arm.side} arm before homing")
    targets: dict[str, list[float]] = {}
    for side, arm in arms.items():
        reading = arm.read_state()
        if reading is None or len(reading) != 7:
            raise RuntimeError(f"no joint feedback from the {arm.side} arm while homing")
        logger.info("%s measured joints %s", side, [round(value, 4) for value in reading])
        # Seat first, while any previous follower stream is still holding.
        # A run leaves the arm in follower mode, and move_j from that mode
        # moves the proximal joints only; the wrist stays on the last
        # move_js target. Leave follower after the seat, reseat in joint
        # mode, then one trajectory to the config home.
        arm.move_j(list(reading))
        arm.set_position_mode()
        arm.move_j(list(reading))
        arm.set_speed_percent(SPEED_PERCENT)
        sleep(prime_s)
        target = clamp_to_joint_envelope(list(HOMES[side])).tolist()
        logger.info("%s home joints %s", side, [round(value, 4) for value in target])
        arm.move_j(target)
        targets[side] = target
    for arm in arms.values():
        NeroGripper(arm, sleep=sleep).move(GRIPPER_INIT_WIDTH_M, GRIPPER_FORCE)
    deadline = clock() + settle_timeout_s
    while clock() < deadline:
        if all(_near_target(arms[side], targets[side]) for side in targets):
            break
        sleep(0.05)
    else:
        raise RuntimeError(f"arms did not reach home within settle_timeout_s={settle_timeout_s:g}s")
    sleep(wait_s)


def _near_target(arm: NeroArm, target: Sequence[float]) -> bool:
    """True when the latest joint feedback is within the home settle tolerance."""
    reading = arm.read_state()
    if reading is None or len(reading) != len(target):
        return False
    return max(abs(value - goal) for value, goal in zip(reading, target, strict=True)) <= (
        RESET_SETTLE_TOL_RAD
    )


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
