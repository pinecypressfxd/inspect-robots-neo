"""Operator bring-up tools: home the arms or drop them limp on demand.

Ports the bring-up checkout's ``test_move_j.py --stage move_j`` and
``disable_arms.py`` flows onto this plugin's wrapper: enable with a hard
failure check, firmware normal mode, ``move_j`` to the config home poses,
and a disable loop that polls per-joint enable flags until every joint
reports limp or a timeout expires.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Mapping, Sequence

from inspect_robots_nero._arm import NeroArm, clamp_to_joint_envelope
from inspect_robots_nero._config import CAN_CHANNELS, HOME_LEFT, HOME_RIGHT

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
    enable_retries: int = 3,
    retry_pause_s: float = 0.5,
) -> None:
    """Enable every arm, switch to position mode, and move_j it to home.

    ``enable()`` returns falsy both for a refused enable and for an arm that
    is already enabled (firmware semantics), so a falsy return retries a few
    times and then accepts the state when the per-joint flags say enabled.
    """
    for arm in arms.values():
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
            raise RuntimeError(f"failed to enable the {arm.side} arm before move_j")
    for arm in arms.values():
        arm.set_position_mode()
    for side, arm in arms.items():
        arm.move_j(clamp_to_joint_envelope(list(HOMES[side])).tolist())
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
    for arm in arms.values():
        arm.set_position_mode()
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
