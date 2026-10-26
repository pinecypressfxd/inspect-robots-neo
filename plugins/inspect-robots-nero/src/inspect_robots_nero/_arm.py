"""Nero arm wrapper over the pyAgxArm vendor SDK, one instance per side.

Mirrors the bring-up HAL's call sequence (create config, connect, settle,
enable, follower mode, ``move_js`` streaming, ``get_joint_angles`` feedback)
with the vendor object injectable for tests.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from inspect_robots_nero._config import (
    FIRMWARE_JOINT_LIMITS,
    FIRMWARE_VERSION,
    JOINT_LIMIT_SAFETY_MARGIN_RAD,
)

# Values must match the AgxArmFactory registry keys for robot="nero",
# comm="can" exactly (lowercase); an uppercase key fails with
# "Driver not registered" at create_arm time.
_FIRMWARE_NAMES = {
    "default": "default",
    "v111": "v111",
    "1.11": "v111",
    "111": "v111",
    "v112": "v112",
    "1.12": "v112",
    "112": "v112",
    "v120": "v120",
    "1.20": "v120",
    "120": "v120",
    "v121": "v121",
    "1.21": "v121",
    "121": "v121",
}


def joint_envelope() -> tuple[np.ndarray, np.ndarray]:
    """Per-joint (low, high) command envelope: firmware limits shrunk by margin."""
    low = np.array([limit[0] + JOINT_LIMIT_SAFETY_MARGIN_RAD for limit in FIRMWARE_JOINT_LIMITS])
    high = np.array([limit[1] - JOINT_LIMIT_SAFETY_MARGIN_RAD for limit in FIRMWARE_JOINT_LIMITS])
    return low, high


def clamp_to_joint_envelope(joints: Sequence[float] | np.ndarray) -> np.ndarray:
    """Clamp commanded joints into the envelope so the firmware never faults.

    The firmware disables the arm (a gravity fall) when commanded past its
    joint limits; IK output can sit exactly on the URDF edge, which mirrors
    those limits, so this clamp is the last line before every wire command.
    """
    low, high = joint_envelope()
    return np.clip(np.asarray(joints, dtype=np.float64), low, high)


class NeroArm:
    """One CAN-connected Nero arm; the vendor robot is injectable for tests."""

    def __init__(
        self,
        side: str,
        channel: str,
        *,
        firmware: str = FIRMWARE_VERSION,
        robot: Any | None = None,
        connect_sleep_s: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        if firmware not in _FIRMWARE_NAMES:
            known = ", ".join(sorted(_FIRMWARE_NAMES))
            raise ValueError(f"unknown firmware version {firmware!r}; known: {known}")
        self.side = side
        self.channel = channel
        self._firmware = _FIRMWARE_NAMES[firmware]
        self._robot = robot
        self._connected = False
        self._connect_sleep_s = connect_sleep_s
        self._sleep = sleep

    @property
    def raw_robot(self) -> Any | None:
        """The vendor robot object, or None before the first connect."""
        return self._robot if self._connected else None

    def connect(self) -> None:
        """Open the CAN connection (no vendor SDK when a robot was injected)."""
        if self._connected:
            return
        if self._robot is None:
            from pyAgxArm import AgxArmFactory, create_agx_arm_config

            config = create_agx_arm_config(
                robot="nero",
                comm="can",
                firmeware_version=self._firmware,
                channel=self.channel,
                interface="socketcan",
            )
            self._robot = AgxArmFactory.create_arm(config)
        self._robot.connect()
        self._sleep(self._connect_sleep_s)
        self._connected = True

    def enable(self) -> bool:
        """Enable the arm's motors."""
        assert self._robot is not None, "connect() before enable()"
        return bool(self._robot.enable())

    def reset_motion_controller(self) -> None:
        """Clear a latched motion-controller state (e-stop / damping).

        ``electronic_emergency_stop`` latches; until this reset the firmware
        silently ignores motion commands, so every recovery-from-disable path
        (homing after an E-STOP) must reset first.
        """
        assert self._robot is not None, "connect() before reset_motion_controller()"
        self._robot.reset()

    def electronic_emergency_stop(self) -> None:
        """Apply damping to all joints (controlled deceleration, no shock).

        The firmware raises joint kd so any subsequent motion (including the
        gravity fall after disable) is resisted; this is the rig's slow-fall
        primitive and must precede every deliberate disable.
        """
        assert self._robot is not None, "connect() before electronic_emergency_stop()"
        self._robot.electronic_emergency_stop()

    def damped_disable(
        self,
        *,
        poll_s: float = 0.05,
        still_s: float = 0.5,
        descent_timeout_s: float = 12.0,
        still_eps_rad: float = 0.01,
    ) -> bool:
        """Hold damping for the whole gravity descent, then disable.

        ``electronic_emergency_stop`` raises joint kd and latches. Clearing
        that latch after a short pause, then disabling, damps only the first
        moment and lets the rest of the fall go limp. This waits, while the
        latch is still held, until the joints stop moving or the descent
        timeout expires, and only then resets and disables so Home can enable
        again.
        """
        with contextlib.suppress(Exception):
            self.electronic_emergency_stop()
            self._wait_until_still(
                poll_s=poll_s,
                still_s=still_s,
                timeout_s=descent_timeout_s,
                eps_rad=still_eps_rad,
            )
            self.reset_motion_controller()
        return self.disable()

    def _wait_until_still(
        self, *, poll_s: float, still_s: float, timeout_s: float, eps_rad: float
    ) -> None:
        """Poll joint angles until they hold still, or ``timeout_s`` elapses."""
        needed = max(1, int(still_s / poll_s))
        limit = max(needed, int(timeout_s / poll_s))
        stable = 0
        previous = self.read_state()
        for _ in range(limit):
            self._sleep(poll_s)
            current = self.read_state()
            if (
                current is None
                or previous is None
                or len(current) != len(previous)
                or max(abs(a - b) for a, b in zip(current, previous, strict=True)) > eps_rad
            ):
                stable = 0
            else:
                stable += 1
                if stable >= needed:
                    return
            previous = current

    def disable(self) -> bool:
        """Disable the arm's motors."""
        assert self._robot is not None, "connect() before disable()"
        return bool(self._robot.disable())

    def set_speed_percent(self, percent: int) -> None:
        """Clamp the firmware speed limit to a percentage."""
        assert self._robot is not None, "connect() before set_speed_percent()"
        self._robot.set_speed_percent(int(percent))

    def set_normal_mode(self) -> None:
        """Prefer the follower mode the move_js streaming path expects."""
        assert self._robot is not None, "connect() before set_normal_mode()"
        if hasattr(self._robot, "set_follower_mode"):
            self._robot.set_follower_mode()
        else:
            self._robot.set_normal_mode()

    def set_position_mode(self) -> None:
        """Switch to the firmware normal mode that ``move_j`` trajectories expect."""
        assert self._robot is not None, "connect() before set_position_mode()"
        self._robot.set_normal_mode()

    def move_j(self, joints: Sequence[float]) -> None:
        """Command one firmware-smoothed joint move (requires position mode)."""
        assert self._robot is not None, "connect() before move_j()"
        self._robot.move_j([float(value) for value in joints])

    def enable_status(self) -> list[bool] | None:
        """Per-joint enable flags (7 bools), or None without feedback."""
        if self._robot is None or not self._connected:
            return None
        status = self._robot.get_joints_enable_status_list()
        if status is None:
            return None
        values = list(status)
        if len(values) != 7 or not all(isinstance(value, bool) for value in values):
            raise ValueError(f"enable status must hold 7 bools, got {values!r}")
        return values

    def move_js(self, joints: Sequence[float]) -> None:
        """Stream one follower-mode joint target (no firmware smoothing)."""
        assert self._robot is not None, "connect() before move_js()"
        self._robot.move_js([float(value) for value in joints])

    def read_state(self) -> tuple[float, ...] | None:
        """Latest joint angles, or None when the vendor returns nothing.

        The vendor returns ``MessageAbstract[list[float]] | None``; the
        payload lives behind its ``.msg`` property.
        """
        if self._robot is None or not self._connected:
            return None
        feedback = self._robot.get_joint_angles()
        if feedback is None:
            return None
        values = getattr(feedback, "msg", feedback)
        if values is None:
            return None
        return tuple(float(value) for value in values)

    def close(self) -> None:
        """Disconnect from CAN; safe to call once."""
        if self._robot is not None:
            self._robot.disconnect()
