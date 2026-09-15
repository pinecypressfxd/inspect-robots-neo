"""Nero arm wrapper over the pyAgxArm vendor SDK, one instance per side.

Mirrors the bring-up HAL's call sequence (create config, connect, settle,
enable, follower mode, ``move_js`` streaming, ``get_joint_angles`` feedback)
with the vendor object injectable for tests.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from inspect_robots_nero._config import FIRMWARE_VERSION

_FIRMWARE_NAMES = {
    "default": "DEFAULT",
    "v111": "V111",
    "1.11": "V111",
    "111": "V111",
    "v112": "V112",
    "1.12": "V112",
    "112": "V112",
}


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

    def move_js(self, joints: Sequence[float]) -> None:
        """Stream one follower-mode joint target (no firmware smoothing)."""
        assert self._robot is not None, "connect() before move_js()"
        self._robot.move_js([float(value) for value in joints])

    def read_state(self) -> tuple[float, ...] | None:
        """Latest joint angles, or None when the vendor returns nothing."""
        if self._robot is None or not self._connected:
            return None
        angles = self._robot.get_joint_angles()
        if angles is None:
            return None
        return tuple(float(value) for value in angles)

    def close(self) -> None:
        """Disconnect from CAN; safe to call once."""
        if self._robot is not None:
            self._robot.disconnect()
