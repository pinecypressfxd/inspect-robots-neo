"""Pika gripper wrapper over the vendor effector attached to a Nero arm."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from inspect_robots_nero._arm import NeroArm


class NeroGripper:
    """Width-commanded gripper; the effector is injectable for tests."""

    def __init__(
        self,
        arm: NeroArm,
        *,
        effector: Any | None = None,
        start_sleep_s: float = 0.3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._arm = arm
        self._effector = effector
        self._start_sleep_s = start_sleep_s
        self._sleep = sleep

    def start(self) -> None:
        """Attach the AGX gripper effector to the paired arm (once)."""
        if self._effector is not None:
            return
        raw = self._arm.raw_robot
        if raw is None:
            raise RuntimeError("gripper start requires the paired arm to be connected first")
        self._effector = raw.init_effector(raw.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self._sleep(self._start_sleep_s)

    def move(self, width: float, force: float) -> None:
        """Command one gripper width in meters with a force ratio."""
        self.start()
        assert self._effector is not None
        move_metric = getattr(self._effector, "move_gripper_m", None)
        if callable(move_metric):
            move_metric(value=float(width), force=float(force))
            return
        self._effector.move_gripper(width=float(width), force=float(force))

    def read_state(self) -> float | None:
        """Latest gripper width in meters, or None before start/without feedback."""
        if self._effector is None:
            return None
        status = self._effector.get_gripper_status()
        if status is None:
            return None
        message = status.msg
        width = getattr(message, "width", None)
        return float(width if width is not None else message.value)

    def stop(self) -> None:
        """Detach the effector handle (the hardware keeps its last width)."""
        self._effector = None
