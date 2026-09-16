"""Test fakes standing in for the pyAgxArm vendor objects."""

from __future__ import annotations

from types import SimpleNamespace


class FakeEffector:
    """Records gripper commands and reports a scripted width."""

    def __init__(self) -> None:
        self.moves: list[tuple[float, float]] = []
        self.width = 0.09

    def move_gripper_m(self, *, value: float, force: float) -> None:
        self.moves.append((float(value), float(force)))
        self.width = float(value)

    def get_gripper_status(self) -> SimpleNamespace:
        return SimpleNamespace(msg=SimpleNamespace(width=self.width, force=0.3), timestamp=0.0)


class LegacyEffector(FakeEffector):
    """A vendor build exposing only the legacy move_gripper API."""

    # Absence marker: the wrapper probes with getattr(..., None)/callable(), so
    # the attribute must be unusable rather than a method that raises.
    move_gripper_m = None  # type: ignore[assignment]

    def move_gripper(self, *, width: float, force: float) -> None:
        self.moves.append((float(width), float(force)))
        self.width = float(width)


class FakeAgxRobot:
    """Stands in for the AgxArmFactory product; records the calls we make."""

    def __init__(self) -> None:
        self.connected = False
        self.disconnected = False
        self.enabled = False
        self.enable_result = True
        self.speed: int | None = None
        self.mode: str | None = None
        self.move_js_calls: list[tuple[float, ...]] = []
        self.move_j_calls: list[tuple[float, ...]] = []
        # None: derive the per-joint enable flags from `enabled`; a list lets
        # tests script joints that lag behind (or never confirm) a disable.
        self.joints_enable: list[bool] | None = None
        self.angles: tuple[float, ...] = (0.1, 0.2, 0.0, 0.5, 0.0, 0.0, 0.3)
        self.effector_kind: str | None = None
        self.OPTIONS = SimpleNamespace(EFFECTOR=SimpleNamespace(AGX_GRIPPER="AGX_GRIPPER"))

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.disconnected = True

    def enable(self) -> bool:
        self.enabled = self.enable_result
        return self.enable_result

    def disable(self) -> bool:
        self.enabled = False
        return True

    def set_speed_percent(self, percent: int) -> None:
        self.speed = int(percent)

    def set_follower_mode(self) -> None:
        self.mode = "follower"

    def set_normal_mode(self) -> None:
        self.mode = "normal"

    def move_j(self, joints: list[float]) -> None:
        self.move_j_calls.append(tuple(float(value) for value in joints))
        # A firmware-smoothed move lands on its target: adopt it as readback.
        self.angles = tuple(float(value) for value in joints)

    def get_joints_enable_status_list(self) -> list[bool]:
        if self.joints_enable is not None:
            return list(self.joints_enable)
        return [self.enabled] * 7

    def move_js(self, joints: list[float]) -> None:
        self.move_js_calls.append(tuple(float(value) for value in joints))
        # Adopt the commanded joints as the readback: Task 5's home-settle loop
        # polls read_state() until the arm reports the commanded pose.
        self.angles = tuple(float(value) for value in joints)

    def get_joint_angles(self) -> tuple[float, ...]:
        return self.angles

    def init_effector(self, kind: str) -> FakeEffector:
        self.effector_kind = kind
        return FakeEffector()
