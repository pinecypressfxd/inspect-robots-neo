"""绝对末端目标的插补. 步长与 Nero 每个控制拍的限幅相同."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

DIM_LABELS: tuple[str, ...] = (
    "left_x",
    "left_y",
    "left_z",
    "left_r1",
    "left_r2",
    "left_r3",
    "left_r4",
    "left_r5",
    "left_r6",
    "left_gripper",
    "right_x",
    "right_y",
    "right_z",
    "right_r1",
    "right_r2",
    "right_r3",
    "right_r4",
    "right_r5",
    "right_r6",
    "right_gripper",
)


# 夹爪组件在 TCP 下方 0.03 m. 水平移动时这个最低点会刮到箱沿.
# 本次试验开始高度约在箱沿上方 0.05 m, 竖直下降不超过这个量.
_DESCEND_LIMIT_M = 0.05


def _step(label: str) -> float:
    if label.endswith("_gripper") or label[-2:] in ("_x", "_y", "_z"):
        return 0.01
    return 0.05


def applied_targets(
    current: Mapping[str, float],
    targets: Mapping[str, float],
    home: Mapping[str, float],
) -> tuple[dict[str, float], tuple[str, ...]]:
    """把低于工作带的 z 收到开始高度以下 0.05 m."""
    _check(targets)
    applied = {label: float(goal) for label, goal in targets.items()}
    limited: list[str] = []
    for side in ("left", "right"):
        label = f"{side}_z"
        if label not in applied:
            continue
        floor = float(home[label]) - _DESCEND_LIMIT_M
        if applied[label] < floor:
            applied[label] = floor
            limited.append(label)
    return applied, tuple(limited)


def interpolate(
    current: Mapping[str, float],
    targets: Mapping[str, float],
    home: Mapping[str, float] | None = None,
) -> tuple[list[list[float]], int]:
    """把绝对目标拆成不超过单拍步长的路径.

    同一只手同时改 xy 和 z 时, 先抬到巡航高度, 再水平移动, 最后垂直下降.
    巡航高度是当前、目标和本次开始高度里的最大值.
    """
    _check(targets)
    reference = home if home is not None else current
    rows: list[list[float]] = []
    pose = {label: float(current[label]) for label in DIM_LABELS}
    for phase in _phases(pose, targets, reference):
        rows.extend(_line(pose, phase))
        for label, goal in phase.items():
            pose[label] = float(goal)
    return rows, len(rows)


def _check(targets: Mapping[str, float]) -> None:
    unknown = set(targets) - set(DIM_LABELS)
    if unknown:
        raise ValueError(f"unknown dimensions: {sorted(unknown)}")


def _phases(
    current: Mapping[str, float],
    targets: Mapping[str, float],
    home: Mapping[str, float],
) -> list[dict[str, float]]:
    raise_z: dict[str, float] = {}
    move_xy: dict[str, float] = {}
    lower_z: dict[str, float] = {}
    other: dict[str, float] = {}
    for label, goal in targets.items():
        value = float(goal)
        if label.endswith(("_x", "_y")):
            move_xy[label] = value
            continue
        if label.endswith("_z"):
            side = label[: -len("_z")]
            moves_xy = f"{side}_x" in targets or f"{side}_y" in targets
            if not moves_xy:
                other[label] = value
                continue
            cruise = max(float(current[label]), value, float(home[label]))
            if cruise - float(current[label]) > 1e-9:
                raise_z[label] = cruise
            if cruise - value > 1e-9:
                lower_z[label] = value
            continue
        other[label] = value
    phases: list[dict[str, float]] = []
    if raise_z:
        phases.append({**raise_z, **other})
        other = {}
    if move_xy:
        phases.append(move_xy)
    if lower_z or other:
        phases.append({**lower_z, **other})
    return phases or [dict(targets)]


def _line(current: Mapping[str, float], targets: Mapping[str, float]) -> list[list[float]]:
    steps = 1
    for label, goal in targets.items():
        delta = abs(float(goal) - float(current[label]))
        steps = max(steps, math.ceil(delta / _step(label) - 1e-12))
    rows: list[list[float]] = []
    for index in range(1, steps + 1):
        row: list[float] = []
        for label in DIM_LABELS:
            start = float(current[label])
            if label in targets:
                goal = float(targets[label])
                row.append(start + (goal - start) * index / steps)
            else:
                row.append(start)
        rows.append(row)
    return rows


def reached(row: Sequence[float], targets: Mapping[str, float]) -> bool:
    """目标各维都落在一个单拍步长以内时视为到达."""
    for label, goal in targets.items():
        actual = float(row[DIM_LABELS.index(label)])
        if abs(actual - float(goal)) > _step(label) + 1e-9:
            return False
    return True
