"""拷贝自 evo 的末端插补，禁止 import evo"""

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


def _step(label: str) -> float:
    if label.endswith("_gripper") or label[-2:] in ("_x", "_y", "_z"):
        return 0.01
    return 0.05


def interpolate(
    current: Mapping[str, float], targets: Mapping[str, float]
) -> tuple[list[list[float]], int]:
    """把绝对目标拆成不超过单拍步长的直线."""
    unknown = set(targets) - set(DIM_LABELS)
    if unknown:
        raise ValueError(f"unknown dimensions: {sorted(unknown)}")
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
    return rows, steps


def reached(row: Sequence[float], targets: Mapping[str, float]) -> bool:
    """目标各维都落在一个单拍步长以内时视为到达."""
    for label, goal in targets.items():
        actual = float(row[DIM_LABELS.index(label)])
        if abs(actual - float(goal)) > _step(label) + 1e-9:
            return False
    return True
