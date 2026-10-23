from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class StepView:
    instruction: str
    images: dict
    camera_times: dict
    body_time: float
    eef_state: np.ndarray
    joint_pos: np.ndarray


class Port(Protocol):
    def observation(self) -> StepView: ...

    def move(self, targets: dict) -> dict: ...

    def set_gripper(self, side: str, width: float) -> dict: ...

    def auto_check(self) -> dict: ...
