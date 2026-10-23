"""Queue absolute Nero targets until the policy drains each motion."""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping

import numpy as np

from inspect_robots_pai.ports import StepView

from .motion import DIM_LABELS, interpolate

_CAMERAS = (
    ("left_rgbd", "left"),
    ("right_rgbd", "right"),
    ("chest_rgbd", "chest"),
)


class LivePort:
    """Hold the latest observation and hand interpolated rows to act()."""

    def __init__(self) -> None:
        self._observation: object | None = None
        self._queue: list[list[float]] = []
        self._cond = threading.Condition()

    def set_observation(self, observation: object) -> None:
        """Remember the observation the next port read should expose."""
        self._observation = observation

    def observation(self) -> StepView:
        """Map the stored observation into a step view."""
        obs = self._observation
        images: dict[str, np.ndarray] = {}
        camera_times = {"left": 0.0, "right": 0.0, "chest": 0.0}
        instruction = ""
        body_time = 0.0
        state: Mapping[str, np.ndarray] | None = None
        if obs is not None:
            raw_images = getattr(obs, "images", {})
            raw_times = getattr(obs, "image_times", {})
            for source, name in _CAMERAS:
                image = raw_images.get(source)
                if image is not None:
                    images[name] = image
                if name in raw_times:
                    camera_times[name] = float(raw_times[name])
                elif source in raw_times:
                    camera_times[name] = float(raw_times[source])
            text = getattr(obs, "instruction", None)
            if text is not None:
                instruction = str(text)
            body_time = float(getattr(obs, "state_time", 0.0))
            state = getattr(obs, "state", None)
        return StepView(
            instruction=instruction,
            images=images,
            camera_times=camera_times,
            body_time=body_time,
            eef_state=_vector(state, "eef_state", 20),
            joint_pos=_vector(state, "joint_pos", 16),
        )

    def move(self, targets: Mapping[str, float]) -> dict[str, object]:
        """Interpolate targets, queue every row, and block until drained."""
        rows, _count = interpolate(self._pose_map(), targets)
        with self._cond:
            self._queue.extend(rows)
            self._cond.notify_all()
            # wait() releases the condition so act() can drain the queue.
            while self._queue:
                self._cond.wait()
        last = [float(value) for value in rows[-1]]
        return {"held": False, "steps": len(rows), "target_vector": last}

    def set_gripper(self, side: str, width: float) -> dict[str, object]:
        """Set one gripper width and move to it."""
        return self.move({f"{side}_gripper": float(width)})

    def auto_check(self) -> dict[str, object]:
        """Return the fixed no-geometry result."""
        return {"ok": True, "detail": "no geometric check"}

    def drain(self) -> list[list[float]]:
        """Take every queued row and wake the blocked move."""
        with self._cond:
            rows = [list(row) for row in self._queue]
            self._queue.clear()
            self._cond.notify_all()
            return rows

    def wait(self, timeout: float, done: threading.Event) -> None:
        """Block until rows are queued, the trial ends, or timeout elapses."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while not self._queue and not done.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self._cond.wait(remaining)

    def wake(self) -> None:
        """Wake a policy waiting for rows or thread completion."""
        with self._cond:
            self._cond.notify_all()

    def _pose_map(self) -> dict[str, float]:
        values = self.observation().eef_state
        pose = dict.fromkeys(DIM_LABELS, 0.0)
        for index, label in enumerate(DIM_LABELS):
            if index < len(values):
                pose[label] = float(values[index])
        return pose


def _vector(state: Mapping[str, np.ndarray] | None, key: str, length: int) -> np.ndarray:
    if state is not None and key in state:
        return np.asarray(state[key], dtype=np.float64)
    return np.zeros(length, dtype=np.float64)
