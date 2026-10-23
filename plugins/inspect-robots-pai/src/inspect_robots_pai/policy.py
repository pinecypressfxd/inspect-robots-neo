"""Adapt one Nero program to Inspect Robots action chunks."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from inspect_robots.types import Action, ActionChunk, Observation
from inspect_robots_pai.domains.nero.live import LivePort
from inspect_robots_pai.domains.nero.motion import DIM_LABELS
from inspect_robots_pai.execution.trial import run_trial


class PaiPolicy:
    """Run a program on a daemon thread and serve its queued motions."""

    def __init__(self, program: str | Path, trial_dir: str | Path) -> None:
        self.program = program
        self.trial_dir = trial_dir
        self.port = LivePort()
        self._thread: threading.Thread | None = None
        self._termination = "returned"
        self._finished = threading.Event()

    def act(self, observation: Observation) -> ActionChunk:
        """Return interpolated actions, a hold, or a stop request."""
        self.port.set_observation(observation)
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        deadline = time.monotonic() + 2.0
        while True:
            rows = self.port.drain()
            if rows:
                return _chunk(rows)
            if self._finished.is_set():
                return _stop(observation, self._termination)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return _hold(observation)
            self.port.wait(remaining, self._finished)

    def _run(self) -> None:
        try:
            result = run_trial(self.program, self.trial_dir, self.port)
            self._termination = result.termination
        except Exception:
            self._termination = "error"
        finally:
            self._finished.set()
            self.port.wake()


def pai_policy(
    program: str | Path | None = None,
    trial_dir: str | Path | None = None,
) -> PaiPolicy:
    """Build a policy. Both program and trial_dir are required."""
    if program is None or trial_dir is None:
        raise ValueError("pai_policy requires program and trial_dir")
    return PaiPolicy(program, trial_dir)


def _chunk(rows: list[list[float]]) -> ActionChunk:
    actions = [Action(np.asarray(row, dtype=np.float64), {}) for row in rows]
    return ActionChunk(actions)


def _hold(observation: Observation) -> ActionChunk:
    return ActionChunk([Action(_pose(observation), {})])


def _stop(observation: Observation, termination: str) -> ActionChunk:
    meta = {"request_stop": True, "stop_reason": termination}
    return ActionChunk([Action(_pose(observation), meta)])


def _pose(observation: Observation) -> np.ndarray:
    pose = np.zeros(len(DIM_LABELS), dtype=np.float64)
    values = observation.state.get("eef_state")
    if values is None:
        return pose
    count = min(len(DIM_LABELS), len(values))
    pose[:count] = np.asarray(values[:count], dtype=np.float64)
    return pose
