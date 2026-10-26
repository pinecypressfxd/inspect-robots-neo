"""ROS last-action sampling: anchor pose, and the pose 100 ms earlier."""

from __future__ import annotations

import numpy as np

from inspect_robots_vla._history import ACTION_DT_NS, sample_last_action_window

_INTERVAL = 3 * ACTION_DT_NS


def _pose(value: float) -> np.ndarray:
    return np.full(14, value, dtype=np.float32)


def test_window_uses_the_latest_sample_at_or_before_the_target() -> None:
    anchor = 300_000_000
    target = anchor - _INTERVAL
    samples = (
        (target - 20_000_000, _pose(1.0)),
        (target + 10_000_000, _pose(2.0)),
        (anchor, _pose(3.0)),
    )
    window = sample_last_action_window(samples, anchor_ns=anchor)
    assert window is not None
    assert float(window[0][0]) == 1.0
    assert float(window[1][0]) == 3.0


def test_window_refuses_a_gap_past_one_action_step() -> None:
    anchor = 300_000_000
    samples = (
        (anchor - _INTERVAL - ACTION_DT_NS - 1, _pose(1.0)),
        (anchor, _pose(3.0)),
    )
    assert sample_last_action_window(samples, anchor_ns=anchor) is None


def test_window_refuses_when_the_anchor_sample_is_missing() -> None:
    samples = ((100_000_000, _pose(1.0)),)
    assert sample_last_action_window(samples, anchor_ns=200_000_000) is None
