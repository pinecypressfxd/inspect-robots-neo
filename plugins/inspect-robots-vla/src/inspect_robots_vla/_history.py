"""Sample the two last-action frames the ROS planner sends to the model.

The anchor frame is the proprioception sample at the observation. The older
frame is the latest sample at or before ``anchor - stride * action_dt``
(100 ms). A sample further than one action step away, a missing anchor, or
two targets that land on the same sample refuses the request. The planner
then skips inference instead of repeating the current pose as history.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence

import numpy as np

# 30 Hz action step, matching ros2_nero_inference.yaml action_dt_ns.
ACTION_DT_NS = 33_333_333


def sample_last_action_window(
    samples: Sequence[tuple[int, np.ndarray]],
    *,
    anchor_ns: int,
    window_size: int = 2,
    sample_interval_ns: int = 3 * ACTION_DT_NS,
    max_target_error_ns: int = ACTION_DT_NS,
) -> tuple[np.ndarray, ...] | None:
    """Oldest-to-newest history frames, or None when ROS would skip.

    ``samples`` are ``(timestamp_ns, state)`` in ascending timestamp order.
    Timestamps after ``anchor_ns`` are ignored. The anchor timestamp itself
    must be one of the samples.
    """
    if window_size <= 0 or sample_interval_ns <= 0 or max_target_error_ns < 0:
        raise ValueError("history window, interval, and tolerance must be non-negative")
    if anchor_ns < 0:
        raise ValueError("anchor_ns must be non-negative")
    covered = tuple(sample for sample in samples if sample[0] <= anchor_ns)
    if not covered:
        return None
    if covered[-1][0] != anchor_ns:
        return None
    timestamps = tuple(sample[0] for sample in covered)
    oldest = timestamps[0]
    selected: list[np.ndarray] = []
    selected_ns: list[int] = []
    for offset in reversed(range(window_size)):
        target_ns = anchor_ns - offset * sample_interval_ns
        if target_ns < oldest:
            return None
        index = bisect_right(timestamps, target_ns) - 1
        stamp = timestamps[index]
        if target_ns - stamp > max_target_error_ns:
            return None
        if selected_ns and stamp <= selected_ns[-1]:
            return None
        selected_ns.append(stamp)
        selected.append(np.asarray(covered[index][1], dtype=np.float32))
    return tuple(selected)
