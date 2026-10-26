"""Play a VLA chunk the way the ROS inference consumer does.

``ros2_nero_inference.yaml`` stamps each model step at ``action_hz`` (30 Hz)
and lets the controller keep moving while the next chunk is in flight. A new
chunk is held until ``preview_s`` after its observation, then blended over
``transition_s``. The blend is one fixed chord, from the pose being commanded
at the start of the window to the incoming pose at the end of it. Mixing the
two curves at the sample time makes the target step sideways every time a
chunk arrives. Each control tick commands the pose ``lookahead_s`` ahead.
The default lead is 150 ms, matching ``ros2_nero_inference.yaml``
``lookahead_ms``. The first chunk is shifted forward by that lead, the
same way the ROS controller reanchors a warm-start trajectory onto the
first sample. Without the shift the opening tick would command the pose
150 ms into the chunk and the arm would jump.
The chord itself is stamped on trajectory time: its start is the active
pose at the preview deadline and its end is the incoming pose one
transition later. Each control tick then reads that chord at
``now + lookahead_s``, the same sample the ROS controller takes. After the
chord the incoming chunk is used at its absolute coordinates. Those
coordinates are anchored on the observation that produced the chunk;
shifting them onto the pose already being commanded drops that correction
and the hand misses. A newer chunk that arrives before its preview deadline
replaces the chord in progress. The new chord starts at the pose the
current trajectory has at that deadline, so the wrist is not pulled back
to an earlier chunk. A result that arrives after the deadline is dropped,
the same ``missed_preview_deadline`` reject in ``Controller.push_trajectory``.
The deadline is the first waypoint plus the preview: one action step after
the observation, then 300 ms. This module is that playback clock in the
embodiment's 20-dim absolute pose layout. It does not talk to the network.
"""

from __future__ import annotations

import threading

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ._anchor import rot6d_to_rpy, rpy_to_rot6d

_POSE_DIM = 20
_ARMS = (
    (slice(0, 3), slice(3, 9), 9),
    (slice(10, 13), slice(13, 19), 19),
)


class TrackedPlayback:
    """Cursor over one absolute chunk, with a preview-then-blend handoff.

    Waypoint ``i`` is due at ``origin + (i + 1) / action_hz``, matching the
    planner's ``(step_index + 1) * action_dt``. Sampling past either end holds
    the end pose. ``take`` advances the cursor by one control period per row
    and reads the active trajectory at ``cursor + lookahead_s``. A handoff
    replaces that read with the fixed chord until the window ends.
    """

    def __init__(
        self,
        *,
        action_hz: float,
        preview_s: float,
        transition_s: float,
        lookahead_s: float = 0.0,
    ) -> None:
        if action_hz <= 0 or preview_s < 0 or transition_s < 0 or lookahead_s < 0:
            raise ValueError(
                "action_hz must be > 0 and the blend windows and lookahead must be >= 0"
            )
        self._dt = 1.0 / float(action_hz)
        self._preview_s = float(preview_s)
        self._transition_s = float(transition_s)
        self._lookahead_s = float(lookahead_s)
        self._targets: np.ndarray | None = None
        self._origin = 0.0
        self._incoming: tuple[np.ndarray, float] | None = None
        self._blend_start: float | None = None
        self._chord: tuple[np.ndarray, np.ndarray] | None = None
        # On-time chunk waiting to cut off the chord already in progress:
        # incoming targets, their origin, blend start, and the two chord ends.
        self._replacement: tuple[np.ndarray, float, float, np.ndarray, np.ndarray] | None = None
        self._lock = threading.RLock()
        self.cursor = 0.0

    def advance(self, dt: float) -> np.ndarray:
        """Move the playback clock by ``dt`` seconds and return that pose.

        The 200 Hz arm loop calls this every 5 ms. The sample is the baked
        trajectory at ``cursor + lookahead``, the same read the ROS controller
        takes, so a 30 Hz policy stall does not stop the clock or replace the
        path with a straight chord.
        """
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError(f"dt must be a finite number > 0, got {dt!r}")
        with self._lock:
            if self._targets is None:
                raise RuntimeError("advance() before a trajectory is installed")
            self.cursor += float(dt)
            return np.asarray(self._sample(self.cursor), dtype=np.float32).copy()

    def cursor_time(self) -> float:
        """Playback time in seconds. Safe to read from the policy thread."""
        with self._lock:
            return float(self.cursor)

    def install_first(self, targets: np.ndarray) -> None:
        """Start playback so the first sample is waypoint 0.

        The cursor's first step plus ``lookahead_s`` would otherwise land
        several waypoints into the chunk. Shifting the origin by the lead
        makes that first sample time equal the first waypoint, matching
        ``Controller._reanchor_first_active_trajectory``.
        """
        with self._lock:
            self._install_first(targets)

    def _install_first(self, targets: np.ndarray) -> None:
        self._targets = _checked_targets(targets)
        self._origin = self._lookahead_s
        self._incoming = None
        self._blend_start = None
        self._chord = None
        self._replacement = None
        self.cursor = 0.0

    def install_next(self, targets: np.ndarray, obs_time: float) -> None:
        """Schedule ``targets``, observed at ``obs_time``, or drop them.

        The first waypoint is one action step after the observation. ROS
        rejects the chunk once the cursor is past that waypoint plus
        ``preview_s`` (``missed_preview_deadline``). An on-time result
        replaces a chord already in progress: the new blend is armed
        immediately, and the arm does not finish the previous 300 ms window.
        The new chord starts from the pose being commanded, not from the raw
        waypoint under it.
        """
        with self._lock:
            if self._targets is None:
                self._install_first(targets)
                return
            observed = float(obs_time)
            deadline = observed + self._dt + self._preview_s
            if self.cursor > deadline:
                return
            chord_running = self._blend_start is not None and self.cursor >= self._blend_start
            if chord_running:
                self._arm_replacement(targets, observed)
                return
            self._replacement = None
            self._schedule(targets, observed)

    def _arm_replacement(self, targets: np.ndarray, obs_time: float) -> None:
        """Cut the current chord off at this chunk's preview time.

        Until that time the arm stays on the chord it is already following.
        The new chord's first end is that pose, so the hand does not jump
        back to a chunk that has already been left behind.
        """
        blend_start = obs_time + self._preview_s
        if blend_start < self.cursor:
            blend_start = self.cursor
        incoming = _checked_targets(targets)
        blend_end = blend_start + self._transition_s
        # Ends are trajectory-time poses. The 150 ms lead is applied later,
        # when a tick reads ``now + lookahead_s``.
        start_pose = np.asarray(self._baked_pose(blend_start), dtype=np.float32).copy()
        end_pose = _sample_waypoints(incoming, obs_time, self._dt, blend_end)
        self._replacement = (incoming, obs_time, blend_start, start_pose, end_pose)

    def _schedule(self, targets: np.ndarray, obs_time: float) -> None:
        """Arm a preview-then-chord handoff onto ``targets``.

        ``targets`` stay in the absolute frame produced by anchoring the
        chunk on the observation. The chord only covers the transition
        window; after it, playback follows those absolute waypoints.
        """
        assert self._targets is not None
        blend_start = obs_time + self._preview_s
        if blend_start < self.cursor:
            blend_start = self.cursor
        origin = float(obs_time)
        incoming = _checked_targets(targets)
        blend_end = blend_start + self._transition_s
        # Capture the start before installing this chord, so a chord already
        # armed at that timestamp stays the start pose.
        start_pose = np.asarray(self._baked_pose(blend_start), dtype=np.float32).copy()
        self._chord = (
            start_pose,
            _sample_waypoints(incoming, origin, self._dt, blend_end),
        )
        self._incoming = (incoming, origin)
        self._blend_start = blend_start
        self._replacement = None

    def take(self, count: int) -> np.ndarray:
        """The next ``count`` control ticks, each one ``1 / action_hz`` later."""
        if count < 1:
            raise ValueError("count must be >= 1")
        with self._lock:
            if self._targets is None:
                raise RuntimeError("take() before a trajectory is installed")
            rows = np.empty((count, _POSE_DIM), dtype=np.float32)
            for index in range(count):
                self.cursor += self._dt
                rows[index] = self._sample(self.cursor)
            return rows

    def peek_ahead(self) -> np.ndarray:
        """The pose the next ``take(1)`` will command, without advancing.

        The 200 Hz joint loop glides from the pose just taken to this one
        over the control period. ROS samples that same span on the wall
        clock; waiting until the next tick to start the glide leaves the
        arm one period behind.
        """
        if self._targets is None:
            raise RuntimeError("peek_ahead() before a trajectory is installed")
        return self._pose_at(self.cursor + self._dt)

    def _sample(self, now: float) -> np.ndarray:
        if self._replacement is not None and now >= self._replacement[2]:
            self._apply_replacement()
        incoming = self._incoming
        blend_start = self._blend_start
        if incoming is not None and blend_start is not None:
            blend_end = blend_start + self._transition_s
            if now >= blend_end:
                targets, origin = incoming
                self._targets = targets
                self._origin = origin
                self._incoming = None
                self._blend_start = None
                self._chord = None
        return self._pose_at(now)

    def _apply_replacement(self) -> None:
        """Install the on-time chunk over the chord that was playing."""
        replacement = self._replacement
        if replacement is None:
            return
        incoming, origin, blend_start, start_pose, end_pose = replacement
        self._replacement = None
        self._incoming = (incoming, origin)
        self._blend_start = blend_start
        self._chord = (start_pose, end_pose)

    def _pose_at(self, now: float) -> np.ndarray:
        """Command at control time ``now``: the baked trajectory 150 ms ahead."""
        return self._baked_pose(now + self._lookahead_s)

    def _baked_pose(self, timestamp: float) -> np.ndarray:
        """Pose on the trajectory at ``timestamp``, before the control lead."""
        replacement = self._replacement
        if replacement is not None and timestamp >= replacement[2]:
            incoming, origin, blend_start, start_pose, end_pose = replacement
            return self._chord_pose(timestamp, incoming, origin, blend_start, start_pose, end_pose)
        assert self._targets is not None
        incoming = self._incoming
        blend_start = self._blend_start
        if incoming is None or blend_start is None or self._chord is None:
            return _sample_waypoints(self._targets, self._origin, self._dt, timestamp)
        targets, origin = incoming
        return self._chord_pose(
            timestamp, targets, origin, blend_start, self._chord[0], self._chord[1]
        )

    def _chord_pose(
        self,
        timestamp: float,
        incoming: np.ndarray,
        origin: float,
        blend_start: float,
        start_pose: np.ndarray,
        end_pose: np.ndarray,
    ) -> np.ndarray:
        """Sample one fixed chord, then the incoming chunk after it."""
        assert self._targets is not None
        blend_end = blend_start + self._transition_s
        if timestamp >= blend_end:
            return _sample_waypoints(incoming, origin, self._dt, timestamp)
        if timestamp < blend_start:
            return _sample_waypoints(self._targets, self._origin, self._dt, timestamp)
        span = blend_end - blend_start
        alpha = 1.0 if span <= 0 else (timestamp - blend_start) / span
        return _blend_pose(start_pose, end_pose, alpha)


def _checked_targets(targets: np.ndarray) -> np.ndarray:
    array = np.asarray(targets, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != _POSE_DIM or array.shape[0] < 1:
        raise ValueError(f"targets must be (m, {_POSE_DIM}), got {array.shape}")
    return array


def _sample_waypoints(
    targets: np.ndarray,
    origin: float,
    dt: float,
    time_s: float,
) -> np.ndarray:
    first = origin + dt
    if time_s <= first:
        return np.asarray(targets[0], dtype=np.float32)
    last = origin + targets.shape[0] * dt
    if time_s >= last:
        return np.asarray(targets[-1], dtype=np.float32)
    slot = (time_s - origin) / dt
    index = int(slot) - 1
    index = min(max(index, 0), targets.shape[0] - 2)
    alpha = slot - (index + 1)
    return _blend_pose(targets[index], targets[index + 1], float(alpha))


def _blend_pose(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        return np.asarray(left, dtype=np.float32)
    if alpha >= 1.0:
        return np.asarray(right, dtype=np.float32)
    mixed = np.empty(_POSE_DIM, dtype=np.float64)
    for xyz, rot, grip in _ARMS:
        mixed[xyz] = (1.0 - alpha) * left[xyz] + alpha * right[xyz]
        start = Rotation.from_euler("xyz", rot6d_to_rpy(left[rot]))
        end = Rotation.from_euler("xyz", rot6d_to_rpy(right[rot]))
        slerped = Slerp([0.0, 1.0], Rotation.concatenate([start, end]))([alpha])
        mixed[rot] = rpy_to_rot6d(np.asarray(slerped.as_euler("xyz")[0]))
        mixed[grip] = (1.0 - alpha) * left[grip] + alpha * right[grip]
    return mixed.astype(np.float32)
