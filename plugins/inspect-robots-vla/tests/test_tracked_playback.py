"""Tracked VLA playback: 30 Hz waypoints, 5 Hz chunks, 150 ms lookahead.

Mirrors ``ros2_nero_inference.yaml`` (action_hz 30, inference_hz 5, preview
300 ms, transition 300 ms, lookahead_ms 150) without the 200 Hz joint
interpolator.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from inspect_robots import Observation
from inspect_robots.errors import ConfigError
from inspect_robots_vla._client import VlaChunk
from inspect_robots_vla._config import ACTION_DIM_VLA
from inspect_robots_vla._trajectory import TrackedPlayback
from inspect_robots_vla.policy import VlaPolicy, VlaPolicyConfig


class _Call:
    def __init__(self, task: str, request_id: int) -> None:
        self.task = task
        self.request_id = request_id


class _FakeClient:
    def __init__(self, script: list[VlaChunk]) -> None:
        self.script = script
        self.calls: list[_Call] = []

    def infer(
        self,
        images: object,
        state: np.ndarray,
        task: str,
        *,
        request_id: int,
        replay_proprioception_json: str | None = None,
        source_observation_timestamp_ns: int | None = None,
    ) -> VlaChunk:
        del images, state, replay_proprioception_json, source_observation_timestamp_ns
        self.calls.append(_Call(task, request_id))
        return self.script.pop(0)

    def close(self) -> None:
        return None


def _eef_state() -> np.ndarray:
    state = np.zeros(20)
    state[0:3] = (0.30, -0.10, 0.20)
    state[3:9] = _identity_rot6d()
    state[9] = 0.05
    state[10:13] = (0.28, 0.12, 0.19)
    state[13:19] = _identity_rot6d()
    state[19] = 0.04
    return state


def _primed_history(eef: np.ndarray, *, anchor_ns: int = 200_000_000) -> dict[str, object]:
    """A window ROS would accept: the anchor, and the pose one stride earlier."""
    older = np.array(eef, copy=True)
    return {
        "proprio_time_ns": anchor_ns,
        "proprio_history": (
            (anchor_ns - 99_999_999, older),
            (anchor_ns, np.array(eef, copy=True)),
        ),
    }


def _observation() -> Observation:
    frame = np.zeros((4, 8, 3), dtype=np.uint8)
    eef = _eef_state()
    return Observation(
        images={"left_rgbd": frame, "right_rgbd": frame, "chest_rgbd": frame},
        state={"eef_state": eef},
        extra=_primed_history(eef),
    )


class _Handle:
    def __init__(self, alive: bool) -> None:
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


def _inline(work: Callable[[], None]) -> _Handle:
    work()
    return _Handle(False)


def _hang(work: Callable[[], None]) -> _Handle:
    del work
    return _Handle(True)


def _identity_rot6d() -> np.ndarray:
    return np.concatenate(
        [
            Rotation.identity().as_matrix()[:, 0],
            Rotation.identity().as_matrix()[:, 1],
        ]
    )


def _pose(left_x: float) -> np.ndarray:
    pose = np.zeros(20, dtype=np.float64)
    pose[0] = left_x
    pose[3:9] = _identity_rot6d()
    pose[10] = 0.2
    pose[13:19] = _identity_rot6d()
    return pose


def _zeros(steps: int) -> VlaChunk:
    return VlaChunk(
        request_id=1,
        deltas=np.zeros((steps, ACTION_DIM_VLA), dtype=np.float32),
    )


def _shifted(steps: int) -> VlaChunk:
    deltas = np.zeros((steps, ACTION_DIM_VLA), dtype=np.float32)
    deltas[:, 0] = 0.05
    return VlaChunk(request_id=2, deltas=deltas)


def test_midpoint_sample_lerps_position() -> None:
    playback = TrackedPlayback(action_hz=30.0, preview_s=0.3, transition_s=0.3, lookahead_s=0.0)
    playback.install_first(np.stack([_pose(0.0), _pose(0.2)]))
    playback.cursor = playback._dt * 0.5
    sampled = playback.take(1)[0]
    assert sampled[0] == pytest.approx(0.1)


def test_first_sample_is_waypoint_zero_despite_the_lookahead() -> None:
    playback = TrackedPlayback(action_hz=30.0, preview_s=0.3, transition_s=0.3, lookahead_s=0.15)
    playback.install_first(np.stack([_pose(index * 0.01) for index in range(50)]))
    # The 150 ms lead is absorbed by the warm-start shift, so the opening
    # tick is waypoint 0. Five ticks later (150 ms) the cursor reaches waypoint 4.
    assert playback.take(1)[0][0] == pytest.approx(0.0)
    assert playback.take(4)[-1][0] == pytest.approx(0.04)


def test_lookahead_does_not_skip_ahead_on_the_first_tick() -> None:
    playback = TrackedPlayback(action_hz=30.0, preview_s=0.3, transition_s=0.3, lookahead_s=0.6)
    playback.install_first(np.stack([_pose(index * 0.01) for index in range(50)]))
    assert playback.take(1)[0][0] == pytest.approx(0.0)
    # After the reanchor, one control tick advances one waypoint.
    # Tick 19 is waypoint 18.
    assert playback.take(18)[-1][0] == pytest.approx(0.18)


def test_handoff_blends_toward_the_incoming_absolute_pose() -> None:
    playback = TrackedPlayback(action_hz=30.0, preview_s=0.0, transition_s=0.3, lookahead_s=0.0)
    playback.install_first(np.stack([_pose(0.20) for _ in range(50)]))
    playback.cursor = 0.2
    # ROS keeps the incoming chunk's absolute coordinates after the chord.
    # The chord leaves x=0.20 and approaches the new chunk (near 0, walking
    # forward), instead of shifting that chunk onto the pose already commanded.
    playback.install_next(np.stack([_pose(index * 0.01) for index in range(50)]), playback.cursor)
    samples = playback.take(10)
    assert float(samples[0][0]) < 0.20
    assert float(samples[-1][0]) < float(samples[0][0])
    assert float(samples[-1][0]) == pytest.approx(0.09, abs=0.02)


def test_lookahead_reads_the_chord_halfway_at_the_window_start() -> None:
    playback = TrackedPlayback(action_hz=30.0, preview_s=0.3, transition_s=0.3, lookahead_s=0.15)
    playback.install_first(np.stack([_pose(0.0) for _ in range(50)]))
    playback.install_next(np.stack([_pose(0.20) for _ in range(50)]), 0.0)
    # The chord is [0.30, 0.60] from x=0 to x=0.20. ROS samples now+150 ms,
    # so the tick at the window start is already the midpoint.
    playback.cursor = 0.30 - playback._dt
    assert float(playback.take(1)[0][0]) == pytest.approx(0.10)
    # One lead before the window, the sample lands on the chord start.
    playback.cursor = 0.15 - playback._dt
    assert float(playback.take(1)[0][0]) == pytest.approx(0.0, abs=0.02)


def test_on_time_chunk_replaces_a_chord_without_kicking_back() -> None:
    playback = TrackedPlayback(action_hz=30.0, preview_s=0.0, transition_s=0.3, lookahead_s=0.1)
    playback.install_first(np.stack([_pose(0.0) for _ in range(4)]))
    playback.cursor = 0.2
    ramp = np.stack([_pose(0.05 + index * 0.01) for index in range(50)])
    playback.install_next(ramp, playback.cursor)
    moved = float(playback.take(2)[-1][0])
    assert moved > 0.0

    playback.install_next(ramp, playback.cursor)
    # Preview has already elapsed (it is zero), so this result is on time and
    # the chord is replaced now. The new chord starts at the pose just
    # commanded, not at the held tail of the four-pose chunk.
    followed = float(playback.take(1)[0][0])
    assert followed >= moved - 1e-4
    assert followed == pytest.approx(moved, abs=0.02)


def test_on_time_chunk_cuts_off_the_previous_transition() -> None:
    playback = TrackedPlayback(action_hz=30.0, preview_s=0.3, transition_s=0.3, lookahead_s=0.0)
    playback.install_first(np.stack([_pose(0.0) for _ in range(50)]))
    playback.install_next(np.stack([_pose(0.20) for _ in range(50)]), 0.0)
    # First chord occupies [0.30, 0.60]. Step to 0.40, inside it.
    playback.cursor = 0.40 - playback._dt
    inside = float(playback.take(1)[0][0])
    assert 0.0 < inside < 0.20

    # Observed at 0.15, so the preview deadline is 0.45, before the old
    # chord would have finished at 0.60. ROS arms that blend now.
    playback.install_next(np.stack([_pose(0.0) for _ in range(50)]), 0.15)
    playback.cursor = 0.45 - playback._dt
    at_switch = float(playback.take(1)[0][0])
    # Up to the deadline the arm is still on the old chord, so it has moved
    # closer to 0.20. After the deadline the new chunk, anchored at 0, pulls
    # it back. Waiting for the old window would still be climbing here.
    assert at_switch > inside
    later = float(playback.take(3)[-1][0])
    assert later < at_switch


class _RecordingClient(_FakeClient):
    def __init__(self, script: list[VlaChunk]) -> None:
        super().__init__(script)
        self.states: list[np.ndarray] = []

    def infer(
        self,
        images: object,
        state: np.ndarray,
        task: str,
        *,
        request_id: int,
        replay_proprioception_json: str | None = None,
        source_observation_timestamp_ns: int | None = None,
    ) -> VlaChunk:
        self.states.append(np.asarray(state, dtype=np.float32).copy())
        return super().infer(
            images,
            state,
            task,
            request_id=request_id,
            replay_proprioception_json=replay_proprioception_json,
            source_observation_timestamp_ns=source_observation_timestamp_ns,
        )


def _observation_at(left_x: float, *samples: tuple[int, float]) -> Observation:
    frame = np.zeros((4, 8, 3), dtype=np.uint8)
    history: list[tuple[int, np.ndarray]] = []
    for stamp, value in samples:
        state = _eef_state()
        state[0] = value
        history.append((stamp, state))
    current = history[-1][1]
    return Observation(
        images={"left_rgbd": frame, "right_rgbd": frame, "chest_rgbd": frame},
        state={"eef_state": np.array(current, copy=True)},
        extra={"proprio_time_ns": history[-1][0], "proprio_history": tuple(history)},
    )


def test_advance_samples_the_waypoint_path_every_5ms() -> None:
    """A 5 ms step reads the baked trajectory, not the chord between 30 Hz ends."""
    playback = TrackedPlayback(action_hz=30.0, preview_s=0.3, transition_s=0.3, lookahead_s=0.0)
    playback.install_first(np.stack([_pose(index * 0.3) for index in range(8)]))
    # Waypoint 0 is due at 1/30 s. Step to 40 ms, inside the next segment.
    for _ in range(8):
        playback.advance(0.005)
    landed = float(playback.advance(0.005)[0])
    assert 0.0 < landed < 0.3


def test_late_chunk_is_dropped() -> None:
    """ROS rejects a chunk once receive time passes the first waypoint plus preview."""
    playback = TrackedPlayback(action_hz=30.0, preview_s=0.3, transition_s=0.3, lookahead_s=0.0)
    playback.install_first(np.stack([_pose(0.0) for _ in range(50)]))
    deadline = playback._dt + 0.3
    playback.cursor = deadline + 1e-4
    playback.install_next(np.stack([_pose(0.5) for _ in range(50)]), 0.0)
    assert playback._incoming is None
    assert float(playback.take(1)[0][0]) == pytest.approx(0.0)


def test_tracked_state_history_samples_100ms_before_the_anchor() -> None:
    """The older frame is the measured pose 100 ms before this observation.

    A window that does not reach that far, or whose nearest sample is more
    than one action step away, skips the request. The arm holds the measured
    pose instead of sending the current pose twice.
    """
    fake = _RecordingClient([_zeros(8), _zeros(8)])
    policy = VlaPolicy(
        "http://127.0.0.1:10055",
        prompt="t",
        client=fake,
        control_hz=30.0,
        inference_hz=5.0,
        schedule="tracked",
        prefetch_spawn=_inline,
    )
    held = policy.act(_observation_at(0.10, (0, 0.10)))
    assert fake.calls == []
    assert float(np.asarray(held.actions[0].data)[0]) == pytest.approx(0.10)

    anchor = 200_000_000
    policy.act(_observation_at(0.05, (0, 0.0), (anchor, 0.05)))
    assert fake.calls == []

    previous_ns = anchor - 99_999_999
    first_chunk = policy.act(_observation_at(0.07, (previous_ns, 0.02), (anchor, 0.07)))
    assert len(fake.states) == 1
    first = fake.states[0]
    assert first.shape == (28,)
    assert first[0] == pytest.approx(0.02)
    assert first[14] == pytest.approx(0.07)

    later = anchor + 200_000_000
    between = later - 50_000_000
    older = later - 99_999_999 - 20_000_000
    # The arm clock, not the policy call, opens the next 5 Hz request.
    sample = first_chunk.actions[0].meta["trajectory_sample"]
    sample(0.2)
    policy.act(_observation_at(0.09, (older, 0.01), (between, 0.04), (later, 0.09)))
    assert len(fake.states) == 2
    second = fake.states[1]
    assert second[0] == pytest.approx(0.01)
    assert second[14] == pytest.approx(0.09)


def test_tracked_act_keeps_moving_while_the_next_chunk_is_in_flight() -> None:
    fake = _FakeClient([_zeros(8), _shifted(8), _shifted(8)])
    policy = VlaPolicy(
        "http://127.0.0.1:10055",
        prompt="put the cup on the pad",
        client=fake,
        control_hz=30.0,
        inference_hz=5.0,
        schedule="tracked",
        prefetch_spawn=_inline,
    )
    assert policy.config.replan_interval == 1
    config = policy.config
    assert isinstance(config, VlaPolicyConfig)
    assert config.lookahead_s == pytest.approx(0.15)
    assert config.preview_s == pytest.approx(0.3)

    first = policy.act(_observation())
    assert len(first) == 1
    assert len(fake.calls) == 1
    sample = first.actions[0].meta["trajectory_sample"]
    held_x = float(np.asarray(first.actions[0].data)[0])
    # Six 30 Hz steps is 200 ms. The policy call itself does not move the clock.
    for _ in range(6):
        sample(1.0 / 30.0)
    assert len(fake.calls) == 1

    second = policy.act(_observation())
    assert len(fake.calls) == 2
    assert np.asarray(second.actions[0].data)[0] == pytest.approx(held_x)
    # The request is installed on the next policy turn. Sampling alone keeps
    # the arm on the chunk already playing.
    policy.act(_observation())

    xs = [float(sample(1.0 / 30.0)[0]) for _ in range(8)]
    # The preview opens at 300 ms, but the 150 ms lead reads into that chord
    # before the control clock reaches it. The first of these ticks is still
    # on the old chunk; the last has already started toward the new one.
    assert xs[0] == pytest.approx(held_x)
    assert xs[-1] > xs[0]


def test_tracked_act_does_not_block_on_an_unfinished_request() -> None:
    fake = _FakeClient([_zeros(4)])
    policy = VlaPolicy(
        "http://127.0.0.1:10055",
        prompt="t",
        client=fake,
        control_hz=30.0,
        inference_hz=5.0,
        schedule="tracked",
        prefetch_spawn=_hang,
    )
    policy.act(_observation())
    policy.act(_observation())
    assert len(fake.calls) == 1


def test_unknown_schedule_is_rejected() -> None:
    with pytest.raises(ConfigError, match="schedule"):
        VlaPolicy("http://127.0.0.1:10055", prompt="t", schedule="open")
