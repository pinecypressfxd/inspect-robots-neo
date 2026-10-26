"""VlaPolicy: the pure ``vla`` VLA adapter over the :10055 wire.

One ``act()`` is one client round trip: the observation's camera frames (per
``submit_images``, sorted into the wire's image order), the 20-dim EE state,
and the task string go to the client's ``infer``; the returned delta chunk is
re-anchored onto that EE state as absolute targets and wrapped in an
[`ActionChunk`][inspect_robots.types.ActionChunk]. Every emitted action then
passes the rollout's approver chain like any other policy's; this module adds
no safety logic of its own (plan 0084).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from inspect_robots import (
    Action,
    ActionChunk,
    ActionSemantics,
    Box,
    EmbodimentInfo,
    Observation,
    ObservationSpace,
    PolicyBase,
    PolicyConfig,
    PolicyInfo,
    Scene,
    StateField,
)
from inspect_robots.errors import ConfigError, PolicyError

from ._anchor import (
    EEF_STATE_DIM,
    anchor_chunk,
    eef_state_to_umi_rpy_state,
    umi_last_action_state,
)
from ._client import VlaChunk, VlaClient, VlaServiceError
from ._config import (
    CHUNK_STEPS,
    STATE_KEY,
    SUBMIT_IMAGES,
    VLA_BASE_URL,
    VLA_CONTROL_HZ,
    VLA_INFERENCE_HZ,
    VLA_LOOKAHEAD_S,
    VLA_POLL_INTERVAL_S,
    VLA_POLL_TIMEOUT_S,
    VLA_PREVIEW_S,
    VLA_SUBMIT_TIMEOUT_S,
    VLA_TRANSITION_S,
)
from ._history import sample_last_action_window
from ._replay import replay_proprioception_json
from ._trajectory import TrackedPlayback

_SCHEDULES = ("chunk", "tracked")


class _PrefetchHandle(Protocol):
    """The bit of a worker thread ``act()`` waits on: still running, or not."""

    def is_alive(self) -> bool:
        """False once the worker has stored its result or its error."""
        ...


_PrefetchResult = tuple[np.ndarray, int, float, float, np.ndarray]


class VlaWire(Protocol):
    """The client seam ``VlaPolicy`` needs; ``VlaClient`` satisfies it.

    The Protocol (not the concrete client) is the constructor's type so tests
    and the hybrid policy can inject a scripted fake without inheritance.
    """

    def infer(
        self,
        images: Mapping[str, np.ndarray] | None,
        state: np.ndarray,
        task: str,
        *,
        request_id: int,
        replay_proprioception_json: str | None = None,
        source_observation_timestamp_ns: int | None = None,
    ) -> VlaChunk:
        """Submit one observation and block until its chunk arrives."""
        ...

    def close(self) -> None:
        """Release the underlying connections."""
        ...


@dataclass(frozen=True)
class VlaPolicyConfig(PolicyConfig):
    """Inference-time configuration recorded in the eval log.

    Extends the core ``PolicyConfig``; ``eval()`` serializes configs with
    ``dataclasses.asdict``, so these fields land in ``EvalSpec.policy_config``.
    """

    base_url: str = VLA_BASE_URL
    prompt: str | None = None
    submit_images: tuple[str, ...] = SUBMIT_IMAGES
    state_key: str = STATE_KEY
    control_hz: float = VLA_CONTROL_HZ
    timeout_s: float = VLA_SUBMIT_TIMEOUT_S
    poll_interval_s: float = VLA_POLL_INTERVAL_S
    poll_timeout_s: float = VLA_POLL_TIMEOUT_S
    schedule: str = "chunk"
    inference_hz: float = VLA_INFERENCE_HZ
    preview_s: float = VLA_PREVIEW_S
    transition_s: float = VLA_TRANSITION_S
    lookahead_s: float = VLA_LOOKAHEAD_S


def _parse_submit_images(value: Sequence[str] | str) -> tuple[str, ...]:
    """Normalize ``submit_images`` (tuple or the ``-P`` comma string) to names.

    The wire carries 2..3 frames; any other count is a checkpoint-config
    mismatch, so it fails at construction rather than mid-rollout.
    """
    names = (
        tuple(item.strip() for item in value.split(",")) if isinstance(value, str) else tuple(value)
    )
    if not all(names):
        raise ConfigError(
            f"submit_images entries must be non-empty camera names, got {list(names)}.\n"
            "fix: pass -P submit_images=left_rgbd,right_rgbd,chest_rgbd"
        )
    if len(names) != len(set(names)):
        raise ConfigError(
            f"submit_images holds duplicate camera names, got {list(names)}.\n"
            "fix: list each camera once"
        )
    if not 2 <= len(names) <= 3:
        raise ConfigError(
            f"submit_images must name 2..3 cameras (the checkpoint's image count), "
            f"got {len(names)}: {list(names)}.\n"
            "fix: pass -P submit_images=left_rgbd,right_rgbd,chest_rgbd"
        )
    return names


class VlaPolicy(PolicyBase):
    """The pure ``vla`` policy: observation in, anchored chunk out.

    The task string sent to the service is the configured ``prompt`` (the
    trained task string, sent verbatim), falling back to the observation's
    live instruction, then to the scene instruction recorded at ``reset()``.
    The anchor state is ``observation.state[state_key]``; when the embodiment
    omits that key, the last commanded target of the previous chunk stands in
    (assuming that chunk played to its end). A transient service failure earns
    exactly one retry, but a failure on the heels of a failed attempt surfaces
    immediately. Construction never touches the network, so
    ``inspect-robots list policies`` works with no service running.
    """

    def __init__(
        self,
        base_url: str = VLA_BASE_URL,
        *,
        prompt: str | None = None,
        submit_images: Sequence[str] | str = SUBMIT_IMAGES,
        state_key: str = STATE_KEY,
        control_hz: float = VLA_CONTROL_HZ,
        timeout_s: float = VLA_SUBMIT_TIMEOUT_S,
        poll_interval_s: float = VLA_POLL_INTERVAL_S,
        poll_timeout_s: float = VLA_POLL_TIMEOUT_S,
        schedule: str = "chunk",
        inference_hz: float = VLA_INFERENCE_HZ,
        preview_s: float = VLA_PREVIEW_S,
        transition_s: float = VLA_TRANSITION_S,
        lookahead_s: float = VLA_LOOKAHEAD_S,
        trace_dir: str | None = None,
        replay: bool = False,
        client: VlaWire | None = None,
        prefetch_spawn: Callable[[Callable[[], None]], _PrefetchHandle] | None = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise ConfigError(
                f"base_url must be an http:// or https:// URL, got {base_url!r}.\n"
                "fix: pass -P base_url=http://127.0.0.1:10055"
            )
        if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
            raise ConfigError(
                f"prompt must be a non-empty string or None, got {prompt!r}.\n"
                'fix: pass -P prompt="the trained task string", or omit -P prompt= '
                "to fall back to the scene instruction"
            )
        if not isinstance(state_key, str) or not state_key:
            raise ConfigError(f"state_key must be a non-empty string, got {state_key!r}")
        if schedule not in _SCHEDULES:
            raise ConfigError(
                f"schedule must be one of {list(_SCHEDULES)}, got {schedule!r}.\n"
                "fix: pass -P schedule=tracked to follow the ROS inference consumer, "
                "or -P schedule=chunk to play each chunk open-loop"
            )
        for name, value in (
            ("control_hz", control_hz),
            ("timeout_s", timeout_s),
            ("poll_interval_s", poll_interval_s),
            ("poll_timeout_s", poll_timeout_s),
            ("inference_hz", inference_hz),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ConfigError(f"{name} must be a finite number > 0, got {value!r}")
        for name, value in (
            ("preview_s", preview_s),
            ("transition_s", transition_s),
            ("lookahead_s", lookahead_s),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not np.isfinite(value)
                or value < 0
            ):
                raise ConfigError(f"{name} must be a finite number >= 0, got {value!r}")
        steps_per_infer = round(float(control_hz) / float(inference_hz))
        if steps_per_infer < 1:
            raise ConfigError(
                f"control_hz={control_hz} is slower than inference_hz={inference_hz}; "
                "the tracked schedule needs at least one control step per inference"
            )
        if not isinstance(replay, bool):
            raise ConfigError(
                f"replay must be true or false, got {replay!r}.\n"
                "fix: pass -P replay=true to consume the UMI replay service"
            )
        if trace_dir is not None and (not isinstance(trace_dir, str) or not trace_dir.strip()):
            raise ConfigError(
                f"trace_dir must be a non-empty path or omitted, got {trace_dir!r}.\n"
                "fix: pass -P trace_dir=/path/to/traces"
            )
        self._trace_dir = trace_dir
        self._replay = replay
        self._submit_images = _parse_submit_images(submit_images)
        self._state_key = state_key
        self._control_hz = float(control_hz)
        self._prompt = prompt
        owned: VlaClient | None = None
        if client is None:
            owned = VlaClient(
                base_url,
                timeout_s=timeout_s,
                poll_interval_s=poll_interval_s,
                poll_timeout_s=poll_timeout_s,
            )
            self._client: VlaWire = owned
        else:
            self._client = client
        self._owned_client = owned
        self.info = PolicyInfo(
            name="vla",
            action_space=Box(
                shape=(EEF_STATE_DIM,),
                semantics=ActionSemantics(
                    control_mode="eef_abs_pose",
                    rotation_repr="rot6d",
                    gripper="continuous",
                    frame="base",
                ),
            ),
            observation_space=ObservationSpace(state_keys=frozenset({state_key})),
            control_hz=self._control_hz,
        )
        self._schedule = schedule
        self._steps_per_infer = steps_per_infer
        self._inference_hz = float(inference_hz)
        self._preview_s = float(preview_s)
        self._transition_s = float(transition_s)
        self._lookahead_s = float(lookahead_s)
        self._prefetch_spawn = prefetch_spawn or _spawn_prefetch
        self.config = VlaPolicyConfig(
            # Tracked mode is called once per control tick so the 100 ms state
            # history is actually observed. Inference still starts every
            # ``steps_per_infer`` ticks.
            action_horizon=1 if schedule == "tracked" else CHUNK_STEPS,
            replan_interval=1 if schedule == "tracked" else None,
            base_url=base_url,
            prompt=prompt,
            submit_images=self._submit_images,
            state_key=state_key,
            control_hz=self._control_hz,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
            poll_timeout_s=poll_timeout_s,
            schedule=schedule,
            inference_hz=float(inference_hz),
            preview_s=self._preview_s,
            transition_s=self._transition_s,
            lookahead_s=self._lookahead_s,
        )
        # Seeded from the wall clock so a fresh process never reuses ids the
        # long-lived service has already answered: the straggler skip only
        # drops results below the requested id. Strictly increasing for life.
        self._request_id = time.time_ns()
        self._scene_instruction: str | None = None
        self._last_target: np.ndarray | None = None
        self._prev_umi_state: np.ndarray | None = None
        self._infer_cursor = 0.0
        self._previous_attempt_failed = False
        self._state_field: StateField | None = None
        self._dim_labels: tuple[str, ...] | None = None
        self._playback: TrackedPlayback | None = None
        self._prefetch_lock = threading.Lock()
        self._prefetch_generation = 0
        self._prefetch_handle: _PrefetchHandle | None = None
        self._prefetch_result: _PrefetchResult | None = None
        self._prefetch_error: PolicyError | None = None

    @property
    def state_field(self) -> StateField | None:
        """The bound embodiment's declared field for ``state_key``, if any."""
        return self._state_field

    @property
    def dim_labels(self) -> tuple[str, ...] | None:
        """The bound embodiment's action dimension labels, if declared."""
        return self._dim_labels

    def bind(self, embodiment_info: EmbodimentInfo) -> None:
        """Record the embodiment's EE state field and action dim labels.

        The policy keeps its own declared spaces (they are what compatibility
        checks); only the state field and labels are recorded, for diagnostics
        and the hybrid layer that reuses this adapter.
        """
        spec = embodiment_info.observation_space.state
        self._state_field = (
            None
            if spec is None
            else next((field for field in spec.fields if field.key == self._state_key), None)
        )
        semantics = embodiment_info.action_space.semantics
        self._dim_labels = None if semantics is None else semantics.dim_labels

    def reset(self, scene: Scene) -> None:
        """Capture the scene instruction and clear per-trial tracking state.

        ``_request_id`` is deliberately not reset: ids must never repeat for a
        long-lived service, or a straggler result from an earlier trial would
        be mistaken for this request's answer.
        """
        self._scene_instruction = scene.instruction or None
        self._last_target = None
        self._prev_umi_state = None
        self._infer_cursor = 0.0
        self._drop_prefetch()

    def act(self, observation: Observation) -> ActionChunk:
        """One round trip, or one tracked control window.

        ``schedule=chunk`` blocks on the service and returns the whole anchored
        chunk. ``schedule=tracked`` blocks only for the first chunk, then
        returns the next ``control_hz / inference_hz`` ticks of the trajectory
        while the following chunk is requested in the background.
        """
        if self._schedule == "tracked":
            return self._act_tracked(observation)
        return self._act_chunk(observation)

    def close(self) -> None:
        """Close the owned wire client; an injected client stays the caller's."""
        self._drop_prefetch()
        if self._owned_client is not None:
            self._owned_client.close()
            self._owned_client = None

    def _act_chunk(self, observation: Observation) -> ActionChunk:
        """Block for one chunk and return every anchored target."""
        targets, request_id, latency, _submitted = self._infer_targets(observation)
        self._last_target = targets[-1]
        return ActionChunk(
            actions=[Action(data=row) for row in targets],
            control_hz=self._control_hz,
            inference_latency_s=latency,
            meta={"request_id": request_id},
        )

    def _act_tracked(self, observation: Observation) -> ActionChunk:
        """Hand the arm a 5 ms trajectory sampler and request chunks at 5 Hz.

        The returned action does not advance the playback clock. The embodiment
        calls ``trajectory_sample`` from its 200 Hz loop, so camera alignment
        on this thread does not pause the arm or replace the path with a chord.
        """
        history = self._history_window(observation)
        request_id, latency = self._drain_prefetch()
        if self._playback is None:
            if history is None:
                return self._hold_pose(observation)
            targets, request_id, latency, _submitted = self._infer_targets(
                observation, history=history
            )
            self._playback = TrackedPlayback(
                action_hz=self._control_hz,
                preview_s=self._preview_s,
                transition_s=self._transition_s,
                lookahead_s=self._lookahead_s,
            )
            self._playback.install_first(targets)
            self._infer_cursor = self._playback.cursor_time()
        elif (
            self._prefetch_handle is None
            and history is not None
            and self._playback.cursor_time() - self._infer_cursor + 1e-9 >= 1.0 / self._inference_hz
        ):
            self._start_prefetch(observation, history=history)
            self._infer_cursor = self._playback.cursor_time()
        assert self._playback is not None
        pose = self._pose_at_cursor()
        self._last_target = pose
        return ActionChunk(
            actions=[Action(data=pose, meta={"trajectory_sample": self._sample_trajectory})],
            control_hz=self._control_hz,
            inference_latency_s=latency,
            meta={"request_id": request_id},
        )

    def _pose_at_cursor(self) -> np.ndarray:
        """The pose the arm would command now, without moving the clock."""
        assert self._playback is not None
        playback = self._playback
        with playback._lock:
            return np.asarray(playback._pose_at(playback.cursor), dtype=np.float32).copy()

    def _sample_trajectory(self, dt: float) -> np.ndarray:
        """One 200 Hz read of the baked trajectory. Called off the policy thread."""
        assert self._playback is not None
        return self._playback.advance(dt)

    def _hold_pose(self, observation: Observation) -> ActionChunk:
        """Hold the measured pose while history cannot fill a model request."""
        pose = np.asarray(self._anchor_state(observation), dtype=np.float32)
        return ActionChunk(
            actions=[Action(data=pose, meta={"servo_next": pose})],
            control_hz=self._control_hz,
        )

    def _history_window(self, observation: Observation) -> tuple[np.ndarray, np.ndarray] | None:
        """Previous and current 14-dim frames, or None when ROS would skip.

        The embodiment attaches 200 Hz measured poses on ``proprio_history``.
        The anchor is that observation's proprio timestamp. The older frame is
        100 ms earlier, within one 30 Hz step. A short or gapped window skips
        the request.
        """
        extra = observation.extra
        raw = extra.get("proprio_history") if extra else None
        anchor = extra.get("proprio_time_ns") if extra else None
        if not isinstance(raw, tuple) or not isinstance(anchor, int) or not raw:
            return None
        samples: list[tuple[int, np.ndarray]] = []
        for stamp, pose in raw:
            samples.append(
                (int(stamp), eef_state_to_umi_rpy_state(np.asarray(pose, dtype=np.float64)))
            )
        window = sample_last_action_window(samples, anchor_ns=anchor)
        if window is None:
            return None
        previous, current = window
        return previous, current

    def _submit_state(self, eef_state: np.ndarray) -> np.ndarray:
        """28-dim last-action state for open-loop chunk mode.

        Tracked mode samples the embodiment history instead. Chunk mode still
        sends the previous request's pose, then the current one.
        """
        current = eef_state_to_umi_rpy_state(eef_state)
        submitted = umi_last_action_state(current, self._prev_umi_state)
        self._prev_umi_state = current
        return submitted

    def _infer_targets(
        self,
        observation: Observation,
        *,
        history: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, int, float, np.ndarray]:
        """Block for one anchored chunk. The last item is the submitted UMI state."""
        task = self._task_string(observation)
        images = {
            name: np.array(frame, copy=True)
            for name, frame in self._submit_frames(observation).items()
        }
        eef_state = np.array(self._anchor_state(observation), copy=True)
        if history is None:
            submitted = self._submit_state(eef_state)
        else:
            previous, current = history
            submitted = umi_last_action_state(current, previous)
        self._request_id += 1
        request_id = self._request_id
        started = time.perf_counter()
        replay_json, source_ns = self._replay_submit(observation)
        targets = self._request_chunk(
            images,
            eef_state,
            submitted,
            task,
            request_id,
            replay_json=replay_json,
            source_ns=source_ns,
        )
        latency = time.perf_counter() - started
        if self._trace_dir is not None:
            from inspect_robots_vla._trace import append_inference

            append_inference(
                self._trace_dir,
                request_id,
                latency,
                poses=np.asarray(targets, dtype=np.float64).tolist(),
            )
        return targets, request_id, latency, submitted

    def _request_chunk(
        self,
        images: Mapping[str, np.ndarray],
        eef_state: np.ndarray,
        submitted: np.ndarray,
        task: str,
        request_id: int,
        *,
        replay_json: str | None = None,
        source_ns: int | None = None,
    ) -> np.ndarray:
        """One client round trip plus anchor, with the existing single retry."""
        attempts = 1 if self._previous_attempt_failed else 2
        chunk: VlaChunk | None = None
        failure: VlaServiceError | None = None
        for _ in range(attempts):
            try:
                chunk = self._client.infer(
                    images,
                    submitted,
                    task,
                    request_id=request_id,
                    replay_proprioception_json=replay_json,
                    source_observation_timestamp_ns=source_ns,
                )
                break
            except VlaServiceError as exc:
                failure = exc
        if chunk is None:
            self._previous_attempt_failed = True
            assert failure is not None
            outcome = (
                "failed twice (one retry)"
                if attempts == 2
                else "failed; the previous attempt also failed, so no retry"
            )
            raise PolicyError(f"vla VLA request {request_id} {outcome}: {failure}") from failure
        self._previous_attempt_failed = False
        try:
            return anchor_chunk(eef_state, chunk)
        except VlaServiceError as exc:
            # A malformed chunk is bad data, not a dead service: the next
            # act() keeps its retry.
            raise PolicyError(
                f"vla VLA request {request_id} returned a chunk that cannot be anchored: {exc}"
            ) from exc

    def _replay_submit(self, observation: Observation) -> tuple[str | None, int | None]:
        """ROS ``umi_replay_http`` fields. Live vla-only leaves both empty."""
        if not self._replay:
            return None, None
        eef_state = np.asarray(observation.state[self._state_key], dtype=np.float64)
        joint_pos = np.asarray(observation.state.get("joint_pos", ()), dtype=np.float64)
        stamp = observation.extra.get("proprio_time_ns")
        source_ns = int(stamp) if isinstance(stamp, int) else time.monotonic_ns()
        try:
            payload = replay_proprioception_json(eef_state, joint_pos, timestamp_ns=source_ns)
        except ValueError as exc:
            raise PolicyError(f"vla replay proprioception cannot be built: {exc}") from exc
        return payload, source_ns

    def _start_prefetch(
        self, observation: Observation, *, history: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Request the next chunk without blocking the control ticks."""
        if self._playback is None:
            return
        task = self._task_string(observation)
        images = {
            name: np.array(frame, copy=True)
            for name, frame in self._submit_frames(observation).items()
        }
        eef_state = np.array(self._anchor_state(observation), copy=True)
        previous, current = history
        submitted = umi_last_action_state(current, previous)
        self._request_id += 1
        request_id = self._request_id
        obs_time = self._playback.cursor
        generation = self._prefetch_generation
        replay_json, source_ns = self._replay_submit(observation)

        def run() -> None:
            started = time.perf_counter()
            try:
                targets = self._request_chunk(
                    images,
                    eef_state,
                    submitted,
                    task,
                    request_id,
                    replay_json=replay_json,
                    source_ns=source_ns,
                )
            except PolicyError as exc:
                with self._prefetch_lock:
                    if generation == self._prefetch_generation:
                        self._prefetch_error = exc
                return
            latency = time.perf_counter() - started
            with self._prefetch_lock:
                if generation != self._prefetch_generation:
                    return
                self._prefetch_result = (targets, request_id, latency, obs_time, eef_state)

        self._prefetch_handle = self._prefetch_spawn(run)

    def _drain_prefetch(self) -> tuple[int | None, float | None]:
        """Install a finished background chunk. Raise if that request failed."""
        handle = self._prefetch_handle
        if handle is None or handle.is_alive():
            return None, None
        self._prefetch_handle = None
        with self._prefetch_lock:
            error = self._prefetch_error
            result = self._prefetch_result
            self._prefetch_error = None
            self._prefetch_result = None
        if error is not None:
            raise error
        if result is None or self._playback is None:
            return None, None
        targets, request_id, latency, obs_time, _eef_state = result
        self._playback.install_next(targets, obs_time)
        return request_id, latency

    def _drop_prefetch(self) -> None:
        """Forget any in-flight chunk. A late result from it is ignored."""
        with self._prefetch_lock:
            self._prefetch_generation += 1
            self._prefetch_result = None
            self._prefetch_error = None
        self._prefetch_handle = None
        self._playback = None

    def _task_string(self, observation: Observation) -> str:
        """The prompt first (the trained string), then live, then scene instruction."""
        for candidate in (self._prompt, observation.instruction, self._scene_instruction):
            if candidate:
                return candidate
        raise PolicyError(
            "vla has no task string to submit: pass -P prompt=... or run "
            "scenes that carry instructions"
        )

    def _submit_frames(self, observation: Observation) -> dict[str, np.ndarray]:
        """The submit_images frames in sorted key order (the wire's image order)."""
        missing = [name for name in self._submit_images if name not in observation.images]
        if missing:
            raise PolicyError(
                f"observation lacks camera(s) {missing} named by submit_images; "
                f"available: {sorted(observation.images)}.\n"
                "fix: point -P submit_images= at cameras the embodiment provides"
            )
        # Declared order is the wire's slot order (left, right, chest):
        # the service maps imageN positionally, never alphabetically.
        return {name: observation.images[name] for name in self._submit_images}

    def _anchor_state(self, observation: Observation) -> np.ndarray:
        """The 20-dim EE anchor: observed, else the last commanded target."""
        observed = observation.state.get(self._state_key)
        if observed is not None:
            return np.asarray(observed)
        if self._last_target is None:
            raise PolicyError(
                f"observation has no state[{self._state_key!r}] to anchor on, and no "
                "previous chunk supplies a fallback target.\n"
                "fix: pair this policy with an embodiment that provides the key, or "
                "pass -P state_key=<its 20-dim EE state key>"
            )
        return self._last_target


def _spawn_prefetch(work: Callable[[], None]) -> _PrefetchHandle:
    """Run one chunk request beside the control loop."""
    thread = threading.Thread(target=work, name="vla-prefetch", daemon=True)
    thread.start()
    return thread
