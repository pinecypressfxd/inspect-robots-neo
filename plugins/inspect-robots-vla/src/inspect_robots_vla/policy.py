"""VlaPolicy: the pure ``umi-replay`` VLA adapter over the :10055 wire.

One ``act()`` is one client round trip: the observation's camera frames (per
``submit_images``, sorted into the wire's image order), the 20-dim EE state,
and the task string go to the client's ``infer``; the returned delta chunk is
re-anchored onto that EE state as absolute targets and wrapped in an
[`ActionChunk`][inspect_robots.types.ActionChunk]. Every emitted action then
passes the rollout's approver chain like any other policy's; this module adds
no safety logic of its own (plan 0084).
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
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

from ._anchor import EEF_STATE_DIM, anchor_chunk, eef_state_to_umi_rpy_state
from ._client import VlaChunk, VlaClient, VlaServiceError
from ._config import (
    CHUNK_STEPS,
    CONTROL_HZ,
    STATE_KEY,
    SUBMIT_IMAGES,
    VLA_BASE_URL,
    VLA_POLL_INTERVAL_S,
    VLA_POLL_TIMEOUT_S,
    VLA_SUBMIT_TIMEOUT_S,
)


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
    control_hz: float = CONTROL_HZ
    timeout_s: float = VLA_SUBMIT_TIMEOUT_S
    poll_interval_s: float = VLA_POLL_INTERVAL_S
    poll_timeout_s: float = VLA_POLL_TIMEOUT_S


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
    """The pure ``umi-replay`` policy: observation in, anchored chunk out.

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
        control_hz: float = CONTROL_HZ,
        timeout_s: float = VLA_SUBMIT_TIMEOUT_S,
        poll_interval_s: float = VLA_POLL_INTERVAL_S,
        poll_timeout_s: float = VLA_POLL_TIMEOUT_S,
        client: VlaWire | None = None,
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
        for name, value in (
            ("control_hz", control_hz),
            ("timeout_s", timeout_s),
            ("poll_interval_s", poll_interval_s),
            ("poll_timeout_s", poll_timeout_s),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ConfigError(f"{name} must be a finite number > 0, got {value!r}")
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
            name="umi-replay",
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
        self.config = VlaPolicyConfig(
            action_horizon=CHUNK_STEPS,
            base_url=base_url,
            prompt=prompt,
            submit_images=self._submit_images,
            state_key=state_key,
            control_hz=self._control_hz,
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
            poll_timeout_s=poll_timeout_s,
        )
        # Seeded from the wall clock so a fresh process never reuses ids the
        # long-lived service has already answered: the straggler skip only
        # drops results below the requested id. Strictly increasing for life.
        self._request_id = time.time_ns()
        self._scene_instruction: str | None = None
        self._last_target: np.ndarray | None = None
        self._previous_attempt_failed = False
        self._state_field: StateField | None = None
        self._dim_labels: tuple[str, ...] | None = None

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

    def act(self, observation: Observation) -> ActionChunk:
        """One round trip: frames, state, and task out; anchored targets back."""
        task = self._task_string(observation)
        images = self._submit_frames(observation)
        eef_state = self._anchor_state(observation)
        self._request_id += 1
        request_id = self._request_id
        started = time.perf_counter()
        # A transient service failure earns exactly one retry, unless the
        # previous act() also failed, when the service is presumed down and
        # the failure surfaces immediately.
        attempts = 1 if self._previous_attempt_failed else 2
        chunk: VlaChunk | None = None
        failure: VlaServiceError | None = None
        for _ in range(attempts):
            try:
                chunk = self._client.infer(
                    images, eef_state_to_umi_rpy_state(eef_state), task, request_id=request_id
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
            raise PolicyError(
                f"umi-replay VLA request {request_id} {outcome}: {failure}"
            ) from failure
        self._previous_attempt_failed = False
        try:
            targets = anchor_chunk(eef_state, chunk)
        except VlaServiceError as exc:
            # A malformed chunk is bad data, not a dead service: the next
            # act() keeps its retry.
            raise PolicyError(
                f"umi-replay VLA request {request_id} returned a chunk that cannot "
                f"be anchored: {exc}"
            ) from exc
        self._last_target = targets[-1]
        return ActionChunk(
            actions=[Action(data=row) for row in targets],
            control_hz=self._control_hz,
            inference_latency_s=time.perf_counter() - started,
            meta={"request_id": request_id},
        )

    def close(self) -> None:
        """Close the owned wire client; an injected client stays the caller's."""
        if self._owned_client is not None:
            self._owned_client.close()
            self._owned_client = None

    def _task_string(self, observation: Observation) -> str:
        """The prompt first (the trained string), then live, then scene instruction."""
        for candidate in (self._prompt, observation.instruction, self._scene_instruction):
            if candidate:
                return candidate
        raise PolicyError(
            "umi-replay has no task string to submit: pass -P prompt=... or run "
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
        return {name: observation.images[name] for name in sorted(self._submit_images)}

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
