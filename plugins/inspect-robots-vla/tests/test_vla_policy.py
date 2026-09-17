"""VlaPolicy tests: the pure ``umi-replay`` adapter over an injected fake client.

The fake speaks only the seam the policy uses (``infer``/``close``), so these
tests pin the adapter's contract: task/state/image submission, anchored
chunk shape, the retry-then-PolicyError path, and the immediate failure after
a consecutive failure. Wire bytes are test_client.py's job.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from inspect_robots import (
    ActionSemantics,
    Box,
    EmbodimentInfo,
    Observation,
    ObservationSpace,
    Policy,
    Scene,
    StateField,
    StateSpec,
)
from inspect_robots.errors import ConfigError, PolicyError
from inspect_robots_vla import VlaPolicy, umi_replay
from inspect_robots_vla._anchor import anchor_chunk
from inspect_robots_vla._client import VlaChunk, VlaServiceError
from inspect_robots_vla._config import (
    ACTION_DIM_VLA,
    CHUNK_STEPS,
    SUBMIT_IMAGES,
    VLA_BASE_URL,
)
from inspect_robots_vla.policy import VlaPolicyConfig

_LEFT_RPY = (0.10, -0.20, 0.30)
_RIGHT_RPY = (-0.40, 0.25, -0.15)


@dataclasses.dataclass
class _Call:
    """One infer() round trip exactly as the policy issued it."""

    images: dict[str, np.ndarray]
    state: np.ndarray
    task: str
    request_id: int


class _FakeClient:
    """Scripted stand-in for VlaClient: each infer() pops the next scripted step."""

    def __init__(self, script: Sequence[VlaChunk | Exception]) -> None:
        self.script = list(script)
        self.calls: list[_Call] = []
        self.closed = False

    def infer(
        self,
        images: Mapping[str, np.ndarray] | None,
        state: np.ndarray,
        task: str,
        *,
        request_id: int,
    ) -> VlaChunk:
        self.calls.append(
            _Call(
                images={} if images is None else dict(images),
                state=np.array(state),
                task=task,
                request_id=request_id,
            )
        )
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    def close(self) -> None:
        self.closed = True


def _columns6(matrix: np.ndarray) -> np.ndarray:
    """Stack the first two rotation-matrix columns: the Zhou 6D form."""
    return np.concatenate([matrix[:, 0], matrix[:, 1]])


def _eef_state() -> np.ndarray:
    """A plausible dual-arm EE state in the nero 20-dim layout (float64)."""
    state = np.zeros(20)
    state[0:3] = (0.30, -0.10, 0.20)
    state[3:9] = _columns6(Rotation.from_euler("xyz", _LEFT_RPY).as_matrix())
    state[9] = 0.05
    state[10:13] = (0.28, 0.12, 0.19)
    state[13:19] = _columns6(Rotation.from_euler("xyz", _RIGHT_RPY).as_matrix())
    state[19] = 0.04
    return state


def _hwc(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(4, 8, 3), dtype=np.uint8)


_DEFAULT_FRAMES = {"left_rgbd": _hwc(1), "right_rgbd": _hwc(2), "chest_rgbd": _hwc(3)}


def _chunk(deltas: np.ndarray, request_id: int = 1) -> VlaChunk:
    return VlaChunk(request_id=request_id, deltas=np.asarray(deltas, dtype=np.float32))


def _zero_chunk(request_id: int = 1) -> VlaChunk:
    return _chunk(np.zeros((2, ACTION_DIM_VLA)), request_id=request_id)


def _scripted_deltas() -> np.ndarray:
    """(3, 14) deltas with hand-checkable xyz cumsums and absolute grippers."""
    deltas = np.zeros((3, ACTION_DIM_VLA), dtype=np.float32)
    deltas[:, 0] = (0.01, 0.01, 0.01)  # left x per-step delta
    deltas[:, 6] = (0.02, 0.03, 0.04)  # left gripper absolute
    deltas[:, 7] = (-0.02, 0.0, -0.02)  # right x per-step delta
    return deltas


def _service_error() -> VlaServiceError:
    return VlaServiceError("VLA service at http://127.0.0.1:10055 dropped the /submit connection")


def _observation(
    *,
    frames: Mapping[str, np.ndarray] | None = None,
    with_state: bool = True,
    instruction: str | None = None,
) -> Observation:
    return Observation(
        images=dict(_DEFAULT_FRAMES if frames is None else frames),
        state={"eef_state": _eef_state()} if with_state else {},
        instruction=instruction,
    )


def test_info_declares_chunked_eef_output() -> None:
    policy = umi_replay(prompt="t", client=_FakeClient([]), control_hz=24.0)

    assert isinstance(policy, Policy)
    assert policy.info.name == "umi-replay"
    assert policy.info.control_hz == 24.0
    assert policy.info.action_space.shape == (20,)
    semantics = policy.info.action_space.semantics
    assert semantics is not None
    assert semantics.control_mode == "eef_abs_pose"
    assert semantics.rotation_repr == "rot6d"
    assert policy.info.observation_space.state_keys == frozenset({"eef_state"})
    assert policy.config.action_horizon == CHUNK_STEPS
    config = policy.config
    assert isinstance(config, VlaPolicyConfig)
    assert config.base_url == VLA_BASE_URL
    assert config.submit_images == SUBMIT_IMAGES
    assert config.prompt == "t"


def test_act_submits_prompt_state_images_and_anchors_the_chunk() -> None:
    fake = _FakeClient([_chunk(_scripted_deltas()), _zero_chunk()])
    policy = umi_replay(prompt="pick up the cup", client=fake)
    state = _eef_state()

    chunk = policy.act(_observation())

    call = fake.calls[0]
    assert call.task == "pick up the cup"
    np.testing.assert_allclose(call.state, state, atol=1e-9)
    assert list(call.images) == ["chest_rgbd", "left_rgbd", "right_rgbd"]  # sorted
    for name, frame in call.images.items():
        np.testing.assert_array_equal(frame, _DEFAULT_FRAMES[name])
    assert call.request_id > 0

    assert len(chunk) == 3
    for action in chunk.actions:
        assert np.asarray(action.data).shape == (20,)
        assert np.asarray(action.data).dtype == np.float32
    assert chunk.control_hz == policy.info.control_hz
    assert chunk.inference_latency_s is not None
    assert chunk.inference_latency_s >= 0.0
    assert chunk.meta["request_id"] == call.request_id

    # Hand-checked anchoring: xyz cumsum onto the anchor, gripper absolute.
    rows = [np.asarray(action.data) for action in chunk.actions]
    np.testing.assert_allclose(rows[0][0:3], (0.31, -0.10, 0.20), atol=1e-6)
    np.testing.assert_allclose(rows[2][0:3], (0.33, -0.10, 0.20), atol=1e-6)
    np.testing.assert_allclose(rows[0][3:9], state[3:9], atol=1e-6)
    np.testing.assert_allclose(rows[1][10:13], (0.26, 0.12, 0.19), atol=1e-6)
    np.testing.assert_array_equal(
        [row[9] for row in rows], np.asarray((0.02, 0.03, 0.04), dtype=np.float32)
    )

    # The next inference carries a strictly greater request id.
    policy.act(_observation())
    assert fake.calls[1].request_id > fake.calls[0].request_id


def test_submit_images_string_form_selects_and_sorts_cameras() -> None:
    fake = _FakeClient([_zero_chunk()])
    policy = umi_replay(prompt="t", submit_images="right_rgbd,left_rgbd", client=fake)

    policy.act(_observation())

    (call,) = fake.calls
    assert list(call.images) == ["left_rgbd", "right_rgbd"]
    np.testing.assert_array_equal(call.images["left_rgbd"], _DEFAULT_FRAMES["left_rgbd"])


def test_task_string_prefers_the_configured_prompt() -> None:
    fake = _FakeClient([_zero_chunk()])
    policy = umi_replay(prompt="trained prompt", client=fake)
    policy.reset(Scene(id="s", instruction="scene says"))

    policy.act(_observation(instruction="live says"))

    assert fake.calls[0].task == "trained prompt"


def test_task_string_falls_back_to_live_then_scene_instruction() -> None:
    for instruction, expected in ((None, "scene says"), ("live says", "live says")):
        fake = _FakeClient([_zero_chunk()])
        policy = VlaPolicy(client=fake)
        policy.reset(Scene(id="s", instruction="scene says"))

        policy.act(_observation(instruction=instruction))

        assert fake.calls[0].task == expected


def test_no_task_string_raises_policy_error() -> None:
    fake = _FakeClient([])
    policy = VlaPolicy(client=fake)  # no prompt, never reset with a scene

    with pytest.raises(PolicyError, match="prompt"):
        policy.act(_observation())
    assert fake.calls == []


def test_state_fallback_uses_the_last_commanded_target() -> None:
    fake = _FakeClient([_chunk(_scripted_deltas()), _zero_chunk()])
    policy = umi_replay(prompt="pour", client=fake)

    policy.act(_observation())
    expected_last = anchor_chunk(_eef_state(), _chunk(_scripted_deltas()))[-1]

    policy.act(_observation(with_state=False))

    np.testing.assert_allclose(fake.calls[1].state, expected_last, atol=1e-6)


def test_first_call_without_state_key_raises_policy_error() -> None:
    fake = _FakeClient([])
    policy = umi_replay(prompt="t", client=fake)

    with pytest.raises(PolicyError, match="eef_state"):
        policy.act(_observation(with_state=False))
    assert fake.calls == []


def test_reset_clears_the_fallback_anchor_and_scene_instruction() -> None:
    fake = _FakeClient([_zero_chunk(), _zero_chunk()])
    policy = VlaPolicy(client=fake)
    policy.reset(Scene(id="s1", instruction="first task"))
    policy.act(_observation())

    policy.reset(Scene(id="s2", instruction="second task"))
    with pytest.raises(PolicyError, match="eef_state"):
        policy.act(_observation(with_state=False))
    policy.act(_observation())
    assert fake.calls[1].task == "second task"


def test_missing_submit_camera_raises_policy_error() -> None:
    frames = {name: frame for name, frame in _DEFAULT_FRAMES.items() if name != "chest_rgbd"}
    fake = _FakeClient([])
    policy = umi_replay(prompt="t", client=fake)

    with pytest.raises(PolicyError, match="chest_rgbd"):
        policy.act(_observation(frames=frames))
    assert fake.calls == []


def test_transient_error_is_retried_once_then_policy_error() -> None:
    fake = _FakeClient([_service_error(), _service_error()])
    policy = umi_replay(prompt="t", client=fake)

    with pytest.raises(PolicyError, match="dropped the /submit connection"):
        policy.act(_observation())

    assert len(fake.calls) == 2
    assert fake.calls[0].request_id == fake.calls[1].request_id
    assert fake.calls[0].task == "t"


def test_successful_retry_returns_the_chunk_and_rearms_the_retry() -> None:
    fake = _FakeClient(
        [_service_error(), _zero_chunk(), _service_error(), _zero_chunk(request_id=2)]
    )
    policy = umi_replay(prompt="t", client=fake)

    policy.act(_observation())
    policy.act(_observation())

    assert len(fake.calls) == 4  # both calls used their retry


def test_second_consecutive_failure_raises_without_retrying() -> None:
    fake = _FakeClient([_service_error(), _service_error(), _service_error()])
    policy = umi_replay(prompt="t", client=fake)

    with pytest.raises(PolicyError):
        policy.act(_observation())
    assert len(fake.calls) == 2

    with pytest.raises(PolicyError, match="previous attempt also failed"):
        policy.act(_observation())
    assert len(fake.calls) == 3  # exactly one new infer, no retry


def test_anchor_failure_raises_policy_error_and_keeps_the_retry_armed() -> None:
    bad = _chunk(np.zeros((2, ACTION_DIM_VLA)))
    bad.deltas[0, 0] = np.nan
    fake = _FakeClient([bad, _service_error(), _zero_chunk()])
    policy = umi_replay(prompt="t", client=fake)

    with pytest.raises(PolicyError, match="non-finite"):
        policy.act(_observation())
    assert len(fake.calls) == 1  # a bad chunk is data, not a transient service error

    policy.act(_observation())
    assert len(fake.calls) == 3  # the next call still gets its retry


def test_bind_records_the_state_field_and_dim_labels() -> None:
    labels = tuple(f"dim{i}" for i in range(20))
    info = EmbodimentInfo(
        name="nero",
        action_space=Box(
            shape=(20,),
            semantics=ActionSemantics(
                control_mode="eef_abs_pose",
                rotation_repr="rot6d",
                gripper="continuous",
                dim_labels=labels,
            ),
        ),
        observation_space=ObservationSpace(
            state=StateSpec(fields=(StateField("eef_state", (20,)), StateField("joint_pos", (16,))))
        ),
    )
    policy = umi_replay(prompt="t", client=_FakeClient([]))

    policy.bind(info)
    assert policy.state_field is not None
    assert policy.state_field.key == "eef_state"
    assert policy.dim_labels == labels
    assert policy.info.action_space.shape == (20,)  # own declaration untouched

    policy.bind(
        EmbodimentInfo(
            name="bare", action_space=Box(shape=(20,)), observation_space=ObservationSpace()
        )
    )
    assert policy.state_field is None
    assert policy.dim_labels is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"prompt": "t", "submit_images": "left_rgbd"}, r"2\.\.3"),
        ({"prompt": "t", "submit_images": ("left_rgbd", "right_rgbd", "left_rgbd")}, "duplicate"),
        ({"prompt": "t", "submit_images": ("left_rgbd", "")}, "non-empty"),
        ({"prompt": "t", "state_key": ""}, "state_key"),
        ({"prompt": "t", "control_hz": 0}, "control_hz"),
        ({"prompt": "t", "poll_timeout_s": 0}, "poll_timeout_s"),
        ({"prompt": 42}, "prompt"),
        ({"prompt": "t", "base_url": "127.0.0.1:10055"}, "base_url"),
    ],
)
def test_construction_rejects_bad_kwargs(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        VlaPolicy(**kwargs)


def test_close_closes_only_the_owned_client() -> None:
    fake = _FakeClient([])
    policy = umi_replay(prompt="t", client=fake)
    policy.close()
    assert not fake.closed  # an injected client stays the caller's

    owned = VlaPolicy(prompt="t")  # builds a real wire client, no connection yet
    assert owned._owned_client is not None
    owned.close()
    assert owned._owned_client is None
    owned.close()  # idempotent
