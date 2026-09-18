"""Re-anchoring tests: delta cumsum onto an EE anchor, rot6d round trips, tracking error.

The 20-dim layout under test is the nero embodiment order
[left xyz3, left rot6d, left gripper, right xyz3, right rot6d, right gripper],
with rot6d in the Zhou 6D form (first two rotation-matrix columns), matching
``inspect_robots_nero._kinematics``. Expected values are built from scipy
directly so the convention is checked, not assumed.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from inspect_robots_vla._anchor import (
    anchor_chunk,
    rot6d_to_rpy,
    rpy_to_rot6d,
    tracking_error,
)
from inspect_robots_vla._client import VlaChunk, VlaServiceError
from inspect_robots_vla._config import (
    ACTION_DIM_VLA,
    TRACKING_ABORT_POS_M,
    TRACKING_ABORT_ROT_DEG,
)

_LEFT_RPY = (0.10, -0.20, 0.30)
_RIGHT_RPY = (-0.40, 0.25, -0.15)


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


def _chunk(deltas: np.ndarray, request_id: int = 1) -> VlaChunk:
    return VlaChunk(request_id=request_id, deltas=np.asarray(deltas, dtype=np.float32))


def test_rpy_rot6d_round_trips() -> None:
    np.testing.assert_allclose(rpy_to_rot6d(np.zeros(3)), [1, 0, 0, 0, 1, 0], atol=1e-12)
    rpys = np.array(
        [
            (0.0, 0.0, 0.0),
            _LEFT_RPY,
            (-2.0, 1.2, 0.7),
            (3.0, -1.4, -2.5),
        ]
    )
    for rpy in rpys:
        np.testing.assert_allclose(rot6d_to_rpy(rpy_to_rot6d(rpy)), rpy, atol=1e-9)
    for matrix in Rotation.random(8, random_state=7).as_matrix():
        six = _columns6(matrix)
        np.testing.assert_allclose(rpy_to_rot6d(rot6d_to_rpy(six)), six, atol=1e-9)


def test_rot6d_scale_is_normalized_away() -> None:
    six = rpy_to_rot6d(np.asarray(_LEFT_RPY))
    np.testing.assert_allclose(rot6d_to_rpy(2.5 * six), _LEFT_RPY, atol=1e-9)


def test_zero_deltas_reproduce_the_anchor_at_every_step() -> None:
    state = _eef_state()
    targets = anchor_chunk(state, _chunk(np.zeros((5, ACTION_DIM_VLA))))
    assert targets.shape == (5, 20)
    assert targets.dtype == np.float32
    # xyz and rot6d reproduce the anchor at every step.
    np.testing.assert_allclose(targets[:, 0:9], np.broadcast_to(state[0:9], (5, 9)), atol=1e-6)
    np.testing.assert_allclose(targets[:, 10:19], np.broadcast_to(state[10:19], (5, 9)), atol=1e-6)
    # Grippers are absolute passthrough, so a zeroed chunk commands closed (0) grippers.
    np.testing.assert_array_equal(targets[:, 9], np.zeros(5, dtype=np.float32))
    np.testing.assert_array_equal(targets[:, 19], np.zeros(5, dtype=np.float32))


def test_se3_composition_matches_hand_computed_targets() -> None:
    state = _eef_state()
    deltas = np.zeros((3, ACTION_DIM_VLA))
    deltas[:, 0] = (0.01, 0.00, 0.02)  # left xyz, in the ANCHOR's frame
    deltas[:, 3] = (0.01, 0.0, 0.0)  # left rpy delta
    deltas[:, 6] = (0.02, 0.03, 0.04)  # left gripper absolute (normalized)
    deltas[:, 7] = (-0.02, 0.0, 0.0)  # right xyz
    targets = anchor_chunk(state, _chunk(deltas))

    anchor_rot = Rotation.from_euler("xyz", _LEFT_RPY).as_matrix()
    for step in range(3):
        # xyz: anchor + R_anchor @ delta (each step vs the anchor, no cumsum)
        expected_xyz = np.asarray(state[0:3]) + anchor_rot @ deltas[step, 0:3]
        np.testing.assert_allclose(targets[step, 0:3], expected_xyz, atol=1e-6)
        # rotation: R_anchor @ R_delta
        expected_rot = anchor_rot @ Rotation.from_euler("xyz", deltas[step, 3:6]).as_matrix()
        expected_rot6d = np.concatenate([expected_rot[:, 0], expected_rot[:, 1]])
        np.testing.assert_allclose(targets[step, 3:9], expected_rot6d, atol=1e-6)
    # grippers normalized -> meters
    np.testing.assert_allclose(targets[:, 9], np.asarray((0.02, 0.03, 0.04)) * 0.09, atol=1e-6)
    # right arm mirrors the layout at offsets 10..19
    right_rot = Rotation.from_euler("xyz", _RIGHT_RPY).as_matrix()
    expected_right = np.asarray(state[10:13]) + (right_rot @ deltas[:, 7:10].T).T
    np.testing.assert_allclose(targets[:, 10:13], expected_right, atol=1e-6)


def test_gripper_is_absolute_not_delta() -> None:
    state = _eef_state()  # anchor grippers 0.05 / 0.04 must be ignored
    deltas = np.zeros((2, ACTION_DIM_VLA))
    deltas[:, 6] = (0.02, 0.08)
    deltas[:, 13] = (0.01, 0.03)
    targets = anchor_chunk(state, _chunk(deltas))
    np.testing.assert_allclose(targets[:, 9], np.asarray((0.02, 0.08)) * 0.09, atol=1e-6)
    np.testing.assert_allclose(targets[:, 19], np.asarray((0.01, 0.03)) * 0.09, atol=1e-6)


def test_reanchoring_on_the_same_state_accumulates_no_drift() -> None:
    state = _eef_state()
    rng = np.random.default_rng(3)
    first = _chunk(rng.normal(scale=0.05, size=(4, ACTION_DIM_VLA)))
    second = _chunk(rng.normal(scale=0.05, size=(4, ACTION_DIM_VLA)), request_id=2)
    before = anchor_chunk(state, second)
    endpoint = anchor_chunk(state, first)[-1]
    after = anchor_chunk(state, second)

    np.testing.assert_array_equal(before, after)
    # The second chunk restarts at the anchor, not at chunk one's endpoint.
    anchor_rot = Rotation.from_euler("xyz", _LEFT_RPY).as_matrix()
    np.testing.assert_allclose(
        after[0, 0:3], state[0:3] + anchor_rot @ second.deltas[0, 0:3], atol=1e-6
    )
    assert not np.allclose(endpoint[0:3], state[0:3])


def test_anchor_chunk_rejects_non_finite_state() -> None:
    state = _eef_state()
    state[4] = np.nan
    with pytest.raises(VlaServiceError, match="non-finite"):
        anchor_chunk(state, _chunk(np.zeros((2, ACTION_DIM_VLA))))


def test_anchor_chunk_rejects_non_finite_deltas() -> None:
    deltas = np.zeros((2, ACTION_DIM_VLA))
    deltas[1, 8] = np.inf
    with pytest.raises(VlaServiceError, match="non-finite"):
        anchor_chunk(_eef_state(), _chunk(deltas))


def test_anchor_chunk_rejects_wrong_shapes() -> None:
    with pytest.raises(VlaServiceError, match="shape"):
        anchor_chunk(_eef_state()[:14], _chunk(np.zeros((2, ACTION_DIM_VLA))))
    with pytest.raises(VlaServiceError, match=r"\(m, 14\)"):
        anchor_chunk(_eef_state(), _chunk(np.zeros((2, 7))))


def test_anchor_chunk_rejects_degenerate_rot6d() -> None:
    state = _eef_state()
    state[3:9] = 0.0
    with pytest.raises(VlaServiceError, match="degenerate"):
        anchor_chunk(state, _chunk(np.zeros((2, ACTION_DIM_VLA))))


def test_tracking_error_zero_for_identical_poses() -> None:
    state = _eef_state()
    assert tracking_error(state, state) == (0.0, 0.0)


def test_tracking_error_at_position_boundary() -> None:
    target = _eef_state()
    target[0] += TRACKING_ABORT_POS_M
    pos_m, rot_deg = tracking_error(target, _eef_state())
    assert pos_m == pytest.approx(TRACKING_ABORT_POS_M, abs=1e-9)
    assert rot_deg == pytest.approx(0.0, abs=1e-9)
    # Exactly at the threshold (to float precision): a strict > threshold abort
    # leaves the skill segment running.
    assert pos_m < TRACKING_ABORT_POS_M + 1e-9


def test_tracking_error_at_rotation_boundary() -> None:
    observed = _eef_state()
    target = observed.copy()
    rotate = Rotation.from_rotvec([0.0, 0.0, np.radians(TRACKING_ABORT_ROT_DEG)])
    anchor = Rotation.from_euler("xyz", _LEFT_RPY)
    target[3:9] = _columns6((rotate * anchor).as_matrix())

    pos_m, rot_deg = tracking_error(target, observed)

    assert pos_m == pytest.approx(0.0, abs=1e-9)
    assert rot_deg == pytest.approx(TRACKING_ABORT_ROT_DEG, abs=1e-9)
    # Exactly at the threshold (to float precision): a strict > threshold abort
    # leaves the skill segment running.
    assert rot_deg < TRACKING_ABORT_ROT_DEG + 1e-9


def test_tracking_error_reports_the_worst_arm() -> None:
    observed = _eef_state()
    target = observed.copy()
    target[0] += 0.010  # left arm 1 cm
    target[10] += 0.020  # right arm 2 cm
    target[3:9] = _columns6(
        (
            Rotation.from_rotvec([0.0, 0.0, np.radians(10.0)])
            * Rotation.from_euler("xyz", _LEFT_RPY)
        ).as_matrix()
    )
    target[13:19] = _columns6(
        (
            Rotation.from_rotvec([0.0, 0.0, np.radians(4.0)])
            * Rotation.from_euler("xyz", _RIGHT_RPY)
        ).as_matrix()
    )

    pos_m, rot_deg = tracking_error(target, observed)

    assert pos_m == pytest.approx(0.020, abs=1e-9)
    assert rot_deg == pytest.approx(10.0, abs=1e-9)


def test_tracking_error_rejects_non_finite() -> None:
    observed = _eef_state()
    observed[11] = np.nan
    with pytest.raises(VlaServiceError, match="non-finite"):
        tracking_error(_eef_state(), observed)


def test_tracking_error_rejects_wrong_shape() -> None:
    with pytest.raises(VlaServiceError, match="shape"):
        tracking_error(np.zeros(19), _eef_state())


def test_tracking_error_rejects_degenerate_rot6d() -> None:
    target = _eef_state()
    target[13:19] = 0.0
    with pytest.raises(VlaServiceError, match="degenerate"):
        tracking_error(target, _eef_state())
