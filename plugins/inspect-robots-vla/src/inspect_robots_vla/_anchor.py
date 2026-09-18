"""Re-anchor VLA delta chunks onto an EE anchor in the nero 20-dim layout.

The service's ``xyz_rpy`` actions are per-step delta EE poses with absolute
grippers, while the embodiment consumes absolute targets
[left xyz3, left rot6d, left gripper, right xyz3, right rot6d, right gripper]
(meters, Zhou 6D first-two-columns form, gripper widths in meters) exactly as
the nero embodiment defines it. This module integrates one chunk per arm: xyz
deltas cumsum straight onto the anchor positions, rpy deltas cumsum in rpy
space onto the anchor orientation and convert to rot6d per step, and grippers
pass through the chunk's absolute values. Re-anchoring each chunk at the
freshly observed ``eef_state`` is what keeps delta integration from drifting
across chunks.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from ._client import VlaChunk, VlaServiceError

#: The embodiment's fully-open gripper width in meters; the service's
#: gripper convention is normalized [0, 1].
from ._config import ACTION_DIM_VLA

#: The embodiment's fully-open gripper width in meters; the service's
#: gripper convention is normalized [0, 1].
UMI_GRIPPER_MAX_M = 0.09

#: Width of the embodiment EE state/action vector both arms share.
EEF_STATE_DIM = 20
#: A rot6d column below this norm cannot be orthonormalized.
_UNIT_EPS = 1e-12

# Per arm: (chunk column slice, xyz slice, rot6d slice, gripper index).
_LEFT_ARM = (slice(0, 7), slice(0, 3), slice(3, 9), 9)
_RIGHT_ARM = (slice(7, 14), slice(10, 13), slice(13, 19), 19)
_ARMS = (_LEFT_ARM, _RIGHT_ARM)


def rpy_to_rot6d(rpy: np.ndarray) -> np.ndarray:
    """Convert one xyz-euler rpy triple ``(3,)`` to the Zhou 6D form ``(6,)``.

    The 6D form is the rotation matrix's first two columns stacked (entries in
    [-1, 1]), matching the nero embodiment's orientation representation.
    """
    return np.asarray(_rot6d_rows(np.asarray(rpy, dtype=np.float64).reshape(1, 3))[0])


def _rot6d_rows(rpy: np.ndarray) -> np.ndarray:
    """Convert ``(m, 3)`` xyz-euler rpy triples to ``(m, 6)`` Zhou 6D rows."""
    matrices = Rotation.from_euler("xyz", rpy).as_matrix()
    return np.concatenate([matrices[:, :, 0], matrices[:, :, 1]], axis=1)


def _unit_column(vector: np.ndarray) -> np.ndarray:
    """Normalize one rot6d half, rejecting columns of ~zero norm."""
    norm = float(np.linalg.norm(vector))
    if norm < _UNIT_EPS:
        raise VlaServiceError(
            "degenerate rot6d: a rotation column has ~zero norm, so the pose "
            "cannot be orthonormalized; remedy: check the eef_state source"
        )
    return vector / norm


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Gram-Schmidt one Zhou 6D vector back to an orthonormal matrix (scales drop)."""
    x = _unit_column(rot6d[0:3])
    y = _unit_column(rot6d[3:6] - np.dot(rot6d[3:6], x) * x)
    return np.column_stack([x, y, np.cross(x, y)])


def rot6d_to_rpy(rot6d: np.ndarray) -> np.ndarray:
    """Convert one Zhou 6D rotation ``(6,)`` to its xyz-euler rpy triple ``(3,)``.

    Non-orthonormal inputs are Gram-Schmidt normalized first, so uniform
    scaling of the 6-vector maps to the same rpy.
    """
    matrix = _rot6d_to_matrix(np.asarray(rot6d, dtype=np.float64))
    return np.asarray(Rotation.from_matrix(matrix).as_euler("xyz"))


def _checked_state(name: str, values: np.ndarray) -> np.ndarray:
    """Return ``values`` as finite float64 with the exact 20-dim EE shape."""
    state = np.asarray(values, dtype=np.float64)
    if state.shape != (EEF_STATE_DIM,):
        raise VlaServiceError(
            f"anchor input {name!r} must have shape ({EEF_STATE_DIM},), got {state.shape}"
        )
    if not np.isfinite(state).all():
        raise VlaServiceError(f"anchor input {name!r} holds non-finite values (NaN or inf)")
    return state


def _checked_deltas(chunk: VlaChunk) -> np.ndarray:
    """Return the chunk deltas as finite float64 of per-arm width 14."""
    deltas = np.asarray(chunk.deltas, dtype=np.float64)
    if deltas.ndim != 2 or deltas.shape[1] != ACTION_DIM_VLA:
        raise VlaServiceError(
            f"chunk deltas must be (m, {ACTION_DIM_VLA}), got shape {deltas.shape}"
        )
    if not np.isfinite(deltas).all():
        raise VlaServiceError("chunk deltas hold non-finite values (NaN or inf)")
    return deltas


def anchor_chunk(eef_state: np.ndarray, chunk: VlaChunk) -> np.ndarray:
    """Integrate one delta chunk onto an anchor into absolute ``(m, 20)`` targets.

    ``eef_state`` is the embodiment 20-dim EE state observed at chunk start.
    Per arm and step: the service's delta is a relative transform composed
    onto the anchor by matrix multiplication (anchor_rot @ delta_rot for
    orientation, anchor_xyz + anchor_rot @ delta_xyz for position -- the
    delta translation is expressed in the anchor's own frame); grippers pass
    through as absolute values scaled to meters. The output is float32 in
    the embodiment layout. Raises VlaServiceError on wrong shapes,
    non-finite values, or degenerate rot6d columns.
    """
    state = _checked_state("eef_state", eef_state)
    deltas = _checked_deltas(chunk)
    targets = np.zeros((deltas.shape[0], EEF_STATE_DIM), dtype=np.float64)
    for arm_slice, xyz_slice, rot_slice, grip_index in _ARMS:
        arm = deltas[:, arm_slice]
        # Each step is a relative SE(3) transform against the chunk-start
        # anchor, composed by matrix multiplication (the service's own
        # relative_rpy_actions_to_absolute): the delta translation lives in
        # the ANCHOR'S frame (R_anchor @ delta_xyz) and the rotation composes
        # (R_anchor @ R_delta) -- never additive cumsum, which both misrotates
        # the motion by the anchor orientation and compounds step over step.
        anchor_rot = _rot6d_to_matrix(state[rot_slice])
        anchor_xyz = state[xyz_slice]
        step_mats = Rotation.from_euler("xyz", arm[:, 3:6]).as_matrix()
        for step, (delta_mat, delta_xyz) in enumerate(zip(step_mats, arm[:, 0:3], strict=True)):
            rot = anchor_rot @ delta_mat
            targets[step, xyz_slice] = anchor_xyz + anchor_rot @ delta_xyz
            targets[step, rot_slice] = np.concatenate([rot[:, 0], rot[:, 1]])
        # The service's gripper is normalized [0, 1]; ours is meters.
        targets[:, grip_index] = np.clip(arm[:, 6], 0.0, 1.0) * UMI_GRIPPER_MAX_M
    return targets.astype(np.float32)


def tracking_error(target20: np.ndarray, observed20: np.ndarray) -> tuple[float, float]:
    """Worst-arm ``(position m, rotation deg)`` error between two 20-dim poses.

    Position error is the xyz norm per arm; rotation error is the angle of the
    relative rotation ``target @ observed.T`` reconstructed from each arm's
    rot6d. The maximum over both arms is returned: the number an abort check
    compares against ``TRACKING_ABORT_POS_M`` / ``TRACKING_ABORT_ROT_DEG``.
    Raises VlaServiceError on wrong shapes, non-finite values, or degenerate
    rot6d columns.
    """
    target = _checked_state("target20", target20)
    observed = _checked_state("observed20", observed20)
    pos_m = 0.0
    rot_deg = 0.0
    for _, xyz_slice, rot_slice, _ in _ARMS:
        pos_m = max(pos_m, float(np.linalg.norm(target[xyz_slice] - observed[xyz_slice])))
        relative = (
            Rotation.from_matrix(_rot6d_to_matrix(target[rot_slice]))
            * Rotation.from_matrix(_rot6d_to_matrix(observed[rot_slice])).inv()
        )
        rot_deg = max(rot_deg, float(np.degrees(relative.magnitude())))
    return pos_m, rot_deg


def umi_last_action_state(
    current14: np.ndarray, previous14: np.ndarray | None = None
) -> np.ndarray:
    """Flatten [previous, current] 14-dim XYZ+RPY states into the 28-dim input.

    The checkpoint runs umi_state_mode="last_action" with history 2: it
        expects the previous frame followed by the current frame and relativizes
        internally. A missing previous frame duplicates the current one, matching
        the training pipeline's index clipping.
    """
    prev = current14 if previous14 is None else np.asarray(previous14, dtype=np.float32)
    out = np.concatenate([prev, current14]).astype(np.float32)
    if out.shape != (28,):
        raise VlaServiceError(f"umi last-action state must hold 28 values, got {out.shape}")
    return out


def eef_state_to_umi_rpy_state(eef_state: np.ndarray) -> np.ndarray:
    """Convert the embodiment 20-dim [xyz, rot6d, grip]x2 into the VLA 14-dim.

    The serve_rlt_inference checkpoint takes XYZ+RPY+gripper per arm (dims
    0:7 left, 7:14 right, grippers absolute); rotation representation is
    converted from rot6d.
    """
    if eef_state.shape != (20,):
        raise VlaServiceError(f"eef_state must hold 20 values, got {eef_state.shape}")

    def arm(state: np.ndarray) -> np.ndarray:
        xyz = state[0:3]
        rpy = rot6d_to_rpy(state[3:9])
        grip = float(state[9]) / UMI_GRIPPER_MAX_M
        return np.concatenate([xyz, rpy, [grip]])

    left = arm(eef_state[0:10])
    right = arm(eef_state[10:20])
    out = np.concatenate([left, right]).astype(np.float32)
    if not np.all(np.isfinite(out)):
        raise VlaServiceError("eef_state contains non-finite values")
    return out
