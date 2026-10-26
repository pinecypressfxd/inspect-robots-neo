"""Proprioception JSON the UMI replay HTTP service requires on ``/submit``.

The ROS client sends this only when ``transport`` is ``umi_replay_http``.
The service anchors the recorded episode on these poses and refuses a
request that omits the field. Pose layout matches the nero 20-dim EE state:
xyz, Zhou rot6d, gripper width in meters.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial.transform import Rotation

from inspect_robots_vla._anchor import UMI_GRIPPER_MAX_M, _rot6d_to_matrix

_LEFT = (slice(0, 3), slice(3, 9), 9, slice(0, 7))
_RIGHT = (slice(10, 13), slice(13, 19), 19, slice(7, 14))


def replay_proprioception_json(
    eef_state: np.ndarray,
    joint_pos: np.ndarray,
    *,
    timestamp_ns: int,
) -> str:
    """Encode the current dual-arm pose in the ROS replay proprioception shape."""
    eef = np.asarray(eef_state, dtype=np.float64)
    joints = np.asarray(joint_pos, dtype=np.float64)
    if eef.shape != (20,) or joints.shape[0] < 14:
        raise ValueError(
            "replay proprioception needs a 20-dim eef_state and 14 joint angles, "
            f"got eef {eef.shape} joints {joints.shape}"
        )
    if not np.isfinite(eef).all() or not np.isfinite(joints[:14]).all():
        raise ValueError("replay proprioception holds non-finite values")
    payload = {
        "timestamp_ns": int(timestamp_ns),
        "ee_poses": {},
        "gripper_open": {},
        "joint_positions": {},
    }
    for side, (xyz, rot6d, grip, joint_slice) in (
        ("left", _LEFT),
        ("right", _RIGHT),
    ):
        quat_xyzw = Rotation.from_matrix(_rot6d_to_matrix(eef[rot6d])).as_quat()
        width = float(eef[grip]) / UMI_GRIPPER_MAX_M
        payload["ee_poses"][side] = {
            "effector_id": side,
            "position_m": [float(value) for value in eef[xyz]],
            "rotation_quat_wxyz": [float(quat_xyzw[3]), *[float(v) for v in quat_xyzw[:3]]],
            "frame_id": "base",
        }
        payload["gripper_open"][side] = float(np.clip(width, 0.0, 1.0))
        payload["joint_positions"][side] = [float(value) for value in joints[joint_slice]]
    return json.dumps(payload, separators=(",", ":"))
