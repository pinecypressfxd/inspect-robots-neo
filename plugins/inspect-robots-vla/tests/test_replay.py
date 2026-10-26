"""The replay submit field uses the same pose the vla-only loop already has."""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial.transform import Rotation

from inspect_robots_vla._anchor import rpy_to_rot6d
from inspect_robots_vla._replay import replay_proprioception_json


def test_replay_proprioception_matches_the_ros_json_shape() -> None:
    eef = np.zeros(20)
    eef[0:3] = (0.4, 0.1, 0.9)
    eef[3:9] = rpy_to_rot6d(np.zeros(3))
    eef[9] = 0.045
    eef[10:13] = (0.4, -0.1, 0.9)
    eef[13:19] = rpy_to_rot6d(np.array([0.0, -np.pi / 2, 0.0]))
    eef[19] = 0.09
    joints = np.arange(16, dtype=np.float64)

    payload = json.loads(replay_proprioception_json(eef, joints, timestamp_ns=123))

    assert payload["timestamp_ns"] == 123
    assert payload["ee_poses"]["left"]["position_m"] == [0.4, 0.1, 0.9]
    assert payload["ee_poses"]["left"]["rotation_quat_wxyz"] == [1.0, 0.0, 0.0, 0.0]
    assert payload["ee_poses"]["left"]["frame_id"] == "base"
    assert payload["gripper_open"] == {"left": 0.5, "right": 1.0}
    assert payload["joint_positions"]["left"] == list(range(7))
    assert payload["joint_positions"]["right"] == list(range(7, 14))
    quat = payload["ee_poses"]["right"]["rotation_quat_wxyz"]
    xyzw = [quat[1], quat[2], quat[3], quat[0]]
    assert Rotation.from_quat(xyzw).as_euler("xyz") == pytest_approx_rpy()


def pytest_approx_rpy():
    import pytest

    return pytest.approx((0.0, -np.pi / 2, 0.0))
