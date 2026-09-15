"""FK/IK and rot6d behavior for the nero kinematics module."""

from __future__ import annotations

import importlib.resources

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from inspect_robots_nero._config import HOME_LEFT, HOME_RIGHT
from inspect_robots_nero._kinematics import (
    NeroKinematics,
    NeroKinematicsError,
    matrix_to_rot6d,
    rot6d_to_matrix,
)


def _urdf_path() -> str:
    return str(importlib.resources.files("inspect_robots_nero") / "assets" / "dual_nero_pika.urdf")


def _kinematics() -> NeroKinematics:
    return NeroKinematics(_urdf_path())


@pytest.mark.parametrize(
    "rpy",
    [(0.0, 0.0, 0.0), (np.pi / 2, 0.0, 0.0), (0.1, -0.2, 0.3), (-1.0, 0.5, 2.0)],
)
def test_rot6d_round_trip(rpy: tuple[float, float, float]) -> None:
    rotation = Rotation.from_euler("xyz", rpy).as_matrix()
    restored = rot6d_to_matrix(matrix_to_rot6d(rotation))
    assert np.allclose(restored, rotation, atol=1e-9)


def test_rot6d_recovers_from_scaled_columns() -> None:
    rotation = Rotation.from_euler("xyz", (0.3, 0.4, 0.5)).as_matrix()
    scaled = matrix_to_rot6d(rotation) * 7.0
    assert np.allclose(rot6d_to_matrix(scaled), rotation, atol=1e-9)


def test_model_has_fourteen_locked_joints_and_tcp_frames() -> None:
    kinematics = _kinematics()
    q = kinematics.q_from(HOME_LEFT, HOME_RIGHT)
    assert q.shape == (14,)
    left, right = kinematics.q_split(q)
    assert left.shape == (7,) and right.shape == (7,)


def test_fk_is_finite_and_homogeneous() -> None:
    poses = _kinematics().fk(_kinematics().q_from(HOME_LEFT, HOME_RIGHT))
    for side in ("left", "right"):
        pose = poses[side]
        assert pose.shape == (4, 4)
        assert np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0))
        assert bool(np.all(np.isfinite(pose)))


def test_ik_recovers_fk_poses_from_a_disturbed_seed() -> None:
    kinematics = _kinematics()
    rng = np.random.default_rng(7)
    for _ in range(3):
        q_goal = kinematics.q_from((rng.uniform(-1.0, 1.0),) * 7, (rng.uniform(-1.0, 1.0),) * 7)
        targets = kinematics.fk(q_goal)
        q_seed = kinematics.q_from(HOME_LEFT, HOME_RIGHT)
        q_solution = kinematics.solve(targets["left"], targets["right"], q_seed)
        reached = kinematics.fk(q_solution)
        for side in ("left", "right"):
            position_error = float(np.linalg.norm(reached[side][:3, 3] - targets[side][:3, 3]))
            orientation_error = float(np.linalg.norm(reached[side][:3, :3] - targets[side][:3, :3]))
            assert position_error < 2e-3
            assert orientation_error < 5e-2


def test_ik_respects_the_joint_step_budget() -> None:
    kinematics = _kinematics()
    q_seed = kinematics.q_from(HOME_LEFT, HOME_RIGHT)
    targets = kinematics.fk(q_seed)
    far = targets["left"].copy()
    far[:3, 3] += (0.05, 0.0, 0.0)
    solution = kinematics.solve(far, targets["right"], q_seed)
    budget = 60 * 0.1
    assert float(np.max(np.abs(solution - q_seed))) <= budget + 1e-9


def test_unreachable_target_raises() -> None:
    kinematics = _kinematics()
    q_seed = kinematics.q_from(HOME_LEFT, HOME_RIGHT)
    poses = kinematics.fk(q_seed)
    unreachable = poses["left"].copy()
    unreachable[:3, 3] += (5.0, 0.0, 0.0)
    with pytest.raises(NeroKinematicsError):
        kinematics.solve(unreachable, poses["right"], q_seed)


def test_workspace_bounds_contain_the_home_pose() -> None:
    kinematics = _kinematics()
    bounds = kinematics.sample_workspace_bounds()
    home = kinematics.fk(kinematics.q_from(HOME_LEFT, HOME_RIGHT))
    for side in ("left", "right"):
        low, high = bounds[side]
        position = home[side][:3, 3]
        assert bool(np.all(position >= np.asarray(low) - 1e-9))
        assert bool(np.all(position <= np.asarray(high) + 1e-9))
