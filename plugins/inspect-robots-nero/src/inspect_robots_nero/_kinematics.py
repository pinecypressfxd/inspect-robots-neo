"""Damped least squares IK and FK for the dual Nero arms over pinocchio.

Ported core of the bring-up controller's solver: TCP frames on the gripper
flanges, the four prismatic pika joints locked out of the model, and a
per-iteration joint step cap. The bring-up stack's ProxQP formulation is
deliberately not ported: this embodiment clamps the per-tick joint delta
itself, so per-iteration QP limits would be redundant here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

FLANGE_FRAMES: dict[str, str] = {"left": "left_gripper_flange", "right": "right_gripper_flange"}
ARM_JOINT_NAMES: dict[str, tuple[str, ...]] = {
    "left": tuple(f"left_joint{i}" for i in range(1, 8)),
    "right": tuple(f"right_joint{i}" for i in range(1, 8)),
}
GRIPPER_JOINT_NAMES: tuple[str, ...] = (
    "left_pika_left_joint",
    "left_pika_right_joint",
    "right_pika_left_joint",
    "right_pika_right_joint",
)


class NeroKinematicsError(RuntimeError):
    """Raised when IK cannot reach a target within the iteration budget."""


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("rot6d column has zero norm")
    return vector / norm


def matrix_to_rot6d(rotation: np.ndarray) -> np.ndarray:
    """Flatten a rotation matrix to its first two columns (Zhou et al. 6D form)."""
    matrix = np.asarray(rotation, dtype=np.float64)
    return np.concatenate([matrix[:, 0], matrix[:, 1]])


def rot6d_to_matrix(vector: np.ndarray) -> np.ndarray:
    """Reconstruct orthonormal rotation columns from a 6-vector; scales are normalized away."""
    raw = np.asarray(vector, dtype=np.float64).ravel()
    if raw.shape != (6,):
        raise ValueError(f"rot6d vector must hold 6 values, got shape {raw.shape}")
    x = _normalize(raw[:3])
    y = _normalize(raw[3:] - np.dot(raw[3:], x) * x)
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


class NeroKinematics:
    """FK/IK over the packaged dual-arm URDF with per-arm TCP frames."""

    def __init__(
        self,
        urdf_path: str,
        *,
        tcp_translation_m: tuple[float, float, float] = (0.0, 0.0, 0.18),
        tcp_rotation_rpy_rad: tuple[float, float, float] = (0.0, -1.5707963267948966, 0.0),
        max_iterations: int = 60,
        position_tol_m: float = 1e-3,
        orientation_tol_rad: float = 1e-3,
        max_joint_step_rad: float = 0.1,
        damping: float = 1e-4,
        posture_gain: float = 0.01,
        samples: int = 256,
        seed: int = 0,
    ) -> None:
        import pinocchio as pin
        from scipy.spatial.transform import Rotation

        self._pin = pin
        self._max_iterations = int(max_iterations)
        self._position_tol = float(position_tol_m)
        self._orientation_tol = float(orientation_tol_rad)
        self._max_joint_step = float(max_joint_step_rad)
        self._damping = float(damping)
        self._posture_gain = float(posture_gain)
        self._samples = int(samples)
        self._seed = int(seed)

        model = pin.buildModelFromUrdf(str(urdf_path))
        tcp_transform = pin.SE3(
            Rotation.from_euler("xyz", tcp_rotation_rpy_rad).as_matrix(),
            np.asarray(tcp_translation_m, dtype=np.float64),
        )
        for flange_name in FLANGE_FRAMES.values():
            flange_id = model.getFrameId(flange_name)
            if flange_id >= model.nframes:
                raise ValueError(f"frame {flange_name!r} missing from the dual nero URDF")
            flange_frame = model.frames[flange_id]
            model.addFrame(
                pin.Frame(
                    f"{flange_name}_tcp",
                    flange_frame.parentJoint,
                    flange_id,
                    flange_frame.placement * tcp_transform,
                    pin.FrameType.OP_FRAME,
                )
            )
        locked_ids = [
            model.getJointId(joint_name)
            for joint_name in GRIPPER_JOINT_NAMES
            if model.existJointName(joint_name)
        ]
        if locked_ids:
            model = pin.buildReducedModel(model, locked_ids, pin.neutral(model))
        self._model = model
        self._data = model.createData()
        self._q_index: dict[str, int] = {}
        for index in range(1, model.njoints):
            self._q_index[model.names[index]] = model.joints[index].idx_q
        self._tcp_frames = {
            side: model.getFrameId(f"{flange_name}_tcp")
            for side, flange_name in FLANGE_FRAMES.items()
        }
        self._posture = self.q_from((0.0,) * 7, (0.0,) * 7)

    def q_from(self, left: Sequence[float], right: Sequence[float]) -> np.ndarray:
        """Assemble the reduced-model configuration from per-arm joint lists."""
        pin = self._pin
        q: np.ndarray = pin.neutral(self._model)
        for values, names in ((left, ARM_JOINT_NAMES["left"]), (right, ARM_JOINT_NAMES["right"])):
            if len(values) != 7:
                raise ValueError(f"arm joints must hold 7 values, got {len(values)}")
            for joint_name, value in zip(names, values, strict=True):
                q[self._q_index[joint_name]] = float(value)
        return q

    def q_split(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split a configuration into per-arm joint lists (left 7, right 7)."""
        q_full = np.asarray(q, dtype=np.float64)
        left = np.asarray([q_full[self._q_index[name]] for name in ARM_JOINT_NAMES["left"]])
        right = np.asarray([q_full[self._q_index[name]] for name in ARM_JOINT_NAMES["right"]])
        return left, right

    def _forward(self, q: np.ndarray) -> dict[str, np.ndarray]:
        pin = self._pin
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)
        poses: dict[str, np.ndarray] = {}
        for side, frame_id in self._tcp_frames.items():
            placement = self._data.oMf[frame_id]
            pose = np.eye(4)
            pose[:3, :3] = placement.rotation
            pose[:3, 3] = placement.translation
            poses[side] = pose
        return poses

    def fk(self, q: np.ndarray) -> dict[str, np.ndarray]:
        """Return per-arm TCP poses (4x4) for a configuration."""
        return self._forward(np.asarray(q, dtype=np.float64))

    def solve(
        self,
        left_target: np.ndarray,
        right_target: np.ndarray,
        q_seed: np.ndarray,
    ) -> np.ndarray:
        """Solve both arms' IK; raises NeroKinematicsError when a target is unreachable."""
        pin = self._pin
        q = np.asarray(q_seed, dtype=np.float64).copy()
        targets = {"left": np.asarray(left_target), "right": np.asarray(right_target)}
        lower = np.asarray(self._model.lowerPositionLimit)
        upper = np.asarray(self._model.upperPositionLimit)
        for _ in range(self._max_iterations):
            self._forward(q)
            errors: list[np.ndarray] = []
            jacobians: list[np.ndarray] = []
            converged = True
            for side, target in targets.items():
                frame_id = self._tcp_frames[side]
                placement = self._data.oMf[frame_id]
                position_error = float(np.linalg.norm(placement.translation - target[:3, 3]))
                rotation_error = float(
                    np.linalg.norm(pin.log3(placement.rotation.T @ target[:3, :3]))
                )
                if position_error > self._position_tol or rotation_error > self._orientation_tol:
                    converged = False
                delta = pin.SE3(target[:3, :3], target[:3, 3])
                errors.append(pin.log6(placement.actInv(delta)).vector)
                jacobians.append(
                    pin.computeFrameJacobian(
                        self._model, self._data, q, frame_id, pin.ReferenceFrame.LOCAL
                    )
                )
            if converged:
                return q
            jacobian = np.vstack(jacobians)
            error = np.concatenate(errors)
            regularized = jacobian @ jacobian.T + self._damping * np.eye(jacobian.shape[0])
            dq = jacobian.T @ np.linalg.solve(regularized, error)
            nullspace = np.eye(self._model.nq) - jacobian.T @ np.linalg.solve(regularized, jacobian)
            dq = dq + nullspace @ (self._posture_gain * (self._posture - q))
            dq = np.clip(dq, -self._max_joint_step, self._max_joint_step)
            q = np.clip(q + dq, lower, upper)
        raise NeroKinematicsError(
            "dual-arm IK did not converge within "
            f"{self._max_iterations} iterations (position tol {self._position_tol} m, "
            f"orientation tol {self._orientation_tol} rad); target may be out of the workspace"
        )

    def sample_workspace_bounds(
        self, *, margin_m: float = 0.02
    ) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
        """Sample FK over joint limits and return per-arm xyz bounds with a margin."""
        rng = np.random.default_rng(self._seed)
        lower = np.asarray(self._model.lowerPositionLimit)
        upper = np.asarray(self._model.upperPositionLimit)
        drawn = rng.uniform(low=lower, high=upper, size=(self._samples, self._model.nq))
        positions: dict[str, list[np.ndarray]] = {"left": [], "right": []}
        for row in drawn:
            poses = self._forward(row)
            for side in ("left", "right"):
                positions[side].append(poses[side][:3, 3])
        bounds: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {}
        for side in ("left", "right"):
            stacked = np.vstack(positions[side])
            low = np.maximum(stacked.min(axis=0) - margin_m, -2.0)
            high = np.minimum(stacked.max(axis=0) + margin_m, 2.0)
            bounds[side] = (
                (float(low[0]), float(low[1]), float(low[2])),
                (float(high[0]), float(high[1]), float(high[2])),
            )
        return bounds
