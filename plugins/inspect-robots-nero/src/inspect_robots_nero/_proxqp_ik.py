"""Pinocchio + ProxQP IK, copied from the ROS inference controller.

Source: ``robot_itl_evaluation/services/controller/internal/ik_solver.py``
``PinocchioProxQPIKSolver``. The EePose adapter is not copied; callers pass
4x4 target matrices keyed by TCP frame name.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation

try:  # pragma: no cover - exercised by runtime environments with the dependency.
    import proxsuite
except ModuleNotFoundError:  # pragma: no cover - import remains usable for tests.
    proxsuite = None


logger = logging.getLogger(__name__)
Q_LEFT_ARM = slice(0, 7)
Q_RIGHT_ARM = slice(7, 14)
IK_MAX_ITERATIONS = 10
IK_POSITION_TOLERANCE_M = 1e-3
IK_ORIENTATION_TOLERANCE_RAD = 1e-3
IK_DEFAULT_DT_S = 0.002
IK_MAX_JOINT_STEP_RAD = 0.1
LIMIT_ACTIVE_TOLERANCE = 1e-6
GRIPPER_JOINT_NAMES = (
    "left_pika_left_joint",
    "left_pika_right_joint",
    "right_pika_left_joint",
    "right_pika_right_joint",
)


@dataclass(frozen=True)
class IKResult:
    joint_positions: np.ndarray
    success: bool
    error_norm: float
    converged: bool = False
    initial_error_norm: float = 0.0
    initial_position_error_m: float = 0.0
    initial_orientation_error_rad: float = 0.0
    final_position_error_m: float = 0.0
    final_orientation_error_rad: float = 0.0
    error_reduction_ratio: float = 0.0
    position_error_reduction_ratio: float = 0.0
    orientation_error_reduction_ratio: float = 0.0
    iteration_count: int = 0
    duration_ms: float = 0.0
    joint_delta_norm_rad: float = 0.0
    max_joint_delta_rad: float = 0.0
    max_joint_step_rad: float = 0.0
    joint_step_limit_ratio: float = 0.0
    velocity_limit_active: bool = False
    configuration_limit_active: bool = False
    left_initial_position_error_m: float = 0.0
    left_initial_orientation_error_rad: float = 0.0
    left_final_position_error_m: float = 0.0
    left_final_orientation_error_rad: float = 0.0
    left_position_error_reduction_ratio: float = 0.0
    left_orientation_error_reduction_ratio: float = 0.0
    left_max_joint_delta_rad: float = 0.0
    right_initial_position_error_m: float = 0.0
    right_initial_orientation_error_rad: float = 0.0
    right_final_position_error_m: float = 0.0
    right_final_orientation_error_rad: float = 0.0
    right_position_error_reduction_ratio: float = 0.0
    right_orientation_error_reduction_ratio: float = 0.0
    right_max_joint_delta_rad: float = 0.0


@dataclass(frozen=True)
class _FrameErrorMetrics:
    error_norm: float
    position_error_m: float
    orientation_error_rad: float
    converged: bool
    position_error_by_effector: dict[str, float] = field(default_factory=dict)
    orientation_error_by_effector: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class _JointDeltaMetrics:
    joint_delta_norm_rad: float
    max_joint_delta_rad: float
    left_max_joint_delta_rad: float
    right_max_joint_delta_rad: float
    max_joint_step_rad: float
    joint_step_limit_ratio: float
    velocity_limit_active: bool
    configuration_limit_active: bool


@dataclass(frozen=True)
class _FrameTask:
    frame_name: str
    frame_id: int
    cost: np.ndarray
    lm_damping: float = 0.0


@dataclass(frozen=True)
class _ProxQPDims:
    n: int
    n_eq: int
    n_ineq: int


class _ProxQPWorkspace:
    def __init__(self) -> None:
        self.qp = None
        self._dims: _ProxQPDims | None = None

    def solve(
        self,
        *,
        H: np.ndarray,
        g: np.ndarray,
        C: np.ndarray | None,
        lower: np.ndarray | None,
        upper: np.ndarray | None,
    ) -> np.ndarray:
        if proxsuite is None:
            raise RuntimeError(
                "proxsuite is required for PinocchioProxQPIKSolver. "
                "Install the project dependencies before selecting this solver.",
            )

        H = _as_float_array(H)
        g = _as_float_array(g)
        C = _as_float_array(C) if C is not None else None
        lower = _as_float_array(lower) if lower is not None else None
        upper = _as_float_array(upper) if upper is not None else None
        dims = _ProxQPDims(
            n=H.shape[0],
            n_eq=0,
            n_ineq=0 if C is None else C.shape[0],
        )

        reuse_workspace = self.qp is not None and self._dims == dims
        if not reuse_workspace:
            self.qp = proxsuite.proxqp.dense.QP(dims.n, dims.n_eq, dims.n_ineq)
            self.qp.init(H, g, None, None, C, lower, upper)
            self.qp.settings.eps_abs = 1e-3
            self.qp.settings.max_iter = 1000
            self.qp.settings.max_iter_in = 1500
            self.qp.settings.verbose = False
            self._dims = dims
            self.qp.solve()
        else:
            self.qp.update(H=H, g=g, C=C, l=lower, u=upper)
            self.qp.solve(self.qp.results.x, self.qp.results.y, self.qp.results.z)

        if self.qp.results.x is None:
            raise RuntimeError(
                f"ProxQP failed with status={self.qp.results.info.status}",
            )
        return np.asarray(self.qp.results.x, dtype=float)


class PinocchioProxQPIKSolver:
    """Differential IK solver built directly from Pinocchio primitives."""

    def __init__(
        self,
        urdf_path: str,
        ee_frame_names: list[str],
        frame_position_cost: float,
        frame_orientation_cost: float,
        posture_cost: float,
        damping_cost: float,
        frame_lm_damping: float = 0.0,
        tcp_translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        tcp_rotation_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0),
        dt: float = IK_DEFAULT_DT_S,
        max_iterations: int = IK_MAX_ITERATIONS,
        time_budget_ms: float | None = None,
        max_joint_step_rad: float = IK_MAX_JOINT_STEP_RAD,
    ) -> None:
        model = pin.buildModelFromUrdf(urdf_path)
        self._tcp_frame_map: dict[str, str] = {}

        self._effector_by_tcp: dict[str, str] = {}
        effector_ids = ("left", "right")
        tcp_translation = np.asarray(tcp_translation_m, dtype=float)
        if tcp_translation.shape != (3,):
            raise ValueError("tcp_translation_m must contain exactly 3 values")
        tcp_rotation_rpy = np.asarray(tcp_rotation_rpy_rad, dtype=float)
        if tcp_rotation_rpy.shape != (3,):
            raise ValueError("tcp_rotation_rpy_rad must contain exactly 3 values")
        tcp_transform = pin.SE3(
            Rotation.from_euler("xyz", tcp_rotation_rpy).as_matrix(),
            tcp_translation,
        )
        for index, flange_name in enumerate(ee_frame_names):
            flange_id = model.getFrameId(flange_name)
            if flange_id >= model.nframes:
                available = [f.name for f in model.frames]
                raise ValueError(
                    f"Frame '{flange_name}' not found. Available: {available}",
                )
            flange_frame = model.frames[flange_id]
            tcp_name = f"{flange_name}_tcp"
            tcp_frame = pin.Frame(
                tcp_name,
                flange_frame.parentJoint,
                flange_id,
                flange_frame.placement * tcp_transform,
                pin.FrameType.OP_FRAME,
            )
            model.addFrame(tcp_frame)
            self._tcp_frame_map[flange_name] = tcp_name
            if index < len(effector_ids):
                self._effector_by_tcp[tcp_name] = effector_ids[index]

        locked_joint_ids = _existing_joint_ids(model, GRIPPER_JOINT_NAMES)
        if locked_joint_ids:
            model = pin.buildReducedModel(model, locked_joint_ids, pin.neutral(model))

        self._model = model
        self._ik_data = self._model.createData()
        self._fk_data = self._model.createData()
        self._initial_q = pin.neutral(self._model)
        self._dt = float(dt)
        self._max_iterations = int(max_iterations)
        self._time_budget_ms = time_budget_ms
        self._max_joint_step_rad = float(max_joint_step_rad)
        frame_lm_damping = float(frame_lm_damping)
        if self._dt <= 0.0:
            raise ValueError("dt must be > 0")
        if self._max_joint_step_rad <= 0.0:
            raise ValueError("max_joint_step_rad must be > 0")
        if frame_lm_damping < 0.0:
            raise ValueError("frame_lm_damping must be non-negative")
        self._posture_cost = float(posture_cost)
        self._damping_cost = float(damping_cost)
        self._proxqp_workspace = _ProxQPWorkspace()
        self._model.velocityLimit[:] = self._max_joint_step_rad / self._dt

        frame_cost = np.array(
            [
                frame_position_cost,
                frame_position_cost,
                frame_position_cost,
                frame_orientation_cost,
                frame_orientation_cost,
                frame_orientation_cost,
            ],
            dtype=float,
        )
        self._tasks: dict[str, _FrameTask] = {}
        self._target_poses: dict[str, pin.SE3] = {}
        for tcp_name in self._tcp_frame_map.values():
            frame_id = self._model.getFrameId(tcp_name)
            self._tasks[tcp_name] = _FrameTask(
                frame_name=tcp_name,
                frame_id=frame_id,
                cost=frame_cost.copy(),
                lm_damping=frame_lm_damping,
            )

        self._joint_idx_map: dict[str, int] = {}
        for i in range(1, self._model.njoints):
            joint = self._model.joints[i]
            self._joint_idx_map[self._model.names[i]] = joint.idx_q

        self._configuration_limit = _ConfigurationLimit(self._model)
        self._velocity_limit = _VelocityLimit(self._model)

    def nq(self) -> int:
        return self._model.nq

    def tcp_frame_name(self, flange_name: str) -> str:
        return self._tcp_frame_map[flange_name]

    def build_initial_q(self, initial_joints: dict[str, float]) -> np.ndarray:
        q = pin.neutral(self._model)
        for name, val in initial_joints.items():
            if name in self._joint_idx_map:
                q[self._joint_idx_map[name]] = val
            else:
                logger.warning("Joint '%s' from config not found in model", name)
        self._initial_q = q.copy()
        return q

    def forward_kinematics(self, q: np.ndarray, frame_name: str) -> np.ndarray:
        pin.forwardKinematics(self._model, self._fk_data, q)
        pin.updateFramePlacements(self._model, self._fk_data)
        fid = self._model.getFrameId(frame_name)
        if fid >= self._model.nframes:
            raise ValueError(f"Frame '{frame_name}' not found in model")
        oMf = self._fk_data.oMf[fid]
        T = np.eye(4)
        T[:3, :3] = oMf.rotation
        T[:3, 3] = oMf.translation
        return T

    def solve(
        self,
        target_poses: dict[str, np.ndarray],
        current_q: np.ndarray,
        *,
        max_iterations: int | None = None,
        time_budget_ms: float | None = None,
    ) -> IKResult:
        started_ns = time.perf_counter_ns()
        q_start = np.asarray(current_q, dtype=float).copy()
        q_new = q_start.copy()
        initial_error = _FrameErrorMetrics(
            error_norm=float("inf"),
            position_error_m=float("inf"),
            orientation_error_rad=float("inf"),
            converged=False,
        )
        final_error = initial_error
        iteration_count = 0
        try:
            self._set_target_pair(target_poses)
            self._check_limits(q_new)
            self._update_kinematics(q_new)
            initial_error = self._frame_error_metrics()
            final_error = initial_error
            max_iter = self._max_iterations if max_iterations is None else max_iterations
            if max_iter < 0:
                raise ValueError("max_iterations must be non-negative")
            budget_ms = self._time_budget_ms if time_budget_ms is None else time_budget_ms
            deadline_ns = (
                None
                if budget_ms is None
                else started_ns + int(float(budget_ms) * 1_000_000)
            )

            if not final_error.converged:
                for iteration in range(max_iter):
                    iteration_count = iteration + 1
                    delta_q = self._solve_delta_q(q_new)
                    q_new = pin.integrate(self._model, q_new, delta_q)
                    self._update_kinematics(q_new)
                    final_error = self._frame_error_metrics()
                    if final_error.converged:
                        break
                    if (
                        deadline_ns is not None
                        and time.perf_counter_ns() >= deadline_ns
                    ):
                        break

            success = _is_acceptable_result(
                q_new,
                initial_error=initial_error,
                final_error=final_error,
            )
            joint_delta = _joint_delta_metrics(
                self._model,
                q_start,
                q_new,
                dt=self._dt,
                max_joint_step_rad=self._max_joint_step_rad,
                configuration_limit=self._configuration_limit,
                velocity_limit=self._velocity_limit,
            )
            return IKResult(
                joint_positions=q_new.copy(),
                success=success,
                error_norm=final_error.error_norm,
                converged=final_error.converged,
                initial_error_norm=initial_error.error_norm,
                initial_position_error_m=initial_error.position_error_m,
                initial_orientation_error_rad=initial_error.orientation_error_rad,
                final_position_error_m=final_error.position_error_m,
                final_orientation_error_rad=final_error.orientation_error_rad,
                error_reduction_ratio=_error_reduction_ratio(
                    initial_error.error_norm,
                    final_error.error_norm,
                ),
                position_error_reduction_ratio=_error_reduction_ratio(
                    initial_error.position_error_m,
                    final_error.position_error_m,
                ),
                orientation_error_reduction_ratio=_error_reduction_ratio(
                    initial_error.orientation_error_rad,
                    final_error.orientation_error_rad,
                ),
                iteration_count=iteration_count,
                duration_ms=(time.perf_counter_ns() - started_ns) / 1e6,
                joint_delta_norm_rad=joint_delta.joint_delta_norm_rad,
                max_joint_delta_rad=joint_delta.max_joint_delta_rad,
                max_joint_step_rad=joint_delta.max_joint_step_rad,
                joint_step_limit_ratio=joint_delta.joint_step_limit_ratio,
                velocity_limit_active=joint_delta.velocity_limit_active,
                configuration_limit_active=joint_delta.configuration_limit_active,
                **_effector_result_fields(initial_error, final_error, joint_delta),
            )
        except Exception as e:
            logger.warning("IK solve failed: %s", e)
            joint_delta = _joint_delta_metrics(
                self._model,
                q_start,
                q_new,
                dt=self._dt,
                max_joint_step_rad=self._max_joint_step_rad,
                configuration_limit=self._configuration_limit,
                velocity_limit=self._velocity_limit,
            )
            return IKResult(
                joint_positions=q_start.copy(),
                success=False,
                error_norm=final_error.error_norm,
                converged=False,
                initial_error_norm=initial_error.error_norm,
                initial_position_error_m=initial_error.position_error_m,
                initial_orientation_error_rad=initial_error.orientation_error_rad,
                final_position_error_m=final_error.position_error_m,
                final_orientation_error_rad=final_error.orientation_error_rad,
                error_reduction_ratio=_error_reduction_ratio(
                    initial_error.error_norm,
                    final_error.error_norm,
                ),
                position_error_reduction_ratio=_error_reduction_ratio(
                    initial_error.position_error_m,
                    final_error.position_error_m,
                ),
                orientation_error_reduction_ratio=_error_reduction_ratio(
                    initial_error.orientation_error_rad,
                    final_error.orientation_error_rad,
                ),
                iteration_count=iteration_count,
                duration_ms=(time.perf_counter_ns() - started_ns) / 1e6,
                joint_delta_norm_rad=joint_delta.joint_delta_norm_rad,
                max_joint_delta_rad=joint_delta.max_joint_delta_rad,
                max_joint_step_rad=joint_delta.max_joint_step_rad,
                joint_step_limit_ratio=joint_delta.joint_step_limit_ratio,
                velocity_limit_active=joint_delta.velocity_limit_active,
                configuration_limit_active=joint_delta.configuration_limit_active,
                **_effector_result_fields(initial_error, final_error, joint_delta),
            )

    def _set_target_pair(self, target_poses: dict[str, np.ndarray]) -> None:
        expected = set(self._tasks)
        received = set(target_poses)
        if received != expected:
            raise ValueError(
                "target_poses must include exactly both TCP frames: "
                f"expected {sorted(expected)}, got {sorted(received)}",
            )
        self._target_poses = {
            frame_name: pin.SE3(target_T[:3, :3], target_T[:3, 3])
            for frame_name, target_T in target_poses.items()
        }

    def _solve_delta_q(self, q: np.ndarray) -> np.ndarray:
        H, g = self._compute_qp_objective(q)
        C, lower, upper = self._compute_qp_inequalities(q)
        return self._proxqp_workspace.solve(H=H, g=g, C=C, lower=lower, upper=upper)

    def _compute_qp_objective(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        H = 1e-12 * np.eye(self._model.nv)
        g = np.zeros(self._model.nv)
        for task in self._tasks.values():
            H_task, g_task = self._frame_task_qp_objective(task)
            H += H_task
            g += g_task
        H_posture, g_posture = self._posture_qp_objective(q)
        H += H_posture
        g += g_posture
        H += (self._damping_cost**2) * np.eye(self._model.nv)
        return H, g

    def _frame_task_qp_objective(
        self,
        task: _FrameTask,
    ) -> tuple[np.ndarray, np.ndarray]:
        error = self._frame_error(task)
        jacobian = self._frame_jacobian(task)
        H, g = _weighted_task_qp_objective(jacobian, error, task.cost)
        if task.lm_damping:
            weighted_error = task.cost * (-error)
            mu = task.lm_damping * weighted_error @ weighted_error
            H += mu * np.eye(jacobian.shape[1])
        return H, g

    def _posture_qp_objective(
        self,
        q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        error = pin.difference(self._model, self._initial_q, q)
        jacobian = np.eye(self._model.nv)
        cost = np.full(self._model.nv, self._posture_cost)
        return _weighted_task_qp_objective(jacobian, error, cost)

    def _compute_qp_inequalities(
        self,
        q: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        C_list = []
        lower_list = []
        upper_list = []
        for limit in (self._configuration_limit, self._velocity_limit):
            matvec = limit.compute(q, self._dt)
            if matvec is None:
                continue
            C, lower, upper = matvec
            C_list.append(C)
            lower_list.append(lower)
            upper_list.append(upper)
        if not C_list:
            return None, None, None
        return np.vstack(C_list), np.hstack(lower_list), np.hstack(upper_list)

    def _frame_error_metrics(self) -> _FrameErrorMetrics:
        total_error = 0.0
        max_position_error = 0.0
        max_orientation_error = 0.0
        position_error_by_effector: dict[str, float] = {}
        orientation_error_by_effector: dict[str, float] = {}
        converged = True
        for task in self._tasks.values():
            error = self._frame_error(task)
            position_error = float(np.linalg.norm(error[:3]))
            orientation_error = float(np.linalg.norm(error[3:]))
            effector_id = self._effector_by_tcp.get(task.frame_name)
            if effector_id is not None:
                position_error_by_effector[effector_id] = position_error
                orientation_error_by_effector[effector_id] = orientation_error
            total_error += float(np.linalg.norm(error))
            max_position_error = max(max_position_error, position_error)
            max_orientation_error = max(max_orientation_error, orientation_error)
            if (
                position_error > IK_POSITION_TOLERANCE_M
                or orientation_error > IK_ORIENTATION_TOLERANCE_RAD
            ):
                converged = False
        return _FrameErrorMetrics(
            error_norm=total_error,
            position_error_m=max_position_error,
            orientation_error_rad=max_orientation_error,
            converged=converged,
            position_error_by_effector=position_error_by_effector,
            orientation_error_by_effector=orientation_error_by_effector,
        )

    def _frame_error(self, task: _FrameTask) -> np.ndarray:
        target = self._target_poses.get(task.frame_name)
        if target is None:
            raise ValueError(f"No target set for frame '{task.frame_name}'")
        frame_to_world = self._ik_data.oMf[task.frame_id]
        target_to_frame = frame_to_world.actInv(target)
        return pin.log(target_to_frame).vector

    def _frame_jacobian(self, task: _FrameTask) -> np.ndarray:
        target = self._target_poses.get(task.frame_name)
        if target is None:
            raise ValueError(f"No target set for frame '{task.frame_name}'")
        frame_to_world = self._ik_data.oMf[task.frame_id]
        frame_to_target = target.actInv(frame_to_world)
        jacobian_in_frame = pin.getFrameJacobian(
            self._model,
            self._ik_data,
            task.frame_id,
            pin.ReferenceFrame.LOCAL,
        )
        return -pin.Jlog6(frame_to_target) @ jacobian_in_frame

    def _update_kinematics(self, q: np.ndarray) -> None:
        pin.forwardKinematics(self._model, self._ik_data, q)
        pin.computeJointJacobians(self._model, self._ik_data, q)
        pin.updateFramePlacements(self._model, self._ik_data)

    def _check_limits(self, q: np.ndarray) -> None:
        has_limit = _has_configuration_limit(self._model)
        lower = self._model.lowerPositionLimit
        upper = self._model.upperPositionLimit
        if np.any(q[has_limit] < lower[has_limit] - 1e-10) or np.any(
            q[has_limit] > upper[has_limit] + 1e-10,
        ):
            raise ValueError("current_q is outside configuration limits")

class _ConfigurationLimit:
    def __init__(self, model: pin.Model, config_limit_gain: float = 0.5) -> None:
        self._model = model
        self._config_limit_gain = config_limit_gain
        has_limit = _has_configuration_limit(model)
        indices = []
        for joint in model.joints:
            if joint.idx_q < 0:
                continue
            q_slice = slice(joint.idx_q, joint.idx_q + joint.nq)
            if has_limit[q_slice].all():
                indices.extend(range(joint.idx_v, joint.idx_v + joint.nv))
        self._indices = np.asarray(indices, dtype=int)
        self._projection = (
            None if len(indices) == 0 else np.eye(model.nv)[self._indices]
        )

    def compute(
        self,
        q: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        del dt
        if self._projection is None:
            return None
        delta_q_max = pin.difference(self._model, q, self._model.upperPositionLimit)
        delta_q_min = pin.difference(self._model, q, self._model.lowerPositionLimit)
        upper = self._config_limit_gain * delta_q_max[self._indices]
        lower = self._config_limit_gain * delta_q_min[self._indices]
        return self._projection, lower, upper


class _VelocityLimit:
    def __init__(self, model: pin.Model) -> None:
        self._model = model
        has_limit = np.logical_and(
            model.velocityLimit < 1e20,
            model.velocityLimit > 1e-10,
        )
        indices = []
        for joint in model.joints:
            if joint.idx_v < 0:
                continue
            v_slice = slice(joint.idx_v, joint.idx_v + joint.nv)
            if has_limit[v_slice].all():
                indices.extend(range(joint.idx_v, joint.idx_v + joint.nv))
        self._indices = np.asarray(indices, dtype=int)
        self._projection = (
            None if len(indices) == 0 else np.eye(model.nv)[self._indices]
        )

    def compute(
        self,
        q: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        del q
        if self._projection is None:
            return None
        velocity_limit = dt * self._model.velocityLimit[self._indices]
        return self._projection, -velocity_limit, velocity_limit


def _weighted_task_qp_objective(
    jacobian: np.ndarray,
    error: np.ndarray,
    cost: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weight = np.diag(cost)
    weighted_jacobian = weight @ jacobian
    weighted_error = weight @ (-error)
    H = weighted_jacobian.T @ weighted_jacobian
    g = -weighted_error.T @ weighted_jacobian
    return H, g


def _has_configuration_limit(model: pin.Model) -> np.ndarray:
    return np.logical_and(
        model.hasConfigurationLimit(),
        np.logical_and(
            model.upperPositionLimit < 1e20,
            model.upperPositionLimit > model.lowerPositionLimit + 1e-10,
        ),
    )


def _is_acceptable_result(
    q: np.ndarray,
    *,
    initial_error: _FrameErrorMetrics,
    final_error: _FrameErrorMetrics,
) -> bool:
    if not np.all(np.isfinite(q)):
        return False
    if final_error.converged:
        return True
    if not np.isfinite(initial_error.error_norm) or not np.isfinite(
        final_error.error_norm,
    ):
        return False
    return final_error.error_norm <= initial_error.error_norm


def _error_reduction_ratio(initial_error_norm: float, final_error_norm: float) -> float:
    if not np.isfinite(initial_error_norm) or initial_error_norm <= 0.0:
        return 0.0
    return float((initial_error_norm - final_error_norm) / initial_error_norm)


def _joint_delta_metrics(
    model: pin.Model,
    q_start: np.ndarray,
    q_final: np.ndarray,
    *,
    dt: float,
    max_joint_step_rad: float,
    configuration_limit: _ConfigurationLimit,
    velocity_limit: _VelocityLimit,
) -> _JointDeltaMetrics:
    delta_q = pin.difference(model, q_start, q_final)
    abs_delta = np.abs(delta_q)
    max_delta = float(np.max(abs_delta)) if abs_delta.size else 0.0
    step_limit = float(max_joint_step_rad)
    return _JointDeltaMetrics(
        joint_delta_norm_rad=float(np.linalg.norm(delta_q)),
        max_joint_delta_rad=max_delta,
        left_max_joint_delta_rad=_max_abs_delta(abs_delta, Q_LEFT_ARM),
        right_max_joint_delta_rad=_max_abs_delta(abs_delta, Q_RIGHT_ARM),
        max_joint_step_rad=step_limit,
        joint_step_limit_ratio=0.0 if step_limit <= 0.0 else max_delta / step_limit,
        velocity_limit_active=_velocity_limit_active(
            model,
            delta_q,
            dt=dt,
            velocity_limit=velocity_limit,
        ),
        configuration_limit_active=_configuration_limit_active(
            model,
            q_start,
            delta_q,
            configuration_limit=configuration_limit,
        ),
    )


def _effector_result_fields(
    initial_error: _FrameErrorMetrics,
    final_error: _FrameErrorMetrics,
    joint_delta: _JointDeltaMetrics,
) -> dict[str, float]:
    fields: dict[str, float] = {}
    for effector_id in ("left", "right"):
        initial_position = initial_error.position_error_by_effector.get(
            effector_id,
            0.0,
        )
        initial_orientation = initial_error.orientation_error_by_effector.get(
            effector_id,
            0.0,
        )
        final_position = final_error.position_error_by_effector.get(effector_id, 0.0)
        final_orientation = final_error.orientation_error_by_effector.get(
            effector_id,
            0.0,
        )
        fields[f"{effector_id}_initial_position_error_m"] = initial_position
        fields[f"{effector_id}_initial_orientation_error_rad"] = initial_orientation
        fields[f"{effector_id}_final_position_error_m"] = final_position
        fields[f"{effector_id}_final_orientation_error_rad"] = final_orientation
        fields[f"{effector_id}_position_error_reduction_ratio"] = (
            _error_reduction_ratio(initial_position, final_position)
        )
        fields[f"{effector_id}_orientation_error_reduction_ratio"] = (
            _error_reduction_ratio(initial_orientation, final_orientation)
        )
    fields["left_max_joint_delta_rad"] = joint_delta.left_max_joint_delta_rad
    fields["right_max_joint_delta_rad"] = joint_delta.right_max_joint_delta_rad
    return fields


def _max_abs_delta(abs_delta: np.ndarray, q_slice: slice) -> float:
    selected = abs_delta[q_slice]
    return float(np.max(selected)) if selected.size else 0.0


def _velocity_limit_active(
    model: pin.Model,
    delta_q: np.ndarray,
    *,
    dt: float,
    velocity_limit: _VelocityLimit,
) -> bool:
    indices = velocity_limit._indices
    if indices.size == 0:
        return False
    bounds = dt * model.velocityLimit[indices]
    return bool(
        np.any(np.abs(delta_q[indices]) >= bounds - LIMIT_ACTIVE_TOLERANCE),
    )


def _configuration_limit_active(
    model: pin.Model,
    q_start: np.ndarray,
    delta_q: np.ndarray,
    *,
    configuration_limit: _ConfigurationLimit,
) -> bool:
    indices = configuration_limit._indices
    if indices.size == 0:
        return False
    delta_q_max = pin.difference(model, q_start, model.upperPositionLimit)
    delta_q_min = pin.difference(model, q_start, model.lowerPositionLimit)
    upper = configuration_limit._config_limit_gain * delta_q_max[indices]
    lower = configuration_limit._config_limit_gain * delta_q_min[indices]
    selected_delta = delta_q[indices]
    moving = np.abs(selected_delta) > LIMIT_ACTIVE_TOLERANCE
    return bool(
        np.any(moving & (selected_delta >= upper - LIMIT_ACTIVE_TOLERANCE))
        or np.any(moving & (selected_delta <= lower + LIMIT_ACTIVE_TOLERANCE)),
    )


def _existing_joint_ids(model: pin.Model, joint_names: tuple[str, ...]) -> list[int]:
    joint_ids: list[int] = []
    for joint_name in joint_names:
        joint_id = model.getJointId(joint_name)
        if joint_id < model.njoints:
            joint_ids.append(joint_id)
    return joint_ids


def _home_joint_dict(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> dict[str, float]:
    joints: dict[str, float] = {}
    for index, value in enumerate(left):
        joints[f"left_joint{index + 1}"] = float(value)
    for index, value in enumerate(right):
        joints[f"right_joint{index + 1}"] = float(value)
    return joints


def _as_float_array(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float64)
