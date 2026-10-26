"""Turn inspect episode traces into the robot ITL reference-report payloads.

The page is the original report. These builders only change where the series
come from: commanded pose, issued-joint FK, measured pose, IK scalars, and
model chunks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from inspect_robots_nero._config import (
    GRIPPER_WIDTH_MAX_M,
    IK_DAMPING_COST,
    IK_FRAME_LM_DAMPING,
    IK_FRAME_ORIENTATION_COST,
    IK_FRAME_POSITION_COST,
    IK_POSTURE_COST,
)
from inspect_robots_nero._kinematics import rot6d_to_matrix

_POSE_DIMS = (
    ("x", "x (m)"),
    ("y", "y (m)"),
    ("z", "z (m)"),
    ("roll", "roll (rad)"),
    ("pitch", "pitch (rad)"),
    ("yaw", "yaw (rad)"),
)
_JOINT_DIMS = (
    *((f"joint{index}", f"joint{index} (rad)") for index in range(1, 8)),
    ("gripper", "gripper open ratio"),
)
_IK_DIMS = (
    ("duration_ms", "IK duration (ms)", (("duration_ms", "duration_ms"),)),
    ("iteration_count", "IK iterations", (("iteration_count", "iteration_count"),)),
    (
        "position_error_m",
        "IK position error (m)",
        (("initial_position_error_m", "initial"), ("final_position_error_m", "final")),
    ),
    (
        "orientation_error_rad",
        "IK orientation error (rad)",
        (
            ("initial_orientation_error_rad", "initial"),
            ("final_orientation_error_rad", "final"),
        ),
    ),
    (
        "position_target_margin_m",
        "IK position error minus tolerance (m)",
        (
            ("position_target_margin_m", "final - tolerance"),
            ("target_margin_zero", "tolerance boundary"),
        ),
    ),
    (
        "orientation_target_margin_rad",
        "IK orientation error minus tolerance (rad)",
        (
            ("orientation_target_margin_rad", "final - tolerance"),
            ("target_margin_zero", "tolerance boundary"),
        ),
    ),
    (
        "joint_delta_rad",
        "IK joint delta / step limit (rad)",
        (("max_joint_delta_rad", "max_delta"), ("max_joint_step_rad", "max_step")),
    ),
    (
        "joint_jump",
        "Controller joint jump (rad)",
        (("joint_jump_max_delta_rad", "max_delta"), ("joint_jump_threshold_rad", "threshold")),
    ),
)
_ENV_FIELDS = (
    "env_loop_hz",
    "left_arm_command_publish_hz",
    "right_arm_command_publish_hz",
    "gripper_command_publish_hz",
    "aligned_obs_hz",
    "group_span_ms",
    "robot_state_age_ms",
    "obs_to_request_ms",
    "remote_model_latency_ms",
    "communication_latency_ms",
    "planner_trajectory_hz",
    "planner_request_hz",
)
_ACTION_DT_S = 1.0 / 30.0
_POSE_TOLERANCE_M = 1e-3
_ORI_TOLERANCE_RAD = 1e-3


def pose_compare(root: Path, episodes: list[str], effector: str) -> dict[str, Any]:
    """Pose page: planner chunks, commanded target, issued FK, measured pose."""
    views = [_pose_view(root, name, effector) for name in episodes]
    warnings = [item for view in views for item in view["warnings"]]
    return {
        "episodes": [view["episode"] for view in views],
        "episode_views": views,
        "effector": effector,
        "warnings": warnings,
    }


def joint_compare(root: Path, episodes: list[str], effector: str) -> dict[str, Any]:
    """Joint page: IK solution against measured joints."""
    views = [_joint_view(root, name, effector) for name in episodes]
    warnings = [item for view in views for item in view["warnings"]]
    return {
        "episodes": [view["episode"] for view in views],
        "episode_views": views,
        "effector": effector,
        "warnings": warnings,
    }


def ik_metrics(root: Path, episodes: list[str], effector: str) -> dict[str, Any]:
    """IK page: solver scalars recorded on each control tick."""
    views = [_ik_view(root, name, effector) for name in episodes]
    warnings = [item for view in views for item in view["warnings"]]
    durations = [
        value
        for view in views
        for dimension in view["dimensions"]
        if dimension["key"] == "duration_ms"
        for series in dimension["series"]
        if series["key"] == "duration_ms"
        for value in series["values"]
    ]
    summary = views[0]["metric_summary"] if views else {}
    return {
        "episodes": [view["episode"] for view in views],
        "episode_views": views,
        "effector": effector,
        "warnings": warnings,
        "metric_summary": {
            **summary,
            "sample_count": sum(int(view["metric_summary"]["sample_count"]) for view in views),
            "mean_duration_ms": _mean(durations),
            "p95_duration_ms": _percentile(durations, 95),
            "max_duration_ms": max(durations) if durations else None,
            "joint_jump_reject_count": sum(
                int(view["metric_summary"]["joint_jump_reject_count"]) for view in views
            ),
        },
    }


def env_loop_report(root: Path, episode: str) -> dict[str, Any]:
    """Env-loop cards filled from control-tick spacing and inference latency."""
    rows = _motion(root / episode)
    inferences = _jsonl(root / episode / "inference.jsonl")
    origin = float(rows[0]["t"]) if rows else 0.0
    samples = _env_samples(rows, inferences, origin)
    return {
        "episode": episode,
        "ros2_episode": episode,
        "summary": {
            "sample_count": len(samples),
            "metrics": {field: _metric_stats(samples, field) for field in _ENV_FIELDS},
            "slowest_stage_counts": {},
        },
        "recorder_report": _recorder_report(episode, len(rows), len(inferences)),
        "samples": samples,
        "warnings": [] if rows else [f"{episode}: no motion samples"],
    }


def _pose_view(root: Path, name: str, effector: str) -> dict[str, Any]:
    rows = _motion(root / name)
    chunks = _jsonl(root / name / "chunks.jsonl")
    origin = _origin(rows, chunks)
    offset = 0 if effector == "left" else 10
    warnings: list[str] = []
    if rows and "issued" not in rows[0]:
        warnings.append(f"{name}: issued FK was not recorded; run another episode")
    commanded = _pose_series(rows, "commanded", offset, origin)
    issued = _pose_series(rows, "issued", offset, origin)
    measured = _pose_series(rows, "measured", offset, origin)
    jumps = _jump_series(rows, commanded, origin)
    planner = _chunk_series(chunks, offset, origin)
    return {
        "episode": name,
        "effector": effector,
        "warnings": warnings,
        "dimensions": [
            {
                "key": key,
                "label": label,
                "planner_chunks": planner[key],
                "ik_target": commanded[key],
                "joint_target_fk": issued[key],
                "robot": measured[key],
                "joint_jump": jumps[key],
            }
            for key, label in _POSE_DIMS
        ],
    }


def _joint_view(root: Path, name: str, effector: str) -> dict[str, Any]:
    rows = _motion(root / name)
    origin = _origin(rows, [])
    start = 0 if effector == "left" else 7
    grip = 9 if effector == "left" else 19
    target = {key: {"times": [], "values": []} for key, _ in _JOINT_DIMS}
    current = {key: {"times": [], "values": []} for key, _ in _JOINT_DIMS}
    jumps = {key: {"times": [], "values": [], "max_deltas": []} for key, _ in _JOINT_DIMS}
    for row in rows:
        time_s = float(row["t"]) - origin
        solution = row.get("solution") or []
        joints = row.get("joints") or []
        for index, (key, _) in enumerate(_JOINT_DIMS[:-1]):
            if start + index < len(solution):
                target[key]["times"].append(time_s)
                target[key]["values"].append(float(solution[start + index]))
            if start + index < len(joints):
                current[key]["times"].append(time_s)
                current[key]["values"].append(float(joints[start + index]))
        commanded = row.get("commanded") or []
        if grip < len(commanded):
            ratio = float(commanded[grip]) / GRIPPER_WIDTH_MAX_M
            target["gripper"]["times"].append(time_s)
            target["gripper"]["values"].append(ratio)
            current["gripper"]["times"].append(time_s)
            current["gripper"]["values"].append(ratio)
        if row.get("held") and target["joint1"]["values"]:
            for key, _ in _JOINT_DIMS:
                if not target[key]["values"]:
                    continue
                jumps[key]["times"].append(time_s)
                jumps[key]["values"].append(target[key]["values"][-1])
                jumps[key]["max_deltas"].append(float(row.get("jump") or 0.0))
    return {
        "episode": name,
        "effector": effector,
        "warnings": [],
        "dimensions": [
            {
                "key": key,
                "label": label,
                "target": target[key],
                "current": current[key],
                "joint_jump": jumps[key],
            }
            for key, label in _JOINT_DIMS
        ],
    }


def _ik_view(root: Path, name: str, effector: str) -> dict[str, Any]:
    rows = _motion(root / name)
    origin = _origin(rows, [])
    prefix = f"{effector}_"
    dimensions = []
    for key, label, fields in _IK_DIMS:
        series = []
        for field, series_label in fields:
            times: list[float] = []
            values: list[float] = []
            for row in rows:
                value = _ik_value(row.get("ik") or {}, field, prefix)
                if value is None:
                    continue
                times.append(float(row["t"]) - origin)
                values.append(value)
            series.append({"key": field, "label": series_label, "times": times, "values": values})
        dimensions.append({"key": key, "label": label, "series": series})
    rejects = sum(
        1 for row in rows if (row.get("ik") or {}).get("joint_jump_rejected") or row.get("held")
    )
    durations = [
        float(row["ik"]["duration_ms"])
        for row in rows
        if isinstance(row.get("ik"), dict) and row["ik"].get("duration_ms") is not None
    ]
    steps = [
        float(row["ik"]["max_joint_step_rad"])
        for row in rows
        if isinstance(row.get("ik"), dict) and row["ik"].get("max_joint_step_rad") is not None
    ]
    return {
        "episode": name,
        "effector": effector,
        "warnings": [] if any(row.get("ik") for row in rows) else [f"{name}: no IK diagnostics"],
        "dimensions": dimensions,
        "ten_iteration_failures": [],
        "evidence_summary": {"ten_iteration_failure_count": 0, "groups": {}},
        "time_window": {
            "start_s": 0.0,
            "end_s": 0.0 if not rows else float(rows[-1]["t"]) - origin,
            "start_event": "start_key_s",
            "end_event": "last_sample",
            "end_source": "last_ik_sample",
        },
        "metric_summary": {
            "sample_count": len(rows),
            "ik_solver_name": "proxqp",
            "mean_duration_ms": _mean(durations),
            "p95_duration_ms": _percentile(durations, 95),
            "max_duration_ms": max(durations) if durations else None,
            "max_joint_step_rad": max(steps) if steps else None,
            "task_weights": {
                "frame_position": IK_FRAME_POSITION_COST,
                "frame_orientation": IK_FRAME_ORIENTATION_COST,
                "posture": IK_POSTURE_COST,
                "damping": IK_DAMPING_COST,
            },
            "task_regularization": {"frame_lm_damping": IK_FRAME_LM_DAMPING},
            "joint_jump_reject_count": rejects,
        },
    }


def _pose_series(
    rows: list[dict[str, Any]],
    key: str,
    offset: int,
    origin: float,
) -> dict[str, dict[str, list[float]]]:
    series = {name: {"times": [], "values": []} for name, _ in _POSE_DIMS}
    for row in rows:
        pose = row.get(key)
        if not isinstance(pose, list) or len(pose) < offset + 10:
            continue
        values = _pose_values(pose, offset)
        time_s = float(row["t"]) - origin
        for name, _ in _POSE_DIMS:
            series[name]["times"].append(time_s)
            series[name]["values"].append(values[name])
    return series


def _jump_series(
    rows: list[dict[str, Any]],
    commanded: dict[str, dict[str, list[float]]],
    origin: float,
) -> dict[str, dict[str, list[float]]]:
    del origin
    series = {name: {"times": [], "values": [], "max_deltas": []} for name, _ in _POSE_DIMS}
    index = 0
    for row in rows:
        if not isinstance(row.get("commanded"), list):
            continue
        if row.get("held"):
            for name, _ in _POSE_DIMS:
                series[name]["times"].append(commanded[name]["times"][index])
                series[name]["values"].append(commanded[name]["values"][index])
                series[name]["max_deltas"].append(float(row.get("jump") or 0.0))
        index += 1
    return series


def _chunk_series(
    chunks: list[dict[str, Any]],
    offset: int,
    origin: float,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in _POSE_DIMS}
    for chunk in chunks:
        poses = chunk.get("poses")
        if not isinstance(poses, list):
            continue
        start = float(chunk["t"])
        series = {name: {"times": [], "values": []} for name, _ in _POSE_DIMS}
        for index, pose in enumerate(poses):
            if not isinstance(pose, list) or len(pose) < offset + 10:
                continue
            values = _pose_values(pose, offset)
            time_s = start + (index + 1) * _ACTION_DT_S - origin
            for name, _ in _POSE_DIMS:
                series[name]["times"].append(time_s)
                series[name]["values"].append(values[name])
        for name, _ in _POSE_DIMS:
            output[name].append({"request_id": chunk.get("request_id"), **series[name]})
    return output


def _pose_values(pose: list[float], offset: int) -> dict[str, float]:
    try:
        matrix = rot6d_to_matrix(np.asarray(pose[offset + 3 : offset + 9], dtype=np.float64))
        roll, pitch, yaw = Rotation.from_matrix(matrix).as_euler("xyz")
    except ValueError:
        roll, pitch, yaw = 0.0, 0.0, 0.0
    return {
        "x": float(pose[offset]),
        "y": float(pose[offset + 1]),
        "z": float(pose[offset + 2]),
        "roll": float(roll),
        "pitch": float(pitch),
        "yaw": float(yaw),
    }


def _ik_value(ik: dict[str, Any], field: str, prefix: str) -> float | None:
    if field == "target_margin_zero":
        return 0.0
    if field == "position_target_margin_m":
        value = _raw_ik(ik, "final_position_error_m", prefix)
        return None if value is None else value - _POSE_TOLERANCE_M
    if field == "orientation_target_margin_rad":
        value = _raw_ik(ik, "final_orientation_error_rad", prefix)
        return None if value is None else value - _ORI_TOLERANCE_RAD
    return _raw_ik(ik, field, prefix)


def _raw_ik(ik: dict[str, Any], field: str, prefix: str) -> float | None:
    selected = ik.get(prefix + field)
    value = ik.get(field) if selected is None else selected
    if value is None or isinstance(value, bool):
        return None
    return float(value)


def _env_samples(
    rows: list[dict[str, Any]],
    inferences: list[dict[str, Any]],
    origin: float,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    previous: float | None = None
    for index, row in enumerate(rows):
        if index % 10:
            continue
        stamp = float(row["t"])
        period = None if previous is None else stamp - previous
        previous = stamp
        hz = None if not period or period <= 0 else 1.0 / period
        samples.append(
            {
                "time_s": stamp - origin,
                "env_loop_hz": hz,
                "left_arm_command_publish_hz": hz,
                "right_arm_command_publish_hz": hz,
                "gripper_command_publish_hz": hz,
            }
        )
    request_times = [float(row["t"]) for row in inferences if "t" in row]
    for index, row in enumerate(inferences):
        stamp = float(row["t"])
        gap = None if index == 0 else stamp - request_times[index - 1]
        samples.append(
            {
                "time_s": stamp - origin,
                "remote_model_latency_ms": float(row.get("latency_s") or 0.0) * 1000.0,
                "planner_request_hz": None if not gap or gap <= 0 else 1.0 / gap,
            }
        )
    samples.sort(key=lambda sample: float(sample["time_s"]))
    return samples


def _metric_stats(samples: list[dict[str, Any]], field: str) -> dict[str, float | None]:
    values = [
        float(sample[field])
        for sample in samples
        if isinstance(sample.get(field), int | float) and not isinstance(sample.get(field), bool)
    ]
    targets = {
        "env_loop_hz": 200.0,
        "left_arm_command_publish_hz": 200.0,
        "right_arm_command_publish_hz": 200.0,
        "gripper_command_publish_hz": 200.0,
        "planner_request_hz": 5.0,
        "planner_trajectory_hz": 30.0,
    }
    target = targets.get(field)
    return {
        "mean": _mean(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "target": target,
        "breach_rate": None,
    }


def _recorder_report(episode: str, motions: int, inferences: int) -> dict[str, Any]:
    return {
        "summary": {
            "latest": {},
            "integrity": {},
            "raw": {
                "recording_status": "completed" if motions else "unavailable",
                "close_reason": episode,
                "verification_status": "inspect_trace",
                "dataset_eligible": False,
                "diagnostics": {
                    "error_count": 0,
                    "warning_count": 0,
                    "errors": [],
                    "warnings": [],
                },
                "lerobot_readiness": {
                    "status": "not_checked",
                    "candidate_frame_count": motions,
                    "accepted_frame_count": inferences,
                },
                "artifact_coverage": {},
                "camera_coverage": {"cameras": []},
                "robot_state_coverage": {"left": {"rows": motions}, "right": {"rows": motions}},
                "data_volume": {},
            },
        }
    }


def _motion(episode: Path) -> list[dict[str, Any]]:
    return _jsonl(episode / "motion.jsonl")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _origin(rows: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> float:
    stamps = [float(row["t"]) for row in rows if "t" in row]
    stamps.extend(float(row["t"]) for row in chunks if "t" in row)
    return min(stamps) if stamps else 0.0


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
