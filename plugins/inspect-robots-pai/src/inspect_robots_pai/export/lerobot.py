"""Export successful Pai trials to a LeRobot v2.1 dataset."""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import zlib
from pathlib import Path
from typing import Any

import numpy as np

FPS = 30
EPISODES_PER_CHUNK = 1000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
CAMERAS = ("left", "right", "chest")
CAMERA_IDS = {
    "left": "left_rgbd",
    "right": "right_rgbd",
    "chest": "chest_rgbd",
}
VECTOR_COLUMNS = (
    "observation.state",
    "action",
    "auxiliary.observation.joints",
    "auxiliary.action.joints",
)


def export_lerobot(run_dir: str | Path, out_dir: str | Path) -> int:
    """Write one LeRobot v2.1 dataset and return how many episodes were written."""
    pa, pq = _import_pyarrow()
    run = Path(run_dir)
    out = Path(out_dir)
    task = _task_text(run)
    selected = _selected_trials(run)
    ffmpeg = shutil.which("ffmpeg")
    videos_skipped = ffmpeg is None and bool(selected)
    shape: tuple[int, int, int] | None = None
    episode_rows: list[list[dict[str, Any]]] = []
    global_index = 0
    for episode_index, (trial, steps) in enumerate(selected):
        last = len(steps) - 1
        rows = [
            _row(
                step,
                episode_index,
                frame_index,
                global_index,
                task_index=0,
                is_last=frame_index == last,
            )
            for frame_index, step in enumerate(steps)
        ]
        global_index += len(rows)
        _write_parquet(pa, pq, out, episode_index, rows)
        episode_rows.append(rows)
        if shape is None:
            first = trial / "frames" / "left" / f"{int(steps[0]['frame']):06d}.png"
            shape = _read_png(first).shape
        if ffmpeg is None:
            continue
        for camera in CAMERAS:
            frames = [
                _read_png(trial / "frames" / camera / f"{int(step['frame']):06d}.png")
                for step in steps
            ]
            error = _encode_video(frames, _video_path(out, episode_index, camera), ffmpeg)
            if error is not None:
                videos_skipped = True
    _write_meta(out, task, episode_rows, shape, videos_skipped=videos_skipped)
    return len(selected)


def _import_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required to export LeRobot datasets; "
            "install the export extra: pip install 'inspect-robots-pai[export]'"
        ) from exc
    return pa, pq


def _task_text(run_dir: Path) -> str:
    path = run_dir / "contract.json"
    if not path.is_file():
        return ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("task", ""))


def _selected_trials(run_dir: Path) -> list[tuple[Path, list[dict[str, Any]]]]:
    root = run_dir / "trials"
    if not root.is_dir():
        return []
    trials = [path for path in root.iterdir() if path.is_dir() and path.name.isdigit()]
    selected: list[tuple[Path, list[dict[str, Any]]]] = []
    for trial in sorted(trials, key=lambda path: int(path.name)):
        if not _success(trial):
            continue
        steps = _load_steps(trial)
        if steps is None:
            continue
        selected.append((trial, steps))
    return selected


def _success(trial: Path) -> bool:
    path = trial / "diagnostics.jsonl"
    if not path.is_file():
        return False
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return False
    record = json.loads(lines[0])
    return isinstance(record, dict) and record.get("success") is True


def _load_steps(trial: Path) -> list[dict[str, Any]] | None:
    path = trial / "steps.jsonl"
    if not path.is_file():
        return None
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict) or not _vectors_ok(record):
            return None
        rows.append(record)
    if not rows or not _frames_ready(trial, rows):
        return None
    return rows


def _vectors_ok(row: dict[str, Any]) -> bool:
    expected = {"eef_state": 20, "target_vector": 20, "joint_pos": 16}
    for key, length in expected.items():
        values = row.get(key)
        if not isinstance(values, list) or len(values) != length:
            return False
    return "frame" in row


def _frames_ready(trial: Path, rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        frame = int(row["frame"])
        for camera in CAMERAS:
            if not (trial / "frames" / camera / f"{frame:06d}.png").is_file():
                return False
    return True


def _row(
    step: dict[str, Any],
    episode_index: int,
    frame_index: int,
    global_index: int,
    task_index: int,
    *,
    is_last: bool,
) -> dict[str, Any]:
    state = _pose16(np.asarray(step["eef_state"], dtype=np.float64))
    action = _pose16(np.asarray(step["target_vector"], dtype=np.float64))
    joints = _joints16(np.asarray(step["joint_pos"], dtype=np.float64))
    payload: dict[str, Any] = {
        "observation.state": state,
        "action": action,
        "auxiliary.observation.joints": joints,
        "auxiliary.action.joints": list(joints),
        "timestamp": frame_index / FPS,
        "frame_index": frame_index,
        "episode_index": episode_index,
        "index": global_index + frame_index,
        "task_index": task_index,
        "next.done": is_last,
    }
    payload.update(_alignment(step, frame_index))
    return payload


def _quat_wxyz(matrix: np.ndarray) -> list[float]:
    """Convert one rotation matrix to wxyz. A non-positive trace is identity."""
    trace = float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2])
    if trace > 0:
        scale = (trace + 1.0) ** 0.5 * 2
        return [
            0.25 * scale,
            float(matrix[2, 1] - matrix[1, 2]) / scale,
            float(matrix[0, 2] - matrix[2, 0]) / scale,
            float(matrix[1, 0] - matrix[0, 1]) / scale,
        ]
    return [1.0, 0.0, 0.0, 0.0]


def _pose16(vector: np.ndarray) -> list[float]:
    """Map one 20-d rot6d pose to xyz, wxyz, and a 0-1 gripper, for both arms."""
    out: list[float] = []
    for base in (0, 10):
        out.extend(float(value) for value in vector[base : base + 3])
        rot6d = np.asarray(vector[base + 3 : base + 9], dtype=np.float64)
        out.extend(_quat_wxyz(_rotation(rot6d)))
        width = float(vector[base + 9])
        out.append(float(np.clip(width / 0.09, 0.0, 1.0)))
    return out


def _rotation(rot6d: np.ndarray) -> np.ndarray:
    """Recover a rotation matrix. A near-zero column becomes the identity."""
    x = np.asarray(rot6d[:3], dtype=np.float64)
    y_raw = np.asarray(rot6d[3:], dtype=np.float64)
    if float(np.linalg.norm(x)) < 1e-12 or float(np.linalg.norm(y_raw)) < 1e-12:
        return np.eye(3)
    x = x / max(float(np.linalg.norm(x)), 1e-12)
    y = y_raw - float(np.dot(y_raw, x)) * x
    y_norm = float(np.linalg.norm(y))
    if y_norm < 1e-12:
        return np.eye(3)
    y = y / max(y_norm, 1e-12)
    return np.column_stack((x, y, np.cross(x, y)))


def _joints16(joint_pos: np.ndarray) -> list[float]:
    """Reorder measured joints so each arm's gripper follows its seven joints."""
    left = [float(value) for value in joint_pos[:7]]
    right = [float(value) for value in joint_pos[7:14]]
    left_open = float(np.clip(float(joint_pos[14]) / 0.09, 0.0, 1.0))
    right_open = float(np.clip(float(joint_pos[15]) / 0.09, 0.0, 1.0))
    return [*left, left_open, *right, right_open]


def _alignment(step: dict[str, Any], frame_index: int) -> dict[str, Any]:
    body_ns = _ns(step.get("body_time", 0.0))
    payload: dict[str, Any] = {
        "auxiliary.alignment.profile_family": "rgbd",
        "auxiliary.alignment.profile_view_count": 3,
        "auxiliary.alignment.policy_id": "scheme_d_rgbd_3_view_v1",
        "auxiliary.alignment.watermark_ns": body_ns,
        "auxiliary.alignment.observation_time_ns": body_ns,
        "auxiliary.alignment.state_left_source_sample_time_ns": body_ns,
        "auxiliary.alignment.state_right_source_sample_time_ns": body_ns,
        "auxiliary.alignment.action_left_source_sample_time_ns": body_ns,
        "auxiliary.alignment.action_right_source_sample_time_ns": body_ns,
        "auxiliary.alignment.max_camera_lag_ms": 0.0,
        "auxiliary.alignment.max_state_lag_ms": 0.0,
        "auxiliary.alignment.max_action_lag_ms": 0.0,
    }
    for camera, camera_id in CAMERA_IDS.items():
        stamp = _ns(_camera_time(step, camera))
        payload[f"auxiliary.alignment.{camera}_camera_id"] = camera_id
        payload[f"auxiliary.alignment.{camera}_source_time_ns"] = stamp
        payload[f"auxiliary.alignment.{camera}_receive_time_ns"] = stamp
        payload[f"auxiliary.alignment.{camera}_sequence_id"] = frame_index
    return payload


def _camera_time(step: dict[str, Any], camera: str) -> float:
    times = step.get("camera_times")
    if not isinstance(times, dict):
        return 0.0
    if camera in times:
        return float(times[camera])
    hardware = CAMERA_IDS[camera]
    if hardware in times:
        return float(times[hardware])
    return 0.0


def _ns(seconds: object) -> int:
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        return 0
    return round(float(seconds) * 1_000_000_000)


def _write_parquet(
    pa: Any, pq: Any, out_dir: Path, episode_index: int, rows: list[dict[str, Any]]
) -> None:
    path = out_dir / DATA_PATH.format(
        episode_chunk=episode_index // EPISODES_PER_CHUNK,
        episode_index=episode_index,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=_schema(pa)), path)


def _schema(pa: Any) -> Any:
    vector = pa.list_(pa.float32(), 16)
    fields = [
        pa.field("observation.state", vector),
        pa.field("action", vector),
        pa.field("auxiliary.observation.joints", vector),
        pa.field("auxiliary.action.joints", vector),
        pa.field("timestamp", pa.float64()),
        pa.field("frame_index", pa.int64()),
        pa.field("episode_index", pa.int64()),
        pa.field("index", pa.int64()),
        pa.field("task_index", pa.int64()),
        pa.field("next.done", pa.bool_()),
        pa.field("auxiliary.alignment.profile_family", pa.string()),
        pa.field("auxiliary.alignment.profile_view_count", pa.int64()),
        pa.field("auxiliary.alignment.policy_id", pa.string()),
        pa.field("auxiliary.alignment.watermark_ns", pa.int64()),
        pa.field("auxiliary.alignment.observation_time_ns", pa.int64()),
        pa.field("auxiliary.alignment.state_left_source_sample_time_ns", pa.int64()),
        pa.field("auxiliary.alignment.state_right_source_sample_time_ns", pa.int64()),
        pa.field("auxiliary.alignment.action_left_source_sample_time_ns", pa.int64()),
        pa.field("auxiliary.alignment.action_right_source_sample_time_ns", pa.int64()),
        pa.field("auxiliary.alignment.max_camera_lag_ms", pa.float64()),
        pa.field("auxiliary.alignment.max_state_lag_ms", pa.float64()),
        pa.field("auxiliary.alignment.max_action_lag_ms", pa.float64()),
    ]
    for camera in CAMERAS:
        fields.append(pa.field(f"auxiliary.alignment.{camera}_camera_id", pa.string()))
        fields.append(pa.field(f"auxiliary.alignment.{camera}_source_time_ns", pa.int64()))
        fields.append(pa.field(f"auxiliary.alignment.{camera}_receive_time_ns", pa.int64()))
        fields.append(pa.field(f"auxiliary.alignment.{camera}_sequence_id", pa.int64()))
    return pa.schema(fields)


def _video_path(out_dir: Path, episode_index: int, camera: str) -> Path:
    return out_dir / VIDEO_PATH.format(
        episode_chunk=episode_index // EPISODES_PER_CHUNK,
        video_key=f"observation.images.{camera}",
        episode_index=episode_index,
    )


def _encode_video(frames: list[np.ndarray], dest: Path, ffmpeg: str) -> str | None:
    if not frames:
        return "no frames"
    height, width = int(frames[0].shape[0]), int(frames[0].shape[1])
    dest.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        ffmpeg,
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(FPS),
        str(dest),
    ]
    payload = b"".join(np.ascontiguousarray(frame, dtype=np.uint8).tobytes() for frame in frames)
    completed = subprocess.run(argv, input=payload, capture_output=True, check=False)
    if completed.returncode != 0:
        dest.unlink(missing_ok=True)
        detail = completed.stderr.decode("utf-8", "replace").strip()
        return detail or f"ffmpeg exited {completed.returncode}"
    return None


def _read_png(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"not a png: {path}")
    offset = 8
    width = height = color_type = 0
    idat = bytearray()
    while offset + 8 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        tag = data[offset + 4 : offset + 8]
        chunk = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if tag == b"IHDR":
            width, height, _depth, color_type = struct.unpack(">IIBB", chunk[:10])
        elif tag == b"IDAT":
            idat.extend(chunk)
        elif tag == b"IEND":
            break
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    raw = zlib.decompress(bytes(idat))
    row_size = width * channels
    rows = []
    cursor = 0
    for _ in range(height):
        filt = raw[cursor]
        cursor += 1
        row = bytearray(raw[cursor : cursor + row_size])
        cursor += row_size
        if filt != 0:
            raise ValueError(f"unsupported png filter {filt} in {path}")
        rows.append(row)
    array = np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(height, width, channels)
    if channels == 1:
        array = np.repeat(array, 3, axis=2)
    elif channels == 4:
        array = array[:, :, :3]
    elif channels == 2:
        array = np.repeat(array[:, :, :1], 3, axis=2)
    return np.ascontiguousarray(array)


def _write_meta(
    out_dir: Path,
    task: str,
    episode_rows: list[list[dict[str, Any]]],
    shape: tuple[int, int, int] | None,
    *,
    videos_skipped: bool,
) -> None:
    meta = out_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    count = len(episode_rows)
    total_frames = sum(len(rows) for rows in episode_rows)
    info: dict[str, Any] = {
        "codebase_version": "v2.1",
        "robot_type": "nero_vr",
        "total_episodes": count,
        "total_frames": total_frames,
        "total_tasks": 1 if task else 0,
        "total_videos": 0 if videos_skipped else count * len(CAMERAS),
        "total_chunks": 0 if count == 0 else (count - 1) // EPISODES_PER_CHUNK + 1,
        "chunks_size": EPISODES_PER_CHUNK,
        "fps": FPS,
        "splits": {"train": f"0:{count}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "auxiliary_action_joints_source": "measured_joint_pos",
        "features": _features(shape),
    }
    if videos_skipped:
        info["videos_skipped"] = True
    (meta / "info.json").write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": task}) + "\n",
        encoding="utf-8",
    )
    episodes = [
        {"episode_index": index, "tasks": [task], "length": len(rows)}
        for index, rows in enumerate(episode_rows)
    ]
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in episodes),
        encoding="utf-8",
    )
    episode_stats = [_episode_stats(rows) for rows in episode_rows]
    chunks = [
        {"episode_index": index, "stats": stats} for index, stats in enumerate(episode_stats)
    ]
    (meta / "episodes_stats.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in chunks),
        encoding="utf-8",
    )
    (meta / "stats.json").write_text(
        json.dumps(_aggregate_stats(episode_stats), indent=2) + "\n",
        encoding="utf-8",
    )


def _features(shape: tuple[int, int, int] | None) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for name in VECTOR_COLUMNS:
        features[name] = {"dtype": "float32", "shape": [16], "names": None}
    if shape is not None:
        height, width, channels = shape
        for camera in CAMERAS:
            features[f"observation.images.{camera}"] = {
                "dtype": "video",
                "shape": [height, width, channels],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": FPS,
                    "video.codec": "libx264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
    for name, dtype in (
        ("timestamp", "float64"),
        ("frame_index", "int64"),
        ("episode_index", "int64"),
        ("index", "int64"),
        ("task_index", "int64"),
        ("next.done", "bool"),
    ):
        features[name] = {"dtype": dtype, "shape": [1], "names": None}
    return features


def _episode_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[float] | list[int]]]:
    return {name: _vector_stats([row[name] for row in rows]) for name in VECTOR_COLUMNS}


def _vector_stats(values: list[list[float]]) -> dict[str, list[float] | list[int]]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": [float(value) for value in array.mean(axis=0)],
        "std": [float(value) for value in array.std(axis=0)],
        "min": [float(value) for value in array.min(axis=0)],
        "max": [float(value) for value in array.max(axis=0)],
        "count": [int(array.shape[0])],
    }


def _aggregate_stats(
    chunks: list[dict[str, dict[str, list[float] | list[int]]]],
) -> dict[str, dict[str, list[float] | list[int]]]:
    if not chunks:
        return {}
    names = sorted({name for chunk in chunks for name in chunk})
    return {
        name: _merge_stats([chunk[name] for chunk in chunks if name in chunk])
        for name in names
    }


def _merge_stats(
    parts: list[dict[str, list[float] | list[int]]],
) -> dict[str, list[float] | list[int]]:
    counts = np.asarray([int(part["count"][0]) for part in parts], dtype=np.float64)
    total = float(counts.sum())
    means = np.asarray([part["mean"] for part in parts], dtype=np.float64)
    variances = np.asarray([part["std"] for part in parts], dtype=np.float64) ** 2
    weights = counts.reshape(-1, 1)
    mean = np.sum(means * weights, axis=0) / total
    variance = np.sum((variances + (means - mean) ** 2) * weights, axis=0) / total
    return {
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in np.sqrt(variance)],
        "min": [
            float(min(values))
            for values in zip(*(part["min"] for part in parts), strict=True)
        ],
        "max": [
            float(max(values))
            for values in zip(*(part["max"] for part in parts), strict=True)
        ],
        "count": [int(total)],
    }
