"""Convert one finished Inspect Robots run into a LeRobot v2.1 dataset.

Usage (from the plugin checkout):

    python scripts/export_lerobot.py LOG.json [-o OUT_DIR] [--images] [--fps 30]

Reads a saved ``EvalLog`` plus its side-car stores — the per-trial actions
JSONL (``trial_metadata["actions"]``) and the streamed camera frames under
``stats.frames_dir`` — and writes a LeRobot v2.1 dataset:

- ``meta/info.json``, ``meta/tasks.jsonl``, ``meta/episodes.jsonl``
- ``data/chunk-000/episode_XXXXXX.parquet`` (one per exported trial)
- ``videos/chunk-000/observation.images.<camera>/episode_XXXXXX.mp4`` per
  camera per trial, or ``images/<camera>/episode_XXXXXX/frame_XXXXXX.png``
  trees with ``--images``.

Each exported trial becomes one episode with one row per executed action step:
``frame_index`` is the row index, ``timestamp`` is ``frame_index / fps``, and
each camera video holds the observation frame captured at that step (the frame
store writes one reset frame at ``t=0`` and one post-action frame per step, so
a trial with N actions carries frames ``t=0..N`` and the final post-action
frame is not exported). Trials missing frames for any action step are skipped
with a warning. The ``observation.state`` column (16-dim ``joint_pos``) is
written only when every exported trial's recorded transcript carries a
``state[joint_pos]`` line per policy observation that aligns with the action
rows; otherwise the column is dropped everywhere (see the plugin README).

Needs ``pyarrow`` (``pip install pyarrow``); video mode additionally needs an
``ffmpeg`` binary on PATH (the core shared encoder pipes frames into it).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from inspect_robots._html import _trial_camera_streams
from inspect_robots._pngenc import encode_png
from inspect_robots._pointers import read_jsonl_prefix, resolve_log_pointer
from inspect_robots._video import encode_stream, resolve_frames_dir
from inspect_robots.frames import _safe
from inspect_robots.log import EvalLog, SceneResult, read_eval_log

#: Feature-group names and path templates mirror the bring-up converter
#: (neo_manipulation ``raw_to_lerobot.py``); that converter's alignment
#: ``auxiliary.*`` fields describe a rig this plugin does not have and are
#: deliberately not ported.
ACTION_GROUP = "absolute_ee_target_base"
STATE_GROUP = "joint_position"
STATE_DIM = 16
JOINT_LABELS = (
    *(f"{side}_joint_{index}" for side in ("left", "right") for index in range(1, 8)),
    "left_gripper_open",
    "right_gripper_open",
)
EPISODES_PER_CHUNK = 1000
DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
_OBSERVATION_PREFIX = "Current observation."
_JOINT_LINE_RE = re.compile(r"^state\[joint_pos\]: \[([^\]]*)\]$")


@dataclass
class _Actions:
    """One trial's executed actions from the side-car JSONL."""

    labels: tuple[str, ...]
    rows: tuple[tuple[int, tuple[float, ...]], ...]


@dataclass
class _Trial:
    """Everything one exported episode is built from."""

    scene: SceneResult
    epoch: int
    actions: _Actions
    streams: dict[str, list[tuple[int, Path]]] = field(default_factory=dict)
    frames: dict[str, list[tuple[int, Path]]] = field(default_factory=dict)
    states: list[list[float]] | None = None

    @property
    def label(self) -> str:
        """The trial id the frame store and the side-car file name use."""
        return f"{self.scene.scene_id}-e{self.epoch}"

    @property
    def instruction(self) -> str:
        """The task string for this trial's episodes entry."""
        return self.scene.instruction or self.scene.scene_id


def _warn(message: str) -> None:
    """Print one export warning."""
    print(f"warning: {message}", file=sys.stderr)


def _fail(message: str) -> NoReturn:
    """Print one fatal export error and exit 1."""
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def _is_number(value: object) -> bool:
    """True for plain ints and floats; bools are not numbers here."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _import_pyarrow() -> tuple[Any, Any]:
    """Import pyarrow lazily; an ImportError exits 2 with the install hint."""
    try:
        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError as exc:
        print(
            f"error: pyarrow is required for the parquet shards (pip install pyarrow): {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    return pa, pq


def _load_actions(log_path: Path, metadata: Mapping[str, Any], label: str) -> _Actions | None:
    """Load one trial's actions side-car; ``None`` (with a warning) if unusable."""
    target = resolve_log_pointer(log_path, metadata.get("actions"))
    if target is None:
        _warn(f"skipping {label}: no actions side-car recorded for this trial")
        return None
    rows_raw = read_jsonl_prefix(target)
    if not rows_raw:
        _warn(f"skipping {label}: actions side-car {target.name} has no readable rows")
        return None
    header = rows_raw[0]
    action_dim = header.get("action_dim")
    if not isinstance(action_dim, int) or isinstance(action_dim, bool) or action_dim <= 0:
        _warn(f"skipping {label}: actions side-car header has no usable action_dim")
        return None
    labels_value = header.get("labels")
    labels = (
        tuple(str(name) for name in labels_value)
        if isinstance(labels_value, list) and len(labels_value) == action_dim
        else tuple(f"dim_{index}" for index in range(action_dim))
    )
    rows: list[tuple[int, tuple[float, ...]]] = []
    for row in rows_raw[1:]:
        if "action_dim" in row:
            continue
        t, action = row.get("t"), row.get("action")
        if (
            not isinstance(t, int)
            or isinstance(t, bool)
            or not isinstance(action, list)
            or len(action) != action_dim
            or not all(_is_number(value) for value in action)
        ):
            _warn(f"skipping {label}: malformed actions row {t!r}")
            return None
        values = tuple(float(value) for value in action)
        if not all(math.isfinite(value) for value in values):
            _warn(f"skipping {label}: non-finite action at t={t}")
            return None
        rows.append((t, values))
    if not rows:
        _warn(f"skipping {label}: actions side-car holds no action rows")
        return None
    return _Actions(labels=labels, rows=tuple(rows))


def _observation_joint_pos(messages: Sequence[Any]) -> list[list[float]]:
    """Extract the ``joint_pos`` vector from every observation message.

    Only agent-policy observation messages count (first text part starts with
    ``Current observation.``), and only the plain vector rendering
    ``state[joint_pos]: [v, ...]`` parses. The labeled per-joint rendering a
    joint-space policy produces is deliberately not scraped out of prose.
    """
    vectors: list[list[float]] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text":
                continue
            text = part.get("text")
            if not isinstance(text, str) or not text.startswith(_OBSERVATION_PREFIX):
                continue
            for line in text.splitlines():
                match = _JOINT_LINE_RE.match(line)
                if match is None:
                    continue
                try:
                    vector = [float(token) for token in match.group(1).split(",")]
                except ValueError:
                    break
                if len(vector) == STATE_DIM and all(math.isfinite(v) for v in vector):
                    vectors.append(vector)
                break
            break
    return vectors


def _aligned_states(vectors: Sequence[list[float]], row_count: int) -> list[list[float]] | None:
    """Expand per-observation states to per-action-row states, or ``None``.

    One observation per action row maps 1:1 (horizon-1 policies); otherwise a
    uniform chunk size is accepted (``row_count % len(vectors) == 0``, each
    observation covering that many consecutive rows). Anything else has no
    defensible alignment and drops the column.
    """
    count = len(vectors)
    if count == 0 or count > row_count or row_count % count != 0:
        return None
    chunk = row_count // count
    return [vectors[index // chunk] for index in range(row_count)]


def _trial_states(
    log_path: Path, scene: SceneResult, epoch: int, row_count: int
) -> list[list[float]] | None:
    """Per-action-row ``joint_pos`` for one trial, or ``None`` when unavailable.

    The saved log carries no per-step state; the agent policy's recorded
    transcript (inline in the sample, else the side-car via the ``transcript``
    pointer) is the only in-log record of the observed 16-dim vector.
    """
    messages: Any = None
    if epoch < len(scene.policy_transcripts):
        messages = scene.policy_transcripts[epoch]
    if not messages:
        target = resolve_log_pointer(log_path, scene.trial_metadata[epoch].get("transcript"))
        messages = read_jsonl_prefix(target) if target is not None else None
    if not messages:
        return None
    return _aligned_states(_observation_joint_pos(messages), row_count)


def _load_frame(path: Path) -> np.ndarray:
    """Load one stored frame, exiting loudly on truncated or non-uint8 files."""
    try:
        array = np.load(path)
    except Exception as exc:
        _fail(f"unreadable frame {path.name}: {exc}")
    if array.dtype != np.uint8:
        _fail(f"unsupported dtype {array.dtype} in {path.name} (frames are uint8)")
    return np.asarray(array)


def _probe_shape(
    frames: Sequence[tuple[int, Path]], *, normalized_channels: bool
) -> tuple[int, int, int]:
    """First usable frame's ``(height, width, channels)`` for the feature table.

    ``normalized_channels`` mirrors what the shared video encoder emits (2-D
    gray replicated to 3 channels, alpha dropped); image mode reports the
    channels actually stored. Empty leading frames are skipped exactly like
    the encoder's own probe.
    """
    for _step, path in frames:
        array = _load_frame(path)
        if array.size == 0:
            continue
        height, width = int(array.shape[0]), int(array.shape[1])
        if array.ndim == 2:
            return height, width, 3 if normalized_channels else 1
        if array.ndim == 3 and array.shape[2] in (1, 3, 4):
            return height, width, 3 if normalized_channels else int(array.shape[2])
        _fail(f"unsupported shape {array.shape} in {path.name}")
    _fail("no usable frames")


def _episode_videos(
    out_dir: Path,
    episode_index: int,
    trial: _Trial,
    cameras: Sequence[str],
    fps: float,
    ffmpeg: str | None,
) -> dict[str, tuple[int, int, int]]:
    """Encode one episode's per-camera MP4s; returns per-camera shapes."""
    if ffmpeg is None:
        _fail("ffmpeg not found on PATH; install it or pass --images")
    shapes: dict[str, tuple[int, int, int]] = {}
    for camera in cameras:
        frames = trial.frames[camera]
        shape = _probe_shape(frames, normalized_channels=True)
        out_path = out_dir / VIDEO_PATH_TEMPLATE.format(
            episode_chunk=episode_index // EPISODES_PER_CHUNK,
            video_key=f"observation.images.{camera}",
            episode_index=episode_index,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result = encode_stream(frames, out_path, fps, ffmpeg)
        if result.error is not None:
            _fail(f"encoding {camera} video for {trial.label}: {result.error}")
        shapes[camera] = shape
    return shapes


def _episode_images(
    out_dir: Path,
    episode_index: int,
    trial: _Trial,
    cameras: Sequence[str],
) -> dict[str, tuple[int, int, int]]:
    """Write one episode's per-camera PNG trees; returns per-camera shapes."""
    shapes: dict[str, tuple[int, int, int]] = {}
    for camera in cameras:
        frames = trial.frames[camera]
        shape = _probe_shape(frames, normalized_channels=False)
        directory = out_dir / "images" / camera / f"episode_{episode_index:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        for row_index, (_step, path) in enumerate(frames):
            payload = encode_png(_load_frame(path))
            (directory / f"frame_{row_index:06d}.png").write_bytes(payload)
        shapes[camera] = shape
    return shapes


def _episode_rows(
    trial: _Trial,
    episode_index: int,
    task_index: int,
    first_index: int,
    fps: float,
    keep_state: bool,
) -> list[dict[str, Any]]:
    """Build one episode's parquet rows: one per executed action step."""
    rows: list[dict[str, Any]] = []
    total = len(trial.actions.rows)
    for row_index in range(total):
        row: dict[str, Any] = {
            "action": list(trial.actions.rows[row_index][1]),
            "timestamp": row_index / fps,
            "frame_index": row_index,
            "episode_index": episode_index,
            "index": first_index + row_index,
            "task_index": task_index,
            "next.done": row_index == total - 1,
        }
        if keep_state and trial.states is not None:
            row["observation.state"] = list(trial.states[row_index])
        rows.append(row)
    return rows


def _write_episode_parquet(
    pa: Any,
    pq: Any,
    out_dir: Path,
    episode_index: int,
    rows: list[dict[str, Any]],
    action_dim: int,
    keep_state: bool,
) -> None:
    """Write one episode's parquet shard with the fixed LeRobot column layout."""
    fields = []
    if keep_state:
        fields.append(pa.field("observation.state", pa.list_(pa.float32(), list_size=STATE_DIM)))
    fields.append(pa.field("action", pa.list_(pa.float32(), list_size=action_dim)))
    fields.extend(
        [
            pa.field("timestamp", pa.float64()),
            pa.field("frame_index", pa.int64()),
            pa.field("episode_index", pa.int64()),
            pa.field("index", pa.int64()),
            pa.field("task_index", pa.int64()),
            pa.field("next.done", pa.bool_()),
        ]
    )
    path = out_dir / DATA_PATH_TEMPLATE.format(
        episode_chunk=episode_index // EPISODES_PER_CHUNK, episode_index=episode_index
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=pa.schema(fields)), path)


def _scalar_feature(dtype: str) -> dict[str, Any]:
    """One scalar index feature spec in the bring-up converter's shape."""
    return {"dtype": dtype, "shape": [1], "names": None}


def _features(
    cameras: Sequence[str],
    shapes: Mapping[str, tuple[int, int, int]],
    action_labels: Sequence[str],
    keep_state: bool,
    *,
    images: bool,
    fps: float,
) -> dict[str, dict[str, Any]]:
    """Assemble the ``meta/info.json`` feature table."""
    features: dict[str, dict[str, Any]] = {}
    if keep_state:
        features["observation.state"] = {
            "dtype": "float32",
            "shape": [STATE_DIM],
            "names": {STATE_GROUP: list(JOINT_LABELS)},
        }
    features["action"] = {
        "dtype": "float32",
        "shape": [len(action_labels)],
        "names": {ACTION_GROUP: list(action_labels)},
    }
    for camera in cameras:
        height, width, channels = shapes[camera]
        entry: dict[str, Any] = {
            "dtype": "image" if images else "video",
            "shape": [height, width, channels],
            "names": ["height", "width", "channel"],
        }
        if not images:
            entry["video_info"] = {
                "video.fps": fps,
                "video.codec": "libx264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            }
        features[f"observation.images.{camera}"] = entry
    features["timestamp"] = _scalar_feature("float64")
    features["frame_index"] = _scalar_feature("int64")
    features["episode_index"] = _scalar_feature("int64")
    features["index"] = _scalar_feature("int64")
    features["task_index"] = _scalar_feature("int64")
    features["next.done"] = _scalar_feature("bool")
    return features


def _build_parser() -> argparse.ArgumentParser:
    """Build the export CLI parser."""
    parser = argparse.ArgumentParser(
        description="Export one finished run as a LeRobot v2.1 dataset"
    )
    parser.add_argument("log", help="path to the saved EvalLog JSON")
    parser.add_argument(
        "-o", "--out", help="output dataset directory (default: <log dir>/<log stem>-lerobot)"
    )
    parser.add_argument(
        "--images",
        action="store_true",
        help="write per-frame PNG trees instead of MP4 videos",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="dataset fps (default: 30)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one export; returns a process exit code."""
    args = _build_parser().parse_args(argv)
    fps = float(args.fps)
    if not (fps > 0 and math.isfinite(fps)):
        _fail("--fps must be a positive finite number")
    log_path = Path(args.log)
    try:
        log: EvalLog = read_eval_log(str(log_path))
    except (OSError, ValueError, KeyError) as exc:
        _fail(f"could not read log {log_path}: {exc}")
    frames_dir = (
        resolve_frames_dir(log.stats.frames_dir, log_path)
        if log.stats.frames_dir is not None
        else None
    )
    if frames_dir is None:
        _fail(
            f"no frame store found for this log (stats.frames_dir: "
            f"{log.stats.frames_dir!r}); rerun the eval with frame logging on"
        )

    candidates: list[_Trial] = []
    for scene in log.samples:
        for epoch, metadata in enumerate(scene.trial_metadata):
            actions = _load_actions(log_path, metadata, f"{scene.scene_id}-e{epoch}")
            if actions is None:
                continue
            candidates.append(
                _Trial(
                    scene=scene,
                    epoch=epoch,
                    actions=actions,
                    streams=_trial_camera_streams(frames_dir, _safe(f"{scene.scene_id}-e{epoch}")),
                )
            )
    cameras = sorted({camera for trial in candidates for camera in trial.streams})
    if not cameras:
        _fail(f"no stored camera frames for this log's trials under {frames_dir}")
    included: list[_Trial] = []
    for trial in candidates:
        gaps: list[str] = []
        for camera in cameras:
            by_step = dict(trial.streams.get(camera, ()))
            if any(t not in by_step for t, _values in trial.actions.rows):
                gaps.append(camera)
        if gaps:
            _warn(
                f"skipping {trial.label}: no stored frames for some action steps "
                f"(cameras: {', '.join(gaps)})"
            )
            continue
        trial.frames = {
            camera: [(t, dict(trial.streams[camera])[t]) for t, _values in trial.actions.rows]
            for camera in cameras
        }
        included.append(trial)
    if not included:
        _fail("no trial in this log has stored frames for its action steps")

    keep_state = True
    for trial in included:
        trial.states = _trial_states(log_path, trial.scene, trial.epoch, len(trial.actions.rows))
        if trial.states is None:
            keep_state = False
    if keep_state:
        print(f"observation.state: joint_pos recovered for all {len(included)} trial(s)")
    else:
        print(
            "note: observation.state omitted; at least one exported trial has no "
            "step-aligned joint_pos record in its transcript"
        )

    out_dir = Path(args.out) if args.out else log_path.parent / f"{log_path.stem}-lerobot"
    if out_dir.is_file() or (out_dir.is_dir() and any(out_dir.iterdir())):
        _fail(f"output directory is not empty: {out_dir}")
    pa, pq = _import_pyarrow()
    ffmpeg = shutil.which("ffmpeg")

    first_shapes: dict[str, tuple[int, int, int]] = {}
    task_index_of: dict[str, int] = {}
    episode_entries: list[dict[str, Any]] = []
    total_frames = 0
    for episode_index, trial in enumerate(included):
        shapes = (
            _episode_images(out_dir, episode_index, trial, cameras)
            if args.images
            else _episode_videos(out_dir, episode_index, trial, cameras, fps, ffmpeg)
        )
        for camera, shape in shapes.items():
            previous = first_shapes.setdefault(camera, shape)
            if previous != shape:
                _fail(
                    f"{camera} frame shape changed between episodes "
                    f"({previous} -> {shape}) at {trial.label}"
                )
        task_index = task_index_of.setdefault(trial.instruction, len(task_index_of))
        rows = _episode_rows(trial, episode_index, task_index, total_frames, fps, keep_state)
        _write_episode_parquet(
            pa, pq, out_dir, episode_index, rows, len(trial.actions.labels), keep_state
        )
        episode_entries.append(
            {"episode_index": episode_index, "tasks": [trial.instruction], "length": len(rows)}
        )
        total_frames += len(rows)

    info: dict[str, Any] = {
        "codebase_version": "v2.1",
        "robot_type": log.eval.embodiment,
        "total_episodes": len(included),
        "total_frames": total_frames,
        "total_tasks": len(task_index_of),
        "total_videos": 0 if args.images else len(included) * len(cameras),
        "total_chunks": (len(included) - 1) // EPISODES_PER_CHUNK + 1,
        "chunks_size": EPISODES_PER_CHUNK,
        "fps": int(fps) if fps.is_integer() else fps,
        "splits": {"train": f"0:{len(included)}"},
        "data_path": DATA_PATH_TEMPLATE,
        "features": _features(
            cameras,
            first_shapes,
            included[0].actions.labels,
            keep_state,
            images=args.images,
            fps=fps,
        ),
    }
    if not args.images:
        info["video_path"] = VIDEO_PATH_TEMPLATE
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "info.json").write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")
    (meta_dir / "tasks.jsonl").write_text(
        "".join(
            json.dumps({"task_index": index, "task": instruction}) + "\n"
            for instruction, index in task_index_of.items()
        ),
        encoding="utf-8",
    )
    (meta_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in episode_entries), encoding="utf-8"
    )
    print(
        f"wrote LeRobot v2.1 dataset: {out_dir} "
        f"({len(included)} episodes, {total_frames} frames, cameras: {', '.join(cameras)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
