import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from inspect_robots._pngenc import encode_png
from inspect_robots_pai.domains.nero.motion import DIM_LABELS
from inspect_robots_pai.export.lerobot import export_lerobot

TASK = "put the water bottle on the black pad."
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def _png() -> bytes:
    return encode_png(np.zeros((2, 2), dtype=np.uint8))


def _vectors() -> tuple[list[float], list[float], list[float]]:
    eef = [0.0] * 20
    eef[9] = 0.09
    target = [0.0] * 20
    target[0] = 0.1
    target[9] = 0.09
    joints = [0.0] * 16
    joints[14] = 0.09
    return eef, target, joints


def _write_frames(trial: Path, cameras: tuple[str, ...] = ("left", "right", "chest")) -> None:
    payload = _png()
    for camera in cameras:
        dest = trial / "frames" / camera / "000000.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)


def _write_steps(trial: Path, rows: int = 2) -> None:
    eef, target, joints = _vectors()
    lines = []
    for _ in range(rows):
        lines.append(
            json.dumps(
                {
                    "eef_state": eef,
                    "target_vector": target,
                    "joint_pos": joints,
                    "frame": 0,
                    "camera_times": {"left": 1.5, "right": 1.6, "chest": 1.7},
                    "body_time": 2.0,
                }
            )
        )
    (trial / "steps.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_diagnostics(trial: Path, success: bool) -> None:
    trial.mkdir(parents=True, exist_ok=True)
    (trial / "diagnostics.jsonl").write_text(
        json.dumps({"process_ok": True, "success": success}) + "\n",
        encoding="utf-8",
    )


def _trial(run: Path, index: int, success: bool, *, frames: bool = True) -> None:
    trial = run / "trials" / str(index)
    _write_diagnostics(trial, success)
    _write_steps(trial)
    if frames:
        _write_frames(trial)
    else:
        _write_frames(trial, ("left", "right"))


def test_export_lerobot_keeps_only_complete_success(tmp_path: Path) -> None:
    run = tmp_path / "run"
    out = tmp_path / "out"
    run.mkdir()
    (run / "contract.json").write_text(json.dumps({"task": TASK}) + "\n", encoding="utf-8")
    _trial(run, 0, True)
    _trial(run, 1, False)
    _trial(run, 2, True, frames=False)

    assert export_lerobot(run, out) == 1

    info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["codebase_version"] == "v2.1"
    assert info["fps"] == 30
    assert info["robot_type"] == "nero_vr"
    assert info["auxiliary_action_joints_source"] == "measured_joint_pos"
    assert info["data_path"] == DATA_PATH
    assert info["video_path"] == VIDEO_PATH
    for name in ("tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl", "stats.json"):
        assert (out / "meta" / name).is_file()
    task_line = (out / "meta" / "tasks.jsonl").read_text(encoding="utf-8").splitlines()[0]
    task_row = json.loads(task_line)
    assert task_row["task"] == TASK
    episode = json.loads(
        (out / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert episode["episode_index"] == 0
    assert episode["length"] == 2

    table = pq.read_table(out / "data" / "chunk-000" / "episode_000000.parquet")
    rows = table.to_pylist()
    assert len(rows) == 2
    assert not (out / "data" / "chunk-000" / "episode_000001.parquet").exists()

    state = rows[0]["observation.state"]
    assert len(state) == 16
    assert state[0] == pytest.approx(0)
    assert state[3] == pytest.approx(1)
    assert state[7] == pytest.approx(1)
    assert all(math.isfinite(value) for value in state)
    assert rows[0]["action"][0] == pytest.approx(0.1)
    assert rows[0]["auxiliary.observation.joints"][7] == pytest.approx(1)
    assert rows[0]["auxiliary.action.joints"] == pytest.approx(
        rows[0]["auxiliary.observation.joints"]
    )
    assert rows[0]["frame_index"] == 0
    assert rows[1]["frame_index"] == 1
    assert rows[0]["timestamp"] == pytest.approx(0)
    assert rows[1]["timestamp"] == pytest.approx(1 / 30)
    assert rows[0]["next.done"] is False
    assert rows[1]["next.done"] is True
    assert rows[0]["episode_index"] == 0
    assert rows[0]["auxiliary.alignment.left_camera_id"] == "left_rgbd"
    assert rows[0]["auxiliary.alignment.right_camera_id"] == "right_rgbd"
    assert rows[0]["auxiliary.alignment.chest_camera_id"] == "chest_rgbd"
    assert rows[0]["auxiliary.alignment.left_source_time_ns"] == 1_500_000_000
    assert rows[0]["auxiliary.alignment.observation_time_ns"] == 2_000_000_000
    assert rows[0]["auxiliary.alignment.left_sequence_id"] == 0
    assert rows[1]["auxiliary.alignment.left_sequence_id"] == 1
    assert rows[0]["auxiliary.alignment.max_camera_lag_ms"] == pytest.approx(0)
    assert rows[0]["auxiliary.alignment.max_state_lag_ms"] == pytest.approx(0)
    assert rows[0]["auxiliary.alignment.max_action_lag_ms"] == pytest.approx(0)


def test_export_sources_do_not_name_nero_labels() -> None:
    root = Path("plugins/inspect-robots-pai/src/inspect_robots_pai")
    for relative in ("export/lerobot.py", "cli.py"):
        text = (root / relative).read_text(encoding="utf-8")
        assert "domains.nero" not in text
        for name in DIM_LABELS:
            assert name not in text
