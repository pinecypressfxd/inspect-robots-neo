from __future__ import annotations

import importlib.util
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from inspect_robots._pngenc import encode_png


class _Budget(Exception):
    def __init__(self, termination: str) -> None:
        self.termination = termination


@dataclass
class TrialResult:
    process_ok: bool
    termination: str
    diagnostics: Path
    auto_check: Path


def _load_run(program: Path):
    spec = importlib.util.spec_from_file_location(
        "pai_task", program / "__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run


def _write_step(trial: Path, view, targets: dict, frame: int) -> None:
    trial.mkdir(parents=True, exist_ok=True)
    for camera, image in view.images.items():
        dest = trial / "frames" / camera / f"{frame:06d}.png"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(encode_png(image))
    row = {
        "instruction": view.instruction,
        "camera_times": view.camera_times,
        "body_time": view.body_time,
        "eef_state": view.eef_state.tolist(),
        "joint_pos": view.joint_pos.tolist(),
        "targets": targets,
        "frame": frame,
    }
    with (trial / "steps.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def run_trial(program, trial_dir, port, wall_s=600, max_actions=80, abort=None):
    trial = Path(trial_dir)
    trial.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    count = 0
    frame = 0

    def _gate():
        nonlocal count
        if abort is not None and abort():
            raise _Budget("aborted")
        if time.monotonic() - start >= wall_s:
            raise _Budget("wall_clock")
        if count >= max_actions:
            raise _Budget("action_budget")
        count += 1

    class _Counted:
        def observation(self):
            return port.observation()

        def move(self, targets):
            nonlocal frame
            _gate()
            _write_step(trial, port.observation(), dict(targets), frame)
            frame += 1
            return port.move(targets)

        def set_gripper(self, side, width):
            _gate()
            return port.set_gripper(side, width)

        def auto_check(self):
            return port.auto_check()

    termination = "returned"
    ok = True
    try:
        _load_run(Path(program))(_Counted(), trial)
    except _Budget as exc:
        ok = False
        termination = exc.termination
    except Exception:
        ok = False
        termination = "error"
        (trial / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    check = port.auto_check()
    check_path = trial / "auto_check.json"
    check_path.write_text(json.dumps(check), encoding="utf-8")
    diag = trial / "diagnostics.jsonl"
    diag.write_text(
        json.dumps({"process_ok": ok, "termination": termination}) + "\n",
        encoding="utf-8",
    )
    return TrialResult(ok, termination, diag, check_path)


def label_trial(trial_dir, success: bool) -> None:
    path = Path(trial_dir) / "diagnostics.jsonl"
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    row["success"] = bool(success)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
