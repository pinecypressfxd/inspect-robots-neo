"""Episode traces for the reference report viewer.

The embodiment writes ``motion.jsonl``. A policy may append
``inference.jsonl`` beside it by reading the ``current`` pointer. The viewer
reads those files and nothing else, so it does not import the policy or the
arm SDK.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def _floats(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=np.float64).ravel()]


class EpisodeTrace:
    """One trial directory under ``root``, plus a ``current`` pointer."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        episode = self.root / f"episode-{stamp}"
        suffix = 0
        while episode.exists():
            suffix += 1
            episode = self.root / f"episode-{stamp}-{suffix}"
        episode.mkdir()
        self.episode = episode
        self._motion = (episode / "motion.jsonl").open("a", encoding="utf-8")
        (self.root / "current").write_text(episode.name + "\n", encoding="utf-8")

    def record_motion(
        self,
        *,
        commanded: np.ndarray,
        measured: np.ndarray,
        joints: np.ndarray,
        solution: np.ndarray,
        success: bool,
        jump: float,
        held: bool,
        issued: np.ndarray | None = None,
        ik: dict[str, Any] | None = None,
    ) -> None:
        """Append one 200 Hz sample. Disk errors must not stop the arm."""
        row: dict[str, Any] = {
            "t": time.time(),
            "commanded": _floats(commanded),
            "measured": _floats(measured),
            "joints": _floats(joints),
            "solution": _floats(solution),
            "success": bool(success),
            "jump": float(jump),
            "held": bool(held),
        }
        if issued is not None:
            row["issued"] = _floats(issued)
        if ik:
            row["ik"] = ik
        try:
            self._motion.write(json.dumps(row, separators=(",", ":")) + "\n")
            self._motion.flush()
        except OSError:
            return

    def close(self) -> None:
        self._motion.close()


def load_episode(episode_dir: Path) -> dict[str, Any]:
    """Read one episode into arrays the viewer plots."""
    motion = _read_jsonl(episode_dir / "motion.jsonl")
    inference = _read_jsonl(episode_dir / "inference.jsonl")
    if not motion:
        return {"episode": episode_dir.name, "samples": 0, "t": [], "inference": []}
    origin = float(motion[0]["t"])
    return {
        "episode": episode_dir.name,
        "samples": len(motion),
        "t": [float(row["t"]) - origin for row in motion],
        "commanded": _columns(motion, "commanded"),
        "measured": _columns(motion, "measured"),
        "joints": _columns(motion, "joints"),
        "solution": _columns(motion, "solution"),
        "jump": [float(row["jump"]) for row in motion],
        "held": [bool(row["held"]) for row in motion],
        "success": [bool(row["success"]) for row in motion],
        "inference": [
            {
                "t": float(row["t"]) - origin,
                "request_id": int(row["request_id"]),
                "latency_s": float(row["latency_s"]),
            }
            for row in inference
            if "t" in row and "latency_s" in row
        ],
    }


def list_episodes(root: Path) -> list[str]:
    """Episode directory names, newest first."""
    if not root.is_dir():
        return []
    episodes = [path for path in root.iterdir() if (path / "motion.jsonl").is_file()]
    episodes.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [path.name for path in episodes]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _columns(rows: list[dict[str, Any]], key: str) -> list[list[float]]:
    width = len(rows[0][key])
    return [[float(row[key][index]) for row in rows] for index in range(width)]
