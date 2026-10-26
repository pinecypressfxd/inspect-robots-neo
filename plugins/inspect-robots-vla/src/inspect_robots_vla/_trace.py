"""Append inference timings next to an embodiment episode trace.

The policy does not import the embodiment. It only appends a line under the
directory named by the embodiment's ``current`` pointer.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def append_inference(
    trace_dir: str,
    request_id: int,
    latency_s: float,
    poses: list[list[float]] | None = None,
) -> None:
    """Record one completed model request. Missing traces are ignored."""
    try:
        root = Path(trace_dir)
        name = (root / "current").read_text(encoding="utf-8").strip()
        if not name:
            return
        episode = root / name
        if not episode.is_dir():
            return
        row = {"t": time.time(), "request_id": request_id, "latency_s": latency_s}
        with (episode / "inference.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        if poses:
            chunk = {"t": row["t"], "request_id": request_id, "poses": poses}
            with (episode / "chunks.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(chunk, separators=(",", ":")) + "\n")
    except OSError:
        return
