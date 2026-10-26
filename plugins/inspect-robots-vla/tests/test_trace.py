"""Inference lines land beside the embodiment's current episode."""

from __future__ import annotations

import json
from pathlib import Path

from inspect_robots_vla._trace import append_inference


def test_append_inference_follows_the_current_pointer(tmp_path: Path) -> None:
    episode = tmp_path / "episode-1"
    episode.mkdir()
    (tmp_path / "current").write_text("episode-1\n", encoding="utf-8")
    append_inference(str(tmp_path), request_id=4, latency_s=0.21)
    append_inference(str(tmp_path / "missing"), request_id=5, latency_s=0.1)
    row = json.loads((episode / "inference.jsonl").read_text(encoding="utf-8"))
    assert row["request_id"] == 4
    assert row["latency_s"] == 0.21
