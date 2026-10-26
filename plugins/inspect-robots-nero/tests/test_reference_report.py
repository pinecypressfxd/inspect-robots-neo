"""Trace files and the reference-report HTTP surface."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from inspect_robots_nero._report_viewer import ReportViewer
from inspect_robots_nero._trace import EpisodeTrace, list_episodes, load_episode


def test_episode_trace_round_trips_motion_and_inference(tmp_path: Path) -> None:
    trace = EpisodeTrace(tmp_path)
    trace.record_motion(
        commanded=np.arange(20),
        measured=np.arange(20) + 0.1,
        joints=np.zeros(14),
        solution=np.ones(14),
        success=True,
        jump=0.02,
        held=False,
    )
    trace.close()
    inference = trace.episode / "inference.jsonl"
    inference.write_text(
        '{"t": 1.0, "request_id": 3, "latency_s": 0.18}\n',
        encoding="utf-8",
    )
    payload = load_episode(trace.episode)
    assert list_episodes(tmp_path) == [trace.episode.name]
    assert payload["samples"] == 1
    assert payload["commanded"][0] == [0.0]
    assert payload["jump"] == [0.02]
    assert payload["inference"][0]["request_id"] == 3
    assert payload["inference"][0]["latency_s"] == 0.18


def test_viewer_serves_the_episode(tmp_path: Path) -> None:
    trace = EpisodeTrace(tmp_path)
    trace.record_motion(
        commanded=np.zeros(20),
        measured=np.zeros(20),
        joints=np.zeros(14),
        solution=np.zeros(14),
        success=True,
        jump=0.0,
        held=False,
    )
    trace.close()
    server = ReportViewer(tmp_path, "127.0.0.1", 0)
    try:
        import threading

        threading.Thread(target=server.serve_forever, daemon=True).start()
        with urlopen(server.url + "api/episodes") as response:
            episodes = json.load(response)
        with urlopen(server.url) as response:
            page = response.read().decode("utf-8")
    finally:
        server.close()
    assert episodes["episodes"] == [trace.episode.name]
    assert "Robot ITL Reference Report" in page
    assert "/static/plotly.min.js" in page
