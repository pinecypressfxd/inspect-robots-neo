"""The speech worker stays up and returns the transcript for one wav path."""

from __future__ import annotations

import sys
from pathlib import Path

from inspect_robots_intake.asr_transcriber import QwenAsrTranscriber
from inspect_robots_intake.contract import SpeechInput


def test_transcribe_reads_the_worker_json(tmp_path: Path) -> None:
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import json, sys\n"
        "print('ready', flush=True)\n"
        "for line in sys.stdin:\n"
        "    print(json.dumps({'text': 'heard ' + line.strip()}), flush=True)\n",
        encoding="utf-8",
    )
    transcriber = QwenAsrTranscriber(Path(sys.executable), tmp_path, worker)
    try:
        assert transcriber.transcribe(SpeechInput("/tmp/clip.wav")) == "heard /tmp/clip.wav"
    finally:
        transcriber.close()
