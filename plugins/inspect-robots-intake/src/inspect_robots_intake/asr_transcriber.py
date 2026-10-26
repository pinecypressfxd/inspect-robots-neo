"""Speech intake's Qwen3-ASR client. The console only supplies a wav path."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

from .contract import IntakeError, SpeechInput

_READY = "ready\n"


class QwenAsrTranscriber:
    """One long-lived Qwen3-ASR process. ``transcribe`` returns the utterance."""

    def __init__(
        self,
        python: Path,
        model_dir: Path,
        worker: Path,
        *,
        log_path: Path | None = None,
    ) -> None:
        self._python = python
        self._model_dir = model_dir
        self._worker = worker
        self._log_path = log_path
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._stderr: object | None = None

    def transcribe(self, speech: SpeechInput) -> str:
        """Transcribe ``speech.path`` and return the text."""
        with self._lock:
            process = self._ensure()
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(speech.path + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
        if not line:
            self._close()
            raise IntakeError("Qwen3-ASR worker exited before returning text")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise IntakeError(f"Qwen3-ASR worker returned {line!r}") from exc
        if not isinstance(payload, dict):
            raise IntakeError(f"Qwen3-ASR worker returned {line!r}")
        error = payload.get("error")
        if isinstance(error, str) and error:
            raise IntakeError(error)
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise IntakeError("Qwen3-ASR returned an empty transcript")
        return text.strip()

    def close(self) -> None:
        """Stop the worker. Safe to call more than once."""
        with self._lock:
            self._close()

    def _ensure(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        self._close()
        if not self._python.is_file():
            raise IntakeError(
                f"Qwen3-ASR environment is missing: {self._python}. "
                "Install qwen-asr into that environment before using Speak."
            )
        if not self._model_dir.is_dir():
            raise IntakeError(f"Qwen3-ASR weights are missing: {self._model_dir}")
        stderr: object = subprocess.DEVNULL
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            stderr = self._log_path.open("w", encoding="utf-8")
        self._stderr = stderr
        process = subprocess.Popen(
            [str(self._python), str(self._worker), str(self._model_dir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,  # type: ignore[arg-type]
            text=True,
        )
        assert process.stdout is not None
        ready = process.stdout.readline()
        if ready != _READY:
            process.kill()
            detail = ""
            if self._log_path is not None and self._log_path.is_file():
                detail = self._log_path.read_text(encoding="utf-8", errors="replace")[-500:].strip()
            self._close()
            raise IntakeError(detail or f"Qwen3-ASR worker did not start (got {ready!r})")
        self._process = process
        return process

    def _close(self) -> None:
        process = self._process
        self._process = None
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            process.terminate()
        stderr = self._stderr
        self._stderr = None
        if stderr is not None and stderr is not subprocess.DEVNULL:
            stderr.close()  # type: ignore[union-attr]
