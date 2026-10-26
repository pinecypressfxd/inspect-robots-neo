"""Record the live chest camera to one MP4 per take.

The mission console already shows ``chest_rgbd``. This writer samples that
same RGB frame at a fixed rate and pipes it to ffmpeg, so a person can film
"pick up the cup and place it on the pad" without starting a robot run.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

FrameReader = Callable[[], np.ndarray]


class ChestRecorder:
    """One chest recording at a time, counted by the MP4s already in ``directory``."""

    def __init__(
        self,
        directory: Path,
        *,
        fps: float = 30.0,
        ffmpeg: str = "ffmpeg",
        popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.directory = directory
        self.fps = fps
        self._ffmpeg = ffmpeg
        self._popen = popen
        self._clock = clock
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._path: Path | None = None
        self._error: str | None = None

    def status(self) -> dict[str, object]:
        """Whether a take is open, plus how many MP4s are already saved."""
        with self._lock:
            recording = self._thread is not None and self._thread.is_alive()
            path = None if self._path is None else str(self._path)
            error = self._error
        return {
            "recording": recording,
            "path": path,
            "saved": self.saved_count(),
            "directory": str(self.directory),
            "error": error,
        }

    def saved_count(self) -> int:
        """How many finished chest takes are in the directory."""
        if not self.directory.is_dir():
            return 0
        return sum(1 for path in self.directory.glob("chest_*.mp4") if path.is_file())

    def start(self, read_frame: FrameReader) -> str | None:
        """Begin one take. Returns an error string when recording cannot start."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return "chest recording is already running"
            if self._popen is subprocess.Popen and shutil.which(self._ffmpeg) is None:
                return f"{self._ffmpeg} was not found; install ffmpeg to record the chest camera"
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / time.strftime("chest_%Y%m%d-%H%M%S.mp4")
            try:
                probe = np.asarray(read_frame())
            except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
                return f"chest camera is not readable: {exc}"
            if probe.ndim != 3 or probe.shape[2] != 3:
                return f"chest frame must be HWC RGB, got shape {probe.shape}"
            height, width = int(probe.shape[0]), int(probe.shape[1])
            try:
                process = self._popen(
                    _ffmpeg_argv(self._ffmpeg, width, height, self.fps, path),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as exc:
                return f"could not start ffmpeg: {exc}"
            self._process = process
            self._path = path
            self._error = None
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop,
                args=(read_frame, probe),
                name="chest-recorder",
                daemon=True,
            )
            self._thread.start()
        return None

    def stop(self) -> str | None:
        """Finish the open take and return its path, or an error."""
        with self._lock:
            thread = self._thread
            path = self._path
        if thread is None or not thread.is_alive():
            return "chest recording is not running"
        self._stop.set()
        thread.join(timeout=5.0)
        with self._lock:
            if thread.is_alive():
                self._error = "chest recording did not stop"
                return self._error
            saved = None if path is None else str(path)
            self._thread = None
            self._process = None
        return saved

    def _loop(self, read_frame: FrameReader, first: np.ndarray) -> None:
        process = self._process
        assert process is not None and process.stdin is not None
        period = 1.0 / self.fps
        nxt = self._clock()
        frame = first
        try:
            while not self._stop.is_set():
                process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
                nxt += period
                delay = nxt - self._clock()
                if delay > 0 and self._stop.wait(delay):
                    break
                try:
                    frame = np.asarray(read_frame())
                except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
                    with self._lock:
                        self._error = f"chest camera read failed: {exc}"
                    break
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def _ffmpeg_argv(ffmpeg: str, width: int, height: int, fps: float, out_path: Path) -> list[str]:
    return [
        ffmpeg,
        "-hide_banner",
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
        f"{fps:g}",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
