"""Mission console: localhost web page with live cameras and run control.

Serves one self-contained operator page (three camera tiles, status bar,
two-step start, stop, live reasoning feed) plus a JSON API over
stdlib ``http.server``. Start spawns the attended eval on a pty:

    uv run --no-sync inspect-robots "<instruction>" --policy agent \\
        -P model=gpt-5.6-luna -P base_url=<url> -P api_key_env=EXPLABS_API_KEY \\
        -P max_speed_frac=<frac> --embodiment nero -E operator_reset_confirm=False \\
        --grader none

The pty is the control channel: stop writes ``/stop``. Spawned runs use
``--grader none``, so the child never asks for a y/n/p verdict. Stop and Home
then end the child process group so the next Start is accepted.
Run from the repo root so ``uv run`` resolves the workspace and ``logs/`` is
the shared eval-log directory; the child inherits this process's environment,
so ``EXPLABS_API_KEY`` comes from the operator's shell that launched the
console.

While a run holds the cameras, the tiles switch from the live MJPEG streams
to ``/frame/<camera>.jpg``, which serves the newest frame the run stored.
``/api/feed`` tails the run's live snapshot, transcript, action side-car, and
wire capture. Every POST route additionally verifies the request's Host (and
Origin, when present) against the bound address, so a distant page cannot
drive the arms.
"""

from __future__ import annotations

import argparse
import contextlib
import html
import json
import math
import os
import pty
import re
import select
import signal
import subprocess
import sys
import threading
import time
import zlib
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, cast
from urllib.parse import parse_qs

from inspect_robots_nero._camera import D405Camera
from inspect_robots_nero._config import CAMERA_DEFAULTS

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

DEFAULT_PORT = 8400
DEFAULT_HOST = "127.0.0.1"
DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_BASE_URL = "https://api.experientiallabs.ai/v1"
DEFAULT_API_KEY_ENV = "EXPLABS_API_KEY"
DEFAULT_MAX_SPEED_FRAC = 0.05
DEFAULT_HISTORY_URL = "http://127.0.0.1:8300/"
DEFAULT_LOG_DIR = "/tmp"
_TRACE_DIR = Path(__file__).resolve().parents[3] / ".inspect_traces"
_BRAND_DIR = Path(__file__).resolve().with_name("brand")
_BRAND_FILES = {
    "/brand/ground.jpg": ("image/jpeg", "ground.jpg"),
    "/brand/mark.png": ("image/png", "mark.png"),
}
STREAM_FPS = 10.0
FRAME_MAX_AGE_S = 1.0
#: Bounded in-memory tail of each run's pty output (consumed by the feed API).
RING_LINES = 2000
MAX_BODY_BYTES = 1_000_000
#: Transcript messages returned by one feed poll (the tail the page renders).
FEED_MESSAGES = 40
#: Action side-car rows returned by one feed poll.
FEED_ACTIONS = 5
#: One source's seq base map is capped so a very long session cannot grow it.
FEED_SOURCES = 512
#: Drain-thread cap on one unterminated output line before it is dropped.
MAX_LINE_BYTES = 65_536
#: How long a pty control write may wait for the terminal to become writable.
PTY_WRITE_TIMEOUT_S = 2.0
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
#: Binding these means "every interface", so the Host allowlist accepts any name.
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", ""})
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_LOG_LINE_RE = re.compile(r"^log:\s*(\S.*)$")
_CAM_PATH_RE = re.compile(r"^/cam/(?P<name>[^/]+)\.mjpg$")
_FRAME_ROUTE_RE = re.compile(r"^/frame/(?P<name>[^/]+)\.jpg$")
_CHEST_TAKE_RE = re.compile(r"^chest_\d{8}-\d{6}\.mp4$")
_PREVIEW_ROUTE_RE = re.compile(r"^/preview/(?P<name>[A-Za-z0-9_-]+)\.jpg$")
_PREVIEW_DIR = Path("/tmp/nero-console-preview")
_PREVIEW_CAMERAS = frozenset({"left_rgbd", "right_rgbd", "chest_rgbd"})
_FRAME_FILE_RE = re.compile(r"^(?P<prefix>.+)_(?P<step>\d{6,})\.npy$")
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*://")


def _encode_jpeg(frame: np.ndarray) -> bytes:
    """Encode one RGB camera frame as JPEG bytes; cv2 imports only in here."""
    import cv2

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise RuntimeError("cv2.imencode failed to produce a JPEG frame")
    return encoded.tobytes()


def _safe(name: str) -> str:
    """Make ``name`` filesystem-safe the way the core FrameStore does."""
    safe = _SAFE_RE.sub("-", name)
    if safe != name:
        safe = f"{safe}-{zlib.crc32(name.encode()) & 0xFFFFFFFF:08x}"
    return safe


def _detect_log_path(line: str) -> str | None:
    """Extract the eval log path from one stdout ``log:`` line, else None.

    The CLI colors its output when it owns a terminal (it does: the child runs
    on our pty), so ANSI escapes are stripped before matching.
    """
    match = _LOG_LINE_RE.match(_ANSI_RE.sub("", line).strip())
    if match is None:
        return None
    path = match.group(1).strip()
    return path or None


def _newest_matching(paths: Iterable[Path], min_mtime: float) -> Path | None:
    """Newest file in ``paths`` whose mtime is at/after ``min_mtime``; else None."""
    newest: tuple[float, Path] | None = None
    for candidate in paths:
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            continue
        if mtime < min_mtime:
            continue
        if newest is None or mtime > newest[0]:
            newest = (mtime, candidate)
    return None if newest is None else newest[1]


def _newest_live_log(min_mtime: float) -> Path | None:
    """Newest ``logs/*.live.json`` modified at/after ``min_mtime``; else None.

    The live sink writes the run's snapshot continuously, so the newest file
    that is not older than the run's spawn time belongs to that run; a stale
    snapshot from an earlier run is never mistaken for the current one.
    """
    directory = Path("logs")
    if not directory.is_dir():
        return None
    return _newest_matching(directory.glob("*.live.json"), min_mtime)


def _read_json_file(path: Path) -> dict[str, Any] | None:
    """Parse one JSON object file, degrading every failure to ``None``."""
    try:
        with path.open(encoding="utf-8") as handle:
            parsed = json.load(handle)
    except (OSError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    """Parse one JSONL file's object rows, skipping malformed lines."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _write_pty(master_fd: int, data: bytes) -> str | None:
    """Write ``data`` to the pty master, bounded by a writability wait.

    A terminal whose input queue is full would block ``os.write`` forever
    while the caller holds the run-manager lock; waiting at most
    ``PTY_WRITE_TIMEOUT_S`` for writability turns that into an error instead.
    """
    try:
        _readable, writable, _exceptional = select.select([], [master_fd], [], PTY_WRITE_TIMEOUT_S)
    except (OSError, ValueError) as exc:
        return f"could not check the run's terminal: {exc}"
    if not writable:
        return "the run's terminal is not accepting input (full buffer); line not written"
    try:
        os.write(master_fd, data)
    except OSError as exc:
        return f"could not write to the run's terminal: {exc}"
    return None


class LerobotExporter:
    """Auto-export each cleanly finished run to LeRobot v2.1 in the background."""

    def __init__(self, script: Path, log_dir: Path) -> None:
        self._script = script
        self._log_dir = log_dir
        self._lock = threading.Lock()
        self._last = "not run yet"
        self._busy = False

    def describe(self) -> dict[str, Any]:
        """Status fields for the page."""
        with self._lock:
            return {"busy": self._busy, "last": self._last}

    def maybe_export(self, exit_code: int | None, log_path: str | None) -> None:
        """Queue one export when the run finished cleanly with a written log."""
        if exit_code != 0 or not log_path or not Path(log_path).is_file():
            with self._lock:
                self._last = f"skipped (exit {exit_code}, log {log_path or 'none'})"
            return
        with self._lock:
            if self._busy:
                self._last = "skipped (an export is already running)"
                return
            self._busy = True
        thread = threading.Thread(target=self._export, args=(log_path,), daemon=True)
        thread.name = "lerobot-export"
        thread.start()

    def _export(self, log_path: str) -> None:
        started = time.monotonic()
        replay_note = _render_wire_replay_for(log_path)
        try:
            out_log = self._log_dir / (
                f"mission-console-lerobot-{time.strftime('%Y%m%d-%H%M%S')}.log"
            )
            with out_log.open("w", encoding="utf-8") as handle:
                done = subprocess.run(
                    [sys.executable, str(self._script), log_path],
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                    check=False,
                )
            summary = f"exit {done.returncode} ({time.monotonic() - started:.0f}s) -> {out_log}"
        except (OSError, subprocess.SubprocessError) as exc:
            summary = f"failed ({exc})"
        finally:
            print(f"[console] lerobot export: {summary}; replay: {replay_note}")
            with self._lock:
                self._busy = False
                self._last = f"{summary}; replay: {replay_note}"


def _render_wire_replay_for(log_path: str) -> str:
    """Render the run's wire capture to a timestamped replay HTML; note on failure."""
    try:
        from inspect_robots._pointers import resolve_log_pointer
        from inspect_robots._wire_replay import render_wire_replay_page

        log_file = Path(log_path)
        log = json.loads(log_file.read_text(encoding="utf-8"))
        root = log_file.resolve().parent
        capture_dirs: list[Path] = []
        for sample in log.get("samples") or []:
            for entry in sample.get("trial_metadata") or []:
                pointer = entry.get("wire_capture")
                if not isinstance(pointer, str) or not pointer:
                    continue
                calls = resolve_log_pointer(log_file, pointer)
                if calls is not None:
                    # The renderer expects the RUN directory (it globs
                    # <run>/<trial>/calls.jsonl and shares <run>/blobs), which
                    # is two levels above the calls file.
                    capture_dirs.append(calls.parent.parent)
        if not capture_dirs:
            return "none recorded"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out = root / f"wire-replay-{stamp}.html"
        unique = list(dict.fromkeys(capture_dirs))
        instruction = next(
            (str(s.get("instruction")) for s in (log.get("samples") or []) if s.get("instruction")),
            "",
        )
        created = str(log.get("created") or "")
        ended = str(log.get("stats", {}).get("ended_at") or "")
        fields = {"instruction": instruction, "created": created, "ended": ended}
        out.write_text(
            render_wire_replay_page(unique, Path(log_path).name, fields=fields),
            encoding="utf-8",
        )
        return str(out)
    except (OSError, ValueError, KeyError) as exc:
        return f"failed ({exc})"


class ArmToolRunner:
    """Run arm_tools.py (home / emergency disable) in the background, one at a time."""

    def __init__(self, script: Path, log_dir: Path) -> None:
        self._prefix = [sys.executable, str(script)]
        self._log_dir = log_dir
        self._lock = threading.Lock()
        self._busy = False
        self._last = "never run"

    def describe(self) -> dict[str, Any]:
        """Status fields for the page: busy flag plus the last command's result."""
        with self._lock:
            return {"busy": self._busy, "last": self._last}

    def home(self, run_active: bool) -> str | None:
        """Home both arms; refused while a run is active or a tool is running."""
        if run_active:
            return "stop the run before homing the arms"
        return self._spawn(["home"], "home")

    def estop(self) -> str | None:
        """Drop both arms limp immediately (always allowed)."""
        return self._spawn(["disable", "--arm", "both", "--yes"], "estop")

    def _spawn(self, args: list[str], label: str) -> str | None:
        with self._lock:
            if self._busy:
                return "an arm command is already running; wait for it to finish"
            self._busy = True
        threading.Thread(
            target=self._run, args=(args, label), daemon=True, name=f"arm-{label}"
        ).start()
        return None

    def _run(self, args: list[str], label: str) -> None:
        started = time.monotonic()
        try:
            log_path = self._log_dir / (
                f"mission-console-armtool-{time.strftime('%Y%m%d-%H%M%S')}-{label}.log"
            )
            with log_path.open("w", encoding="utf-8") as handle:
                done = subprocess.run(
                    [*self._prefix, *args],
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    timeout=90,
                    check=False,
                )
            summary = (
                f"{label}: exit {done.returncode} ({time.monotonic() - started:.1f}s) -> {log_path}"
            )
            print(f"[console] arm {summary}")
        except (OSError, subprocess.SubprocessError) as exc:
            summary = f"{label}: failed ({exc})"
            print(f"[console] arm {summary}")
        finally:
            with self._lock:
                self._busy = False
                self._last = summary


class CameraPool:
    """One lazily started ``D405Camera`` singleton per configured camera.

    V4L2 mmap streaming is exclusive per device node: while the console
    streams, nothing else can open the cameras, and once a run holds them the
    console's own tiles get a plain 503. ``stop`` releases every device so a
    spawned eval can claim them; the first MJPEG request afterwards starts a
    camera again, and ``restart`` re-opens the cameras a failed start had
    released so the tiles recover without a page reload.
    """

    def __init__(self, overrides: Mapping[str, str]) -> None:
        """Build one not-yet-started camera per ``CAMERA_DEFAULTS`` entry."""
        self._cameras: dict[str, D405Camera] = {}
        for name, spec in CAMERA_DEFAULTS.items():
            settings = cast(dict[str, Any], spec)
            device = overrides.get(name, str(settings["device"]))
            self._cameras[name] = D405Camera(
                name,
                device,
                width=int(settings["width"]),
                height=int(settings["height"]),
                fps=int(settings["fps"]),
                pixel_format=str(settings["pixel_format"]),
            )
        self._lock = threading.Lock()
        self._started: set[str] = set()
        self._was_started: frozenset[str] = frozenset()
        self._held: dict[str, np.ndarray] = {}

    @property
    def names(self) -> tuple[str, ...]:
        """Camera names in tile order (chest centered between the two arms)."""
        order = ("left_rgbd", "chest_rgbd", "right_rgbd")
        return tuple(name for name in order if name in self._cameras) + tuple(
            name for name in self._cameras if name not in order
        )

    def describe(self) -> list[str]:
        """One ``name=device`` string per camera, for the startup banner."""
        return [f"{name}={camera.device}" for name, camera in self._cameras.items()]

    def camera(self, name: str) -> D405Camera:
        """Return the camera for ``name`` without starting it (KeyError if unknown)."""
        return self._cameras[name]

    def start(self, name: str) -> D405Camera:
        """Return the camera for ``name``, starting it on first request.

        Raises ``KeyError`` for an unknown name and ``RuntimeError`` when the
        device cannot be opened (busy, unplugged, cabling); the MJPEG route
        turns both into plain error responses.
        """
        camera = self._cameras[name]
        with self._lock:
            camera.start()
            self._started.add(name)
        return camera

    def any_started(self) -> bool:
        """True while at least one camera is open (devices held)."""
        return bool(self._started)

    def held_frame(self, name: str) -> np.ndarray | None:
        """Last frame copied before the devices were released, if one exists."""
        import numpy as np

        frame = self._held.get(name)
        if frame is None:
            return None
        return np.array(frame, copy=True)

    def stop(self) -> None:
        """Stop every camera and release its device; idempotent, best effort.

        The last frame of each open camera is kept so the tiles can stay up
        until the run has stored a frame of its own.
        """
        import numpy as np

        with self._lock:
            self._was_started = frozenset(self._started)
            self._started.clear()
            for name, camera in self._cameras.items():
                try:
                    frame, _stamp = camera.read(max_age_s=60.0)
                except Exception:
                    frame = None
                if frame is not None:
                    self._held[name] = np.array(frame, copy=True)
                try:
                    camera.stop()
                except Exception as exc:  # hardware teardown must never abort the rest
                    print(f"[console] stopping camera {camera.name!r} failed: {exc}")

    def restart(self) -> None:
        """Re-open the cameras the last ``stop`` released; best effort."""
        for name in sorted(self._was_started):
            try:
                self.start(name)
            except Exception as exc:  # a busy or unplugged node must not stop the rest
                print(f"[console] restarting camera {name!r} failed: {exc}")


class _Run:
    """One spawned run's state; every field is guarded by RunManager's lock."""

    def __init__(
        self,
        instruction: str,
        process: subprocess.Popen[bytes],
        master_fd: int,
        log_file: TextIO,
        started_at: float,
        run_id: int,
    ) -> None:
        self.instruction = instruction
        self.process = process
        self.master_fd = master_fd
        self.log_file = log_file
        self.started_at = started_at
        self.run_id = run_id
        self.tail: deque[str] = deque(maxlen=RING_LINES)
        self.log_path: str | None = None
        self.exit_code: int | None = None
        self.stop_line = "/stop"
        self.kick: str | None = None
        self.kicked = False
        self.tiles = "frames"


class RunManager:
    """Own the single active run: spawn it on a pty, drain it, report status.

    The pty is the only control channel (``/stop`` written to the master fd,
    echoed to the console's stdout); no signal is ever sent to the child.
    Spawned runs pass ``--grader none``, so the process exits when the
    episode ends instead of waiting for a y/n/p verdict. It sees terminal
    EOF when the console exits.
    """

    def __init__(
        self,
        prefix: list[str],
        suffix: list[str],
        log_dir: Path,
        exporter: LerobotExporter | None = None,
    ) -> None:
        """Store the argv halves (instruction is appended between them)."""
        self._prefix = prefix
        self._suffix = suffix
        self._log_dir = log_dir
        self._exporter = exporter
        self._lock = threading.Lock()
        self._run: _Run | None = None
        self._spawn_count = 0
        self._last_error: str | None = None
        self._stop_requested = False

    def start(
        self,
        instruction: str,
        prepare: Callable[[], None] | None = None,
        rollback: Callable[[], None] | None = None,
        suffix: list[str] | None = None,
        command: list[str] | None = None,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        stop_line: str = "/stop",
        kick: str | None = None,
        tiles: str = "frames",
    ) -> str | None:
        """Spawn one run for ``instruction``; return an error message or None.

        ``prepare`` runs after the active-run check but before the spawn, so
        callers can release shared hardware (the cameras) exactly when a run
        is about to claim it. ``rollback`` runs when the spawn itself fails,
        so callers can take the hardware back (restart the cameras) and the
        page can recover instead of sitting idle with dead tiles.
        """
        with self._lock:
            self._adopt_exit_code()
            if self._run is not None and self._run.exit_code is None and self._stop_requested:
                finish = True
            else:
                finish = False
        if finish:
            error = self.force_end()
            if error is not None:
                return error
        with self._lock:
            self._adopt_exit_code()
            if self._run is not None and self._run.exit_code is None:
                return "a run is already active; stop it or wait for it to end"
            if prepare is not None:
                prepare()
            self._spawn_count += 1
            self._stop_requested = False
            try:
                self._log_dir.mkdir(parents=True, exist_ok=True)
                # The handle lives until the drain thread closes it at run end;
                # the spawn count keeps same-second runs from sharing a name.
                log_file: TextIO = open(  # noqa: SIM115 - closed in _drain
                    self._log_dir
                    / (
                        f"mission-console-{time.strftime('%Y%m%d-%H%M%S')}"
                        f"-{self._spawn_count:03d}.log"
                    ),
                    "w",
                    encoding="utf-8",
                    buffering=1,  # line-buffered: a console crash must not lose run output
                )
            except OSError as exc:
                return self._fail_start(
                    rollback, f"could not open the run log under {self._log_dir}: {exc}"
                )
            try:
                master_fd, slave_fd = pty.openpty()
            except OSError as exc:
                log_file.close()
                return self._fail_start(rollback, f"could not allocate the run terminal: {exc}")
            effective = suffix if suffix is not None else self._suffix
            resolved = [arg.replace("__INSTRUCTION__", instruction) for arg in effective]
            if command is None:
                command = [*self._prefix, instruction, *resolved]
            try:
                # ValueError joins OSError: an embedded NUL byte in the
                # instruction (or any argument) fails the exec, not the console.
                process = subprocess.Popen(
                    command,
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    start_new_session=True,
                    cwd=cwd,
                    env=None if env is None else dict(env),
                )
            except (OSError, ValueError) as exc:
                log_file.close()
                os.close(master_fd)
                os.close(slave_fd)
                return self._fail_start(rollback, f"could not spawn {' '.join(command)}: {exc}")
            os.close(slave_fd)
            run = _Run(
                instruction=instruction,
                process=process,
                master_fd=master_fd,
                log_file=log_file,
                started_at=time.time(),
                run_id=self._spawn_count,
            )
            run.stop_line = stop_line
            run.kick = kick
            run.tiles = tiles if tiles in ("frames", "preview") else "frames"
            self._run = run
            self._last_error = None
        threading.Thread(target=self._drain, args=(run,), daemon=True, name="run-drain").start()
        print(f"[console] started: {instruction!r}")
        print(f"[console] command: {' '.join(command)}")
        return None

    def _fail_start(self, rollback: Callable[[], None] | None, message: str) -> str:
        """Record a spawn failure for /api/status, roll back hardware, report."""
        self._last_error = message
        if rollback is not None:
            rollback()
        return message

    def _drain(self, run: _Run) -> None:
        """Drain the pty to the ring tail and log file, then reap the child.

        The log-file write degrades to ring-only on OSError (a full disk must
        not kill the drain), and one unterminated output line above
        ``MAX_LINE_BYTES`` is dropped with a marker instead of growing the
        buffer without bound.
        """
        buffer = ""
        log_disabled = False
        while True:
            try:
                chunk = os.read(run.master_fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            if not log_disabled:
                try:
                    run.log_file.write(text)
                except OSError as exc:
                    log_disabled = True
                    print(f"[console] run log write failed; keeping the tail in memory only: {exc}")
            buffer += text
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                kick_now = False
                with self._lock:
                    run.tail.append(line)
                    detected = _detect_log_path(line)
                    if detected is not None:
                        run.log_path = detected
                        print(f"[console] run log: {detected}")
                    if run.kick and not run.kicked and "Commands: s=start" in line:
                        run.kicked = True
                        kick_now = True
                if kick_now and run.kick is not None:
                    self.write_line(run.kick)
            if len(buffer) > MAX_LINE_BYTES:
                dropped = len(buffer)
                marker = f"[line exceeds {MAX_LINE_BYTES} bytes; dropped {dropped}]"
                with self._lock:
                    run.tail.append(marker)
                buffer = ""
        with contextlib.suppress(OSError):
            run.log_file.flush()
        exit_code = run.process.wait()
        with self._lock:
            run.exit_code = exit_code
            if run.log_path is None:
                fallback = _newest_live_log(run.started_at)
                run.log_path = str(fallback) if fallback is not None else None
            log_path = run.log_path
            os.close(run.master_fd)
            run.log_file.close()
        print(f"[console] ended: exit code {exit_code}; log: {log_path or '(none detected)'}")
        if self._exporter is not None:
            self._exporter.maybe_export(exit_code, log_path)

    def write_line(self, line: str) -> str | None:
        """Write one operator line to the active run's terminal, or an error.

        This is the safety-critical path: it writes to the pty master and
        nothing else, bounded by a writability wait so a full input queue
        cannot hold the manager lock indefinitely. Every attempt (success or
        failure) is echoed to the console's stdout.
        """
        with self._lock:
            run = self._run
            if run is None:
                return "no run has been started"
            if run.exit_code is not None:
                return "the run has already ended"
            error = _write_pty(run.master_fd, f"{line}\n".encode())
        if error is not None:
            print(f"[console] pty write failed ({line!r}): {error}")
            return error
        print(f"[console] pty -> {line}")
        return None

    def status(self) -> dict[str, Any]:
        """Snapshot for ``/api/status``: state, instruction, log, exit, error.

        After Stop is accepted and before the child exits, state is ``stopping``.
        """
        with self._lock:
            run = self._run
            error = self._last_error
            if run is None:
                return {
                    "state": "idle",
                    "instruction": None,
                    "log": None,
                    "exit_code": None,
                    "error": error,
                }
            self._adopt_exit_code()
            state = "running" if run.exit_code is None else "ended"
            if state == "running" and self._stop_requested:
                state = "stopping"
            snapshot = {
                "state": state,
                "instruction": run.instruction,
                "log": run.log_path,
                "exit_code": run.exit_code,
                "error": error,
                "tiles": run.tiles if state in ("running", "stopping") else "live",
            }
            started_at = run.started_at
        if snapshot["log"] is None and state == "running":
            live = _newest_live_log(started_at)
            if live is not None:
                snapshot["log"] = str(live)
        return snapshot

    def mark_stop_requested(self) -> None:
        """Record that the operator stopped the episode; homing is safe after."""
        self._stop_requested = True

    def homing_allowed(self) -> bool:
        """True when no run is active, or Stop was already sent and it is exiting."""
        run = self.status()
        return run.get("state") not in ("running", "external") or self._stop_requested

    def _adopt_exit_code(self) -> None:
        """Store a polled exit code so a dead child does not stay ``running``.

        The drain thread also records the code after the terminal closes. A
        grandchild can keep that read blocked after the direct child has
        already exited; ``poll`` notices the exit without waiting for EOF.
        Caller holds ``self._lock``.
        """
        run = self._run
        if run is None or run.exit_code is not None:
            return
        code = run.process.poll()
        if code is not None:
            run.exit_code = code

    def _signal_group(self, pid: int, sig: int) -> None:
        """Signal the run's session, then the direct child if it has no group."""
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            return
        except OSError:
            os.kill(pid, sig)

    def _active_stop_line(self) -> str:
        """Key the active run expects. ROS origin mode quits on ``q``."""
        with self._lock:
            run = self._run
            if run is None:
                return "/stop"
            return run.stop_line

    def force_end(self) -> str | None:
        """End the active run: ``/stop``, then the whole process group.

        Stop and Home both use this. The child is its own session, so a
        signal to that one process can leave a grandchild holding the
        terminal and the arms. The group signal is what lets the next Start
        proceed.
        """
        self.write_line(self._active_stop_line())
        with self._lock:
            self._adopt_exit_code()
            run = self._run
            if run is None or run.exit_code is not None:
                return None
            pid = run.process.pid
            try:
                self._signal_group(pid, signal.SIGTERM)
            except OSError as exc:
                return f"could not terminate the run: {exc}"
        try:
            run.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                self._signal_group(pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                run.process.wait(timeout=5)
        with self._lock:
            self._adopt_exit_code()
        return None

    def current(self) -> dict[str, Any] | None:
        """Snapshot of the last run for the feed and frame routes; None if never.

        When the console owns no run, a freshly-written ``*.live.json`` (an
        eval launched outside the console, e.g. from a bare CLI command) is
        reported as an ``external`` run so the tiles, tool timeline, and
        feed follow it too: the camera nodes are exclusive, so the console
        cannot stream live video while that process holds them.
        """
        with self._lock:
            run = self._run
            if run is not None and (
                run.exit_code is None or time.monotonic() - run.started_at < 120
            ):
                return {
                    "state": "running" if run.exit_code is None else "ended",
                    "instruction": run.instruction,
                    "started_at": run.started_at,
                    "run_id": run.run_id,
                    "log": run.log_path,
                    "tail": list(run.tail),
                }
        live = _newest_live_log(time.time() - 30.0)
        if live is None:
            return (
                None
                if run is None
                else {
                    "state": "running" if run.exit_code is None else "ended",
                    "instruction": run.instruction,
                    "started_at": run.started_at,
                    "run_id": run.run_id,
                    "log": run.log_path,
                    "tail": list(run.tail),
                }
            )
        return {
            "state": "external",
            "instruction": None,
            "started_at": live.stat().st_mtime,
            "run_id": -1,
            "log": str(live),
            "tail": [],
        }

    def shutdown(self) -> None:
        """Best-effort graceful end for an active run: one ``/stop`` line.

        Runs when the console exits. Never a signal: the write lets the eval
        end its episode, and the closing terminal then gives it EOF.
        """
        with self._lock:
            run = self._run
            if run is None or run.exit_code is not None:
                return
            error = _write_pty(run.master_fd, f"{run.stop_line}\n".encode())
        if error is None:
            print(f"[console] pty -> {run.stop_line} (console shutdown)")
        else:
            print(f"[console] shutdown /stop write failed: {error}; the run will see EOF")


class RunFeed:
    """Assemble ``/api/feed`` payloads from the active (or last) run's artifacts.

    Every transcript message gets a stable, monotonically increasing ``seq``
    derived from its position inside its source (the live snapshot's active
    trial, or a ``logs/transcripts/<stamp>/<trial>.jsonl`` file), so the page
    can append only what it has not seen. Source ids are prefixed with a
    per-run token: every ad-hoc run's trial is ``scene-0-e0``, so without it
    run two would reuse run one's cached seq base and poll in below the
    client's ``since`` watermark. Sources discovered later start above every
    seq handed out before, and the global ``seq`` never goes backwards.
    """

    def __init__(self) -> None:
        """Start with no sources and seq zero."""
        self._lock = threading.Lock()
        self._bases: dict[str, int] = {}
        # Wire-fallback tool entries get stable NEGATIVE seqs: one per
        # distinct decision, never colliding with message seqs (positive).
        self._wire_notes: dict[str, int] = {}
        self._wire_seq_next = -1
        self._max_seq = 0
        # First time each tool seq was observed. A page refresh asks for seq 0
        # and would otherwise stamp every old call with the refresh clock.
        self._seen_at: dict[int, str] = {}

    def build(self, run: Mapping[str, Any] | None, since: int) -> dict[str, Any]:
        """Return one feed snapshot for ``run`` filtered to messages after ``since``."""
        with self._lock:
            live: dict[str, Any] | None = None
            messages: list[dict[str, Any]] = []
            actions: dict[str, Any] | None = None
            wire: dict[str, Any] | None = None
            if run is not None:
                doc, _doc_path = _run_live_doc(run)
                if doc is not None:
                    live = _live_status(doc)
                raw_messages, source = _transcript_tail(doc, run)
                if source is not None:
                    messages = self._assign(raw_messages, source, since)
                actions = _actions_tail(doc, run)
                wire = _wire_note(run)
            tools: list[dict[str, Any]] = []
            for message in messages:
                for tool in message.get("tools") or []:
                    entry = dict(tool)
                    seq = message["seq"]
                    entry["seq"] = seq
                    entry["at"] = self._clock(seq, entry.get("at"))
                    tools.append(entry)
            if wire is not None and wire.get("tool"):
                # One stable seq per distinct wire note: a fresh seq per poll
                # would slip past the client's dedupe and stack duplicates of
                # the same decision into the timeline every second.
                key = f"{wire['tool']}|{wire.get('note', '')}"
                wire_seq = self._wire_notes.get(key)
                if wire_seq is None:
                    if len(self._wire_notes) >= FEED_SOURCES:
                        self._wire_notes.pop(next(iter(self._wire_notes)))
                    self._wire_seq_next -= 1
                    wire_seq = self._wire_seq_next
                    self._wire_notes[key] = wire_seq
                tools.append(
                    {
                        "name": str(wire["tool"]),
                        "arguments": "",
                        "note": str(wire.get("note", "")),
                        "seq": wire_seq,
                        "at": self._clock(wire_seq, None),
                    }
                )
            return {
                "seq": self._max_seq,
                "live": live,
                "messages": messages,
                "actions": actions,
                "wire": wire,
                "tools": tools,
            }

    def _assign(
        self, raw_messages: Sequence[object], source: str, since: int
    ) -> list[dict[str, Any]]:
        """Stamp positional seqs onto one source's compact messages."""
        base = self._bases.get(source)
        if base is None:
            if len(self._bases) >= FEED_SOURCES:
                self._bases.pop(next(iter(self._bases)))
            base = self._max_seq
            self._bases[source] = base
        stamped: list[dict[str, Any]] = []
        for index, message in enumerate(_compact_message(item) for item in raw_messages):
            if message is None:
                continue
            seq = base + index + 1
            if seq > self._max_seq:
                self._max_seq = seq
            if seq > since:
                stamped.append({"seq": seq, **message})
        del stamped[:-FEED_MESSAGES]
        return stamped

    def _clock(self, seq: int, logged: object) -> str:
        """The tool's own clock, else the first time this seq was observed.

        A logged ``at`` (written when the planner accepted the call) wins, so
        a console restart still shows that time. Otherwise the first feed
        poll that sees the seq freezes the clock; later polls, including a
        page refresh that asks for every seq again, reuse it.
        """
        if isinstance(logged, str) and logged:
            return logged
        found = self._seen_at.get(seq)
        if found is None:
            if len(self._seen_at) >= FEED_SOURCES:
                self._seen_at.pop(next(iter(self._seen_at)))
            found = time.strftime("%H:%M:%S")
            self._seen_at[seq] = found
        return found


def _started_at_of(run: Mapping[str, Any]) -> float:
    """The run's spawn time as a float, defaulting to zero for foreign data."""
    started = run.get("started_at")
    return started if isinstance(started, (int, float)) else 0.0


def _run_live_doc(run: Mapping[str, Any]) -> tuple[dict[str, Any] | None, Path | None]:
    """The run's live snapshot (or final log) as a parsed document and path."""
    live_path = _newest_live_log(_started_at_of(run))
    if live_path is not None:
        doc = _read_json_file(live_path)
        if doc is not None:
            return doc, live_path
    log = run.get("log")
    if isinstance(log, str) and log:
        path = Path(log)
        doc = _read_json_file(path)
        if doc is not None:
            return doc, path
    return None, None


def _samples_of(doc: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The document's sample rows, tolerating anything malformed."""
    if doc is None:
        return []
    samples = doc.get("samples")
    if not isinstance(samples, list):
        return []
    return [sample for sample in samples if isinstance(sample, dict)]


def _live_status(doc: Mapping[str, Any]) -> dict[str, Any] | None:
    """One compact status row from a live snapshot or final log document."""
    status: dict[str, Any] = {"status": doc.get("status")}
    stats = doc.get("stats")
    if isinstance(stats, dict):
        for key in ("duration_s", "total_steps", "frames_dir"):
            status[key] = stats.get(key)
    for sample in _samples_of(doc):
        metadata = sample.get("trial_metadata")
        if not isinstance(metadata, list):
            continue
        for entry in metadata:
            if isinstance(entry, dict) and isinstance(entry.get("live"), dict):
                live = cast(dict[str, Any], entry["live"])
                status["step"] = live.get("step")
                status["updated_at"] = live.get("updated_at")
                return status
    return status


def _active_live_transcript(
    doc: Mapping[str, Any] | None, token: str
) -> tuple[Sequence[object], str] | None:
    """The in-flight trial's transcript from a live snapshot, with its source id."""
    for sample in _samples_of(doc):
        metadata = sample.get("trial_metadata")
        if not isinstance(metadata, list):
            continue
        for index, entry in enumerate(metadata):
            if not (isinstance(entry, dict) and isinstance(entry.get("live"), dict)):
                continue
            transcripts = sample.get("policy_transcripts")
            if not isinstance(transcripts, list) or index >= len(transcripts):
                continue
            transcript = transcripts[index]
            if isinstance(transcript, list):
                scene_id = sample.get("scene_id")
                label = scene_id if isinstance(scene_id, str) else "scene"
                return transcript, f"{token}:{label}-e{index}"
    return None


def _pointer_path(doc: Mapping[str, Any] | None, key: str) -> Path | None:
    """Resolve the newest trial's ``key`` side-car pointer against ``logs/``."""
    for sample in reversed(_samples_of(doc)):
        metadata = sample.get("trial_metadata")
        if not isinstance(metadata, list):
            continue
        for entry in reversed(metadata):
            if not isinstance(entry, dict):
                continue
            pointer = entry.get(key)
            if isinstance(pointer, str) and pointer:
                return Path("logs") / pointer
    return None


def _run_token(run: Mapping[str, Any]) -> str:
    """A per-run prefix for feed source ids, so seqs never repeat across runs.

    Every ad-hoc run's transcript is ``scene-0-e0``; without the run token the
    second run's messages would reuse the first run's cached seq base and poll
    in below the client's ``since`` watermark, blanking the feed.
    """
    run_id = run.get("run_id")
    return f"r{run_id}" if isinstance(run_id, int) else "r?"


def _transcript_tail(
    doc: Mapping[str, Any] | None, run: Mapping[str, Any]
) -> tuple[Sequence[object], str | None]:
    """The run's newest transcript messages and their source id.

    Preference order: the live snapshot's active trial (updates mid-trial),
    then the newest ``transcript`` pointer the document records (written at
    trial end), then a newest-file scan under ``logs/transcripts/`` guarded
    by the run's spawn time. Source ids carry the run token so a later run's
    seqs sort above every earlier run's.
    """
    token = _run_token(run)
    active = _active_live_transcript(doc, token)
    if active is not None:
        return active
    pointer = _pointer_path(doc, "transcript")
    if pointer is not None and pointer.is_file():
        return _read_jsonl_rows(pointer), f"{token}:{pointer.stem}"
    directory = Path("logs")
    if not directory.is_dir():
        return (), None
    newest = _newest_matching(directory.glob("transcripts/*/*.jsonl"), _started_at_of(run))
    if newest is None:
        return (), None
    return _read_jsonl_rows(newest), f"{token}:{newest.stem}"


def _actions_tail(doc: Mapping[str, Any] | None, run: Mapping[str, Any]) -> dict[str, Any] | None:
    """The run's last executed action rows from the actions side-car."""
    pointer = _pointer_path(doc, "actions")
    candidates: list[Path] = [pointer] if pointer is not None else []
    directory = Path("logs")
    if directory.is_dir():
        newest = _newest_matching(directory.glob("actions/*/*.jsonl"), _started_at_of(run))
        if newest is not None:
            candidates.append(newest)
    for path in candidates:
        if not path.is_file():
            continue
        rows = _read_jsonl_rows(path)
        header = next((row for row in rows if row.get("kind") == "header"), None)
        steps = [row for row in rows if row.get("kind") != "header"]
        if header is None and not steps:
            continue
        return {
            "file": path.name,
            "action_dim": header.get("action_dim") if header is not None else None,
            "labels": header.get("labels") if header is not None else None,
            "rows": steps[-FEED_ACTIONS:],
        }
    return None


def _chat_content(content: object) -> str | None:
    """Render text from an OpenAI-style content value, collapsing media parts.

    Same parsing shape as the core transcript renderer: a string passes
    through, a list joins its text parts, and every non-text part (images)
    degrades to a ``[image]`` marker.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("text", "")))
        else:
            parts.append("[image]")
    return "\n".join(parts)


def _compact_message(raw: object) -> dict[str, Any] | None:
    """One transcript message as ``{role, text, tools}``, tolerating foreign data."""
    if not isinstance(raw, dict):
        return None
    role = raw.get("role")
    if not isinstance(role, str):
        return None
    message: dict[str, Any] = {"role": role, "text": _chat_content(raw.get("content"))}
    tools: list[dict[str, str]] = []
    calls = raw.get("tool_calls")
    if isinstance(calls, list):
        for call in calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments", "")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            tools.append({"name": str(function.get("name", "unknown")), "arguments": arguments})
    if isinstance(raw.get("at"), str):
        for tool in tools:
            tool["at"] = raw["at"]
    if tools:
        message["tools"] = tools
    return message


def _wire_note(run: Mapping[str, Any]) -> dict[str, Any] | None:
    """The last wire call's tool name and its note/summary argument."""
    directory = Path("logs")
    if not directory.is_dir():
        return None
    newest = _newest_matching(directory.glob("wire/*/*/calls.jsonl"), _started_at_of(run))
    if newest is None:
        return None
    rows = _read_jsonl_rows(newest)
    if not rows:
        return None
    response = rows[-1].get("response")
    if not isinstance(response, dict):
        return None
    for name, arguments in _wire_tool_calls(response):
        note = ""
        if isinstance(arguments, dict):
            for key in ("note", "summary"):
                value = arguments.get(key)
                if isinstance(value, str) and value:
                    note = value
                    break
        return {"tool": name, "note": note}
    return None


def _wire_tool_calls(response: Mapping[str, Any]) -> list[tuple[str, object]]:
    """Extract ``(name, parsed arguments)`` from supported wire response shapes."""
    calls: list[tuple[str, object]] = []
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            raw_calls = message.get("tool_calls")
            if isinstance(raw_calls, list):
                for raw in raw_calls:
                    if not isinstance(raw, dict):
                        continue
                    function = raw.get("function")
                    if not isinstance(function, dict):
                        continue
                    calls.append(
                        (
                            str(function.get("name", "unknown")),
                            _parse_json_arguments(function.get("arguments", "")),
                        )
                    )
    content = response.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                calls.append((str(block.get("name", "unknown")), block.get("input")))
    return calls


def _parse_json_arguments(value: object) -> object:
    """Parse JSON-string tool arguments, passing anything else through."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def _stored_frame(
    frames_dir: str, doc_path: Path, camera: str
) -> tuple[npt.NDArray[np.uint8] | None, str]:
    """The newest stored frame for one camera, or an error explaining its absence.

    Directory discovery is core's ``resolve_frames_dir`` (the recorded frames
    directory string as-is, then the same stamp under the log directory's
    ``frames/``); the per-camera newest-frame pick is console-local, core has
    no such helper.
    """
    import numpy as np

    from inspect_robots._video import resolve_frames_dir

    root = resolve_frames_dir(frames_dir, doc_path)
    if root is None:
        return None, f"the run's frames directory {frames_dir!r} is not present on disk"
    best: tuple[float, int, Path] | None = None
    try:
        entries = list(root.glob(f"*_{_safe(camera)}_*.npy"))
    except OSError:
        entries = []
    for path in entries:
        match = _FRAME_FILE_RE.match(path.name)
        if match is None:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        key = (mtime, int(match.group("step")))
        if best is None or key > (best[0], best[1]):
            best = (mtime, int(match.group("step")), path)
    if best is None:
        return None, f"no stored frames for camera {camera!r} yet"
    try:
        array = cast("npt.NDArray[Any]", np.load(best[2], allow_pickle=False))
        if array.dtype != np.uint8 or array.size == 0:
            raise ValueError("unexpected frame format")
        if not (array.ndim == 2 or (array.ndim == 3 and array.shape[2] in {1, 3, 4})):
            raise ValueError("unexpected frame shape")
    except Exception as exc:
        return None, f"stored frame unreadable: {exc}"
    return cast("npt.NDArray[np.uint8]", array), ""


class MissionRequestHandler(BaseHTTPRequestHandler):
    """Serve the console page, camera streams, stored frames, and the APIs."""

    server: MissionServer

    def do_GET(self) -> None:
        """Route GET: page, status/feed JSON, camera streams, stored frames."""
        path, _, query = self.path.partition("?")
        if path == "/":
            self._send_bytes(200, "text/html; charset=utf-8", self.server.page_html)
        elif path == "/api/status":
            status = self.server.runs.status()
            if status["state"] == "idle":
                display = self.server.runs.current()
                if display is not None and display.get("state") in ("running", "external"):
                    status["state"] = display["state"]
                    status["log"] = display.get("log")
            if status["state"] in ("running", "external") and self.server.pool.any_started():
                # V4L2 nodes are exclusive: a CLI-launched eval cannot open the
                # cameras while the console streams, so the console yields.
                self.server.pool.stop()
                print("[console] cameras released: a run outside the console holds them")
            status["cameras_live"] = self.server.pool.any_started()
            status["arm_tool"] = self.server.armtools.describe()
            status["lerobot"] = self.server.exports.describe()
            if self.server.chest is not None:
                status["chest"] = self.server.chest.status()
            if self.server.last_video:
                status["last_video"] = self.server.last_video
            root = getattr(self.server, "evo_workspace", None)
            outcome_missing = root is not None and not (Path(root) / "outcome.json").is_file()
            waiting = root is not None and (Path(root) / "awaiting_outcome").is_file()
            if (
                self.server.default_policy == "evo"
                and outcome_missing
                and (status["state"] == "ended" or waiting)
            ):
                status["needs_outcome"] = True
            self._send_json(200, status)
        elif path == "/api/feed":
            self._api_feed(query)
        else:
            match = _CAM_PATH_RE.fullmatch(path)
            if match is not None:
                self._serve_camera(match.group("name"))
                return
            match = _FRAME_ROUTE_RE.fullmatch(path)
            if match is not None:
                self._serve_frame(match.group("name"))
                return
            match = _PREVIEW_ROUTE_RE.fullmatch(path)
            if match is not None:
                self._serve_preview(match.group("name"))
                return
            if path == "/api/chest-takes":
                self._api_chest_takes()
                return
            if path in _BRAND_FILES:
                self._serve_brand(path)
                return
            self._send_plain(404, "not found\n")

    def do_POST(self) -> None:
        """Route POST: start, stop, home, and estop, behind the Host/Origin gate."""
        if not self._request_allowed():
            self._send_plain(
                403,
                "forbidden: POST routes require a Host header matching this "
                "console's bound address (and a matching Origin when present)\n",
            )
            return
        path = self.path.split("?", 1)[0]
        if path == "/api/start":
            self._api_start()
        elif path == "/api/stop":
            self._api_stop()
        elif path == "/api/home":
            self._api_home()
        elif path == "/api/estop":
            self._api_estop()
        elif path == "/api/chest":
            self._api_chest()
        elif path == "/api/speech":
            self._api_speech()
        elif path == "/api/guide":
            self._api_guide()
        elif path == "/api/evo-outcome":
            self._api_evo_outcome()
        else:
            self._send_plain(404, "not found\n")

    def _api_feed(self, query: str) -> None:
        """Serve the live feed tail for ``?since=<seq>`` (invalid seq means 0)."""
        try:
            since = int(parse_qs(query).get("since", ["0"])[-1])
        except ValueError:
            since = 0
        feed = self.server.feed.build(self.server.runs.current(), since)
        self._send_json(200, feed)

    def _api_start(self) -> None:
        """Spawn a run from ``{"instruction": ...}``; camera-free by then."""
        body = self._read_json_body()
        if body is None:
            return
        instruction = body.get("instruction")
        if not isinstance(instruction, str):
            instruction = ""
        policy = body.get("policy", self.server.default_policy)
        if policy not in ("agent", "harness", "vla", "vla-replay", "vla-origin", "evo"):
            self._send_json(400, {"error": f"unknown policy {policy!r}"})
            return
        self.server.default_policy = policy
        if policy == "evo":
            mode = body.get("evo_mode", self.server.launch_args.evo_mode)
            if mode not in ("playground", "test"):
                self._send_json(400, {"error": "evo_mode must be playground or test"})
                return
            stamp = time.strftime("%Y%m%d-%H%M%S")
            root = Path(".inspect_traces") / "evo" / f"{stamp}_{os.urandom(2).hex()}"
            root.mkdir(parents=True, exist_ok=True)
            coder = body.get("evo_coder", getattr(self.server.launch_args, "evo_coder", "astra"))
            if coder not in ("astra", "glm"):
                self._send_json(400, {"error": "evo_coder must be astra or glm"})
                return
            self.server.launch_args.evo_mode = mode
            self.server.launch_args.evo_coder = coder
            self.server.launch_args.evo_workspace = str(root.resolve())
            self.server.evo_workspace = root
        if start_reads_video_goal(policy, body.get("intake")):
            try:
                self._use_named_take(body)
                instruction = self._agent_video_goal()
            except Exception as exc:
                print(f"[console] start rejected: {exc}")
                self._send_json(400, {"error": str(exc)})
                return
            suffix = self.server.build_suffix(policy)
            error = self.server.runs.start(
                instruction,
                prepare=self.server.pool.stop,
                rollback=self.server.pool.restart,
                suffix=suffix,
            )
            if error is not None:
                print(f"[console] start rejected: {error}")
                self._send_json(400, {"error": error})
                return
            self._send_json(200, {"ok": True, "prompt": instruction, "policy": policy})
            return
        if not instruction.strip():
            print("[console] start rejected: instruction must be non-empty")
            self._send_json(400, {"error": "instruction must be non-empty"})
            return
        if policy == "vla-origin":
            try:
                launch = vla_origin_launch(instruction.strip())
            except OSError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            error = self.server.runs.start(
                instruction.strip(),
                prepare=self.server.pool.stop,
                rollback=self.server.pool.restart,
                command=launch.command,
                env=launch.env,
                cwd=launch.cwd,
                stop_line="q",
                kick="s",
                tiles="preview",
            )
        else:
            suffix = self.server.build_suffix(policy)
            error = self.server.runs.start(
                instruction.strip(),
                prepare=self.server.pool.stop,
                rollback=self.server.pool.restart,
                suffix=suffix,
            )
        if error is not None:
            print(f"[console] start rejected: {error}")
            self._send_json(400, {"error": error})
            return
        self._send_json(200, {"ok": True})

    def _agent_video_goal(self) -> str:
        """Ask Astra for the goal paragraph of the latest chest demonstration."""
        from inspect_robots_agent._llm import Provider
        from inspect_robots_agent._responses import ResponsesClient
        from inspect_robots_intake.contract import IntakeError, VideoInput
        from inspect_robots_intake.video import MessageVideoPrompter

        path = self.server.last_video
        if not isinstance(path, str) or not path:
            raise IntakeError("record a video prompt before start")
        api_key = os.environ.get(self.server.api_key_env, "")
        if not api_key:
            raise IntakeError(f"{self.server.api_key_env} is not set")
        client = ResponsesClient(
            Provider(
                base_url=self.server.base_url,
                api_key=api_key,
                model=self.server.model,
            )
        )

        def complete(messages: list[dict[str, Any]]) -> str:
            reply = client.complete(messages, tools=[])
            return reply.content or ""

        return MessageVideoPrompter(complete).to_prompt(VideoInput(path))

    def _api_guide(self) -> None:
        """Send one live observation into the running agent or harness memory."""
        body = self._read_json_body()
        if body is None:
            return
        try:
            line = guide_line(body.get("text"))
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        error = self.server.runs.write_line(line)
        if error is not None:
            self._send_json(400, {"error": error})
            return
        self._send_json(200, {"ok": True, "text": line})

    def _api_stop(self) -> None:
        """Stop the episode and free the slot so Start can run again."""
        body = self._read_json_body()
        if body is None:
            return
        self.server.runs.mark_stop_requested()
        error = self.server.runs.force_end()
        if error is None:
            self._send_json(200, {"ok": True})
        else:
            self._send_json(400, {"error": error})

    def _api_home(self) -> None:
        """Home both arms, ending any run that is still holding Start."""
        end_error = self.server.runs.force_end()
        still_active = self.server.runs.status().get("state") in ("running", "stopping")
        error = end_error or self.server.armtools.home(run_active=still_active)
        if error is None:
            self._send_json(200, {"ok": True})
        else:
            self._send_json(400, {"error": error})

    def _api_estop(self) -> None:
        """Emergency stop: force-end any active run, then drop the arms limp."""
        errors = [self.server.runs.force_end(), self.server.armtools.estop()]
        messages = [item for item in errors if item is not None]
        if messages:
            self._send_json(400, {"error": "; ".join(messages)})
        else:
            self._send_json(200, {"ok": True})

    def _api_evo_outcome(self) -> None:
        """Write the operator label. Reflect only if the run process is already gone."""
        body = self._read_json_body()
        if body is None:
            return
        success = body.get("success")
        if not isinstance(success, bool):
            self._send_json(400, {"error": "success must be a bool"})
            return
        root = getattr(self.server, "evo_workspace", None)
        if root is None:
            self._send_json(400, {"error": "no evo workspace"})
            return
        (Path(root) / "outcome.json").write_text(json.dumps({"success": success}), encoding="utf-8")
        if self.server.runs.status().get("state") == "ended":
            self._reflect_stopped_evo(Path(root))
        self._send_json(200, {"ok": True})

    def _reflect_stopped_evo(self, root: Path) -> None:
        """Update skills after E-STOP killed the process. Does not move the robot."""
        plugin = Path(__file__).resolve().parents[2] / "inspect-robots-evo" / "src"
        if str(plugin) not in sys.path:
            sys.path.insert(0, str(plugin))
        try:
            from inspect_robots_evo.episode import reflect_dir
            from inspect_robots_evo.policy import coding_model
            from inspect_robots_evo.workspace import Workspace
        except ImportError as exc:
            print(f"[console] evo reflection skipped: {exc}")
            return
        args = self.server.launch_args
        workspace = Workspace(root, getattr(args, "evo_mode", "playground"))
        model_name, base_url, key_env, wire = _evo_endpoint(args)
        reflect_dir(
            workspace,
            coding_model(
                model_name,
                base_url=base_url,
                api_key=os.environ.get(key_env, ""),
                wire=wire,
            ),
        )

    def _chest_takes(self) -> list[str]:
        """Saved chest clips, newest first. Names stay inside the intake directory."""
        recorder = self.server.chest
        if recorder is None or not recorder.directory.is_dir():
            return []
        files = [
            path
            for path in recorder.directory.glob("chest_*.mp4")
            if path.is_file() and _CHEST_TAKE_RE.fullmatch(path.name)
        ]
        files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return [path.name for path in files]

    def _api_chest_takes(self) -> None:
        """List saved video prompts for the Video prompt menu."""
        self._send_json(200, {"takes": self._chest_takes()})

    def _use_named_take(self, body: Mapping[str, Any]) -> None:
        """Point the next video analysis at one saved clip, when the page named it."""
        from inspect_robots_intake.contract import IntakeError

        name = body.get("video")
        if not isinstance(name, str) or not name:
            return
        recorder = self.server.chest
        if (
            recorder is None
            or not _CHEST_TAKE_RE.fullmatch(name)
            or not (recorder.directory / name).is_file()
        ):
            raise IntakeError(f"unknown video prompt {name!r}")
        self.server.last_video = str(recorder.directory / name)

    def _api_chest(self) -> None:
        """Start or stop one chest take for the video-to-prompt intake."""
        recorder = self.server.chest
        if recorder is None:
            self._send_json(503, {"error": "video intake recorder is not available"})
            return
        from inspect_robots_intake.contract import IntakeError

        if recorder.status()["recording"]:
            try:
                video = recorder.stop()
            except IntakeError as exc:
                self._send_json(409, {"error": str(exc), **recorder.status()})
                return
            body = recorder.status()
            body["video"] = video.path
            self.server.last_video = video.path
            self._send_json(200, body)
            return
        run = self.server.runs.current()
        run_state = None if run is None else run.get("state")
        if run_state in ("running", "external") or self.server.runs.status()["state"] in (
            "running",
            "stopping",
        ):
            self._send_json(
                409,
                {"error": "stop the run before recording; it holds the chest camera"},
            )
            return
        try:
            recorder.start(self._read_chest)
        except IntakeError as exc:
            self._send_json(409, {"error": str(exc), **recorder.status()})
            return
        self._send_json(200, recorder.status())

    def _api_speech(self) -> None:
        """Transcribe one microphone clip and return the text for the prompt box."""
        transcriber = self.server.speech
        if transcriber is None:
            self._send_json(503, {"error": "speech intake is not available"})
            return
        header = self.headers.get("Content-Length")
        try:
            length = int(header) if header else 0
        except ValueError:
            self._send_json(400, {"error": "invalid Content-Length"})
            return
        if length <= 0 or length > 8_000_000:
            self._send_json(400, {"error": "speech clip must be between 1 byte and 8 MB"})
            return
        audio = self.rfile.read(length)
        speech_dir = _TRACE_DIR / "speech_intake"
        speech_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        webm = speech_dir / f"speech_{stamp}.webm"
        wav = speech_dir / f"speech_{stamp}.wav"
        webm.write_bytes(audio)
        try:
            converted = subprocess.run(
                ["ffmpeg", "-y", "-i", str(webm), "-ar", "16000", "-ac", "1", str(wav)],
                capture_output=True,
                check=False,
            )
        except FileNotFoundError:
            self._send_json(503, {"error": "ffmpeg is not installed"})
            return
        if converted.returncode != 0 or not wav.is_file():
            detail = converted.stderr.decode("utf-8", errors="replace")[-300:].strip()
            self._send_json(409, {"error": detail or "could not convert the speech clip to wav"})
            return
        from inspect_robots_intake.contract import IntakeError, SpeechInput

        try:
            text = transcriber.transcribe(SpeechInput(str(wav)))
        except IntakeError as exc:
            self._send_json(409, {"error": str(exc)})
            return
        self._send_json(200, {"text": text, "wav": str(wav)})

    def _read_chest(self) -> npt.NDArray[np.uint8]:
        """Newest chest RGB frame. The intake recorder samples this; it does not open the device."""
        camera = self.server.pool.start("chest_rgbd")
        frame, _stamp = camera.read(max_age_s=FRAME_MAX_AGE_S)
        return frame

    def _request_allowed(self) -> bool:
        """Whether the request is addressed to this console, not a foreign origin.

        Every POST route requires a Host header naming the bound address
        (loopback names accepted for loopback binds; port must match when
        present), and an Origin header, when sent, must match too. This keeps
        a page opened from another origin from driving the arms through the
        operator's browser.
        """
        host_header = self.headers.get("Host")
        if not host_header or not self._authority_allowed(host_header):
            return False
        origin = self.headers.get("Origin")
        return origin is None or self._origin_allowed(origin)

    def _authority_allowed(self, authority: str) -> bool:
        """Whether one ``host[:port]`` authority (or origin URL) is allowed."""
        value = authority.strip().lower()
        if value.startswith("["):
            end = value.find("]")
            if end == -1:
                return False
            host = value[1:end]
            port = value[end + 1 :].lstrip(":")
        else:
            host, separator, port = value.partition(":")
            if not separator:
                port = ""
            if ":" in host:
                return False
        host = host.removesuffix(".")
        if not self.server.wildcard_host and host not in self.server.allowed_hosts:
            return False
        return not (port and port != str(self.server.bound_port))

    def _origin_allowed(self, origin: str) -> bool:
        """Whether one Origin header value matches the bound address."""
        value = _SCHEME_RE.sub("", origin.strip().lower())
        authority = value.partition("/")[0]
        if not authority or authority == "null":
            return False
        return self._authority_allowed(authority)

    def _serve_camera(self, name: str) -> None:
        """Stream one camera as multipart JPEG until the client or camera goes.

        The first frame is grabbed before any header is sent so an unusable
        camera answers with a plain 503 instead of a broken stream; the
        header write itself sits inside the guarded block so a client that
        disconnects during warm-up cannot print a traceback. Failures after
        streaming begins (client disconnect, stale or unplugged device,
        encode error) just end this one stream and leave the server running.
        """
        try:
            camera = self.server.pool.start(name)
            frame, _stamp = camera.read(max_age_s=FRAME_MAX_AGE_S)
        except KeyError:
            self._send_plain(404, f"unknown camera {name!r}\n")
            return
        except (OSError, RuntimeError, ValueError) as exc:
            self._send_plain(503, f"camera {name!r} unavailable: {exc}\n")
            return
        period = 1.0 / STREAM_FPS
        deadline = time.monotonic() + period
        try:
            self.send_response(200)
            self.send_header("Content-Type", 'multipart/x-mixed-replace; boundary="frame"')
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            while True:
                payload = _encode_jpeg(frame)
                self.wfile.write(
                    b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(payload)).encode("ascii")
                    + b"\r\n\r\n"
                    + payload
                    + b"\r\n"
                )
                self.wfile.flush()
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                deadline = time.monotonic() + period
                frame, _stamp = camera.read(max_age_s=FRAME_MAX_AGE_S)
        except (OSError, RuntimeError, ValueError):
            return

    def _serve_frame(self, name: str) -> None:
        """Serve one JPEG for ``name``: the run's newest stored frame, or the device.

        While a run is active it owns the cameras, so this route reads the
        newest frame the run stored (same discovery the core report uses).
        Otherwise it falls back to a live snapshot from the device, starting
        the camera on first request.
        """
        try:
            self.server.pool.camera(name)
        except KeyError:
            self._send_plain(404, f"unknown camera {name!r}\n")
            return
        run = self.server.runs.current()
        if run is not None and run.get("state") in ("running", "external"):
            frame, error = self._stored_run_frame(run, name)
            if frame is None:
                frame = self.server.pool.held_frame(name)
            if frame is None:
                self._send_plain(503, f"camera {name!r} unavailable: {error}\n")
                return
        else:
            try:
                frame, _stamp = self.server.pool.start(name).read(max_age_s=FRAME_MAX_AGE_S)
            except (OSError, RuntimeError, ValueError) as exc:
                self._send_plain(503, f"camera {name!r} unavailable: {exc}\n")
                return
        try:
            payload = _encode_jpeg(frame)
        except (OSError, RuntimeError, ValueError) as exc:
            self._send_plain(503, f"camera {name!r} frame encode failed: {exc}\n")
            return
        self._send_bytes(200, "image/jpeg", payload, {"Cache-Control": "no-store"})

    def _serve_preview(self, name: str) -> None:
        """Serve the 1 Hz JPEG the ROS recorder overwrites for this camera."""
        if name not in _PREVIEW_CAMERAS:
            self._send_plain(404, "not found\n")
            return
        try:
            payload = (_PREVIEW_DIR / f"{name}.jpg").read_bytes()
        except OSError:
            self._send_plain(404, "preview not ready\n")
            return
        self._send_bytes(200, "image/jpeg", payload, {"Cache-Control": "no-store"})

    def _stored_run_frame(
        self, run: Mapping[str, Any], name: str
    ) -> tuple[npt.NDArray[np.uint8] | None, str]:
        """The running run's newest stored frame via its live snapshot."""
        doc, doc_path = _run_live_doc(run)
        frames_dir = None
        if doc is not None:
            stats = doc.get("stats")
            if isinstance(stats, dict):
                recorded = stats.get("frames_dir")
                if isinstance(recorded, str) and recorded:
                    frames_dir = recorded
        if frames_dir is None or doc_path is None:
            return None, "the run has not recorded a frames directory yet"
        return _stored_frame(frames_dir, doc_path, name)

    def _read_json_body(self) -> dict[str, Any] | None:
        """Parse the request body as a JSON object; None after sending a 400."""
        header = self.headers.get("Content-Length")
        if header is None:
            length = 0
        else:
            try:
                length = int(header)
            except ValueError:
                self._send_json(400, {"error": "invalid Content-Length"})
                return None
        if length > MAX_BODY_BYTES:
            self._send_json(400, {"error": "request body too large"})
            return None
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            parsed: Any = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "body must be valid JSON"})
            return None
        if not isinstance(parsed, dict):
            self._send_json(400, {"error": "body must be a JSON object"})
            return None
        return parsed

    def _serve_brand(self, path: str) -> None:
        """Serve one cropped company mark. Unknown names never reach this method."""
        content_type, name = _BRAND_FILES[path]
        file_path = _BRAND_DIR / name
        if not file_path.is_file():
            self._send_plain(404, "not found\n")
            return
        self._send_bytes(200, content_type, file_path.read_bytes())

    def _send_json(self, code: int, payload: Mapping[str, Any]) -> None:
        """Send one JSON response."""
        self._send_bytes(code, "application/json", json.dumps(payload).encode("utf-8"))

    def _send_plain(self, code: int, text: str) -> None:
        """Send one plain-text response (404s and camera 503s)."""
        self._send_bytes(code, "text/plain; charset=utf-8", text.encode("utf-8"))

    def _send_bytes(
        self,
        code: int,
        content_type: str,
        payload: bytes,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        """Send one complete response with an explicit length."""
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        if extra_headers is not None:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        """Log requests to stderr, hiding the once-a-second poll routes."""
        path = self.path.split("?", 1)[0]
        if path == "/api/status" or path == "/api/feed" or _FRAME_ROUTE_RE.fullmatch(path):
            return
        super().log_message(format, *args)


class MissionServer(ThreadingHTTPServer):
    """Threading HTTP server carrying the camera pool, run manager, and page."""

    def __init__(
        self,
        address: tuple[str, int],
        pool: CameraPool,
        runs: RunManager,
        armtools: ArmToolRunner,
        exporter: LerobotExporter,
        page_html: bytes,
        default_policy: str,
        build_suffix: Callable[[str], list[str]],
    ) -> None:
        """Bind ``address``; requests are served by MissionRequestHandler."""
        super().__init__(address, MissionRequestHandler)
        self.pool = pool
        self.runs = runs
        self.armtools = armtools
        self.exports = exporter
        self.default_policy = default_policy
        self.build_suffix = build_suffix
        self.feed = RunFeed()
        self.page_html = page_html
        host = address[0].lower()
        loopback = _LOOPBACK_HOSTS if host in _LOOPBACK_HOSTS else frozenset()
        self.allowed_hosts = frozenset({host} | loopback)
        self.wildcard_host = host in _WILDCARD_HOSTS
        self.bound_port = address[1]
        self.base_url = DEFAULT_BASE_URL
        self.api_key_env = DEFAULT_API_KEY_ENV
        self.model = DEFAULT_MODEL
        self.last_video: str | None = None
        # Video-intake take recorder. The chest device stays in this console;
        # the file it writes is a VideoInput for inspect-robots-intake.
        self.chest: Any = None


_PAGE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Nero mission console</title>
<style>
  :root { color-scheme: dark; }
  body {
    margin: 0; color: #f4f4f2;
    font-family: "Noto Sans CJK SC", "Noto Sans", sans-serif;
    background: #070708 url("/brand/ground.jpg") center top / cover no-repeat fixed;
  }
  header {
    display: flex; align-items: center; gap: 12px; height: 52px; padding: 0 16px;
    background: rgba(7, 7, 8, 0.72); border-bottom: 2px solid #f4f4f2;
  }
  .mark { height: 28px; width: auto; background: #f4f4f2; padding: 4px 5px; }
  h1 { margin: 0; font-size: 16px; font-weight: 700; letter-spacing: 0.14em; }
  header a {
    margin-left: auto; color: #f4f4f2; text-decoration: none;
    border-bottom: 1px solid #f4f4f2;
  }
  .stage { display: grid; grid-template-columns: 84px minmax(0, 1fr); align-items: start; }
  .stage-main { min-width: 0; }
  .signal {
    display: flex; flex-direction: column; align-items: center; gap: 10px;
    position: sticky; top: 8px; margin: 8px 0 8px 12px; padding: 4px 0;
  }
  .housing {
    display: flex; flex-direction: column; gap: 8px; padding: 10px 8px;
    border-radius: 18px; background: #0c0c0c; border: 1px solid #3a3a3a;
  }
  .housing i {
    width: 22px; height: 22px; border-radius: 50%; display: block;
    background: #1c1c1c; box-shadow: inset 0 0 0 1px #111;
  }
  #signallabel { font-size: 13px; font-weight: 700; letter-spacing: 0.12em; color: #f4f4f2; }
  .signal[data-kind="idle"] .housing i:nth-child(2) { background: #5c4a14; }
  .signal[data-kind="run"] .housing i:nth-child(3) {
    background: #30d158; box-shadow: 0 0 14px #30d158;
  }
  .signal[data-kind="wait"] .housing i:nth-child(2) {
    background: #ffd60a; box-shadow: 0 0 14px #ffd60a;
  }
  .signal[data-kind="stop"] .housing i:nth-child(1) {
    background: #ff453a; box-shadow: 0 0 14px #ff453a;
  }
  .signal[data-kind="fault"] .housing i:nth-child(1) {
    background: #ff453a; box-shadow: 0 0 14px #ff453a;
  }
  @media (prefers-reduced-motion: no-preference) {
    .signal[data-kind="wait"] .housing i:nth-child(2),
    .signal[data-kind="fault"] .housing i:nth-child(1) {
      animation: lamp 0.9s steps(2, end) infinite;
    }
  }
  @keyframes lamp { 50% { opacity: 0.2; box-shadow: none; } }
  #status {
    padding: 8px 16px; background: rgba(7, 7, 8, 0.55); color: #c8c8c6;
    font-family: ui-monospace, monospace; font-size: 13px; white-space: pre-wrap;
  }
  .cams { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; padding: 0 12px 8px; }
  .cams figure { margin: 0; background: #000; }
  .cams img {
    width: 100%; aspect-ratio: 4 / 3; object-fit: contain; display: block; background: #000;
  }
  .cams figcaption {
    font-size: 13px; padding: 6px 8px 6px; color: #f4f4f2; background: rgba(7, 7, 8, 0.72);
    border-top: 2px solid #f4f4f2;
  }
  .cams figcaption .mode { float: right; color: #8a8a88; }
  .deck { display: grid; gap: 10px; padding: 12px 16px 4px; }
  .taskrow, .actions { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  #policy, #instruction, #guide {
    height: 42px; box-sizing: border-box; background: #111; color: #f4f4f2;
    border: 1px solid #2c2c2c; border-radius: 2px; padding: 0 10px; font: inherit;
  }
  #instruction, #guide { flex: 1; min-width: 240px; }
  #policy:focus, #instruction:focus, #guide:focus, button:focus-visible {
    outline: 2px solid #f4f4f2; outline-offset: 2px;
  }
  button {
    height: 42px; padding: 0 16px; font: inherit; font-weight: 600; border-radius: 2px;
    cursor: pointer; border: 1px solid transparent;
  }
  #start { background: #f4f4f2; color: #111; }
  #start:hover { background: #fff; }
  #start.armed, #start.armed:hover { background: #111; color: #f4f4f2; }
  #stop, #cancel { background: transparent; color: #f4f4f2; border-color: #3a3a3a; }
  #speech, #chest, #home {
    background: #141414; color: #f4f4f2; border-color: #2a2a2a;
  }
  .video-pop { position: relative; display: flex; }
  #video-more {
    width: 28px; padding: 0; background: #141414; color: #8a8a88;
    border: 1px solid #2a2a2a; border-left: 0;
  }
  #video-more.armed { color: #f4f4f2; border-color: #f4f4f2; }
  .video-takes {
    position: absolute; top: 46px; right: 0; z-index: 6; min-width: 280px;
    max-height: 240px; overflow: auto; background: #111; border: 1px solid #3a3a3a;
  }
  .video-takes button {
    display: block; width: 100%; height: 34px; padding: 0 12px; text-align: left;
    background: transparent; color: #f4f4f2; border: 0; border-radius: 0;
    font-size: 13px; font-weight: 500;
  }
  .video-takes .empty { color: #8a8a88; padding: 10px 12px; }
  #estop {
    margin-left: auto; min-width: 112px; background: #111; color: #ff3b30;
    border-color: #ff3b30; font-weight: 700;
  }
  #estop:hover { background: #ff3b30; color: #111; }
  button.armed { background: #111; color: #f4f4f2; box-shadow: inset 0 -3px 0 #f4f4f2; }
  .video-takes button.armed { box-shadow: inset 3px 0 0 #f4f4f2; }
  #speech.armed, #chest.armed { border-color: #f4f4f2; }
  button:disabled { opacity: 0.38; cursor: default; }
  @media (prefers-reduced-motion: no-preference) {
    #start.armed { animation: slit 1.4s steps(2, end) infinite; }
  }
  @keyframes slit { 50% { box-shadow: inset 0 -3px 0 transparent; } }
  .tool-line { margin: 2px 0; overflow-wrap: anywhere; }
  .tool-name { font-weight: bold; padding: 0 6px; border-radius: 4px;
    background: #d7dce3; color: #20242b; margin-right: 4px; }
  .tool-line.tool-move_to .tool-name { background: #19723b; color: #fff; }
  .tool-line.tool-set_gripper .tool-name { background: #8a5700; color: #fff; }
  .tool-line.tool-delegate_skill .tool-name { background: #7a55b5; color: #fff; }
  .tool-line.tool-done .tool-name,
  .tool-line.tool-give_up .tool-name { background: #a12a2a; color: #fff; }
  .tool-args { color: #d6dee8; font-family: ui-monospace, monospace; font-size: 12px; }
  .tool-note { color: #8a8a88; }
  #usetrained { background: transparent; color: #f4f4f2; border-color: #f4f4f2; }
  #feedpanel { margin: 8px 16px 24px; display: grid; gap: 8px; }
  .feedgrid { display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 8px; }
  .panel { border: 1px solid rgba(255, 255, 255, 0.08); background: rgba(7, 7, 8, 0.62); }
  .panel h2 {
    font-size: 13px; font-weight: 600; margin: 0; padding: 8px 10px; color: #f4f4f2;
    background: transparent; border-bottom: 1px solid #222;
  }
  .panelbody {
    padding: 8px; font-size: 13px; color: #d4d4d2; white-space: pre-wrap;
    overflow-wrap: anywhere; max-height: 170px; overflow-y: auto;
  }
  #feed { padding: 8px; font-size: 13px; color: #bdbdbd; max-height: 340px; overflow-y: auto; }
  .msg { margin: 0 0 8px; padding-left: 8px; border-left: 2px solid #2a2a2a; }
  .msg-user { border-left-color: #f4f4f2; }
  .msg-assistant { border-left-color: #8a8a88; }
  .msg-system { border-left-color: #3a3a3a; }
  .msg-role { display: inline-block; font-size: 12px; color: #8a8a88; margin-right: 8px; }
  .msg-text { white-space: pre-wrap; overflow-wrap: anywhere; color: #f4f4f2; }
  .msg-tool {
    font-family: ui-monospace, monospace; font-size: 12px; color: #d4d4d2;
    white-space: pre-wrap; overflow-wrap: anywhere;
  }
  @media (max-width: 900px) {
    .feedgrid, .cams { grid-template-columns: 1fr; }
    header { height: auto; min-height: 52px; padding: 8px 16px; }
    #estop { margin-left: 0; }
  }
</style>
</head>
<body>
<header>
  <img class="mark" src="/brand/mark.png" alt="">
  <h1>MATRIX</h1>
  <a href="__HISTORY_URL__" target="_blank" rel="noopener">Run history</a>
</header>
<div class="stage">
<div id="signal" class="signal" data-kind="idle" role="status" aria-live="polite">
  <span class="housing" aria-hidden="true"><i></i><i></i><i></i></span>
  <span id="signallabel">空闲</span>
</div>
<div class="stage-main">
<div id="status">connecting...</div>
<div class="cams">
"""

_PAGE_TAIL = """</div>
</div>
</div>
<div class="deck">
  <div class="taskrow">
  <select id="policy">
    <option value="agent">agent (LLM only)</option>
    <option value="harness">harness (HarnessVLA)</option>
    <option value="vla">vla (VLA only)</option>
    <option value="vla-replay">vla-replay (same VLA only, recorded episode)</option>
    <option value="vla-origin">vla-origin (ROS inference script)</option>
    <option value="evo">evo (EmbodiedEvo)</option>
  </select>
  <select id="evomode">
    <option value="playground">playground</option>
    <option value="test">test</option>
  </select>
  <select id="evocoder">
    <option value="astra">astra</option>
    <option value="glm">glm 5.3</option>
  </select>
  <input id="instruction" placeholder="instruction for the run" autocomplete="off">
  <button id="usetrained" hidden
    title="replace the instruction with the checkpoint's exact training task string">
    use trained task string</button>
  </div>
  <div class="actions">
  <input id="guide" placeholder="运行中告诉agent你的观察和经验" autocomplete="off" disabled>
  <button id="guidesend" type="button" disabled>Send to agent</button>
  <button id="start">Start</button>
  <button id="stop">Stop</button>
  <button id="cancel" hidden>Cancel</button>
  <button id="speech" title="first press starts the microphone, second press transcribes it">
    Voice prompt</button>
  <div class="video-pop" id="videopop">
  <button id="chest"
    title="first press records a new chest clip, second press keeps it">
    Video prompt</button>
  <button id="video-more" title="choose a saved video prompt" aria-expanded="false">▾</button>
  <div id="video-takes" class="video-takes" hidden></div>
  </div>
  <button id="home">Home arms</button>
  <button id="estop" title="end the run; both arms descend under damping until they stop">
    E-STOP</button>
  <span id="evooutcome" hidden>
    <button id="evoyes" type="button">success</button>
    <button id="evono" type="button">failure</button>
  </span>
  </div>
</div>
<section id="feedpanel">
  <div class="feedgrid">
    <div class="panel"><h2>run status</h2>
      <div class="panelbody" id="livebody">no data yet</div></div>
    <div class="panel"><h2>last model decision</h2>
      <div class="panelbody" id="wirebody">no data yet</div></div>
    <div class="panel"><h2>last actions</h2>
      <div class="panelbody" id="actionbody">no data yet</div></div>
  </div>
  <div class="panel"><h2>stage timing</h2>
    <div id="timebody" class="panelbody">no timings yet</div></div>
  <div class="panel"><h2>tool timeline</h2>
    <div id="toolbody" class="panelbody">no tool calls yet</div></div>
  <div class="panel"><h2>run terminal (prompts appear here)</h2>
    <div id="ptybody" class="panelbody">no output yet</div></div>
  <div class="panel"><h2>reasoning feed (newest pinned)</h2><div id="feed">no data yet</div></div>
</section>
<script>
const startBtn = document.getElementById("start");
const guideInput = document.getElementById("guide");
const guideSend = document.getElementById("guidesend");
const stopBtn = document.getElementById("stop");
const cancelBtn = document.getElementById("cancel");
const policySel = document.getElementById("policy");
const input = document.getElementById("instruction");
// Pure vla and vla-origin both submit this sentence as the model task.
// Near-variants stall, so it is prefilled and one click puts it back.
const ORIGIN_PROMPT = "please pick the red cup and place it on the black mat";
function prefillTrainedPrompt() {
  if (policySel.value !== "vla" && policySel.value !== "vla-replay"
      && policySel.value !== "vla-origin") return;
  if (input.value.trim() === "" || input.value === __TRAINED_PROMPT__) {
    input.value = ORIGIN_PROMPT;
  }
}
policySel.addEventListener("change", prefillTrainedPrompt);
prefillTrainedPrompt();
// The skill policy only tracks its trained phrasing; a hand-typed near
// variant (a comma, "then") stalls after the trained half. When a vla run
// would submit anything else, offer the exact string in one click.
const useTrainedBtn = document.getElementById("usetrained");
function refreshTrainedHint() {
  useTrainedBtn.hidden = !(
    policySel.value === "vla" &&
    input.value.trim() !== "" && input.value !== ORIGIN_PROMPT
  );
}
policySel.addEventListener("change", refreshTrainedHint);
input.addEventListener("input", refreshTrainedHint);
useTrainedBtn.addEventListener("click", () => {
  input.value = ORIGIN_PROMPT;
  refreshTrainedHint();
  input.focus();
});
refreshTrainedHint();
const speechBtn = document.getElementById("speech");
const chestBtn = document.getElementById("chest");
const videoMore = document.getElementById("video-more");
const takesEl = document.getElementById("video-takes");
const videoPop = document.getElementById("videopop");
const statusBox = document.getElementById("status");
const signal = document.getElementById("signal");
const signalLabel = document.getElementById("signallabel");
const SIGNAL_TEXT = {run: "运行", wait: "请等待", stop: "已结束", fault: "出错", idle: "空闲"};
function signalKind(state, status) {
  const exit = status.exit_code;
  const chest = status.chest;
  const failed = !!(stickyError || status.error
    || (exit !== null && exit !== undefined && exit !== 0));
  if (state === "running" || state === "external") { return "run"; }
  if (state === "stopping" || startPending || (chest && chest.recording)) { return "wait"; }
  if (failed) { return "fault"; }
  if (state === "ended") { return "stop"; }
  return "idle";
}
function setSignal(state, status) {
  if (!signal) { return; }
  const kind = signalKind(state, status);
  signal.dataset.kind = kind;
  if (signalLabel) { signalLabel.textContent = SIGNAL_TEXT[kind]; }
}
const feedEl = document.getElementById("feed");
const liveBody = document.getElementById("livebody");
const wireBody = document.getElementById("wirebody");
const actionBody = document.getElementById("actionbody");
let armed = false;
let startPending = false;
let lastState = "";
let lastError = "";
let tileMode = "live";
let since = 0;
const seenSeq = new Set();

function disarm() {
  cancelBtn.hidden = true;
  armed = false;
  startBtn.textContent = "Start";
  startBtn.classList.remove("armed");
}

function frameUrl(name) {
  return "/frame/" + encodeURIComponent(name) + ".jpg?x=" + Date.now();
}

function holdVisibleTiles() {
  for (const img of document.querySelectorAll(".cams img")) {
    if (!img.naturalWidth || !img.naturalHeight) { continue; }
    try {
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const context = canvas.getContext("2d");
      if (!context) { continue; }
      context.drawImage(img, 0, 0);
      img.src = canvas.toDataURL("image/jpeg", 0.85);
    } catch (err) {
      // A frame the browser will not copy stays as it is.
    }
  }
}

function showStill(img, blob) {
  const next = URL.createObjectURL(blob);
  const previous = img.dataset.objectUrl;
  img.src = next;
  img.dataset.objectUrl = next;
  if (previous) { URL.revokeObjectURL(previous); }
}

function setTileMode(mode) {
  tileMode = mode;
  for (const img of document.querySelectorAll(".cams img")) {
    const label = img.parentElement.querySelector(".mode");
    if (mode === "live") {
      img.src = "";
      img.src = img.dataset.src;
      if (label) { label.textContent = "live stream"; }
    } else if (mode === "frames") {
      if (label) { label.textContent = "run frame (1s)"; }
    } else if (mode === "preview") {
      if (label) { label.textContent = "inference preview (1s)"; }
    } else {
      img.src = "";
      if (label) { label.textContent = ""; }
    }
  }
  if (mode === "frames") { refreshFrameTiles(); }
  if (mode === "preview") { refreshPreviewTiles(); }
}

function refreshFrameTiles() {
  for (const img of document.querySelectorAll(".cams img")) {
    fetch(frameUrl(img.dataset.cam)).then((response) => {
      return response.ok ? response.blob() : null;
    }).then((blob) => {
      if (!blob || tileMode !== "frames") { return; }
      showStill(img, blob);
    }).catch(() => {});
  }
}

function refreshPreviewTiles() {
  for (const img of document.querySelectorAll(".cams img")) {
    const url = "/preview/" + encodeURIComponent(img.dataset.cam) + ".jpg?x=" + Date.now();
    fetch(url).then((response) => response.ok ? response.blob() : null).then((blob) => {
      if (!blob || tileMode !== "preview") { return; }
      showStill(img, blob);
    }).catch(() => {});
  }
}

function renderTiming(lines) {
  const body = document.getElementById("timebody");
  if (!body) { return; }
  const timing = [];
  for (const line of lines || []) {
    const at = line.indexOf("[timing]");
    if (at !== -1) { timing.push(line.slice(at)); }
  }
  body.textContent = timing.length ? timing.join("\\n") : "no timings yet";
}

function render(status) {
  const state = status.state || "idle";
  const parts = ["state: " + state];
  if (status.instruction) { parts.push("instruction: " + status.instruction); }
  if (status.log) { parts.push("log: " + status.log); }
  if (status.exit_code !== null && status.exit_code !== undefined) {
    parts.push("exit: " + status.exit_code);
  }
  if (state === "running" || state === "stopping") { stickyError = ""; }
  if (stickyError) { parts.push("error: " + stickyError); }
  else if (status.error) { parts.push("error: " + status.error); }
  const chest = status.chest;
  if (chestBtn && chest) {
    const saved = chest.saved || 0;
    chestBtn.textContent = chest.recording
      ? ("Stop video (" + saved + " saved)")
      : "Video prompt";
    chestBtn.classList.toggle("armed", !!chest.recording);
    if (chest.recording && chest.path) { parts.push("chest: " + chest.path); }
    else if (chest.directory) { parts.push("chest takes: " + chest.directory); }
    if (chest.error) { parts.push("chest error: " + chest.error); }
  }
  if (videoMore) {
    const recording = !!(chest && chest.recording);
    videoMore.disabled = state === "running" || recording;
    videoMore.classList.toggle("armed", !!selectedTake);
    if (selectedTake) {
      parts.push("next start reuses " + selectedTake);
    } else if (intakeKind === "video" && status.last_video) {
      parts.push("next start uses " + String(status.last_video).split("/").pop());
    }
  }
  setSignal(state, status);
  if (!startPending) { statusBox.textContent = parts.join("   |   "); }
  stopBtn.disabled = state !== "running";
  document.getElementById("evooutcome").hidden = !status.needs_outcome;
  startBtn.disabled = startPending || state === "running";
  const canGuide = state === "running"
    && (policySel.value === "agent" || policySel.value === "harness");
  if (guideInput) { guideInput.disabled = !canGuide; }
  if (guideSend) { guideSend.disabled = !canGuide; }
  const active = state === "running" || state === "stopping" || state === "external";
  const wantPreview = active && status.tiles === "preview";
  if (wantPreview && tileMode !== "preview") {
    setTileMode("preview");
  } else if (active && !wantPreview && tileMode !== "frames") {
    setTileMode("frames");
  } else if (!active && (tileMode === "frames" || tileMode === "preview")) {
    setTileMode("live");
  }
  if ((status.error || "") !== lastError) {
    lastError = status.error || "";
    if (lastError && tileMode === "live") {
      // A failed start released then restarted the cameras; reconnect tiles.
      setTileMode("live");
    }
  }
  if (state !== lastState) {
    disarm();
    lastState = state;
  }
}

function textLine(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div;
}

function messageNode(message) {
  const div = document.createElement("div");
  div.className = "msg msg-" + message.role;
  const role = document.createElement("span");
  role.className = "msg-role";
  role.textContent = message.role;
  div.appendChild(role);
  if (message.text) {
    const text = document.createElement("div");
    text.className = "msg-text";
    text.textContent = message.text;
    div.appendChild(text);
  }
  for (const tool of message.tools || []) {
    const line = document.createElement("div");
    line.className = "msg-tool";
    line.textContent = tool.name + "(" + tool.arguments + ")";
    div.appendChild(line);
  }
  return div;
}

function renderLive(live) {
  if (!live) { liveBody.textContent = "no data yet"; return; }
  const parts = [];
  if (live.status !== undefined && live.status !== null) { parts.push("status: " + live.status); }
  if (live.step !== undefined && live.step !== null) { parts.push("step: " + live.step); }
  if (live.total_steps !== undefined && live.total_steps !== null) {
    parts.push("steps total: " + live.total_steps);
  }
  if (typeof live.duration_s === "number") {
    parts.push("duration: " + live.duration_s.toFixed(0) + "s");
  }
  if (live.updated_at) { parts.push("updated: " + live.updated_at); }
  liveBody.replaceChildren(...(parts.length ? parts.map(textLine) : [textLine("no data yet")]));
}

function renderWire(wire) {
  if (!wire) { wireBody.textContent = "no data yet"; return; }
  wireBody.replaceChildren(textLine(wire.tool + (wire.note ? ": " + wire.note : " (no note)")));
}

function renderActions(actions) {
  if (!actions || !actions.rows || !actions.rows.length) {
    actionBody.textContent = "no data yet";
    return;
  }
  const lines = actions.rows.map(function (row) {
    const values = (row.action || []).map(function (value) {
      return typeof value === "number" ? value.toFixed(3) : String(value);
    });
    return "t=" + row.t + "  [" + values.join(", ") + "]";
  });
  actionBody.replaceChildren(...lines.map(textLine));
}

function renderTools(tools) {
  const body = document.getElementById("toolbody");
  if (!body) { return; }
  // The feed delivers only NEW tool calls (seq-filtered delta), so an empty
  // delta must never wipe the accumulated timeline: the placeholder shows
  // only while nothing has been seen at all.
  if (!tools || !tools.length) {
    if (!body.children.length) { body.textContent = "no tool calls yet"; }
    return;
  }
  if (!body.children.length) { body.replaceChildren(); }
  const seen = new Set([...body.children].map(function (node) { return node.dataset.seq; }));
  for (const tool of tools) {
    if (seen.has(String(tool.seq))) { continue; }
    const line = document.createElement("div");
    line.className = "tool-line tool-" + tool.name;
    line.dataset.seq = tool.seq;
    const stamp = document.createElement("span");
    stamp.className = "tool-note";
    stamp.textContent = (tool.at || new Date().toTimeString().slice(0, 8)) + " ";
    line.appendChild(stamp);
    const label = document.createElement("span");
    label.className = "tool-name";
    label.textContent = tool.name;
    line.appendChild(label);
    if (tool.arguments) {
      const args = document.createElement("span");
      args.className = "tool-args";
      args.textContent = "(" + tool.arguments + ")";
      line.appendChild(args);
    }
    if (tool.note) {
      const note = document.createElement("span");
      note.className = "tool-note";
      note.textContent = " " + tool.note;
      line.appendChild(note);
    }
    body.prepend(line);
  }
  while (body.children.length > 200) { body.lastChild.remove(); }
}

function renderFeed(feed) {
  renderLive(feed.live);
  renderWire(feed.wire);
  renderActions(feed.actions);
  renderTools(feed.tools);
  const messages = feed.messages || [];
  if (!messages.length) { return; }
  if (feedEl.textContent === "no data yet") { feedEl.replaceChildren(); }
  for (const message of messages) {
    if (seenSeq.has(message.seq)) { continue; }
    seenSeq.add(message.seq);
    feedEl.prepend(messageNode(message));
  }
  while (feedEl.children.length > 400) { feedEl.lastChild.remove(); }
}

async function post(path, body) {
  let data = {};
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {})
    });
    data = await response.json().catch(() => ({}));
    if (!response.ok) {
      stickyError = data.error || String(response.status);
      statusBox.textContent = "error: " + stickyError;
    } else if (path === "/api/start") {
      stickyError = "";
    }
    if (path === "/api/chest" && response.ok && data.video) {
      selectedTake = null;
      intakeKind = "video";
      statusBox.textContent = "video prompt: " + data.video;
    }
  } catch (err) {
    statusBox.textContent = "error: " + err;
  }
  poll();
  pollFeed();
  return data;
}

startBtn.addEventListener("click", () => {
  if (!armed) {
    armed = true;
    startBtn.textContent = "Confirm start — arms will move";
    startBtn.classList.add("armed");
    cancelBtn.hidden = false;
    return;
  }
  disarm();
  holdVisibleTiles();
  const body = {
    instruction: input.value,
    policy: policySel.value,
    evo_mode: document.getElementById("evomode").value,
    evo_coder: document.getElementById("evocoder").value,
    intake: intakeKind,
    video: selectedTake
  };
  const videoGoal = (body.policy === "agent" || body.policy === "harness")
    && body.intake === "video";
  if (videoGoal) {
    startPending = true;
    stickyError = "";
    statusBox.textContent = "reading the video prompt...";
    startBtn.disabled = true;
  }
  post("/api/start", body).then((data) => {
    startPending = false;
    if (data && data.prompt) {
      input.value = data.prompt;
      const label = data.policy === "harness" ? "harness goal"
        : data.policy === "agent" ? "agent goal" : "vla prompt";
      statusBox.textContent = label + ": " + data.prompt;
      intakeKind = "speech";
      selectedTake = null;
    }
  });
});
function sendGuide() {
  if (!guideInput || guideInput.disabled) { return; }
  const text = guideInput.value.trim();
  if (!text) { return; }
  post("/api/guide", {text: text}).then((data) => {
    if (data && data.ok) {
      guideInput.value = "";
      statusBox.textContent = "sent to the agent: " + data.text;
    }
  });
}
if (guideSend) { guideSend.addEventListener("click", sendGuide); }
if (guideInput) {
  guideInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      sendGuide();
    }
  });
}
cancelBtn.addEventListener("click", disarm);
input.addEventListener("input", disarm);
stopBtn.addEventListener("click", () => post("/api/stop", {}));
document.getElementById("home").addEventListener("click", () => post("/api/home", {}));
chestBtn.addEventListener("click", () => {
  if (!chestBtn.classList.contains("armed")) {
    selectedTake = null;
    intakeKind = "speech";
    closeTakes();
  }
  post("/api/chest", {});
});
function closeTakes() {
  takesEl.hidden = true;
  videoMore.setAttribute("aria-expanded", "false");
}
async function openTakes() {
  takesEl.hidden = false;
  videoMore.setAttribute("aria-expanded", "true");
  takesEl.replaceChildren();
  let names = [];
  try {
    const response = await fetch("/api/chest-takes");
    const data = await response.json();
    names = (data && data.takes) || [];
  } catch (err) {
    names = [];
  }
  if (!names.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "no saved video prompts";
    takesEl.append(empty);
    return;
  }
  for (const name of names) {
    const item = document.createElement("button");
    item.type = "button";
    item.textContent = name;
    if (name === selectedTake) { item.classList.add("armed"); }
    item.addEventListener("click", (event) => {
      event.stopPropagation();
      selectedTake = selectedTake === name ? null : name;
      intakeKind = selectedTake ? "video" : "speech";
      closeTakes();
      poll();
    });
    takesEl.append(item);
  }
}
videoMore.addEventListener("click", (event) => {
  event.stopPropagation();
  if (takesEl.hidden) { openTakes(); }
  else { closeTakes(); }
});
document.addEventListener("click", (event) => {
  if (!videoPop.contains(event.target)) { closeTakes(); }
});
let speechRecorder = null;
let speechStream = null;
let intakeKind = "speech";
let selectedTake = null;
let stickyError = "";
speechBtn.addEventListener("click", async () => {
  if (speechRecorder && speechRecorder.state === "recording") {
    speechRecorder.stop();
    return;
  }
  try {
    speechStream = await navigator.mediaDevices.getUserMedia({audio: true});
  } catch (err) {
    statusBox.textContent = "error: microphone unavailable: " + err;
    return;
  }
  const chunks = [];
  speechRecorder = new MediaRecorder(speechStream);
  speechRecorder.ondataavailable = (event) => {
    if (event.data && event.data.size) { chunks.push(event.data); }
  };
  speechRecorder.onstop = async () => {
    if (speechStream) { speechStream.getTracks().forEach((track) => track.stop()); }
    speechBtn.disabled = true;
    speechBtn.textContent = "Transcribing...";
    speechBtn.classList.remove("armed");
    const blob = new Blob(chunks, {type: speechRecorder.mimeType || "audio/webm"});
    try {
      const response = await fetch("/api/speech", {
        method: "POST",
        headers: {"Content-Type": blob.type || "application/octet-stream"},
        body: blob
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        statusBox.textContent = "error: " + (data.error || response.status);
      } else {
        input.value = data.text || "";
        intakeKind = "speech";
        selectedTake = null;
        statusBox.textContent = "voice prompt: " + input.value;
      }
    } catch (err) {
      statusBox.textContent = "error: " + err;
    }
    speechBtn.disabled = false;
    speechBtn.textContent = "Voice prompt";
  };
  speechRecorder.start();
  speechBtn.textContent = "Stop voice";
  speechBtn.classList.add("armed");
});
document.getElementById("estop").addEventListener("click", () => post("/api/estop", {}));
document.getElementById("evoyes").addEventListener("click", () => {
  post("/api/evo-outcome", { success: true });
});
document.getElementById("evono").addEventListener("click", () => {
  post("/api/evo-outcome", { success: false });
});

async function poll() {
  try {
    const response = await fetch("/api/status");
    render(await response.json());
    if (tileMode === "frames") { refreshFrameTiles(); }
    else if (tileMode === "preview") { refreshPreviewTiles(); }
  } catch (err) {
    statusBox.textContent = "connection lost: " + err;
    if (signal) { signal.dataset.kind = "fault"; }
    if (signalLabel) { signalLabel.textContent = SIGNAL_TEXT.fault; }
  }
}

async function pollFeed() {
  try {
    const response = await fetch("/api/feed?since=" + since);
    const feed = await response.json();
    renderFeed(feed);
    const pty = document.getElementById("ptybody");
    if (pty) {
      const lines = feed.pty_tail || [];
      pty.textContent = lines.length ? lines.join("\\n") : "no output yet";
      renderTiming(lines);
    }
    if (typeof feed.seq === "number" && feed.seq > since) { since = feed.seq; }
  } catch (err) {
    // The status poll reports connection loss; keep the last rendered feed.
  }
}
poll();
pollFeed();
setInterval(poll, 1000);
setInterval(pollFeed, 1000);
</script>
</body>
</html>
"""


def guide_line(text: object) -> str:
    """One operator observation safe to write on the run's console channel.

    Slash lines are episode commands. Collapsing whitespace keeps the line a
    single message the policy stores.
    """
    if not isinstance(text, str):
        raise ValueError("guidance must be non-empty")
    cleaned = " ".join(text.split())
    if not cleaned:
        raise ValueError("guidance must be non-empty")
    if cleaned.startswith("/"):
        raise ValueError("guidance cannot start with /")
    return cleaned[:240]


def start_reads_video_goal(policy: str, intake: object) -> bool:
    """True when Start must turn the recorded demonstration into the trial goal.

    Agent and harness plan from that paragraph. A pure VLA run keeps the
    typed sentence, because a free-form paragraph stalls the checkpoint.
    """
    return policy in ("agent", "harness") and intake == "video"


def _trained_prompt() -> str:
    """The VLA checkpoint's training task string, or "" without the plugin.

    The console itself stays importable with only core installed; the prefill
    simply stays empty when the vla plugin is absent (vla runs would fail at
    spawn anyway without its registered policy).
    """
    try:
        from inspect_robots_vla._config import TRAINED_SKILL_PROMPT
    except ImportError:
        return ""
    return str(TRAINED_SKILL_PROMPT)


def render_page(camera_names: Sequence[str], history_url: str, trained_prompt: str) -> bytes:
    """Render the console page; every interpolation is escaped exactly once.

    The trained prompt lands inside a script block, so it is embedded as a
    JSON string literal with ``<`` neutralized (HTML entities do not decode
    inside script elements; the replace keeps a future prompt edit from
    closing the block).
    """
    tiles = "".join(
        '<figure><img data-cam="'
        + html.escape(name, quote=True)
        + '" data-src="/cam/'
        + html.escape(name, quote=True)
        + '.mjpg" src="/cam/'
        + html.escape(name, quote=True)
        + '.mjpg" alt="'
        + html.escape(name, quote=True)
        + ' camera tile"><figcaption>'
        + html.escape(name)
        + '<span class="mode">live stream</span></figcaption></figure>'
        for name in camera_names
    )
    head = _PAGE_HEAD.replace("__HISTORY_URL__", html.escape(history_url, quote=True))
    prompt_literal = json.dumps(trained_prompt).replace("<", "\\u003c")
    tail = _PAGE_TAIL.replace("__TRAINED_PROMPT__", prompt_literal)
    return (head + tiles + tail).encode("utf-8")


_RLTOKEN = Path("/home/dell/fengxuedong/projects/neo_manipulation/.worktree/RLtoken")
_ORIGIN_CONFIG = (
    _RLTOKEN / "neo_manipulation/evaluation/robot_itl_evaluation/config/ros2_nero_inference.yaml"
)
_ORIGIN_SCRIPT = _RLTOKEN / "scripts/eval/robot_itl_evaluation/run_ros2_real_robot_inference.sh"


class _VlaOriginLaunch:
    """Argv, env, and cwd for one ROS inference script run."""

    def __init__(self, command: list[str], env: dict[str, str], cwd: str) -> None:
        self.command = command
        self.env = env
        self.cwd = cwd


def overlay_direct_http_task(text: str, task: str) -> str:
    """Replace the live direct_http task and leave every other task line."""
    lines = text.splitlines(keepends=True)
    in_direct_http = False
    replaced = False
    rendered: list[str] = []
    quoted = json.dumps(task)
    for line in lines:
        body = line.lstrip(" ")
        indent = len(line) - len(body)
        key = body.rstrip("\n")
        if indent <= 6 and key.endswith(":"):
            in_direct_http = key.startswith("direct_http:")
        if in_direct_http and not replaced and key.startswith("task:"):
            newline = "\n" if line.endswith("\n") else ""
            rendered.append(f"{' ' * indent}task: {quoted}{newline}")
            replaced = True
            continue
        rendered.append(line)
    if not replaced:
        raise ValueError("direct_http task key was not found in the ROS inference config")
    return "".join(rendered)


def vla_origin_launch(task: str) -> _VlaOriginLaunch:
    """Point the ROS inference script at a copy of the yaml with this task."""
    if not _ORIGIN_SCRIPT.is_file():
        raise OSError(f"ROS inference script is missing: {_ORIGIN_SCRIPT}")
    if not _ORIGIN_CONFIG.is_file():
        raise OSError(f"ROS inference config is missing: {_ORIGIN_CONFIG}")
    destination = Path("/tmp") / f"vla-origin-{os.getpid()}.yaml"
    destination.write_text(
        overlay_direct_http_task(_ORIGIN_CONFIG.read_text(encoding="utf-8"), task),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["ROS2_NERO_EVAL_CONFIG"] = str(destination)
    env["ROS2_COORDINATOR_STDIN"] = "/dev/stdin"
    return _VlaOriginLaunch(
        command=["bash", str(_ORIGIN_SCRIPT)],
        env=env,
        cwd=str(_RLTOKEN),
    )


def _policy_namespace(base: argparse.Namespace, policy: str) -> argparse.Namespace:
    """Copy of ``base`` with ``policy`` swapped, for on-demand suffix builds."""
    patched = argparse.Namespace(**vars(base))
    patched.policy = policy
    return patched


_GLM_MODEL = "glm-5.3"
_GLM_BASE_URL = "https://api.z.ai/api/anthropic/v1"
_GLM_KEY_ENV = "ANTHROPIC_AUTH_TOKEN"


def _evo_endpoint(namespace: argparse.Namespace) -> tuple[str, str, str, str]:
    """Return model, base URL, key env, and wire for the selected evo coder."""
    coder = getattr(namespace, "evo_coder", "astra")
    if coder == "glm":
        return _GLM_MODEL, _GLM_BASE_URL, _GLM_KEY_ENV, "anthropic"
    return namespace.model, namespace.base_url, namespace.api_key_env, "responses"


def build_command(namespace: argparse.Namespace) -> tuple[list[str], list[str]]:
    """Split the spawned eval argv into the halves around the instruction."""
    # Explicit `run --instruction` rather than the positional sugar: the sugar
    # only fires for instructions with interior whitespace, so a space-free
    # CJK instruction would be parsed as a subcommand.
    prefix = ["uv", "run", "--no-sync", "inspect-robots", "run", "--instruction"]
    common = [
        "--store-frames",
        "--grader",
        "none",
        "--embodiment",
        "nero",
        "-E",
        "operator_reset_confirm=False",
    ]
    if namespace.policy == "evo":
        model_name, base_url, key_env, wire = _evo_endpoint(namespace)
        suffix = [
            *common,
            "--policy",
            "evo",
            "-P",
            f"model={model_name}",
            "-P",
            f"wire={wire}",
            "-P",
            f"base_url={base_url}",
            "-P",
            f"api_key_env={key_env}",
            "-P",
            f"mode={namespace.evo_mode}",
            "-P",
            f"workspace={namespace.evo_workspace}",
            "--max-steps",
            "0",
        ]
    elif namespace.policy in ("vla", "vla-replay", "umi-replay"):
        # Pure VLA baseline, paced like ros2_nero_inference.yaml: the base
        # checkpoints emit 30 Hz actions (action_hz), and schedule=tracked
        # asks for the next chunk at 5 Hz without stopping the arm, then
        # blends it after the 300 ms preview across the 300 ms transition.
        # Joint step matches the yaml (1.0 rad); agent keeps the 0.1 default.
        # 5400 steps is 180 s at 30 Hz, the same wall budget the 10 Hz
        # open-loop runs used.
        suffix = [
            *common,
            "--policy",
            "vla",
            "-P",
            "prompt=__INSTRUCTION__",
            "-P",
            "schedule=tracked",
            "-P",
            "control_hz=30",
            "-E",
            "control_hz=200",
            "-E",
            "max_joint_step_rad=1.0",
            "-E",
            f"trace_dir={_TRACE_DIR}",
            "-P",
            f"trace_dir={_TRACE_DIR}",
            "--max-steps",
            "5400",
        ]
        if namespace.policy == "vla-replay":
            # Same embodiment, 30 Hz tracked playback, and 200 Hz IK as vla.
            # Only the action source changes: the UMI replay service.
            suffix.extend(
                [
                    "-P",
                    "replay=true",
                    "-P",
                    "base_url=http://127.0.0.1:10054",
                ]
            )
    elif namespace.policy == "harness":
        # The Kingsoft gateway exposes ChatGPT-6 Astra as --model (mog-10-a).
        # Same 30 Hz clock as the base checkpoints. move_to splits a motion by
        # per-tick distance, so 10 Hz played those ticks three times slower
        # than the embodiment was tuned for. The delegated chunk is the model's
        # 30 Hz waypoints. 5400 steps is 180 s.
        suffix = [
            *common,
            "--policy",
            namespace.policy,
            "-P",
            f"model={namespace.model}",
            "-P",
            "wire=responses",
            "-P",
            f"base_url={namespace.base_url}",
            "-P",
            f"api_key_env={namespace.api_key_env}",
            "-P",
            "control_hz=30",
            "-E",
            "control_hz=30",
            "--max-steps",
            "5400",
        ]
    else:
        # Same gateway constraint the harness branch carries: kspmas rejects
        # function tools + reasoning_effort on /chat/completions, so agent
        # runs also speak /v1/responses.
        suffix = [
            *common,
            "--policy",
            "agent",
            "-P",
            f"model={namespace.model}",
            "-P",
            "wire=responses",
            "-P",
            f"base_url={namespace.base_url}",
            "-P",
            f"api_key_env={namespace.api_key_env}",
            "-P",
            f"max_speed_frac={namespace.max_speed_frac:g}",
            "--max-steps",
            "0",
        ]
    return prefix, suffix


def _parse_camera_overrides(entries: Sequence[str]) -> dict[str, str]:
    """Parse repeatable ``--camera NAME=DEVICE`` arguments into overrides."""
    overrides: dict[str, str] = {}
    for entry in entries:
        name, separator, device = entry.partition("=")
        name, device = name.strip(), device.strip()
        if not separator or not name or not device:
            raise ValueError(f"--camera expects NAME=DEVICE, got {entry!r}")
        if name not in CAMERA_DEFAULTS:
            raise ValueError(f"unknown camera {name!r}; valid names: {sorted(CAMERA_DEFAULTS)}")
        overrides[name] = device
    return overrides


def build_parser() -> argparse.ArgumentParser:
    """Build the console CLI parser."""
    parser = argparse.ArgumentParser(
        description="Nero mission console: live cameras and run control on localhost"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="listen port (default: 8400)"
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="bind address (default: 127.0.0.1; a non-loopback host prints a loud warning)",
    )
    parser.add_argument(
        "--camera",
        action="append",
        metavar="NAME=DEVICE",
        help="override one camera's V4L2 device (repeatable; names as in CAMERA_DEFAULTS)",
    )
    parser.add_argument(
        "--max-speed-frac",
        type=float,
        default=DEFAULT_MAX_SPEED_FRAC,
        help="agent policy speed fraction (default: 0.05)",
    )
    parser.add_argument(
        "--policy",
        choices=["agent", "harness", "vla", "vla-replay", "vla-origin", "evo"],
        default="agent",
        help="policy the spawned runs use (agent needs the LLM flags below)",
    )
    parser.add_argument(
        "--evo-mode",
        choices=["playground", "test"],
        default="playground",
        help="evo workspace mode (default: playground)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="model id for the agent policy (default: gpt-5.6-luna)",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible base URL (default: the experientiallabs proxy)",
    )
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help="environment variable holding the API key (default: EXPLABS_API_KEY)",
    )
    parser.add_argument(
        "--history-url",
        default=DEFAULT_HISTORY_URL,
        help="view --serve URL linked from the page (default: http://127.0.0.1:8300/)",
    )
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help="directory for captured run output logs (default: /tmp)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the mission console until interrupted; return a process exit code."""
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.max_speed_frac) or args.max_speed_frac <= 0:
        print("error: --max-speed-frac must be a positive finite number", file=sys.stderr)
        return 2
    try:
        overrides = _parse_camera_overrides(args.camera or [])
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.host not in _LOOPBACK_HOSTS:
        print(
            "\n".join(
                [
                    "!" * 76,
                    f"!! WARNING: binding the mission console to {args.host}, not loopback.",
                    "!! Anyone who can reach this address can start the robot's arms.",
                    "!! Keep the default 127.0.0.1 unless you control the network in front.",
                    "!" * 76,
                ]
            ),
            file=sys.stderr,
        )

    pool = CameraPool(overrides)
    args.evo_workspace = ""
    args.evo_coder = "astra"
    prefix, suffix = build_command(args)
    exporter = LerobotExporter(Path(__file__).with_name("export_lerobot.py"), Path(args.log_dir))
    runs = RunManager(prefix, suffix, Path(args.log_dir), exporter)
    armtools = ArmToolRunner(Path(__file__).with_name("arm_tools.py"), Path(args.log_dir))
    page_html = render_page(pool.names, args.history_url, _trained_prompt())
    chest_dir = _TRACE_DIR / "video_intake"
    # The take belongs to video intake. This script only supplies chest frames.
    # The plugin is a workspace member that is not an embodiment dependency,
    # so the console adds its source tree when the env has not installed it.
    intake_src = Path(__file__).resolve().parents[2] / "inspect-robots-intake" / "src"
    if str(intake_src) not in sys.path:
        sys.path.insert(0, str(intake_src))
    chest_recorder = None
    speech_transcriber = None
    try:
        from inspect_robots_intake.capture import VideoTakeRecorder

        chest_recorder = VideoTakeRecorder(chest_dir)
    except ImportError as exc:
        print(f"[console] chest recording unavailable: {exc}")
    try:
        from inspect_robots_intake.asr_transcriber import QwenAsrTranscriber

        speech_transcriber = QwenAsrTranscriber(
            Path("/home/dell/ckpts/qwen3-asr/.venv/bin/python"),
            Path("/home/dell/ckpts/Qwen3-ASR-1.7B"),
            intake_src / "inspect_robots_intake" / "qwen_asr_worker.py",
            log_path=_TRACE_DIR / "speech_intake" / "qwen-asr-worker.log",
        )
    except ImportError as exc:
        print(f"[console] speech intake unavailable: {exc}")
    server = MissionServer(
        (args.host, args.port),
        pool,
        runs,
        armtools,
        exporter,
        page_html,
        args.policy,
        lambda policy: build_command(_policy_namespace(args, policy))[1],
    )
    server.launch_args = args
    server.base_url = args.base_url
    server.api_key_env = args.api_key_env
    server.model = args.model
    server.chest = chest_recorder
    server.speech = speech_transcriber
    print(f"[console] mission console on http://{args.host}:{args.port}/")
    if chest_recorder is not None:
        print(f"[console] chest takes for video intake: {chest_dir}")
        newest = max(
            chest_dir.glob("chest_*.mp4"),
            key=lambda path: path.stat().st_mtime,
            default=None,
        )
        if newest is not None:
            server.last_video = str(newest)
            print(f"[console] last video prompt: {newest.name}")
    print(f"[console] cameras: {', '.join(pool.describe())}")
    print(f"[console] start command: {' '.join([*prefix, '<instruction>', *suffix])}")
    print("[console] control is pty-write only (/stop); runs use --grader none")
    print(f"[console] runs inherit this shell's environment; {args.api_key_env} must be set here")
    print("[console] POST routes verify Host/Origin against the bound address")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[console] interrupt; shutting down")
    finally:
        if server.speech is not None:
            server.speech.close()
        if server.chest is not None and server.chest.status()["recording"]:
            from inspect_robots_intake.contract import IntakeError

            try:
                server.chest.stop()
            except IntakeError as exc:
                print(f"[console] chest recording did not finish: {exc}")
        server.server_close()
        runs.shutdown()
        pool.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
