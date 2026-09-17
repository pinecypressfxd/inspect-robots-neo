"""Mission console: localhost web page with live cameras and run control.

Serves one self-contained operator page (three camera tiles, status bar,
two-step start, stop, verdicts, live reasoning feed) plus a JSON API over
stdlib ``http.server``. Start spawns the attended eval on a pty:

    uv run --no-sync inspect-robots "<instruction>" --policy agent \\
        -P model=gpt-6-astra -P base_url=<url> -P api_key_env=EXPLABS_API_KEY \\
        -P max_speed_frac=<frac> --embodiment nero -E operator_reset_confirm=False

The pty is the ONLY control channel: stop writes ``/stop``, verdicts write
``/y`` ``/n`` ``/p`` (or ``/skip``); the child process is never signalled.
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
DEFAULT_MODEL = "gpt-6-astra"
DEFAULT_BASE_URL = "https://api.experientiallabs.ai/v1"
DEFAULT_API_KEY_ENV = "EXPLABS_API_KEY"
DEFAULT_MAX_SPEED_FRAC = 0.05
DEFAULT_HISTORY_URL = "http://127.0.0.1:8300/"
DEFAULT_LOG_DIR = "/tmp"
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
VERDICT_CHOICES = frozenset({"y", "n", "p", "skip"})
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
#: Binding these means "every interface", so the Host allowlist accepts any name.
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::", ""})
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_LOG_LINE_RE = re.compile(r"^log:\s*(\S.*)$")
_CAM_PATH_RE = re.compile(r"^/cam/(?P<name>[^/]+)\.mjpg$")
_FRAME_ROUTE_RE = re.compile(r"^/frame/(?P<name>[^/]+)\.jpg$")
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

    def stop(self) -> None:
        """Stop every camera and release its device; idempotent, best effort."""
        with self._lock:
            self._was_started = frozenset(self._started)
            self._started.clear()
            for camera in self._cameras.values():
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


class RunManager:
    """Own the single active run: spawn it on a pty, drain it, report status.

    The pty is the only control channel (``/stop`` and verdict lines written
    to the master fd, each one echoed to the console's stdout); no signal is
    ever sent to the child. It ends by itself, and sees terminal EOF when the
    console exits.
    """

    def __init__(self, prefix: list[str], suffix: list[str], log_dir: Path) -> None:
        """Store the argv halves (instruction is appended between them)."""
        self._prefix = prefix
        self._suffix = suffix
        self._log_dir = log_dir
        self._lock = threading.Lock()
        self._run: _Run | None = None
        self._spawn_count = 0
        self._last_error: str | None = None

    def start(
        self,
        instruction: str,
        prepare: Callable[[], None] | None = None,
        rollback: Callable[[], None] | None = None,
    ) -> str | None:
        """Spawn one run for ``instruction``; return an error message or None.

        ``prepare`` runs after the active-run check but before the spawn, so
        callers can release shared hardware (the cameras) exactly when a run
        is about to claim it. ``rollback`` runs when the spawn itself fails,
        so callers can take the hardware back (restart the cameras) and the
        page can recover instead of sitting idle with dead tiles.
        """
        with self._lock:
            if self._run is not None and self._run.exit_code is None:
                return "a run is already active; stop it or wait for it to end"
            if prepare is not None:
                prepare()
            self._spawn_count += 1
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
            command = [*self._prefix, instruction, *self._suffix]
            try:
                # ValueError joins OSError: an embedded NUL byte in the
                # instruction (or any argument) fails the exec, not the console.
                process = subprocess.Popen(
                    command,
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    start_new_session=True,
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
                with self._lock:
                    run.tail.append(line)
                    detected = _detect_log_path(line)
                    if detected is not None:
                        run.log_path = detected
                        print(f"[console] run log: {detected}")
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
        """Snapshot for ``/api/status``: state, instruction, log, exit, error."""
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
            state = "running" if run.exit_code is None else "ended"
            snapshot = {
                "state": state,
                "instruction": run.instruction,
                "log": run.log_path,
                "exit_code": run.exit_code,
                "error": error,
            }
            started_at = run.started_at
        if snapshot["log"] is None and state == "running":
            live = _newest_live_log(started_at)
            if live is not None:
                snapshot["log"] = str(live)
        return snapshot

    def current(self) -> dict[str, Any] | None:
        """Snapshot of the last run for the feed and frame routes; None if never."""
        with self._lock:
            run = self._run
            if run is None:
                return None
            return {
                "state": "running" if run.exit_code is None else "ended",
                "instruction": run.instruction,
                "started_at": run.started_at,
                "run_id": run.run_id,
                "log": run.log_path,
            }

    def shutdown(self) -> None:
        """Best-effort graceful end for an active run: one ``/stop`` line.

        Runs when the console exits. Never a signal: the write lets the eval
        end its episode through the normal verdict flow; afterwards the
        closing terminal gives it EOF.
        """
        with self._lock:
            run = self._run
            if run is None or run.exit_code is not None:
                return
            error = _write_pty(run.master_fd, b"/stop\n")
        if error is None:
            print("[console] pty -> /stop (console shutdown)")
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
        self._max_seq = 0

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
            return {
                "seq": self._max_seq,
                "live": live,
                "messages": messages,
                "actions": actions,
                "wire": wire,
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
            self._send_json(200, self.server.runs.status())
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
            self._send_plain(404, "not found\n")

    def do_POST(self) -> None:
        """Route POST: start, stop, and verdict, behind the Host/Origin gate."""
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
        elif path == "/api/verdict":
            self._api_verdict()
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
        if not isinstance(instruction, str) or not instruction.strip():
            self._send_json(400, {"error": "instruction must be a non-empty string"})
            return
        error = self.server.runs.start(
            instruction.strip(),
            prepare=self.server.pool.stop,
            rollback=self.server.pool.restart,
        )
        if error is not None:
            self._send_json(400, {"error": error})
            return
        self._send_json(200, {"ok": True})

    def _api_stop(self) -> None:
        """Graceful stop: the ``/stop`` console line on the run's terminal."""
        body = self._read_json_body()
        if body is None:
            return
        error = self.server.runs.write_line("/stop")
        if error is None:
            self._send_json(200, {"ok": True})
        else:
            self._send_json(400, {"error": error})

    def _api_verdict(self) -> None:
        """Write ``/<choice>`` for ``{"choice": "y"|"n"|"p"|"skip"}``."""
        body = self._read_json_body()
        if body is None:
            return
        choice = body.get("choice")
        if not isinstance(choice, str) or choice.strip().lower() not in VERDICT_CHOICES:
            self._send_json(400, {"error": "choice must be one of: y, n, p, skip"})
            return
        line = f"/{choice.strip().lower()}"
        error = self.server.runs.write_line(line)
        if error is None:
            self._send_json(200, {"ok": True})
        else:
            self._send_json(400, {"error": error})

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
        if run is not None and run.get("state") == "running":
            frame, error = self._stored_run_frame(run, name)
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
        self, address: tuple[str, int], pool: CameraPool, runs: RunManager, page_html: bytes
    ) -> None:
        """Bind ``address``; requests are served by MissionRequestHandler."""
        super().__init__(address, MissionRequestHandler)
        self.pool = pool
        self.runs = runs
        self.feed = RunFeed()
        self.page_html = page_html
        host = address[0].lower()
        loopback = _LOOPBACK_HOSTS if host in _LOOPBACK_HOSTS else frozenset()
        self.allowed_hosts = frozenset({host} | loopback)
        self.wildcard_host = host in _WILDCARD_HOSTS
        self.bound_port = address[1]


_PAGE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Nero mission console</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: system-ui, sans-serif; margin: 0; background: #101418; color: #e8e8e8; }
  header { display: flex; justify-content: space-between; align-items: baseline;
           padding: 10px 16px; background: #171d24; border-bottom: 1px solid #2a3340; }
  h1 { font-size: 18px; margin: 0; }
  header a { color: #7fb4ff; }
  #status { padding: 8px 16px; background: #0b0e12; border-bottom: 1px solid #2a3340;
            font-family: ui-monospace, monospace; font-size: 13px; white-space: pre-wrap; }
  .cams { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; padding: 8px; }
  .cams figure { margin: 0; background: #000; border: 1px solid #2a3340; }
  .cams img { width: 100%; aspect-ratio: 4 / 3; object-fit: contain; display: block;
              background: #000; }
  .cams figcaption { font-size: 12px; padding: 3px 6px; color: #9aa7b4; background: #171d24; }
  .cams figcaption .mode { float: right; color: #5f7186; }
  .controls { display: flex; gap: 8px; padding: 8px 16px; align-items: center; flex-wrap: wrap; }
  #instruction { flex: 1; min-width: 260px; padding: 8px; background: #171d24; color: inherit;
                 border: 1px solid #2a3340; border-radius: 4px; }
  button { padding: 8px 14px; font-size: 14px; border: 1px solid #2a3340; border-radius: 4px;
           background: #202a35; color: inherit; cursor: pointer; }
  button.armed { background: #7a2e2e; border-color: #b04a4a; font-weight: bold; }
  button:disabled { opacity: 0.5; cursor: default; }
  #verdicts { display: flex; gap: 6px; }
  #feedpanel { margin: 8px 16px 24px; display: grid; gap: 8px; }
  .feedgrid { display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 8px; }
  .panel { border: 1px solid #2a3340; border-radius: 4px; background: #141a21; }
  .panel h2 { font-size: 11px; margin: 0; padding: 5px 8px; color: #9aa7b4;
              text-transform: uppercase; letter-spacing: .07em; background: #171d24;
              border-bottom: 1px solid #2a3340; }
  .panelbody { padding: 8px; font-size: 13px; color: #c8d2dd; white-space: pre-wrap;
               overflow-wrap: anywhere; max-height: 170px; overflow-y: auto; }
  #feed { padding: 8px; font-size: 13px; color: #9aa7b4; max-height: 340px; overflow-y: auto; }
  .msg { margin: 0 0 8px; padding-left: 8px; border-left: 3px solid #2a3340; }
  .msg-user { border-left-color: #3d6aa5; }
  .msg-assistant { border-left-color: #7a55b5; }
  .msg-system { border-left-color: #5f7186; }
  .msg-role { display: inline-block; font-size: 11px; text-transform: uppercase;
              letter-spacing: .07em; color: #9aa7b4; margin-right: 8px; }
  .msg-text { white-space: pre-wrap; overflow-wrap: anywhere; color: #c8d2dd; }
  .msg-tool { font-family: ui-monospace, monospace; font-size: 12px; color: #bd9bed;
              white-space: pre-wrap; overflow-wrap: anywhere; }
  @media (max-width: 900px) { .feedgrid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>Nero mission console</h1>
  <a href="__HISTORY_URL__" target="_blank" rel="noopener">Run history</a>
</header>
<div id="status">connecting...</div>
<div class="cams">
"""

_PAGE_TAIL = """</div>
<div class="controls">
  <input id="instruction" placeholder="instruction for the run" autocomplete="off">
  <button id="start">Start</button>
  <button id="stop">Stop</button>
  <span id="verdicts" hidden>
    <button data-choice="y">verdict y</button>
    <button data-choice="n">verdict n</button>
    <button data-choice="p">verdict p</button>
  </span>
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
  <div class="panel"><h2>reasoning feed (newest pinned)</h2><div id="feed">no data yet</div></div>
</section>
<script>
const startBtn = document.getElementById("start");
const stopBtn = document.getElementById("stop");
const input = document.getElementById("instruction");
const statusBox = document.getElementById("status");
const verdicts = document.getElementById("verdicts");
const feedEl = document.getElementById("feed");
const liveBody = document.getElementById("livebody");
const wireBody = document.getElementById("wirebody");
const actionBody = document.getElementById("actionbody");
let armed = false;
let lastState = "";
let lastError = "";
let tileMode = "live";
let since = 0;
const seenSeq = new Set();

function disarm() {
  armed = false;
  startBtn.textContent = "Start";
  startBtn.classList.remove("armed");
}

function frameUrl(name) {
  return "/frame/" + encodeURIComponent(name) + ".jpg?x=" + Date.now();
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
      img.src = frameUrl(img.dataset.cam);
      if (label) { label.textContent = "run frame (1s)"; }
    } else {
      img.src = "";
      if (label) { label.textContent = ""; }
    }
  }
}

function refreshFrameTiles() {
  for (const img of document.querySelectorAll(".cams img")) {
    img.src = frameUrl(img.dataset.cam);
  }
}

function render(status) {
  const state = status.state || "idle";
  const parts = ["state: " + state];
  if (status.instruction) { parts.push("instruction: " + status.instruction); }
  if (status.log) { parts.push("log: " + status.log); }
  if (status.exit_code !== null && status.exit_code !== undefined) {
    parts.push("exit: " + status.exit_code);
  }
  if (status.error) { parts.push("error: " + status.error); }
  statusBox.textContent = parts.join("   |   ");
  verdicts.hidden = state === "idle";
  stopBtn.disabled = state !== "running";
  startBtn.disabled = state === "running";
  if (state === "running" && tileMode !== "frames") {
    // The run owns the cameras; switch the tiles to the run's stored frames.
    setTileMode("frames");
  } else if (state !== "running" && tileMode === "frames") {
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

function renderFeed(feed) {
  renderLive(feed.live);
  renderWire(feed.wire);
  renderActions(feed.actions);
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
  try {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {})
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) { statusBox.textContent = "error: " + (data.error || response.status); }
  } catch (err) {
    statusBox.textContent = "error: " + err;
  }
  poll();
  pollFeed();
}

startBtn.addEventListener("click", () => {
  if (!armed) {
    armed = true;
    startBtn.textContent = "Confirm start — arms will move";
    startBtn.classList.add("armed");
    return;
  }
  disarm();
  post("/api/start", { instruction: input.value });
});
input.addEventListener("input", disarm);
stopBtn.addEventListener("click", () => post("/api/stop", {}));
for (const btn of verdicts.querySelectorAll("button")) {
  btn.addEventListener("click", () => post("/api/verdict", { choice: btn.dataset.choice }));
}

async function poll() {
  try {
    const response = await fetch("/api/status");
    render(await response.json());
    if (tileMode === "frames") { refreshFrameTiles(); }
  } catch (err) {
    statusBox.textContent = "connection lost: " + err;
  }
}

async function pollFeed() {
  try {
    const response = await fetch("/api/feed?since=" + since);
    const feed = await response.json();
    renderFeed(feed);
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


def render_page(camera_names: Sequence[str], history_url: str) -> bytes:
    """Render the console page; every interpolation is escaped exactly once."""
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
    return (head + tiles + _PAGE_TAIL).encode("utf-8")


def build_command(namespace: argparse.Namespace) -> tuple[list[str], list[str]]:
    """Split the spawned eval argv into the halves around the instruction."""
    prefix = ["uv", "run", "--no-sync", "inspect-robots"]
    suffix = [
        "--policy",
        "agent",
        "-P",
        f"model={namespace.model}",
        "-P",
        f"base_url={namespace.base_url}",
        "-P",
        f"api_key_env={namespace.api_key_env}",
        "-P",
        f"max_speed_frac={namespace.max_speed_frac:g}",
        "--embodiment",
        "nero",
        "-E",
        "operator_reset_confirm=False",
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
        "--model",
        default=DEFAULT_MODEL,
        help="model id for the agent policy (default: gpt-6-astra)",
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
    prefix, suffix = build_command(args)
    runs = RunManager(prefix, suffix, Path(args.log_dir))
    page_html = render_page(pool.names, args.history_url)
    server = MissionServer((args.host, args.port), pool, runs, page_html)
    print(f"[console] mission console on http://{args.host}:{args.port}/")
    print(f"[console] cameras: {', '.join(pool.describe())}")
    print(f"[console] start command: {' '.join([*prefix, '<instruction>', *suffix])}")
    print("[console] control is pty-write only (/stop, /y, /n, /p); the child is never signalled")
    print(f"[console] runs inherit this shell's environment; {args.api_key_env} must be set here")
    print("[console] POST routes verify Host/Origin against the bound address")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[console] interrupt; shutting down")
    finally:
        server.server_close()
        runs.shutdown()
        pool.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
