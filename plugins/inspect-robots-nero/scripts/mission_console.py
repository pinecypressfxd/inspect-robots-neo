"""Mission console: localhost web page with live cameras and run control.

Serves one self-contained operator page (three camera tiles, status bar,
two-step start, stop, verdicts) plus a JSON API over stdlib ``http.server``.
Start spawns the attended eval on a pty:

    uv run --no-sync inspect-robots "<instruction>" --policy agent \\
        -P model=gpt-6-astra -P base_url=<url> -P api_key_env=EXPLABS_API_KEY \\
        -P max_speed_frac=<frac> --embodiment nero -E operator_reset_confirm=False

The pty is the ONLY control channel: stop writes ``/stop``, verdicts write
``/y`` ``/n`` ``/p`` (or ``/skip``); the child process is never signalled.
Run from the repo root so ``uv run`` resolves the workspace and ``logs/`` is
the shared eval-log directory; the child inherits this process's environment,
so ``EXPLABS_API_KEY`` comes from the operator's shell that launched the
console.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import pty
import re
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, cast

from inspect_robots_nero._camera import D405Camera
from inspect_robots_nero._config import CAMERA_DEFAULTS

if TYPE_CHECKING:
    import numpy as np

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
VERDICT_CHOICES = frozenset({"y", "n", "p", "skip"})
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_LOG_LINE_RE = re.compile(r"^log:\s*(\S.*)$")
_CAM_PATH_RE = re.compile(r"^/cam/(?P<name>[^/]+)\.mjpg$")


def _encode_jpeg(frame: np.ndarray) -> bytes:
    """Encode one RGB camera frame as JPEG bytes; cv2 imports only in here."""
    import cv2

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise RuntimeError("cv2.imencode failed to produce a JPEG frame")
    return encoded.tobytes()


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


def _newest_live_log(min_mtime: float) -> Path | None:
    """Newest ``logs/*.live.json`` modified at/after ``min_mtime``; else None.

    The live sink writes the run's snapshot continuously, so the newest file
    that is not older than the run's spawn time belongs to that run; a stale
    snapshot from an earlier run is never mistaken for the current one.
    """
    directory = Path("logs")
    if not directory.is_dir():
        return None
    newest: tuple[float, Path] | None = None
    for candidate in directory.glob("*.live.json"):
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            continue
        if mtime < min_mtime:
            continue
        if newest is None or mtime > newest[0]:
            newest = (mtime, candidate)
    return None if newest is None else newest[1]


class CameraPool:
    """One lazily started ``D405Camera`` singleton per configured camera.

    V4L2 mmap streaming is exclusive per device node: while the console
    streams, nothing else can open the cameras, and once a run holds them the
    console's own tiles get a plain 503. ``stop`` releases every device so a
    spawned eval can claim them; the first MJPEG request afterwards starts a
    camera again.
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

    @property
    def names(self) -> tuple[str, ...]:
        """Camera names in tile order (the CAMERA_DEFAULTS declaration order)."""
        return tuple(self._cameras)

    def describe(self) -> list[str]:
        """One ``name=device`` string per camera, for the startup banner."""
        return [f"{name}={camera.device}" for name, camera in self._cameras.items()]

    def start(self, name: str) -> D405Camera:
        """Return the camera for ``name``, starting it on first request.

        Raises ``KeyError`` for an unknown name and ``RuntimeError`` when the
        device cannot be opened (busy, unplugged, cabling); the MJPEG route
        turns both into plain error responses.
        """
        camera = self._cameras[name]
        with self._lock:
            camera.start()
        return camera

    def stop(self) -> None:
        """Stop every camera and release its device; idempotent, best effort."""
        with self._lock:
            for camera in self._cameras.values():
                try:
                    camera.stop()
                except Exception as exc:  # hardware teardown must never abort the rest
                    print(f"[console] stopping camera {camera.name!r} failed: {exc}")


class _Run:
    """One spawned run's state; every field is guarded by RunManager's lock."""

    def __init__(
        self,
        instruction: str,
        process: subprocess.Popen[bytes],
        master_fd: int,
        log_file: TextIO,
        started_at: float,
    ) -> None:
        self.instruction = instruction
        self.process = process
        self.master_fd = master_fd
        self.log_file = log_file
        self.started_at = started_at
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

    def start(self, instruction: str, prepare: Callable[[], None] | None = None) -> str | None:
        """Spawn one run for ``instruction``; return an error message or None.

        ``prepare`` runs after the active-run check but before the spawn, so
        callers can release shared hardware (the cameras) exactly when a run
        is about to claim it.
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
                return f"could not open the run log under {self._log_dir}: {exc}"
            try:
                master_fd, slave_fd = pty.openpty()
            except OSError as exc:
                log_file.close()
                return f"could not allocate the run terminal: {exc}"
            command = [*self._prefix, instruction, *self._suffix]
            try:
                process = subprocess.Popen(
                    command,
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    start_new_session=True,
                )
            except OSError as exc:
                log_file.close()
                os.close(master_fd)
                os.close(slave_fd)
                return f"could not spawn {' '.join(command)}: {exc}"
            os.close(slave_fd)
            run = _Run(
                instruction=instruction,
                process=process,
                master_fd=master_fd,
                log_file=log_file,
                started_at=time.time(),
            )
            self._run = run
        threading.Thread(target=self._drain, args=(run,), daemon=True, name="run-drain").start()
        print(f"[console] started: {instruction!r}")
        print(f"[console] command: {' '.join(command)}")
        return None

    def _drain(self, run: _Run) -> None:
        """Drain the pty to the ring tail and log file, then reap the child."""
        buffer = ""
        while True:
            try:
                chunk = os.read(run.master_fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            run.log_file.write(text)
            buffer += text
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                with self._lock:
                    run.tail.append(line)
                    detected = _detect_log_path(line)
                    if detected is not None:
                        run.log_path = detected
                        print(f"[console] run log: {detected}")
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
        nothing else. Every attempt (success or failure) is echoed to the
        console's stdout.
        """
        with self._lock:
            run = self._run
            if run is None:
                return "no run has been started"
            if run.exit_code is not None:
                return "the run has already ended"
            try:
                os.write(run.master_fd, f"{line}\n".encode())
            except OSError as exc:
                print(f"[console] pty write failed ({line!r}): {exc}")
                return f"could not write to the run's terminal: {exc}"
        print(f"[console] pty -> {line}")
        return None

    def status(self) -> dict[str, Any]:
        """Snapshot for ``/api/status``: state, instruction, log, exit code."""
        with self._lock:
            run = self._run
            if run is None:
                return {"state": "idle", "instruction": None, "log": None, "exit_code": None}
            state = "running" if run.exit_code is None else "ended"
            snapshot = {
                "state": state,
                "instruction": run.instruction,
                "log": run.log_path,
                "exit_code": run.exit_code,
            }
            started_at = run.started_at
        if snapshot["log"] is None and state == "running":
            live = _newest_live_log(started_at)
            if live is not None:
                snapshot["log"] = str(live)
        return snapshot

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
            try:
                os.write(run.master_fd, b"/stop\n")
                print("[console] pty -> /stop (console shutdown)")
            except OSError as exc:
                print(f"[console] shutdown /stop write failed: {exc}; the run will see EOF")


class MissionRequestHandler(BaseHTTPRequestHandler):
    """Serve the console page, the camera MJPEG streams, and the run API."""

    server: MissionServer

    def do_GET(self) -> None:
        """Route GET: page, status JSON, and camera streams."""
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send_bytes(200, "text/html; charset=utf-8", self.server.page_html)
        elif path == "/api/status":
            self._send_json(200, self.server.runs.status())
        else:
            match = _CAM_PATH_RE.fullmatch(path)
            if match is not None:
                self._serve_camera(match.group("name"))
            else:
                self._send_plain(404, "not found\n")

    def do_POST(self) -> None:
        """Route POST: start, stop, and verdict."""
        path = self.path.split("?", 1)[0]
        if path == "/api/start":
            self._api_start()
        elif path == "/api/stop":
            self._api_stop()
        elif path == "/api/verdict":
            self._api_verdict()
        else:
            self._send_plain(404, "not found\n")

    def _api_start(self) -> None:
        """Spawn a run from ``{"instruction": ...}``; camera-free by then."""
        body = self._read_json_body()
        if body is None:
            return
        instruction = body.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            self._send_json(400, {"error": "instruction must be a non-empty string"})
            return
        error = self.server.runs.start(instruction.strip(), prepare=self.server.pool.stop)
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

    def _serve_camera(self, name: str) -> None:
        """Stream one camera as multipart JPEG until the client or camera goes.

        The first frame is grabbed before any header is sent so an unusable
        camera answers with a plain 503 instead of a broken stream; failures
        after that (client disconnect, stale or unplugged device, encode
        error) just end this one stream and leave the server running.
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
        self.send_response(200)
        self.send_header("Content-Type", 'multipart/x-mixed-replace; boundary="frame"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        period = 1.0 / STREAM_FPS
        deadline = time.monotonic() + period
        try:
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

    def _send_bytes(self, code: int, content_type: str, payload: bytes) -> None:
        """Send one complete response with an explicit length."""
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        """Log requests to stderr, hiding the once-a-second status poll."""
        if self.path == "/api/status":
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
        self.page_html = page_html


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
  .controls { display: flex; gap: 8px; padding: 8px 16px; align-items: center; flex-wrap: wrap; }
  #instruction { flex: 1; min-width: 260px; padding: 8px; background: #171d24; color: inherit;
                 border: 1px solid #2a3340; border-radius: 4px; }
  button { padding: 8px 14px; font-size: 14px; border: 1px solid #2a3340; border-radius: 4px;
           background: #202a35; color: inherit; cursor: pointer; }
  button.armed { background: #7a2e2e; border-color: #b04a4a; font-weight: bold; }
  button:disabled { opacity: 0.5; cursor: default; }
  #verdicts { display: flex; gap: 6px; }
  #feed { margin: 8px 16px 24px; padding: 10px; border: 1px dashed #2a3340; border-radius: 4px;
          min-height: 120px; font-size: 13px; color: #9aa7b4; }
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
<section id="feed">
  <div>Reasoning feed: live run reasoning and action history appear here.</div>
</section>
<script>
const startBtn = document.getElementById("start");
const stopBtn = document.getElementById("stop");
const input = document.getElementById("instruction");
const statusBox = document.getElementById("status");
const verdicts = document.getElementById("verdicts");
let armed = false;
let lastState = "";

function disarm() {
  armed = false;
  startBtn.textContent = "Start";
  startBtn.classList.remove("armed");
}

function setTiles(live) {
  for (const img of document.querySelectorAll(".cams img")) {
    img.src = live ? img.getAttribute("data-src") : "";
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
  statusBox.textContent = parts.join("   |   ");
  verdicts.hidden = state === "idle";
  stopBtn.disabled = state !== "running";
  startBtn.disabled = state === "running";
  if (state !== lastState) {
    // Tiles hold the cameras exclusively; the run needs them, so tiles go
    // dark while a run is live and reconnect once it ends.
    setTiles(state !== "running");
    disarm();
    lastState = state;
  }
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
  } catch (err) {
    statusBox.textContent = "connection lost: " + err;
  }
}
poll();
setInterval(poll, 1000);
</script>
</body>
</html>
"""


def render_page(camera_names: Sequence[str], history_url: str) -> bytes:
    """Render the console page; every interpolation is escaped exactly once."""
    tiles = "".join(
        '<figure><img data-src="/cam/'
        + html.escape(name, quote=True)
        + '.mjpg" src="/cam/'
        + html.escape(name, quote=True)
        + '.mjpg" alt="'
        + html.escape(name, quote=True)
        + ' live stream"><figcaption>'
        + html.escape(name)
        + "</figcaption></figure>"
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
