"""Serve the original robot ITL reference report over inspect traces."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from inspect_robots_nero._report_payload import (
    env_loop_report,
    ik_metrics,
    joint_compare,
    pose_compare,
)
from inspect_robots_nero._trace import list_episodes

_STATIC = Path(__file__).with_name("static")


class ReportViewer:
    """HTTP server bound to one trace directory."""

    def __init__(self, root: Path, host: str, port: int) -> None:
        self.root = root
        self.host = host
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                try:
                    self._get(parsed)
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    self._send_json({"error": str(exc)}, status=500)

            def _get(self, parsed: Any) -> None:
                if parsed.path in ("/", "/index.html"):
                    page = _STATIC.joinpath("reference_report.html").read_bytes()
                    self._send(page, "text/html; charset=utf-8")
                    return
                if parsed.path == "/static/plotly.min.js":
                    script = _STATIC.joinpath("plotly.min.js").read_bytes()
                    self._send(script, "application/javascript")
                    return
                query = {key: values[0] for key, values in parse_qs(parsed.query).items()}
                if parsed.path == "/api/episodes":
                    self._send_json({"episodes": list_episodes(viewer.root)})
                    return
                episodes = _episodes(query)
                effector = _effector(query)
                if parsed.path == "/api/pose_compare":
                    self._send_json(pose_compare(viewer.root, episodes, effector))
                    return
                if parsed.path == "/api/joint_compare":
                    self._send_json(joint_compare(viewer.root, episodes, effector))
                    return
                if parsed.path == "/api/ik_solver_metrics":
                    self._send_json(ik_metrics(viewer.root, episodes, effector))
                    return
                if parsed.path == "/api/env_loop_report":
                    name = query.get("episode", "")
                    if not _known(viewer.root, name):
                        self._send_json({"error": "unknown episode"}, status=404)
                        return
                    self._send_json(env_loop_report(viewer.root, name))
                    return
                self._send(b"not found", "text/plain", status=404)

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path in ("/api/ensure_verification", "/api/ensure_lerobot_precheck"):
                    self._send_json({"status": "ok"})
                    return
                self._send(b"not found", "text/plain", status=404)

            def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
                self._send(json.dumps(payload).encode("utf-8"), "application/json", status)

            def log_message(self, fmt: str, *args: Any) -> None:
                del fmt, args

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self.port = int(self._httpd.server_address[1])

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def serve_forever(self) -> None:
        self._httpd.serve_forever()

    def close(self) -> None:
        self._httpd.server_close()


def _episodes(query: dict[str, str]) -> list[str]:
    raw = query.get("episodes") or query.get("episode") or ""
    return [name for name in raw.split(",") if name and "/" not in name and name not in {".", ".."}]


def _effector(query: dict[str, str]) -> str:
    value = query.get("effector", "left")
    return value if value in {"left", "right"} else "left"


def _known(root: Path, name: str) -> bool:
    if not name or "/" in name or name in {".", ".."}:
        return False
    return (root / name / "motion.jsonl").is_file()
