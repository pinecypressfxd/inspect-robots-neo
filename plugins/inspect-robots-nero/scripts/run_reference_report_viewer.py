#!/usr/bin/env python3
"""Serve the inspect reference report for a trace directory."""

from __future__ import annotations

import argparse
import errno
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TRACE_DIR = REPO_ROOT / ".inspect_traces"


def build_parser() -> argparse.ArgumentParser:
    """Arguments for the local reference-report server."""
    parser = argparse.ArgumentParser(description="Serve commanded-vs-measured traces.")
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=62444)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Bind the viewer and block until interrupted."""
    args = build_parser().parse_args(argv)
    src = REPO_ROOT / "plugins" / "inspect-robots-nero" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from inspect_robots_nero._report_viewer import ReportViewer

    try:
        server = ReportViewer(args.trace_dir, args.host, args.port)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(f"[inspect] port already in use: {args.host}:{args.port}", file=sys.stderr)
            return 2
        raise
    print(f"[inspect] reference report viewer at {server.url}", flush=True)
    print(f"[inspect] serving traces from {args.trace_dir}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
