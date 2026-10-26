#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

TRACE_DIR=${TRACE_DIR:-${REPO_ROOT}/.inspect_traces}
REFERENCE_REPORT_VIEWER_HOST=${REFERENCE_REPORT_VIEWER_HOST:-127.0.0.1}
REFERENCE_REPORT_VIEWER_PORT=${REFERENCE_REPORT_VIEWER_PORT:-62444}

echo "[inspect] reference report viewer host=${REFERENCE_REPORT_VIEWER_HOST} port=${REFERENCE_REPORT_VIEWER_PORT}" >&2
echo "[inspect] TRACE_DIR=${TRACE_DIR}" >&2

uv run --no-sync --directory "${REPO_ROOT}" python -u \
  "${SCRIPT_DIR}/run_reference_report_viewer.py" \
  --trace-dir "${TRACE_DIR}" \
  --host "${REFERENCE_REPORT_VIEWER_HOST}" \
  --port "${REFERENCE_REPORT_VIEWER_PORT}"
