"""Row data and shared formatting helpers for the view-library index pages.

The CLI owns filesystem discovery and log parsing; this module carries the
plain row type those passes produce, plus the escaping and number-formatting
conventions the library renderer in ``_library.py`` interpolates. Keeping that
boundary pure makes escaping and presentation independently testable.
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class IndexEntry:
    """One evaluation-log row in the directory index."""

    name: str
    page: str | None
    created: str
    instruction: str
    policy: str
    model: str | None
    status: str
    status_class: str
    metrics: Mapping[str, float]
    errored_trials: int
    termination: str
    error: str | None
    # Whether any trial in the run ended with the "success" termination reason.
    success: bool = False
    # Run start (eval log created) and end (stats.completed_at) ISO stamps
    # with the run duration in seconds; None on logs without stats.
    started_at: str | None = None
    completed_at: str | None = None
    duration_s: float | None = None


_ERROR_LIMIT = 160


def _escape(value: object) -> str:
    """Escape one foreign value at its HTML interpolation boundary."""
    return html.escape(str(value), quote=True)


def _number(value: int | float | None) -> str:
    """Format numeric log values compactly before their interpolation boundary."""
    return "n/a" if value is None else f"{value:.4g}"
