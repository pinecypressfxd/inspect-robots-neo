"""Gemini adapter that stops before any live model call."""

from __future__ import annotations

from pathlib import Path

from inspect_robots_pai.agents.codex import AgentError


class GeminiDiagnoser:
    """Fail closed and leave diagnostics.success untouched."""

    def __call__(self, run_dir: Path, failed_indices: list[int]) -> str:
        """Raise AgentError without writing a success label."""
        raise AgentError("gemini diagnoser is offline")
