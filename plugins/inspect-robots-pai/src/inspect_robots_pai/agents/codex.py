"""Codex adapters that stop before any live model call."""

from __future__ import annotations

from pathlib import Path


class AgentError(Exception):
    """A coding agent failed before it could change the candidate."""


def _models_path() -> Path:
    start = Path(__file__).resolve().parent
    for directory in (start, *start.parents):
        candidate = directory / "configs" / "models.yaml"
        if candidate.is_file():
            return candidate
    raise AgentError("configs/models.yaml was not found beside the package")


def _coder_model() -> str:
    section: str | None = None
    model = ""
    for raw in _models_path().read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not raw[:1].isspace() and stripped.endswith(":"):
            section = stripped[:-1]
            continue
        if section != "coder" or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        if key.strip() == "model":
            model = value.strip().strip("\"'")
    if not model:
        raise AgentError("coder.model is missing")
    return model


class CodexWriter:
    """Read coder.model, then fail without writing the candidate."""

    def __call__(self, candidate: Path, instruction: str) -> None:
        """Raise AgentError after resolving the configured coder model."""
        model = _coder_model()
        raise AgentError(f"codex writer is offline ({model})")


class CodexSynthesizer:
    """Open the candidate read-only, then fail without writing it."""

    def __call__(self, candidate: Path, diagnosis: str) -> str:
        """Read candidate/__init__.py and raise AgentError."""
        init_py = Path(candidate) / "__init__.py"
        with init_py.open("r", encoding="utf-8") as handle:
            handle.read()
        raise AgentError("codex synthesizer is offline")
