"""Nero dual-arm embodiment plugin for Inspect Robots."""

from __future__ import annotations

from typing import Any

from inspect_robots_nero.embodiment import NeroEmbodiment

__all__ = ["NeroEmbodiment", "nero_embodiment"]


def nero_embodiment(**kwargs: Any) -> NeroEmbodiment:
    """Construct the registry-facing nero embodiment from CLI or programmatic arguments."""
    return NeroEmbodiment(**kwargs)


# Consumed by inspect_robots.conformance.missing_runtime_requirements via the
# doctor/preflight path: the vendor SDK is not pip-installable from PyPI.
nero_embodiment.RUNTIME_REQUIREMENTS = {  # type: ignore[attr-defined]
    "pyAgxArm": (
        "install the vendor SDK: pip install -e <neo_manipulation checkout>/third_party/pyAgxArm"
    ),
}
