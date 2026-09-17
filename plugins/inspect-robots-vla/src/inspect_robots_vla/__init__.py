"""VLA policies for Inspect Robots: pure ``umi-replay`` and the Astra hybrid."""

from __future__ import annotations

from typing import Any

from inspect_robots_vla.policy import VlaPolicy

__all__ = ["VlaPolicy", "umi_replay"]


def umi_replay(**kwargs: Any) -> VlaPolicy:
    """Registry factory for the ``umi-replay`` policy (entry point ``umi-replay``).

    Accepts the same keyword arguments as
    [`VlaPolicy`][inspect_robots_vla.policy.VlaPolicy]; the CLI forwards each
    ``-P key=value`` pair here.
    """
    return VlaPolicy(**kwargs)
