"""VLA policies for Inspect Robots: pure ``umi-replay`` and the Astra hybrid.

The pure adapter lives in
[`policy`][inspect_robots_vla.policy] and the planner-executor hybrid in
[`hybrid`][inspect_robots_vla.hybrid] (registered as ``hybrid`` through its
own entry point). The hybrid module is reached lazily, by entry point or the
``__getattr__`` below, so a direct ``import inspect_robots_vla`` never pulls
the agent plugin the hybrid's planner needs. Registry paths are different:
the core registry eagerly loads every entry point in the group, so listing or
resolving policies imports the hybrid module (and the agent plugin it
declares as a dependency) regardless.
"""

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


def __getattr__(name: str) -> Any:
    """Resolve the hybrid exports lazily (PEP 562); see the module docstring."""
    if name in {"HybridPolicy", "hybrid_policy"}:
        from inspect_robots_vla import hybrid

        return getattr(hybrid, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
