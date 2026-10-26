"""VLA policies for Inspect Robots.

The registered adapter is [`policy`][inspect_robots_vla.policy] (``vla``).
The selectable ``hybrid`` policy is retired. [`executor`][inspect_robots_vla.executor]
is the planner executor HarnessVLA subclasses, and is reached lazily by the
``__getattr__`` below so a direct ``import inspect_robots_vla`` never pulls
the agent plugin.
"""

from __future__ import annotations

from typing import Any

from inspect_robots_vla.policy import VlaPolicy

__all__ = ["VlaPolicy", "umi_replay", "vla_policy"]


def vla_policy(**kwargs: Any) -> VlaPolicy:
    """Registry factory for the ``vla`` policy (entry point ``vla``).

    Accepts the same keyword arguments as
    [`VlaPolicy`][inspect_robots_vla.policy.VlaPolicy]; the CLI forwards each
    ``-P key=value`` pair here.
    """
    return VlaPolicy(**kwargs)


#: Compat alias for the pre-rename entry point; kept so old commands and
#: scripts using ``--policy umi-replay`` keep resolving.
umi_replay = vla_policy


def __getattr__(name: str) -> Any:
    """Resolve the executor, and the retired hybrid names, lazily (PEP 562)."""
    if name in {"ExecutorPolicy", "ExecutorPolicyConfig", "HybridPolicy", "hybrid_policy"}:
        from inspect_robots_vla import executor

        if name == "HybridPolicy":
            return executor.ExecutorPolicy
        return getattr(executor, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
