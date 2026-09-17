"""VLA policies for Inspect Robots: pure ``umi-replay`` and the Astra hybrid."""

from __future__ import annotations

from typing import Any

__all__ = ["umi_replay"]


def umi_replay(**kwargs: Any) -> None:
    """Registry factory for the ``umi-replay`` policy (plan 0084, task 3).

    Accepts (and ignores) CLI ``-P`` kwargs so the entry point is discoverable
    while the wire client is the only implemented piece; constructing the
    policy is the next task's deliverable.
    """
    raise NotImplementedError(
        "the umi-replay policy adapter lands with plan 0084 task 3; until then "
        "the wire client is available as inspect_robots_vla._client.VlaClient"
    )
