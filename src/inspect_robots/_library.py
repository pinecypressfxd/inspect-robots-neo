"""Group directory-index rows into tasks for the view library.

Pure row-data in, pure task structures and HTML out; the CLI owns log
discovery and parsing. Grouping key is the shared instruction text, which is
what an ad-hoc run and a registered task both carry.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from inspect_robots._html_index import IndexEntry

_SLUG_MAX = 48


@dataclass(frozen=True)
class LibraryTask:
    """One task: a shared instruction plus its rollout runs."""

    slug: str
    instruction: str
    runs: tuple[IndexEntry, ...]


def task_slug(instruction: str, taken: set[str]) -> str:
    """Derive a filesystem-safe, collision-free page-name slug."""
    base = re.sub(r"[^a-z0-9]+", "-", instruction.lower()).strip("-")[:_SLUG_MAX].rstrip("-")
    base = base or "task"
    candidate = base
    suffix = 2
    while candidate in taken:
        candidate = f"{base}-{suffix}"
        suffix += 1
    taken.add(candidate)
    return candidate


def group_library(entries: Sequence[IndexEntry]) -> tuple[LibraryTask, ...]:
    """Group index rows by shared instruction; runs inside a task are newest-first."""
    grouped: dict[str, list[IndexEntry]] = {}
    for entry in entries:
        if entry.instruction:
            grouped.setdefault(entry.instruction, []).append(entry)
    tasks: list[LibraryTask] = []
    taken: set[str] = set()
    for instruction in sorted(grouped):
        runs = tuple(sorted(grouped[instruction], key=lambda entry: entry.created, reverse=True))
        tasks.append(
            LibraryTask(slug=task_slug(instruction, taken), instruction=instruction, runs=runs)
        )
    return tuple(tasks)


def mean_score(task: LibraryTask) -> float | None:
    """Mean of the runs' ``score`` metric, or None when no run reports one."""
    values = [float(entry.metrics["score"]) for entry in task.runs if "score" in entry.metrics]
    if not values:
        return None
    return sum(values) / len(values)


def success_fraction(task: LibraryTask) -> str:
    """``successful/total`` over the task's runs."""
    return f"{sum(1 for entry in task.runs if entry.success)}/{len(task.runs)}"
