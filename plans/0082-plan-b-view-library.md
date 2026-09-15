# View Task Library (Part B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `inspect-robots view LOG_DIR` into a task-grouped video library: a library index page (tasks sidebar with search and policy chips), per-task pages with rollout tabs over iframe-embedded run pages, and a playback-synced decision card (rationale + per-arm command deltas) inside each run page.

**Architecture:** New pure module `src/inspect_robots/_library.py` beside `_html_index.py` (CLI owns filesystem discovery; the module takes plain row data and returns HTML strings). The directory index becomes the library page; `render_index`'s flat table is superseded and deleted (the 100% coverage gate forbids dead code). The decision card extends the existing per-run page JS contract: `video[data-steps][data-fps]` plus `section.turn[data-step]`, fed by a per-trial JSON payload computed from the trial's `actions/` JSONL side-car.

**Tech Stack:** stdlib only (html.escape, json, re), the repo's existing self-contained-HTML conventions, pytest at the core 100% coverage gate.

**Spec:** `plans/0082-nero-embodiment-and-view-library.md` (Part B).

## Global Constraints

- Core changes: `ruff check .`, `ruff format --check .`, `uv run --no-sync mypy` (strict, src+tests), `uv run --no-sync pytest --cov` with `--cov-fail-under=100` must all pass before every commit.
- Escape every foreign value once at its interpolation boundary (`html.escape(..., quote=True)`), following `_html_index.py`.
- No new dependencies; no server; static HTML + the existing `--serve` refresh cadence.
- Zero-width tasks (empty directory) and unreadable logs (empty instruction) must render, not crash.
- Commit after every task. Branch: the feature branch created at execution start.

---

### Task 1: Grouping core (`_library.py`) and the run `success` flag

**Files:**
- Modify: `src/inspect_robots/_html_index.py` (add `success` field to `IndexEntry`; delete `render_index`, `_row`, `_STYLES`, `_FILTER_KEY`, `_link`, `_error_cell` in Task 2 when the library page replaces them; in this task only ADD the field)
- Modify: `src/inspect_robots/cli.py` (`_index_entry` around line 2156: set `success`)
- Create: `src/inspect_robots/_library.py`
- Test: `tests/test_library.py`

**Interfaces:**
- Produces:
  - `IndexEntry.success: bool = False` (frozen dataclass field with default; existing constructors stay valid)
  - `LibraryTask` frozen dataclass: `slug: str`, `instruction: str`, `runs: tuple[IndexEntry, ...]`
  - `group_library(entries: Sequence[IndexEntry]) -> tuple[LibraryTask, ...]` (skips empty-instruction entries; deterministic slug assignment sorted by instruction)
  - `task_slug(instruction: str, taken: set[str]) -> str`
  - `mean_score(task: LibraryTask) -> float | None` (mean of runs' `metrics["score"]`)
  - `success_fraction(task: LibraryTask) -> str` (`"a/b"`)
- Consumes: `IndexEntry` from `_html_index`.

- [ ] **Step 1: Write the failing tests**

```python
"""Task grouping for the view library."""

from __future__ import annotations

from inspect_robots._html_index import IndexEntry
from inspect_robots._library import (
    group_library,
    mean_score,
    success_fraction,
    task_slug,
)


def _entry(instruction: str, name: str = "a.json", **overrides: object) -> IndexEntry:
    values: dict[str, object] = {
        "name": name,
        "page": name.replace(".json", ".html"),
        "created": "2026-09-15T00:00:00+00:00",
        "instruction": instruction,
        "policy": "agent",
        "model": "openai/gpt-6-astra",
        "status": "completed",
        "status_class": "status-completed",
        "metrics": {"score": 50.0},
        "errored_trials": 0,
        "termination": "success",
        "error": None,
        "success": True,
    }
    values.update(overrides)
    return IndexEntry(**values)  # type: ignore[arg-type]


def test_entries_group_by_shared_instruction() -> None:
    tasks = group_library(
        [
            _entry("put the cup on the pad", "a.json"),
            _entry("put the cup on the pad", "b.json"),
            _entry("fold the cloth", "c.json"),
        ]
    )
    assert [task.instruction for task in tasks] == ["fold the cloth", "put the cup on the pad"]
    cup = tasks[1]
    assert [run.name for run in cup.runs] == ["b.json", "a.json"]  # newest first


def test_slugs_are_stable_and_collision_free() -> None:
    taken: set[str] = set()
    first = task_slug("Put the Cup, on the pad!", taken)
    second = task_slug("Put the Cup, on the pad!", taken)
    assert first == "put-the-cup-on-the-pad"
    assert second == "put-the-cup-on-the-pad-2"
    assert task_slug("   !!!   ", taken) not in ("", first, second)


def test_long_instructions_truncate_slugs() -> None:
    taken: set[str] = set()
    slug = task_slug("x" * 200, taken)
    assert len(slug) <= 48


def test_empty_instruction_entries_are_excluded() -> None:
    tasks = group_library([_entry(""), _entry("real task")])
    assert [task.instruction for task in tasks] == ["real task"]


def test_mean_score_and_success_fraction() -> None:
    task = group_library(
        [
            _entry("t", "a.json", metrics={"score": 40.0}, success=True),
            _entry("t", "b.json", metrics={"score": 60.0}, success=False),
            _entry("t", "c.json", metrics={}, success=False),
        ]
    )[0]
    assert mean_score(task) == 50.0
    assert success_fraction(task) == "1/3"


def test_mean_score_is_none_without_scores() -> None:
    task = group_library([_entry("t", metrics={})])[0]
    assert mean_score(task) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync python -m pytest tests/test_library.py -q`
Expected: FAIL (`No module named 'inspect_robots._library'`).

- [ ] **Step 3: Implement**

`src/inspect_robots/_library.py`:

```python
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
```

Add to `IndexEntry` in `_html_index.py` (after `error`):

```python
    # Whether any trial in the run ended with the "success" termination reason.
    success: bool = False
```

In `cli.py` `_index_entry`, compute and pass it:

```python
    success = any(
        reason == "success"
        for scene in log.samples
        for reason in scene.termination_reasons
        if reason is not None
    )
```

and add `success=success` to the `IndexEntry(...)` call.

- [ ] **Step 4: Run tests and gates, commit**

Run: `uv run --no-sync python -m pytest tests/test_library.py tests/test_html_index.py -q` then the full gate set.
Expected: PASS (existing index tests still pass: the field defaults to False).

```bash
git add src/inspect_robots tests
git commit -m "feat(view): group directory logs into tasks for the library"
```

---

### Task 2: Library index page replaces the flat index

**Files:**
- Modify: `src/inspect_robots/_library.py` (add `render_library`)
- Modify: `src/inspect_robots/_html_index.py` (delete `render_index`, `_row`, `_link`, `_error_cell`, `_STYLES`, `_FILTER_KEY`; keep `IndexEntry`, `_escape`, `_number`, `_ERROR_LIMIT` if still used elsewhere, otherwise delete too)
- Modify: `src/inspect_robots/cli.py` (`_render_view_directory` around line 2337: replace `render_index(...)` with the library + task pages; Task 3 adds `render_task_page`)
- Modify: `tests/test_html_index.py` (rewrite for the library page)
- Test: `tests/test_library.py` (add render tests)

**Interfaces:**
- Produces: `render_library(tasks: Sequence[LibraryTask], *, loose: Sequence[IndexEntry] = (), title: str = "Inspect Robots task library", refresh_seconds: int | None = None) -> str`
- Consumes: Task 1's grouping.

- [ ] **Step 1: Update/add the failing tests**

Rewrite `tests/test_html_index.py` to render the library page via `render_library` (it currently asserts the deleted flat table; port the escaping/refresh/empty assertions). Add to `tests/test_library.py`:

```python
from inspect_robots._library import render_library


def test_library_page_renders_task_links_and_stats() -> None:
    tasks = group_library(
        [
            _entry("put the cup on the pad", "a.json", policy="agent"),
            _entry("put the cup on the pad", "b.json", policy="agent"),
            _entry("fold the cloth", "c.json", metrics={"score": 25.0}, success=False),
        ]
    )
    html = render_library(tasks)
    assert 'href="task-put-the-cup-on-the-pad.html"' in html
    assert "put the cup on the pad" in html
    assert "2 runs" in html and "1/2" in html
    assert "mean score 50" in html or "50" in html
    assert "agent" in html


def test_library_page_escapes_instructions() -> None:
    tasks = group_library([_entry('<script>alert(1)</script>')])
    html = render_library(tasks)
    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
    assert 'href="task-script-alert-script.html"' in html or "task-" in html


def test_library_page_lists_loose_rows() -> None:
    loose = [_entry("", name="broken.json", error="unreadable: boom", page=None)]
    html = render_library([], loose=loose)
    assert "broken.json" in html and "unreadable: boom" in html


def test_library_page_refresh_meta_and_empty_state() -> None:
    html = render_library([], refresh_seconds=5)
    assert '<meta http-equiv="refresh" content="5">' in html
    assert "no evaluation logs found" in html
```

- [ ] **Step 2: Run to verify failure, then implement `render_library`**

Pattern to follow (modeled exactly on the deleted `render_index` in `_html_index.py`: same `_STYLES` palette variables, same localStorage filter pattern, same refresh meta):

```python
def render_library(
    tasks: Sequence[LibraryTask],
    *,
    loose: Sequence[IndexEntry] = (),
    title: str = "Inspect Robots task library",
    refresh_seconds: int | None = None,
) -> str:
    """Return one self-contained HTML document indexing tasks and their runs."""
    policies = sorted({run.policy for task in tasks for run in task.runs if run.policy}
                      | {run.policy for run in loose if run.policy})
    cards = "".join(_task_card(task) for task in tasks)
    loose_rows = "".join(_loose_row(entry) for entry in loose)
    # ... header + toolbar (search input id="filter", chip buttons per policy
    # with data-policy) + <main> with the task card list + loose table + one
    # <script> combining the localStorage text filter and chip toggling.
```

`_task_card(task)`: one row/anchor per task: `href="task-<slug>.html"`, data-policy attributes of its runs' distinct policies, data-text containing instruction+policy text for the JS filter, showing: instruction (escaped, `title=` full text), stats line `N runs · success a/b · mean score X` (`"n/a"` when None), policies.

`_loose_row(entry)`: flat-table row (name, created, error escaped via `html.escape`).

JS contract: text filter hides cards/rows whose data-text misses the query; chip click toggles an excluded-policy set (cards whose policies are all excluded hide); both persist in localStorage keys `inspect-robots-library-filter` / `inspect-robots-library-policies`; follow the escaping-in-JS rules of the deleted page (values interpolated only inside string literals already JSON-safe: embed the filter keys via `_escape`).

- [ ] **Step 3: Wire the CLI**

In `cli.py` `_render_view_directory`, replace the `render_index(...)` call with:

```python
    from inspect_robots._library import group_library, render_library

    tasks = group_library(entries)
    loose = [entry for entry in entries if not entry.instruction]
    bytes_written += _write_html_atomic(
        render_library(tasks, loose=loose, refresh_seconds=refresh_seconds), index_path
    )
```

(Task 3 adds the task pages.) Delete the now-unused `from inspect_robots._html_index import ... render_index` import if present (`_index_entry` still imports `IndexEntry` from `_html_index`).

- [ ] **Step 4: Run tests and gates (100% coverage), commit**

Delete dead helpers from `_html_index.py` so coverage holds at 100%. Run the full gate set including `uv run --no-sync pytest --cov --cov-fail-under=100`.

```bash
git add src/inspect_robots tests
git commit -m "feat(view): library index page groups runs into tasks"
```

---

### Task 3: Task pages with rollout tabs

**Files:**
- Modify: `src/inspect_robots/_library.py` (add `render_task_page`)
- Modify: `src/inspect_robots/cli.py` (`_render_view_directory`: write `task-<slug>.html` per task)
- Test: `tests/test_library.py` (add), `tests/test_html_index.py` (directory-path assertions if it drives `_cmd_view`)

**Interfaces:**
- Produces: `render_task_page(task: LibraryTask, *, refresh_seconds: int | None = None) -> str`; task pages land at `out_dir/task-<slug>.html`.
- Consumes: Task 1 grouping; `IndexEntry.page` (the per-run HTML file names produced by `_directory_page_names`).

- [ ] **Step 1: Failing tests**

```python
from inspect_robots._library import render_task_page


def test_task_page_embeds_rollout_tabs_and_iframe() -> None:
    tasks = group_library(
        [_entry("put the cup on the pad", "a.json"), _entry("put the cup on the pad", "b.json")]
    )
    html = render_task_page(tasks[0])
    assert 'data-src="b.html"' in html and 'data-src="a.html"' in html
    assert "<iframe" in html and "Rollout 1" in html and "Rollout 2" in html
    assert 'href="index.html"' in html
    assert "put the cup on the pad" in html and "1/2" in html


def test_task_page_escapes_instruction() -> None:
    tasks = group_library([_entry("<b>task</b>")])
    html = render_task_page(tasks[0])
    assert "<b>task</b>" not in html
    assert "&lt;b&gt;task&lt;/b&gt;" in html


def test_run_without_page_gets_a_disabled_tab() -> None:
    tasks = group_library([_entry("t", page=None)])
    html = render_task_page(tasks[0])
    assert "disabled" in html
```

- [ ] **Step 2: Implement `render_task_page`**

Header: back link `index.html`, instruction as h1 (escaped), stats line (runs count, `success_fraction`, `mean_score` or `n/a`, policy badges). Tabs: one button per run, `Rollout {i}` (+ status badge), `data-src="{run.page}"` escaped, `disabled` when `run.page is None`, first enabled tab carries the `active` class and seeds the iframe `src`. Body: `<iframe id="rollout-frame" title="rollout player">` (set `src` only for the default tab; other tabs swap `iframe.src = button.dataset.src` on click). Reuse the `_STYLES` palette. Refresh meta identical to `render_library`.

- [ ] **Step 3: Wire the CLI**

In `_render_view_directory`, after the library index write:

```python
    from inspect_robots._library import render_task_page

    for task in tasks:
        bytes_written += _write_html_atomic(
            render_task_page(task, refresh_seconds=refresh_seconds),
            out_dir / f"task-{task.slug}.html",
        )
```

- [ ] **Step 4: Tests, gates (100%), commit**

```bash
git add src/inspect_robots tests
git commit -m "feat(view): per-task pages with rollout tabs over run pages"
```

---

### Task 4: Playback-synced decision card in run pages

**Files:**
- Modify: `src/inspect_robots/_html.py` (side-car loader, per-trial JSON payload, card markup, JS hook)

- Modify: `tests/test_html_view.py` (new tests)

**Interfaces:**
- Produces (markup contract):
  - Per trial block: `<script type="application/json" class="decision-data">{payload}</script>` where payload = `{"steps": [{"t": <int>, "l": [dx, dy, dz], "r": [dx, dy, dz], "lg": <m>, "rg": <m>}, ...]}` (deltas in cm vs the previous executed action; first step zeros; grippers absolute meters, rounded to 4 decimals). Serialized with `json.dumps(..., separators=(",", ":")).replace("</", "<\\/")`.
  - Card: `<aside class="decision-card">` with `data-decision-index`, `data-decision-step`, `data-decision-rationale`, `data-decision-commands` spans.
- Consumes: existing per-run page JS (`video[data-steps][data-fps]`, `section.turn[data-step]`, the flipbook `[data-step-label]` updates); trial metadata `record.metadata["actions"]` (relative path from the log dir); JSONL rows `{"t": int, "action": [floats]}` with a leading header row containing `"action_dim"`.

- [ ] **Step 1: Failing tests (add to `tests/test_html_view.py`)**

Build a golden log whose trial metadata carries `{"actions": "actions/20260915/test.jsonl"}` and write that side-car under the tmp log dir (mirror the existing fixtures in that file for logs + frames):

```python
def _sidecar(dir_path, rows: list[list[float]]) -> None:
    path = dir_path / "actions" / "20260915" / "test.jsonl"
    path.parent.mkdir(parents=True)
    lines = ['{"action_dim": 20, "control_mode": "eef_abs_pose"}']
    for t, action in enumerate(rows):
        lines.append(json.dumps({"t": t, "action": action}))
    path.write_text("\n".join(lines), encoding="utf-8")


def test_decision_card_embeds_command_deltas() -> None:
    # log with one trial, two steps: second action moves left x by 1 cm
    # ... render via render_html(log, log_path=..., ...)
    assert 'class="decision-data"' in html
    assert '"l":[1.0,0,0]' in html or '"l":[1,0,0]' in html
    assert "decision-card" in html
    assert "</script><script" not in html.split('class="decision-data"')[1][:200]


def test_decision_card_degrades_without_sidecar() -> None:
    # render a log whose trial metadata lacks "actions"
    assert "no command data" in html


def test_decision_payload_has_no_raw_close_tag() -> None:
    # instruction/rationale text containing "</script>" stays escaped in the payload
    assert "</script>" not in payload_region
```

Exact fixture setup copies the neighboring tests' golden-log builders; the side-car lives beside the log JSON (`log_path.parent / metadata["actions"]`).

- [ ] **Step 2: Implement**

In `_html.py`:

1. `_load_decision_steps(log_path: Path | None, metadata: Mapping[str, Any]) -> list[dict[str, Any]]`: resolve `metadata.get("actions")` relative to `log_path.parent` if both present; read the JSONL; skip the header row (`"action_dim"` in the row); walk rows computing per-step deltas vs the previous action (first step zeros); return the `steps` payload list (cm conversion `*100` on dims 0-2 and 10-12, rounded to 4 decimals; `lg`/`rg` absolute from dims 9/19). Any read/parse failure returns `[]` (degrade, never crash the page).
2. In the per-trial player block builder (the same place the `video[data-steps]` panel and transcript rail are emitted): emit the card markup above and, when steps exist, the JSON payload script; otherwise render `<div class="decision-sub muted">no command data</div>` inside the card.
3. Extend the block JS: the existing `timeupdate` handler already computes `steps[index]`; after it highlights the active turn, call `updateDecisionCard(block, currentStep, activeTurn)`; the flipbook click path sets the same via the shared function. `updateDecisionCard` reads the payload once (`JSON.parse` of the `decision-data` script's textContent), finds the last step row with `t <= currentStep`, formats commands as `L Δxyz [a b c] cm · R Δxyz [a b c] cm · gripper L 0.04 m · R 0.09 m`, and fills rationale from the active turn's `textContent` trimmed to 240 chars (textContent only, no HTML injection), with `–` placeholders when nothing applies.

- [ ] **Step 3: Tests, gates (100% coverage over the new branches), commit**

```bash
git add src/inspect_robots/_html.py tests/test_html_view.py
git commit -m "feat(view): playback-synced decision card with command deltas"
```

---

### Task 5: Docs, CHANGELOG, full verification

**Files:**
- Modify: `docs/guide/cli.md` (view section)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Docs.** In the `view` section of `docs/guide/cli.md`, after the directory-mode paragraph, add: "Directory renders group runs into a task library: the index lists tasks with search and policy filters, each task page tabs between rollouts, and every run page shows a decision card synced to playback with the policy's rationale and per-arm command deltas from the actions side-car." (2 sentences, no em dashes.)

- [ ] **Step 2: CHANGELOG** under `## [Unreleased]` / `### Changed` (the index page changed shape):

```markdown
- **Core:** `inspect-robots view LOG_DIR` now renders a task library: a task
  index with search and policy filters, per-task pages with rollout tabs, and
  playback-synced decision cards (rationale plus per-arm command deltas from
  the actions side-car) on each run page.
```

- [ ] **Step 3: Full gates.** `ruff check .`, `ruff format --check .`, `uv run --no-sync mypy`, `uv run --no-sync pytest --cov --cov-fail-under=100`. All green, then commit:

```bash
git add docs CHANGELOG.md
git commit -m "docs(view): task library and decision card in the guide"
```

---

## Verification after Task 5 (spec milestone 7-8 complete)

- `uv run --no-sync pytest tests/test_library.py tests/test_html_index.py tests/test_html_view.py -q` green; full `pytest --cov` at 100%.
- Manual: `uv run --no-sync inspect-robots view <a populated LOG_DIR> --open` shows the library; a task page tabs between rollouts; a run page with an actions side-car shows the decision card tracking the video playhead.
