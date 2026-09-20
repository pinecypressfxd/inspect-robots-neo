"""Group directory-index rows into tasks and render the view library.

Pure row-data in, pure task structures and HTML out; the CLI owns log
discovery and parsing. Grouping key is the shared instruction text, which is
what an ad-hoc run and a registered task both carry.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from inspect_robots._html_index import _ERROR_LIMIT, IndexEntry, _escape, _number

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
    values: list[float] = []
    for entry in task.runs:
        value = entry.metrics.get("score")
        # A sanitized non-finite score is written as a JSON null metric (#253):
        # it reports no score rather than crashing the whole library page.
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def success_fraction(task: LibraryTask) -> str:
    """``successful/total`` over the task's runs."""
    return f"{sum(1 for entry in task.runs if entry.success)}/{len(task.runs)}"


_STYLES = """
:root {
  color-scheme: light dark;
  --bg: #f7f8fa;
  --panel: #ffffff;
  --text: #20242b;
  --muted: #68707d;
  --line: #dfe3e8;
  --link: #245ca6;
  --green: #19723b;
  --green-bg: #e9f6ed;
  --red: #a12a2a;
  --red-bg: #fbecec;
  --grey: #626a75;
  --grey-bg: #eef0f2;
  --amber: #8a5700;
  --amber-bg: #fff5d9;
  --neutral: #45546a;
  --neutral-bg: #edf1f6;
}
@media (prefers-color-scheme: light) {
  :root { color-scheme: light; }
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #111419;
    --panel: #191d24;
    --text: #e7eaf0;
    --muted: #a4acb8;
    --line: #343a45;
    --link: #8ebcff;
    --green: #7ed99a;
    --green-bg: #193b27;
    --red: #ff9b9b;
    --red-bg: #492323;
    --grey: #c0c5cd;
    --grey-bg: #343943;
    --amber: #ffd484;
    --amber-bg: #3b2d12;
    --neutral: #b9c9df;
    --neutral-bg: #293342;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
header { border-bottom: 1px solid var(--line); background: var(--panel); }
.header-inner, main { width: min(1500px, calc(100% - 32px)); margin: auto; }
.header-inner { padding: 26px 0 20px; }
h1 { margin: 0; font-size: 24px; font-weight: 650; }
.meta { color: var(--muted); margin-top: 7px; }
main { margin-bottom: 64px; }
.toolbar { margin: 22px 0 14px; }
label { display: block; color: var(--muted); font-size: 12px; margin-bottom: 5px; }
input {
  width: min(460px, 100%);
  padding: 9px 11px;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--panel);
  color: var(--text);
  font: inherit;
}
a { color: var(--link); }
.tasks { display: flex; flex-direction: column; gap: 8px; }
.task-card {
  display: block;
  padding: 12px 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  color: var(--text);
  text-decoration: none;
}
.task-card:hover { background: var(--bg); }
.task-card .instruction { color: var(--link); font-weight: 650; overflow-wrap: anywhere; }
.task-card .stats { color: var(--muted); margin-top: 3px; }
.timing { color: var(--muted); font-size: 12px; margin-top: 2px;
  font-variant-numeric: tabular-nums; }
header .timing { margin-top: 2px; }
.task-card .policies { color: var(--muted); margin-top: 3px; font-size: 12px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.chip {
  padding: 3px 10px;
  border: 1px solid var(--line);
  border-radius: 999px;
  background: var(--panel);
  color: var(--muted);
  font: inherit;
  font-size: 12px;
  cursor: pointer;
}
.chip[aria-pressed="true"] { opacity: .55; text-decoration: line-through; }
h2 { margin: 26px 0 10px; font-size: 15px; font-weight: 650; }
.table-wrap {
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
}
table { width: 100%; border-collapse: collapse; }
th, td { padding: 10px 12px; text-align: left; vertical-align: top; }
th {
  color: var(--muted);
  border-bottom: 1px solid var(--line);
  font-size: 11px;
  font-weight: 650;
  letter-spacing: .04em;
  text-transform: uppercase;
  white-space: nowrap;
}
td { border-top: 1px solid var(--line); }
tbody tr:first-child td { border-top: 0; }
tbody tr:hover { background: var(--bg); }
.when, .log { white-space: nowrap; }
.error-cell { color: var(--red); min-width: 150px; max-width: 280px; overflow-wrap: anywhere; }
.empty { color: var(--muted); padding: 28px 12px; text-align: center; }
""".strip()

_FILTER_KEY = "inspect-robots-library-filter"
_POLICY_KEY = "inspect-robots-library-policies"


def _task_card(task: LibraryTask) -> str:
    """Render one fully escaped task card linking its per-task page."""
    policies = sorted({run.policy for run in task.runs if run.policy})
    data_policy = f' data-policy="{_escape(json.dumps(policies))}"' if policies else ""
    policy_line = f'<div class="policies">{_escape(", ".join(policies))}</div>' if policies else ""
    runs_word = "run" if len(task.runs) == 1 else "runs"
    stats = (
        f"{len(task.runs)} {runs_word} · success {success_fraction(task)} · "
        f"mean score {_number(mean_score(task))}"
    )
    timing = _latest_timing(task)
    timing_line = f'<div class="timing">{timing}</div>' if timing else ""
    return (
        f'<a class="task-card" href="task-{_escape(task.slug)}.html"{data_policy} '
        f'data-text="{_escape(" ".join([task.instruction, *policies]))}">'
        f'<div class="instruction" title="{_escape(task.instruction)}">'
        f"{_escape(task.instruction)}</div>"
        f'<div class="stats">{stats}</div>'
        f"{timing_line}{policy_line}</a>"
    )


def _latest_timing(task: LibraryTask) -> str:
    """The newest run's start/end stamps and duration, e.g. for the card."""
    for run in task.runs:
        if run.started_at and run.completed_at:
            duration = "" if run.duration_s is None else f" · {_number(run.duration_s)}s"
            return f"last run {_escape(run.started_at)} → {_escape(run.completed_at)}{duration}"
    return ""


def _error_cell(error: str | None) -> str:
    """Render a compact error with its complete value available as a tooltip."""
    if not error:
        return ""
    short = error if len(error) <= _ERROR_LIMIT else f"{error[: _ERROR_LIMIT - 1]}…"
    return f'<span title="{_escape(error)}">{_escape(short)}</span>'


def _loose_row(entry: IndexEntry) -> str:
    """Render one fully escaped row for a run that groups into no task."""
    name = _escape(entry.name)
    log = name if entry.page is None else f'<a href="{_escape(entry.page)}">{name}</a>'
    data_text = " ".join(
        part for part in (entry.name, entry.created, entry.policy, entry.error or "") if part
    )
    return (
        f'<tr class="loose-row" data-text="{_escape(data_text)}">'
        f'<td class="when"><time>{_escape(entry.created)}</time></td>'
        f'<td class="log">{log}</td>'
        f'<td class="error-cell">{_error_cell(entry.error)}</td>'
        "</tr>"
    )


def _loose_section(loose: Sequence[IndexEntry]) -> str:
    """Render the ungrouped-logs table, newest first."""
    if not loose:
        return ""
    ordered = sorted(loose, key=lambda entry: entry.created, reverse=True)
    rows = "".join(_loose_row(entry) for entry in ordered)
    return (
        '<section class="loose"><h2>Ungrouped logs</h2>'
        '<div class="table-wrap"><table>'
        "<thead><tr><th>When</th><th>Log</th><th>Error</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div></section>"
    )


def render_library(
    tasks: Sequence[LibraryTask],
    *,
    loose: Sequence[IndexEntry] = (),
    title: str = "Inspect Robots task library",
    refresh_seconds: int | None = None,
) -> str:
    """Return one self-contained HTML document indexing tasks and their runs."""
    policies = sorted(
        {run.policy for task in tasks for run in task.runs if run.policy}
        | {entry.policy for entry in loose if entry.policy}
    )
    chips = "".join(
        f'<button type="button" class="chip" data-policy="{_escape(policy)}" '
        f'aria-pressed="false">{_escape(policy)}</button>'
        for policy in policies
    )
    chip_row = f'<div class="chips">{chips}</div>' if policies else ""
    cards = "".join(_task_card(task) for task in tasks)
    run_total = sum(len(task.runs) for task in tasks)
    tasks_word = "task" if len(tasks) == 1 else "tasks"
    runs_word = "run" if run_total == 1 else "runs"
    empty = "" if tasks or loose else '<p class="empty">no evaluation logs found</p>'
    refresh = ""
    if refresh_seconds is not None:
        # A full refresh resets filter focus/caret at the selected interval;
        # accepted over the extra complexity of a fetch-and-swap index update.
        refresh = f'<meta http-equiv="refresh" content="{refresh_seconds}">\n'
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
{refresh}<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(title)}</title>
<style>{_STYLES}</style>
</head>
<body>
<header><div class="header-inner">
  <h1>{_escape(title)}</h1>
  <div class="meta">{len(tasks)} {tasks_word} · {run_total} {runs_word}</div>
</div></header>
<main>
  <div class="toolbar">
    <label for="filter">Filter tasks</label>
    <input id="filter" type="search" placeholder="instruction, policy…">
    {chip_row}
  </div>
  <div class="tasks">{cards}</div>
  {empty}
  {_loose_section(loose)}
</main>
<script>
const filterKey = "{_escape(_FILTER_KEY)}", policyKey = "{_escape(_POLICY_KEY)}";
const input = document.querySelector("#filter");
const cards = document.querySelectorAll(".task-card");
const loose = document.querySelectorAll(".loose-row");
const chips = document.querySelectorAll(".chip");
const excluded = new Set();
function policiesOf(card) {{
  try {{ return JSON.parse(card.dataset.policy || "[]"); }} catch (_) {{ return []; }}
}}
function applyFilter() {{
  const query = input.value.toLocaleLowerCase();
  cards.forEach(card => {{
    const names = policiesOf(card);
    const policyOk = names.length === 0 || !names.every(name => excluded.has(name));
    card.hidden = !(card.dataset.text.toLocaleLowerCase().includes(query) && policyOk);
  }});
  loose.forEach(row => row.hidden = !row.dataset.text.toLocaleLowerCase().includes(query));
  try {{ localStorage.setItem(filterKey, input.value); }} catch (_) {{}}
}}
function syncChips() {{
  chips.forEach(chip => chip.setAttribute(
    "aria-pressed", excluded.has(chip.dataset.policy) ? "true" : "false"
  ));
}}
try {{ input.value = localStorage.getItem(filterKey) || ""; }} catch (_) {{}}
try {{
  JSON.parse(localStorage.getItem(policyKey) || "[]").forEach(name => excluded.add(String(name)));
}} catch (_) {{}}
syncChips();
input.addEventListener("input", applyFilter);
chips.forEach(chip => chip.addEventListener("click", () => {{
  const name = chip.dataset.policy;
  if (excluded.has(name)) {{ excluded.delete(name); }} else {{ excluded.add(name); }}
  try {{ localStorage.setItem(policyKey, JSON.stringify(Array.from(excluded))); }} catch (_) {{}}
  syncChips();
  applyFilter();
}}));
applyFilter();
</script>
</body>
</html>
"""


_TASK_STYLES = """
.back { display: inline-block; margin-bottom: 8px; font-size: 13px; }
.badges { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.badge {
  padding: 2px 9px;
  border: 1px solid var(--line);
  border-radius: 999px;
  color: var(--muted);
  font-size: 12px;
}
.tabs { display: flex; flex-wrap: wrap; gap: 6px; margin: 22px 0 12px; }
.tab {
  padding: 7px 13px;
  border: 1px solid var(--line);
  border-radius: 7px;
  background: var(--panel);
  color: var(--text);
  font: inherit;
  cursor: pointer;
}
.tab.active { border-color: var(--link); color: var(--link); font-weight: 650; }
.tab[disabled] { opacity: .55; cursor: default; }
.tab .status {
  display: inline-block;
  margin-left: 6px;
  padding: 1px 8px;
  border-radius: 999px;
  font-size: 11px;
}
.status-completed { color: var(--green); background: var(--green-bg); }
.status-running { color: var(--amber); background: var(--amber-bg); }
.status-error { color: var(--red); background: var(--red-bg); }
.status-cancelled { color: var(--grey); background: var(--grey-bg); }
.status-neutral { color: var(--neutral); background: var(--neutral-bg); }
.frame-wrap { border: 1px solid var(--line); border-radius: 8px; background: var(--panel); }
#rollout-frame {
  display: block;
  width: 100%;
  height: calc(100vh - 230px);
  min-height: 480px;
  border: 0;
}
""".strip()


def _tab(index: int, run: IndexEntry, *, active: bool) -> str:
    """Render one rollout tab; unlinked tabs stay disabled with no swap target."""
    classes = "tab active" if active else "tab"
    data_src = "" if run.page is None else f' data-src="{_escape(run.page)}"'
    disabled = "" if run.page is not None else " disabled"
    return (
        f'<button type="button" class="{classes}"{data_src}{disabled} '
        f'title="{_escape(run.name)} · {_escape(run.created)}">'
        f"Rollout {index + 1} "
        f'<span class="status {run.status_class}">{_escape(run.status)}</span></button>'
    )


def render_task_page(task: LibraryTask, *, refresh_seconds: int | None = None) -> str:
    """Return one self-contained HTML page tabbing between the task's run pages."""
    policies = sorted({run.policy for run in task.runs if run.policy})
    badges = (
        "".join(f'<span class="badge">{_escape(policy)}</span>' for policy in policies)
        if policies
        else ""
    )
    badges_row = f'<div class="badges">{badges}</div>' if policies else ""
    runs_word = "run" if len(task.runs) == 1 else "runs"
    stats = (
        f"{len(task.runs)} {runs_word} · success {success_fraction(task)} · "
        f"mean score {_number(mean_score(task))}"
    )
    timing = _latest_timing(task)
    timing_row = f'<div class="timing">{timing}</div>' if timing else ""
    # Runs are newest first, so the first run with a page is the default tab.
    first_enabled = next(
        (index for index, run in enumerate(task.runs) if run.page is not None), None
    )
    default_page = None if first_enabled is None else task.runs[first_enabled].page
    tabs = "".join(
        _tab(index, run, active=index == first_enabled) for index, run in enumerate(task.runs)
    )
    frame_src = "" if default_page is None else f' src="{_escape(default_page)}"'
    iframe = f'<iframe id="rollout-frame" title="rollout player"{frame_src}></iframe>'
    refresh = ""
    if refresh_seconds is not None:
        # A full refresh resets the selected tab to the newest rollout at the
        # selected interval; accepted over patching the iframe in place.
        refresh = f'<meta http-equiv="refresh" content="{refresh_seconds}">\n'
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
{refresh}<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(task.instruction)}</title>
<style>{_STYLES}{_TASK_STYLES}</style>
</head>
<body>
<header><div class="header-inner">
  <a class="back" href="index.html">&larr; Task library</a>
  <h1>{_escape(task.instruction)}</h1>
  <div class="meta">{stats}</div>
  {timing_row}
  {badges_row}
</div></header>
<main>
  <div class="tabs">{tabs}</div>
  <div class="frame-wrap">{iframe}</div>
</main>
<script>
const frame = document.getElementById("rollout-frame");
document.querySelectorAll("button.tab").forEach(tab => tab.addEventListener("click", () => {{
  document.querySelectorAll("button.tab").forEach(
    other => other.classList.toggle("active", other === tab)
  );
  frame.src = tab.dataset.src;
}}));
</script>
</body>
</html>
"""
