"""Task grouping for the view library."""

from __future__ import annotations

from pathlib import Path

from inspect_robots._html_index import IndexEntry
from inspect_robots._library import (
    group_library,
    mean_score,
    render_library,
    render_task_page,
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
            _entry("put the cup on the pad", "a.json", created="2026-09-14T00:00:00+00:00"),
            _entry("put the cup on the pad", "b.json", created="2026-09-15T00:00:00+00:00"),
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


def test_mean_score_is_none_when_the_only_score_metric_is_null() -> None:
    """Regression guard for #253-style logs: a non-finite score is sanitized
    to a JSON null metric, which must read as "no score" and not crash."""
    task = group_library([_entry("t", metrics={"score": None})])[0]
    assert mean_score(task) is None


def test_library_page_renders_task_links_and_stats() -> None:
    tasks = group_library(
        [
            _entry("put the cup on the pad", "a.json", policy="agent"),
            _entry("put the cup on the pad", "b.json", policy="agent", success=False),
            _entry("fold the cloth", "c.json", metrics={"score": 25.0}, success=False),
        ]
    )
    html = render_library(tasks)

    assert 'href="task-put-the-cup-on-the-pad.html"' in html
    assert "put the cup on the pad" in html
    assert "2 runs" in html and "1/2" in html
    assert "mean score 50" in html
    assert "agent" in html


def test_library_page_escapes_instructions() -> None:
    tasks = group_library([_entry("<script>alert(1)</script>")])
    html = render_library(tasks)

    assert "<script>alert" not in html
    assert "&lt;script&gt;" in html
    assert 'href="task-script-alert-1-script.html"' in html


def test_library_page_lists_loose_rows() -> None:
    loose = [_entry("", name="broken.json", error="unreadable: boom", page=None)]
    html = render_library([], loose=loose)

    assert "broken.json" in html and "unreadable: boom" in html


def test_library_page_refresh_meta_and_empty_state() -> None:
    html = render_library([], refresh_seconds=5)

    assert '<meta http-equiv="refresh" content="5">' in html
    assert "no evaluation logs found" in html


def test_task_page_embeds_rollout_tabs_and_iframe() -> None:
    tasks = group_library(
        [
            _entry("put the cup on the pad", "a.json", created="2026-09-14T00:00:00+00:00"),
            _entry("put the cup on the pad", "b.json", success=False),
        ]
    )
    html = render_task_page(tasks[0])

    assert 'data-src="b.html"' in html and 'data-src="a.html"' in html
    assert "<iframe" in html and "Rollout 1" in html and "Rollout 2" in html
    assert 'href="index.html"' in html
    assert "put the cup on the pad" in html and "1/2" in html


def test_task_page_seeds_the_newest_enabled_tab_into_the_iframe() -> None:
    tasks = group_library(
        [
            _entry("t", "old.json", created="2026-09-14T00:00:00+00:00"),
            _entry("t", "new.json"),
        ]
    )
    html = render_task_page(tasks[0])

    assert html.count('class="tab active"') == 1
    assert '<button type="button" class="tab active" data-src="new.html"' in html
    assert '<iframe id="rollout-frame" title="rollout player" src="new.html">' in html
    assert "frame.src = tab.dataset.src" in html


def test_task_page_escapes_instruction() -> None:
    tasks = group_library([_entry("<b>task</b>")])
    html = render_task_page(tasks[0])

    assert "<b>task</b>" not in html
    assert "&lt;b&gt;task&lt;/b&gt;" in html


def test_run_without_page_gets_a_disabled_tab() -> None:
    tasks = group_library([_entry("t", page=None)])
    html = render_task_page(tasks[0])

    assert "disabled" in html
    assert 'data-src="' not in html
    assert '<iframe id="rollout-frame" title="rollout player">' in html


def test_task_page_falls_back_to_the_next_run_when_the_newest_has_no_page() -> None:
    tasks = group_library(
        [
            _entry("t", "gone.json", page=None),
            _entry("t", "kept.json", created="2026-09-14T00:00:00+00:00"),
        ]
    )
    html = render_task_page(tasks[0])

    assert '<button type="button" class="tab" disabled title=' in html
    assert 'data-src="kept.html"' in html
    assert '<iframe id="rollout-frame" title="rollout player" src="kept.html">' in html


def test_task_page_meta_line_and_policy_badges() -> None:
    tasks = group_library(
        [
            _entry("t", "a.json", policy="agent"),
            _entry("t", "b.json", policy="pi0", metrics={}, success=False),
        ]
    )
    html = render_task_page(tasks[0])

    assert "2 runs · success 1/2 · mean score 50" in html
    assert '<span class="badge">agent</span>' in html
    assert '<span class="badge">pi0</span>' in html


def test_task_page_singular_run_and_missing_mean_score() -> None:
    tasks = group_library([_entry("t", metrics={})])
    html = render_task_page(tasks[0])

    assert "1 run · success 1/1 · mean score n/a" in html


def test_task_page_status_badges_carry_run_status() -> None:
    tasks = group_library(
        [
            _entry("t", "a.json", status="running", status_class="status-running"),
            _entry(
                "t",
                "b.json",
                created="2026-09-14T00:00:00+00:00",
                status="error",
                status_class="status-error",
            ),
        ]
    )
    html = render_task_page(tasks[0])

    assert '<span class="status status-running">running</span>' in html
    assert '<span class="status status-error">error</span>' in html


def test_task_page_refresh_meta_matches_the_library_page() -> None:
    tasks = group_library([_entry("t")])
    html = render_task_page(tasks[0], refresh_seconds=5)

    assert '<meta http-equiv="refresh" content="5">' in html
    assert '<meta http-equiv="refresh"' not in render_task_page(tasks[0])


def test_task_page_tab_click_script_is_wired_once() -> None:
    tasks = group_library([_entry("t")])
    html = render_task_page(tasks[0])

    assert html.count('addEventListener("click"') == 1
    assert 'document.querySelectorAll("button.tab")' in html


def test_task_card_and_page_show_last_run_timing(tmp_path: Path) -> None:
    from inspect_robots._library import render_library, render_task_page

    entry = _entry(
        "put the cup on the pad",
        started_at="2026-09-20T08:00:00+00:00",
        completed_at="2026-09-20T08:01:30+00:00",
        duration_s=90.0,
    )
    html = render_library(group_library([entry]))
    assert "last run 2026-09-20T08:00:00+00:00 → 2026-09-20T08:01:30+00:00" in html
    assert "· 90s" in html
    page = render_task_page(group_library([entry])[0])
    assert "last run 2026-09-20T08:00:00+00:00 →" in page


def test_task_card_omits_timing_without_stats() -> None:
    from inspect_robots._library import render_library

    html = render_library(group_library([_entry("t")]))  # no timing fields set
    assert "last run" not in html
