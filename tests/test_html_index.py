"""Pure rendering tests for the self-contained task-library index page."""

from __future__ import annotations

from inspect_robots._html_index import IndexEntry
from inspect_robots._library import group_library, render_library


def _entry(
    name: str,
    *,
    page: str | None = "run.html",
    created: str = "2026-07-30T12:00:00Z",
    instruction: str = "pick up the cube",
    policy: str = "agent",
    metrics: dict[str, float] | None = None,
    error: str | None = None,
    success: bool = True,
) -> IndexEntry:
    return IndexEntry(
        name=name,
        page=page,
        created=created,
        instruction=instruction,
        policy=policy,
        model="provider/models/claude-test",
        status="completed",
        status_class="status-completed",
        metrics={"score": 50.0} if metrics is None else metrics,
        errored_trials=0,
        termination="succeeded",
        error=error,
        success=success,
    )


def _document(
    entries: list[IndexEntry],
    *,
    loose: list[IndexEntry] | None = None,
    refresh_seconds: int | None = None,
) -> str:
    return render_library(
        group_library(entries), loose=loose or [], refresh_seconds=refresh_seconds
    )


def test_escaping_of_instructions_title_and_loose_names() -> None:
    document = _document(
        [_entry("linked.json", instruction="<script>alert(1)</script>")],
        loose=[
            _entry("gone.json", instruction="", page=None, error='<boom> & "gone"'),
        ],
    )

    assert "<script>alert" not in document
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in document
    assert "&lt;boom&gt; &amp; &quot;gone&quot;" in document
    assert 'title="&lt;script&gt;alert(1)&lt;/script&gt;"' in document


def test_title_and_header_meta_escape_and_count_tasks_and_runs() -> None:
    document = render_library(
        group_library([_entry("a.json")]),
        title="runs <b>&amp;</b> more",
    )

    assert "<title>runs &lt;b&gt;&amp;amp;&lt;/b&gt; more</title>" in document
    assert "<h1>runs &lt;b&gt;&amp;amp;&lt;/b&gt; more</h1>" in document
    assert '<div class="meta">1 task · 1 run</div>' in document

    two = _document(
        [
            _entry("a.json"),
            _entry("b.json", instruction="fold the cloth"),
            _entry("c.json", instruction="fold the cloth"),
        ]
    )
    assert '<div class="meta">2 tasks · 3 runs</div>' in two


def test_cards_link_task_pages_and_show_the_stats_line() -> None:
    document = _document(
        [
            _entry("a.json"),
            _entry("b.json", metrics={"score": 0.666666}, success=False),
        ]
    )

    assert '<a class="task-card" href="task-pick-up-the-cube.html"' in document
    assert "2 runs · success 1/2 · mean score 25.33" in document
    single = _document([_entry("only.json", metrics={})])
    assert "1 run · success 1/1 · mean score n/a" in single


def test_card_stats_line_uses_four_significant_figures() -> None:
    document = _document([_entry("a.json", metrics={"score": 0.666666})])

    assert "mean score 0.6667" in document


def test_policy_chips_and_card_policy_data() -> None:
    document = _document(
        [
            _entry("a.json", policy="agent"),
            _entry("b.json", policy="pi0", instruction="fold the cloth"),
        ]
    )

    assert '<div class="chips">' in document
    assert (
        '<button type="button" class="chip" data-policy="agent" '
        'aria-pressed="false">agent</button>' in document
    )
    assert 'data-policy="[&quot;agent&quot;]"' in document
    assert 'data-policy="[&quot;pi0&quot;]"' in document
    # Card data-text carries the instruction and policies for the text filter.
    assert 'data-text="pick up the cube agent"' in document


def test_runs_without_policy_get_no_chip_and_no_policy_attributes() -> None:
    document = _document([_entry("a.json", policy="")])

    assert "<button" not in document
    assert 'data-policy="' not in document
    assert '<div class="policies">' not in document


def test_loose_rows_link_pages_and_stay_plain_without_one() -> None:
    document = _document(
        [],
        loose=[
            _entry("linked.json", instruction="", page="linked.html"),
            _entry("gone.json", instruction="", page=None, error="unreadable: boom"),
        ],
    )

    assert '<td class="log"><a href="linked.html">linked.json</a></td>' in document
    assert '<td class="log">gone.json</td>' in document
    assert "no evaluation logs found" not in document


def test_loose_rows_are_newest_first() -> None:
    document = _document(
        [],
        loose=[
            _entry("old.json", instruction="", page=None, created="2026-07-29T12:00:00Z"),
            _entry("new.json", instruction="", page=None, created="2026-07-30T12:00:00Z"),
        ],
    )

    assert document.index("new.json") < document.index("old.json")


def test_empty_library_renders_the_empty_state() -> None:
    document = render_library([], loose=[])

    assert "<!doctype html>" in document
    assert "no evaluation logs found" in document


def test_static_library_has_no_meta_refresh() -> None:
    document = _document([_entry("run.json")], refresh_seconds=None)

    assert '<meta http-equiv="refresh"' not in document


def test_served_library_has_exact_meta_refresh() -> None:
    document = _document([_entry("run.json")], refresh_seconds=60)

    assert '<meta http-equiv="refresh" content="60">' in document


def test_filter_script_and_persisted_keys_are_present() -> None:
    document = _document([_entry("run.json")])

    assert 'id="filter"' in document
    assert "localStorage.setItem" in document
    assert "localStorage.getItem" in document
    assert "inspect-robots-library-filter" in document
    assert "inspect-robots-library-policies" in document
    assert "dataset.text.toLocaleLowerCase().includes(query)" in document
    assert "excluded.has(name)" in document


def test_chip_toggle_script_is_wired_once() -> None:
    document = _document([_entry("run.json")])

    assert document.count('addEventListener("click"') == 1
    assert 'chips.forEach(chip => chip.addEventListener("click"' in document
    assert "syncChips();" in document
    assert "aria-pressed" in document


def test_long_error_is_truncated_with_full_escaped_tooltip() -> None:
    error = 'failed <badly> "' + "x" * 200
    document = _document(
        [],
        loose=[_entry("broken.json", instruction="", page=None, error=error)],
    )

    assert f'title="failed &lt;badly&gt; &quot;{"x" * 200}"' in document
    assert "…" in document


def test_palette_variables_carry_over_from_the_flat_index() -> None:
    document = _document([_entry("run.json")])

    assert "--link: #245ca6;" in document
    assert "@media (prefers-color-scheme: dark)" in document
    assert '.chip[aria-pressed="true"]' in document
