"""Self-contained HTML replay of one agent-policy wire capture.

The agent policy records every LLM request attempt under
``<log_dir>/wire/<run>/<trial>/calls.jsonl`` with decoded images deduplicated at
``<log_dir>/wire/<run>/blobs/<sha256>.png``.  :func:`render_wire_replay` turns
one capture directory into a single offline page: every trial in its own
section with calls newest-last, each call showing only observation images not
yet seen in that trial's file, the assistant tool calls with pretty-printed
arguments, reasoning text, and status/duration badges.  Foreign data degrades
instead of crashing: malformed JSONL lines and unreadable trials are skipped, a
missing blob file drops just that image, and an absent response renders a
``no response`` badge.  Every foreign value is HTML-escaped exactly once, at
its interpolation boundary.
"""

from __future__ import annotations

import base64
import html
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_BLOB_RE = re.compile(r"\$blob:([0-9a-f]{64})")

_CSS = """
:root {
  color-scheme: light dark;
  --bg: #f7f8fa; --panel: #ffffff; --text: #20242b; --muted: #68707d;
  --line: #dfe3e8; --green: #19723b; --green-bg: #e9f6ed; --red: #a12a2a;
  --red-bg: #fbecec; --grey: #626a75; --grey-bg: #eef0f2; --assistant: #7a55b5;
  --amber: #8a5700; --amber-line: #d69b2d; --amber-bg: #fff5d9;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #111419; --panel: #191d24; --text: #e7eaf0; --muted: #a4acb8;
    --line: #343a45; --green: #7ed99a; --green-bg: #193b27; --red: #ff9b9b;
    --red-bg: #492323; --grey: #c0c5cd; --grey-bg: #343943; --assistant: #bd9bed;
    --amber: #ffd484; --amber-line: #b77a16; --amber-bg: #3b2d12;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.55 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
main { width: min(1120px, calc(100% - 32px)); margin: 0 auto 64px; }
h1 { margin: 28px 0 0; font-size: 24px; font-weight: 650; }
.meta { color: var(--muted); margin: 6px 0 0; overflow-wrap: anywhere; }
.note { color: var(--muted); margin: 24px 0; }
.capture {
  margin: 22px 0; padding: 20px 22px; background: var(--panel);
  border: 1px solid var(--line); border-radius: 9px;
}
h2 { margin: 0; font-size: 18px; font-weight: 650; overflow-wrap: anywhere; }
.trial { margin-top: 18px; }
h3 {
  margin: 0 0 10px; font-size: 13px; text-transform: uppercase;
  letter-spacing: .06em; color: var(--muted); overflow-wrap: anywhere;
}
.chip {
  margin-left: 8px; border-radius: 999px; padding: 2px 9px; font-size: 12px;
  color: var(--muted); background: var(--grey-bg); text-transform: none;
}
.call {
  margin: 14px 0; padding: 12px 14px; background: var(--bg);
  border: 1px solid var(--line); border-radius: 8px;
}
.call-head { display: flex; gap: 10px; flex-wrap: wrap; align-items: baseline; }
.call-id { font-weight: 700; }
.endpoint {
  font: 13px ui-monospace, SFMono-Regular, Consolas, monospace;
  color: var(--muted); overflow-wrap: anywhere;
}
.badge {
  display: inline-block; border-radius: 999px; padding: 2px 9px; font-size: 12px;
  color: var(--grey); background: var(--grey-bg);
}
.badge.ok { color: var(--green); background: var(--green-bg); }
.badge.error { color: var(--red); background: var(--red-bg); }
.badge.none { color: var(--amber); background: var(--amber-bg); }
.obs {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 8px; margin-top: 10px;
}
.obs img {
  width: 100%; max-width: 448px; height: auto;
  border: 1px solid var(--line); border-radius: 6px;
}
.label {
  display: block; margin-top: 10px; font-size: 11px; font-weight: 750;
  text-transform: uppercase; letter-spacing: .07em; color: var(--muted);
}
.reasoning {
  margin-top: 2px; padding: 8px 11px; color: var(--amber); background: var(--amber-bg);
  border-left: 3px solid var(--amber-line); white-space: pre-wrap; overflow-wrap: anywhere;
}
.answer { margin-top: 2px; white-space: pre-wrap; overflow-wrap: anywhere; }
.tool { margin-top: 10px; padding: 10px 12px; background: var(--panel); border-radius: 6px; }
.tool-name {
  font-weight: 700; color: var(--assistant);
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
}
pre.args, pre.raw {
  margin: 6px 0 0; padding: 0; white-space: pre-wrap; overflow-wrap: anywhere;
  font: 13px/1.5 ui-monospace, SFMono-Regular, Consolas, monospace;
}
"""


def _escape(value: object) -> str:
    """Escape one foreign value at its HTML interpolation boundary."""
    return html.escape(str(value), quote=True)


def render_wire_replay(capture_dir: Path) -> str:
    """Render one wire capture directory as a single self-contained HTML page."""
    return render_wire_replay_page((capture_dir,), capture_dir.name)


def render_wire_replay_page(capture_dirs: Sequence[Path], log_name: str) -> str:
    """Compose every capture directory an eval log referenced into one page.

    Directories without readable trials contribute nothing; when none do, the
    page body carries a ``no readable wire calls`` note instead of failing.
    """
    sections = [
        section
        for section in (_capture_section(directory) for directory in capture_dirs)
        if section is not None
    ]
    body = "".join(sections)
    note = "" if body else '<p class="note">no readable wire calls</p>'
    title = _escape(f"Wire replay: {log_name}")
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{title}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n<main>\n"
        f'<h1>Wire replay</h1>\n<p class="meta">{_escape(log_name)}</p>\n'
        f"{note}{body}\n</main>\n</body>\n</html>\n"
    )


def _capture_section(capture_dir: Path) -> str | None:
    """Render one capture directory, or ``None`` when no trial file is readable."""
    sections = [
        section
        for section in (
            _trial_section(calls) for calls in sorted(capture_dir.glob("*/calls.jsonl"))
        )
        if section is not None
    ]
    if not sections:
        return None
    return (
        f'<section class="capture">\n<h2>{_escape(capture_dir.name)}</h2>\n'
        + "".join(sections)
        + "</section>\n"
    )


def _trial_section(calls_path: Path) -> str | None:
    """Render one trial's calls oldest-first, or ``None`` without readable rows."""
    rows = _read_rows(calls_path)
    if not rows:
        return None
    blob_dir = calls_path.parent.parent / "blobs"
    seen: set[str] = set()
    articles = "".join(_call_article(row, blob_dir, seen) for row in rows)
    label = "1 call" if len(rows) == 1 else f"{len(rows)} calls"
    return (
        f'<section class="trial">\n<h3>{_escape(calls_path.parent.name)} '
        f'<span class="chip">{label}</span></h3>\n{articles}</section>\n'
    )


def _read_rows(calls_path: Path) -> list[dict[str, Any]]:
    """Parse one sidecar's rows, skipping malformed and non-object lines."""
    try:
        text = calls_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _call_article(row: dict[str, Any], blob_dir: Path, seen: set[str]) -> str:
    """Render one captured attempt with its new images and assistant output."""
    response = row.get("response")
    head = (
        f'<span class="call-id">call {_escape(_label(row.get("call")))} · '
        f"attempt {_escape(_label(row.get('attempt')))}</span>"
        f'<span class="endpoint">{_escape(_label(row.get("endpoint")))}</span>'
        + _status_badge(row.get("status"))
        + f'<span class="badge">{_escape(_duration_text(row.get("duration_s")))}</span>'
    )
    if response is None:
        head += '<span class="badge none">no response</span>'
    parts = [
        part
        for part in (
            _images_html(row.get("request"), blob_dir, seen),
            _labeled_block("reasoning", "\n\n".join(_reasoning_texts(response))),
            _labeled_block("answer", "\n\n".join(_assistant_texts(response))),
            _tools_html(response),
            _raw_response_html(response),
        )
        if part
    ]
    body = "".join(parts)
    return f'<article class="call">\n<div class="call-head">{head}</div>\n{body}</article>\n'


def _images_html(request: object, blob_dir: Path, seen: set[str]) -> str:
    """Embed each not-yet-seen observation blob referenced by one request."""
    images: list[str] = []
    for sha in _blob_tokens(request):
        if sha in seen:
            continue
        seen.add(sha)
        image = _image_html(sha, blob_dir)
        if image is not None:
            images.append(image)
    if not images:
        return ""
    return f'<div class="obs">{"".join(images)}</div>\n'


def _image_html(sha: str, blob_dir: Path) -> str | None:
    """Embed one blob as a data URL, skipping unreadable blob files."""
    try:
        raw = (blob_dir / f"{sha}.png").read_bytes()
    except OSError:
        return None
    encoded = base64.b64encode(raw).decode("ascii")
    return (
        f'<img loading="lazy" alt="observation {sha[:12]}" src="data:image/png;base64,{encoded}">'
    )


def _blob_tokens(value: object) -> list[str]:
    """Return every blob sha referenced by a captured value, in order."""
    if isinstance(value, str):
        return [match.group(1) for match in _BLOB_RE.finditer(value)]
    if isinstance(value, list):
        return [sha for item in value for sha in _blob_tokens(item)]
    if isinstance(value, dict):
        return [sha for item in value.values() for sha in _blob_tokens(item)]
    return []


def _labeled_block(label: str, text: str) -> str:
    """Render one non-empty text run under its uppercase label."""
    if not text:
        return ""
    return f'<div class="{label}"><span class="label">{label}</span>{_escape(text)}</div>\n'


def _tools_html(response: object) -> str:
    """Render the assistant tool calls of one response, if it holds any."""
    calls = _tool_calls(response) if isinstance(response, dict) else []
    if not calls:
        return ""
    tools = "".join(
        f'<div class="tool"><span class="tool-name">{_escape(name)}</span>'
        f'<pre class="args">{_escape(arguments)}</pre></div>\n'
        for name, arguments in calls
    )
    return tools


def _raw_response_html(response: object) -> str:
    """Render a non-object response body (a truncated raw wire payload)."""
    if not isinstance(response, str):
        return ""
    escaped = _escape(response)
    return f'<div><span class="label">response</span><pre class="raw">{escaped}</pre></div>\n'


def _tool_calls(response: dict[str, Any]) -> list[tuple[str, str]]:
    """Extract ``(name, pretty-printed arguments)`` from supported response shapes."""
    calls: list[tuple[str, str]] = []
    message = _assistant_message(response)
    if message is not None:
        raw_calls = message.get("tool_calls")
        if isinstance(raw_calls, list):
            for raw in raw_calls:
                if not isinstance(raw, dict):
                    continue
                function = raw.get("function")
                if not isinstance(function, dict):
                    continue
                calls.append(
                    (
                        str(function.get("name", "unknown")),
                        _pretty_arguments(function.get("arguments", "")),
                    )
                )
    for block in _content_blocks(response):
        if isinstance(block, dict) and block.get("type") == "tool_use":
            calls.append(
                (str(block.get("name", "unknown")), _pretty_arguments(block.get("input", {})))
            )
    return calls


def _pretty_arguments(value: object) -> str:
    """Pretty-print tool arguments, tolerating unparsable JSON strings."""
    parsed: object = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
    return json.dumps(parsed, indent=2, ensure_ascii=False, sort_keys=True)


def _reasoning_texts(response: object) -> list[str]:
    """Extract reasoning or thinking text from supported response shapes."""
    texts: list[str] = []
    if isinstance(response, dict):
        message = _assistant_message(response)
        if message is not None:
            for key in ("reasoning_content", "reasoning"):
                value = message.get(key)
                if isinstance(value, str) and value:
                    texts.append(value)
        for block in _content_blocks(response):
            if isinstance(block, dict) and block.get("type") == "thinking":
                value = block.get("thinking")
                if isinstance(value, str) and value:
                    texts.append(value)
    return texts


def _assistant_texts(response: object) -> list[str]:
    """Extract visible assistant text from supported response shapes."""
    texts: list[str] = []
    if isinstance(response, dict):
        message = _assistant_message(response)
        if message is not None:
            content = message.get("content")
            if isinstance(content, str) and content:
                texts.append(content)
        for block in _content_blocks(response):
            if isinstance(block, dict) and block.get("type") == "text":
                value = block.get("text")
                if isinstance(value, str) and value:
                    texts.append(value)
    return texts


def _assistant_message(response: dict[str, Any]) -> dict[str, Any] | None:
    """Return the OpenAI-style assistant message of a response, if well-formed."""
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            return message
    return None


def _content_blocks(response: dict[str, Any]) -> list[object]:
    """Return the Anthropic-style content blocks of a response, if any."""
    blocks = response.get("content")
    return blocks if isinstance(blocks, list) else []


def _label(value: object) -> str:
    """Format one optional scalar header field."""
    return "-" if value is None else str(value)


def _duration_text(value: object) -> str:
    """Format one captured duration, degrading non-numeric foreign values."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.3f}s"
    return "-"


def _status_badge(status: object) -> str:
    """Render the captured HTTP-ish status as a colored badge."""
    kind = "neutral"
    if isinstance(status, int) and not isinstance(status, bool):
        kind = "ok" if 200 <= status < 300 else "error"
    return f'<span class="badge {kind}">{_escape(_label(status))}</span>'
