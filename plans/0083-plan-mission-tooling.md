# 0083 Mission Tooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox syntax.

**Goal:** Wire replay view (core), LeRobot v2.1 exporter (plugin), and a plugin-level mission console with live cameras and start/stop control.

**Spec:** `plans/0083-mission-tooling.md` (binding decisions live there).

**Global Constraints**

- Core changes (WS3): `ruff check .`, `ruff format --check .`, `mypy` strict (src+tests), `pytest --cov --cov-fail-under=100` before every commit. No new core deps.
- Plugin changes (WS1/WS2): plugin gates (ruff/format/mypy src+tests/pytest); new third-party imports (pyarrow) must be lazy with a clear install hint; stdlib-only for the console.
- Prose: no em dashes in README/docs additions. Escape-once at interpolation boundaries in every generated HTML.
- Commit per task; branch `feat/0082-nero-and-view-library`.

---

### Task 1 (WS3): `inspect --replay` wire replay HTML

**Files:** create `src/inspect_robots/_wire_replay.py`; modify `src/inspect_robots/cli.py` (`inspect` subcommand: add `--replay` flag); test `tests/test_wire_replay.py`.

**Interfaces:**
- `render_wire_replay(capture_dir: Path) -> str`: reads `scene-*/calls.jsonl` (fields per call: `request`, `response` (dict or absent), `status`, `duration_s`); image parts reference `data:image/png;base64,$blob:<sha256>` → embed `blobs/<sha256>.png` as data URLs. Output: one self-contained HTML, newest-last per scene, per call showing new observation images (dedup by hash), assistant tool calls (`name` + pretty-printed `arguments`), reasoning text, status/duration badge.
- CLI: `inspect-robots inspect LOG --replay` resolves `sample.trial_metadata["wire_capture"]` (relative to the log's directory; same resolution pattern as the decision card's `actions` key via `resolve_log_pointer` in `_pointers.py`) for each trial, renders, writes `<log_dir>/wire-replay.html`, prints the path. Missing key or unreadable dir → clear message, exit 0 (degrade, like transcript).
- Contract details: tolerate absent `response` (in-flight/cancelled call) with a "no response" badge; never crash on malformed lines (skip); bound embedded size by skipping duplicate blobs.

**Steps:** failing tests with a fixture wire capture (tmp dir: 2 calls, 1 blob, one response-less call, one malformed line) → implement → core gates → commit "feat(view): wire replay HTML from inspect --replay".

### Task 2 (WS1): LeRobot v2.1 exporter

**Files:** create `plugins/inspect-robots-nero/scripts/export_lerobot.py`; modify plugin README (usage + optional deps).

**Interfaces:**
- CLI: `python scripts/export_lerobot.py LOG.json [-o OUT_DIR] [--images] [--fps 30]`.
- Output layout (LeRobot v2.1): `meta/info.json` (features: `observation.state` float32 (16,) from `joint_pos`; `action` float32 (20,); `observation.images.<camera>` video entries per camera; `timestamp` float64; `index`/`episode_index`/`task_index` int64; `fps`; `total_episodes/frames/videos`), `meta/tasks.jsonl` (instruction per task), `meta/episodes.jsonl` (per-trial lengths/task index), `data/chunk-000/episode_000000.parquet` one per trial (pyarrow, lazy import, `pip install pyarrow` hint on ImportError), `videos/chunk-000/<camera>/episode_000000.mp4` via `inspect_robots._video`'s shared encoder (fallback `--images`: `images/<camera>/episode_XXXXXX/-frameXXXXXX.png`).
- Data sources per trial: actions side-car rows (`trial_metadata["actions"]`, header has action_dim + labels; step `t` ↔ frame index) and per-step frames from the log's frame store (same discovery `_html.py` uses for flipbooks). Trials without recorded frames → skip episode with a warning.
- Verify layout against `raw_to_lerobot.py`'s `DATA_PATH_TEMPLATE`/feature tables (read it during implementation; do NOT port alignment aux fields).
- No automated tests (script, hardware-data-shaped); ruff + plugin mypy must pass on it; README documents a smoke run against a real completed log.

**Steps:** implement → run against `logs/adhoc_586b388c.json` era fixture or the first completed real run available (if none has frames yet, validate parquet schema with a synthetic log built in-repo by hand and say so) → gates → commit "feat(nero): LeRobot v2.1 export script".

### Task 3 (WS2a): mission console server — cameras + page + start/stop

**Files:** create `plugins/inspect-robots-nero/scripts/mission_console.py` (+ optional `mission_page.html` embedded or separate).

**Interfaces:**
- `python scripts/mission_console.py [--port 8400] [--host 127.0.0.1]` → stdlib `http.server` + threads.
- Routes: `GET /` page (three camera tiles top row like the bring-up monitor, status bar, instruction input + Start [two-step confirm], Stop, verdict buttons /y /n /p, reasoning feed panel, link to the `view --serve` history); `GET /cam/{left_rgbd|right_rgbd|chest_rgbd}.mjpg` multipart JPEG ~10 fps from plugin `D405Camera` singletons (device from `_config.CAMERA_DEFAULTS`, `--device` overrides per camera); `POST /api/start` `{instruction}` (reject if a run is active); `POST /api/stop`; `POST /api/verdict` `{choice}`; `GET /api/status` JSON (idle/running + current instruction + log path).
- Run manager: `pty.spawn`-style (`pty.openpty` + `subprocess`) of `inspect-robots "<instruction>" --policy agent -P model=gpt-6-astra -P base_url=... -P api_key_env=EXPLABS_API_KEY -P max_speed_frac=<cfg> --embodiment nero -E operator_reset_confirm=False` with the agent flags overridable via CLI args/env template; stop = write `/stop\n` to the pty; verdict = `/y\n` etc.; process reaped in a thread; log dir parsed from stdout (`log: ` line) or newest `*.live.json`.
- Safety: Start requires the page's explicit confirm step (client-side two-step + server rejects empty instruction); all writes to pty are logged to console stdout; bind 127.0.0.1 default.

**Steps:** implement server+page (no unit tests, hardware script; ruff + plugin mypy clean) → manual checklist documented in README (cameras visible, start/stop/verdict round-trip) → commit "feat(nero): mission console server with live cameras and run control".

### Task 4 (WS2b): mission console — live reasoning feed + polish

**Files:** modify `mission_console.py` (+page).

**Interfaces:**
- `GET /api/feed?since=<seq>` JSON: merged tail of the active run — live-log status, last N transcript messages (via `logs/transcripts/<stamp>/scene-0-e0.jsonl`, text+tool-call summaries, images elided), last actions rows (cm deltas precomputed client-side or raw), last wire response note; monotonic `seq`.
- Page JS: 1 s poll, append-only feed, newest decision pinned; status colors; disable controls per state; Esc/stop semantics documented; gracefully degrade panels when files absent.
- README: end-to-end usage (start console → confirm → watch → verdict → export LeRobot → view history).

**Steps:** implement → README + manual checklist → gates → commit "feat(nero): mission console live reasoning feed".

### Task 5: docs, CHANGELOG, notes

- CHANGELOG `## [Unreleased]`: `**Core:** inspect --replay wire replay HTML` + `**Plugins:** inspect-robots-nero mission console and LeRobot v2.1 export`.
- Plugin README: mission console + exporter sections.
- Update `~/fengxuedong/projects/notes/llm_drive_robot/01-调试与测试步骤.md` with the new workflow (console-first operation).
- Gates; commit "docs: mission tooling in guide and changelog".
