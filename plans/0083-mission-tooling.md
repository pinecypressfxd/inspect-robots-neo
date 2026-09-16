# 0083: Mission tooling for the nero rig

Three approved workstreams (design agreed 2026-09-16, priority 3 → 1 → 2):

- **WS3 Wire replay**: the agent policy's wire capture (default on,
  `wire_capture=True`) already records every LLM request/response under
  `logs/wire/<stamp>/` with content-addressed image blobs and the path in
  `sample.trial_metadata["wire_capture"]`. Deliverable: a first-class replay
  view — `inspect-robots inspect LOG --replay` renders one self-contained HTML
  (per turn: observation images, tool call + arguments, reasoning text).
- **WS1 LeRobot export**: `plugins/inspect-robots-nero/scripts/export_lerobot.py`
  converts a finished run (log JSON + frame store + actions side-car) into a
  LeRobot v2.1 dataset: `meta/{info.json, episodes.jsonl, tasks.jsonl}`,
  `data/chunk-000/episode_XXXXXX.parquet` (action 20-dim, observation.state
  16-dim joint_pos, timestamps/indices), per-camera
  `videos/chunk-000/<cam>/episode_XXXXXX.mp4` (or `--images` PNG folders).
  Feature layout mirrors the bring-up `raw_to_lerobot.py` minus alignment
  auxiliaries. Parquet needs pyarrow: plugin-side optional dependency, never
  core (core stays NumPy-only).
- **WS2 Mission console (plugin-level, option A)**:
  `plugins/inspect-robots-nero/scripts/mission_console.py`, stdlib-only
  (`http.server`, `threading`, `pty`, `json`). Binds 127.0.0.1. Layout mirrors
  the bring-up monitor: three live camera tiles (MJPEG ~10 fps from the
  plugin's D405Camera), status line, start form (new instruction + explicit
  human confirm), stop / verdict buttons, and a reasoning feed below (1 s
  poll: live-log status + transcript tail + latest actions + last wire
  response). Start spawns the eval as a pty subprocess with
  `-E operator_reset_confirm=False` (the web confirm replaces the terminal
  gate); stop writes the `/stop` console line; verdicts write `/y`, `/n`,
  `/p` lines. History stays in core `view --serve` (linked, not duplicated).

## Safety

- The web Start button is two-step (instruction, then explicit confirm); it is
  the operator gate. The spawned eval never auto-confirms beyond that.
- Stop is graceful (`/stop` line → verdict flow in-page); no SIGKILL paths.
- The console binds localhost only and owns the cameras exclusively
  (documented conflict with a concurrently running eval).

## Verification

- WS3: core gates + 100% coverage; replay renders from a fixture wire capture.
- WS1: exporter smoke against a synthetic log; parquet/mp4 layout diffed
  against the schema tables in `raw_to_lerobot.py`.
- WS2: manual hardware checklist in the plugin README (cameras, start, stop,
  verdict, feed); ruff/mypy clean on the script.
