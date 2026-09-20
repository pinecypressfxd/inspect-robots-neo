# inspect-robots-nero

## Safety

> [!WARNING]
> This adapter sends commands to physical dual arms over CAN. Keep a tested
> emergency stop within reach, keep the bench clear of people and liquids, and
> do not run an unsupervised evaluation until the full command path has been
> checked on the hardware.

- First runs: `-P max_speed_frac=0.05` (agent policy) and keep `speed_percent`
  at its default 20.
- The IK clamps per-tick joint deltas, and core guardrails (Clamp + DeltaLimit)
  are on by default; do not run with `--disable-guardrails` on this embodiment.
- Loss of CAN feedback raises an `EmbodimentFault` and ends the trial; verify
  your `can_left`/`can_right` links come up before attaching a payload.

## What it drives

Dual Nero arms on a shared shelf (channels `can_left`/`can_right`, firmware
v112, 7 joints each, pika grippers, width 0 to 0.09 m) and three RealSense
D405 color streams read as V4L2 devices (left/right/chest). Constants mirror
the working bring-up config; the scalar arguments (`control_hz`, `cameras`,
`camera_max_age_s`, `reset_settle_timeout_s`, `operator_reset_confirm`) are
overridable with `-E`, while workspace bounds and `max_step` are programmatic
constructor arguments (pass them in Python when constructing `NeroEmbodiment`),
and the rest are fixed constants carried from that config.

## Install

pip install inspect-robots-nero, plus the vendor SDK from the lab checkout:

    pip install -e <neo_manipulation checkout>/third_party/pyAgxArm

`inspect-robots doctor --embodiment nero` reports a missing `pyAgxArm` install
with this remedy. Any workspace `uv sync` prunes venv packages that are not in
the lock, which silently removes the editable `pyAgxArm` (and `pyarrow`, which
the LeRobot exporter needs); after syncing, reinstall both:

    uv pip install -e <neo_manipulation checkout>/third_party/pyAgxArm
    uv pip install pyarrow

The URDF (`assets/dual_nero_pika.urdf`) ships with the
package, copied from the neo_manipulation repository (kinematics only; no mesh
assets are required).

## Quickstart (attended, single arm)

    export OPENAI_API_KEY=...
    inspect-robots "put the cup on the pad with left arm" \
      --policy agent -P model=openai/gpt-5.6-luna -P max_speed_frac=0.05 \
      --embodiment nero

The Astra model id above was current as of September 2026; confirm with
`curl -s https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY" | grep -i astra`
and record the chosen id here. Confirmed lab configuration (September 2026)
runs Astra through the experientiallabs proxy, where the model id is
`gpt-5.6-luna` (no provider prefix; a custom `base_url` passes the model id
through verbatim):

    export EXPLABS_API_KEY=...
    inspect-robots "put the cup on the pad with left arm" \
      --policy agent -P model=gpt-5.6-luna \
      -P base_url=https://api.experientiallabs.ai/v1 \
      -P api_key_env=EXPLABS_API_KEY -P max_speed_frac=0.05 \
      --embodiment nero

Other proxied endpoints work the same way with `-P base_url=... -P api_key_env=NAME`.

On reset the run prompts to arrange the scene, then drives both arms to their
home positions before the first observation. End an episode with Esc; the
default operator grader asks for the verdict.

## Operator tools

Two standalone commands cover power-on and power-off, mirroring the bring-up
checkout's `test_move_j.py --stage move_j` and `disable_arms.py`. After the
arms are powered on, home both:

    python scripts/arm_tools.py home [--arm left|right|both] [--wait 2.0]

The command enables each arm (failing loudly if an enable is refused),
switches to the firmware position mode, and issues one `move_j` to the config
home pose per arm. To drop the arms limp immediately:

    python scripts/arm_tools.py disable --arm both --yes

Without `--yes` it only warns and exits 2 (`Arms will go limp`). With it, it
disables and polls the per-joint enable flags until every joint reports limp
or `--timeout` (default 5 s) expires; a timeout exits 1. For
operator-confirmed incremental `move_js` steps on one arm, use
`scripts/bench_smoke.py`.

## LeRobot export

`scripts/export_lerobot.py` converts a finished run into a LeRobot v2.1
dataset for downstream training:

    python plugins/inspect-robots-nero/scripts/export_lerobot.py logs/adhoc_586b388c.json [-o OUT_DIR] [--images] [--fps 30]

The default output directory is `<log dir>/<log stem>-lerobot` and must not
already hold files. Each exported trial becomes one episode with one row per
executed action step (`action` 20-dim from the actions side-car,
`observation.state` 16-dim `joint_pos`, `timestamp = frame_index / fps`), one
parquet shard under `data/chunk-000/`, and one MP4 per camera under
`videos/chunk-000/observation.images.<camera>/` encoded through the core
shared ffmpeg encoder. `--images` writes `images/<camera>/episode_XXXXXX/`
PNG trees instead of videos. Feature layout and path templates mirror the
bring-up converter's `raw_to_lerobot.py`. Emitted meta files: `info.json`,
`tasks.jsonl`, `episodes.jsonl`, `episodes_stats.jsonl`, and `stats.json`
(per-episode and global min/max/mean/std over the exported float columns).
Omitted: the converter's `info.json` `camera_alignment` block and its
alignment `auxiliary.*` features and stats, which describe a rig this plugin
does not have.

Optional dependencies: `pip install pyarrow` (required; the exporter prints
this hint and exits 2 without it) and an `ffmpeg` binary on PATH for video
mode (or pass `--images`).

Behavior notes:

- Trials whose action steps lack stored frames are skipped with a warning; a
  run where no trial has frames exits 1. The frame captured after the final
  action (no matching action row) is not exported, keeping videos and parquet
  rows one-to-one.
- `observation.state` is written only when every exported trial's recorded
  transcript carries a `state[joint_pos]` line per policy observation aligned
  with the action rows (the saved log itself stores no per-step state); the
  values inherit the transcript's 4-decimal rounding. When
  any trial lacks that record, the column is dropped from the parquet shards
  and `info.json` features entirely, and the exporter prints a note; train on
  `action` plus images in that case.

## Camera overrides

    -E cameras=left_rgbd=/dev/v4l/by-path/pci-0000:80:14.0-usb-0:11.2:1.0-video-index4

Cross-camera frame alignment is not reproduced in this adapter; v1 accepts the
millisecond-scale skew between the three streams (the bring-up stack's
alignment gate is not part of this plugin). Frames older than
`-E camera_max_age_s` (default 0.5) are rejected loudly.

## Mission console

`scripts/mission_console.py` is a stdlib-only operator console for attended
nero runs: three live camera tiles on top, a status line, an instruction form
with a two-step start, stop and verdict buttons, and a link to the running
history viewer. Run it from the repo root:

    python plugins/inspect-robots-nero/scripts/mission_console.py [--port 8400] [--host 127.0.0.1] \
        [--camera NAME=DEVICE ...] [--max-speed-frac 0.05] \
        [--model gpt-5.6-luna] [--base-url URL] [--api-key-env EXPLABS_API_KEY] \
        [--history-url http://127.0.0.1:8300/] [--log-dir /tmp]

`--camera` takes the embodiment's `name=device` form (repeatable) and only
overrides that camera's device node. Start spawns, on a pty:

    uv run --no-sync inspect-robots "<instruction>" --policy agent \
        -P model=gpt-5.6-luna -P base_url=https://api.experientiallabs.ai/v1 \
        -P api_key_env=EXPLABS_API_KEY -P max_speed_frac=0.05 \
        --embodiment nero -E operator_reset_confirm=False

The child inherits the console's environment, so export `EXPLABS_API_KEY` in
the shell that launches the console. Its terminal output is captured to a
bounded in-memory tail plus a file under `--log-dir` (default `/tmp`), and the
run's eval log path is detected from the process's `log:` line, falling back
to the newest `logs/*.live.json` written after the run started. The JSON API
(`GET /api/status`, `GET /api/feed`, `POST /api/start` `/api/stop`
`/api/verdict`) drives the same controls the page offers.

Below the controls the page polls `/api/feed` once a second and renders: the
run's live-log status (status, current step, total steps, duration), the
newest transcript tail as one compact line per message (role and text, camera
images elided to an `[image]` marker, tool calls as `name(arguments)`; the
last 40 messages, newest pinned), the last five executed action rows from the
run's action side-car (raw action vectors), and the last wire call's tool
name with its note or summary argument. Every panel degrades to "no data
yet" until the run writes the corresponding artifact.

Safety and semantics:

- Start is two-step on the page: the first click only arms the button, the
  second (labeled with an explicit arms-will-move warning) actually posts.
  The server separately rejects an empty instruction and any start while a
  run is active. This web confirm replaces the terminal reset gate, which is
  why the spawned run passes `-E operator_reset_confirm=False`. The terminal
  gate's Esc key is not reachable from the page; the Stop button (the `/stop`
  line) is the page's way to end an episode early.
- The console binds 127.0.0.1 by default. A non-loopback `--host` prints a
  loud warning: anyone who can reach the page can start the arms. Every POST
  route additionally requires the request's Host header to match the bound
  address (loopback names accepted for loopback binds, port matching when
  present) and, when the browser sends an Origin header, that it matches too;
  anything else gets a 403. This check is a real barrier only for loopback
  binds; a wildcard bind (`0.0.0.0`) accepts any host name, and there the
  port check merely catches misaddressed requests, not foreign ones.
- Camera exclusivity: the tiles hold the three V4L2 nodes while streaming,
  and V4L2 mmap streaming is exclusive per node. The console releases the
  cameras when a run starts so the spawned eval can claim them. During a run
  the tiles do not go dark: they switch to `GET /frame/<camera>.jpg`, which
  serves the newest frame the run stored (from the frame directory its
  `.live.json` records), refreshed once a second, and switch back to the
  live MJPEG streams when the run ends. A frame requested while a run holds
  the devices but has not stored that camera yet answers a plain 503, and so
  does a dead or busy camera. An eval started outside the console while
  tiles are live fails to open the cameras: stop the console first.
- Stop and verdicts are graceful: the buttons write the `/stop` and `/y`,
  `/n`, `/p` (or `/skip`) console lines to the run's terminal, which is the
  only control channel, and every write is echoed on the console's stdout.
  The child process is never signalled or killed; it ends by itself (process
  exit, any code, marks the run ended in `/api/status`), and closing the
  console sends one final `/stop` before the terminal closes.
- Failed starts recover: a spawn that cannot start (unusable instruction,
  missing binary, full disk for the run log) returns a 400 with the reason,
  restarts the cameras it had released, and surfaces the error in
  `/api/status` so the page reconnects its tiles instead of looking stuck.

Manual checklist before first hardware use: all three camera tiles show live
video; a start round-trip (arm, confirm, the status line flips to running
and the tiles switch to run frames); the reasoning feed shows the model's
messages, actions, and last decision while the run is live; Stop ends the
episode; a verdict button resolves the verdict prompt; the ended state shows
the exit code and log path and the tiles return to live video.

End to end, one mission looks like this: start the console, type the
instruction, click Start twice (the second click, labeled as such, moves the
arms), watch the feed panels and the run-frame tiles while the model works,
press Stop to end the episode early if needed, click a verdict button when
the run asks, export the finished log with
`python plugins/inspect-robots-nero/scripts/export_lerobot.py logs/<stamp>.json` for training data, and
review the whole history at the `--history-url` (default
`http://127.0.0.1:8300/`, served by `inspect-robots view logs --serve`).
