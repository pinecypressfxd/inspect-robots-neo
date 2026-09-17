# inspect-robots-vla

VLA policies for [Inspect Robots](https://github.com/robocurve/inspect-robots)
speaking the live `serve_rlt_inference` HTTP wire (NPZ submit/poll on
`127.0.0.1:10055`):

- `umi-replay`: the pure VLA policy. One `act()` is one wire round trip: the
  submit-camera frames (HWC uint8, encoded CHW in sorted key order), the 20-dim
  EE state, and the task string go to `POST /submit`; the returned delta-EE
  chunk (20 steps x 14 dims on the live checkpoint) is re-anchored onto the
  observed EE state and emitted as an absolute-target `ActionChunk`. This is
  the pure-VLA baseline and the hybrid's execution inner loop.
- `hybrid`: a Helix-style planner/executor split. A frontier LLM (through the
  [agent plugin](../inspect-robots-agent/)'s chat wire) decomposes the goal with
  the `delegate_skill`, `done`, and `give_up` tools; each delegated subgoal runs
  as a `umi-replay` segment whose VLA task string is the subgoal itself, and the
  planner regains control at every checkpoint, tracking abort, or time cap.

## Safety

> [!WARNING]
> VLA chunks ride the same four clamp layers as any `move_to` command: the
> workspace box, the core Clamp + DeltaLimit guardrails, the per-tick joint
> rate clamp (0.1 rad), and the joint envelope (firmware limits inset by
> 0.05 rad). Any target the VLA emits is clamped or rejected by those layers
> before it reaches the arms; do not run with `--disable-guardrails`.

Clamping lag self-heals: every chunk re-anchors on the freshly observed EE
state, so drift never crosses a chunk boundary. When the hybrid's hands stop
tracking the commanded targets (default 3 cm / 20 deg, measured against the
last commanded step after each executed chunk), the segment ends and the
tracking-abort report hands control back to the planner as
`skill_interrupted`, which decides to retry, correct course, or give up rather
than commanding through the error.

## Usage

Both policies pair with the [nero embodiment](../inspect-robots-nero/) (or any
embodiment providing a 20-dim `eef_state` key and the submit cameras) and an
inference service listening on :10055. Every constructor argument is `-P`
overridable (see [Tunables](#tunables)).

Baseline, the trained task string sent verbatim to the VLA:

    inspect-robots "put the cup on the pad with left arm" \
      --policy umi-replay -P prompt="put the cup on the pad" \
      --embodiment nero

Hybrid, Astra planning through the experientiallabs proxy while the VLA
executes each delegated skill:

    export EXPLABS_API_KEY=...
    inspect-robots "put the cup on the pad with left arm" \
      --policy hybrid -P model=gpt-6-astra \
      -P base_url=https://api.experientiallabs.ai/v1 \
      -P api_key_env=EXPLABS_API_KEY \
      --embodiment nero

The hybrid reads its goal from `-P prompt=` or the scene instruction and writes
the per-segment VLA task strings itself, so `-P prompt=` is the goal, not the
trained string. The nero mission console spawns the `agent` policy with fixed
flags and has no `-P` passthrough: run these policies through the CLI directly.

The hybrid planner conversation reaches the eval log through the standard
`transcript()` hook (images stubbed) and the live stream through
`transcript_delta()`.

## Dependencies

`inspect-robots-agent` is a declared dependency of this package: the hybrid
policy's planner reuses its chat client, the same reuse the capx plugin makes.
The core registry eagerly loads every policy entry point, so any registry path
(`inspect-robots list policies`, `eval`, ...) imports the hybrid module and,
with it, the agent plugin; the lazy import in `inspect_robots_vla.__init__`
only keeps a direct `import inspect_robots_vla` free of it. There is no torch
and no vendor SDK anywhere: the wire client speaks plain HTTP over httpx and
the anchor math is scipy.

## Install

In the Inspect Robots workspace this plugin is a uv member:
`uv sync --all-packages --extra dev` from the repo root installs it editable
and registers both entry points. Standalone: `pip install inspect-robots-vla`
(once published).

## Tunables

Defaults from `src/inspect_robots_vla/_config.py`; every row is a constructor
argument, overridable per run with `-P key=value`.

| `-P` flag | Default | Policies | Meaning |
|---|---|---|---|
| `base_url` (`vla_base_url` for hybrid) | `http://127.0.0.1:10055` | both | inference service base URL |
| `prompt` | scene instruction | both | task string; umi-replay sends it verbatim, hybrid uses it as the planner goal |
| `submit_images` | `left_rgbd,right_rgbd,chest_rgbd` | both | cameras whose frames go into the submit NPZ (2..3, sorted key order on the wire) |
| `state_key` | `eef_state` | both | observation state key used as the 20-dim EE anchor |
| `control_hz` | `30` | both | declared control rate |
| `timeout_s` | `10` | both | `/submit` HTTP timeout in seconds |
| `poll_interval_s` | `0.05` | both | `/result/latest` poll period in seconds |
| `poll_timeout_s` | `30` | both | how long `infer()` waits for the chunk in seconds |
| `model` | `INSPECT_ROBOTS_MODEL` env | hybrid | planner model id |
| `base_url` | provider default | hybrid | planner chat base URL (OpenAI-compatible) |
| `api_key_env` | provider default | hybrid | environment variable holding the planner API key |
| `checkpoint_interval_s` | `5` | hybrid | planner cadence between segments; `0` decides after every chunk |
| `max_skill_seconds` | `60` | hybrid | cap on one `delegate_skill` segment in seconds |
| `tracking_abort_pos_m` | `0.03` | hybrid | position tracking-abort threshold in meters |
| `tracking_abort_rot_deg` | `20` | hybrid | rotation tracking-abort threshold in degrees |
| `budget_llm_calls` | `100` | hybrid | whole-trial planner call budget |

Chunk length (20 steps) and the wire action format (`xyz_rpy`, 14-dim per
step) are checkpoint facts, not tunables: the client only decodes that format.
