# 0082: Nero dual-arm embodiment for the Astra policy, and a view task library

Two deliverables on one data path, approved as one design (§1-§4, 2026-09-15):

- **Part A** adds `plugins/inspect-robots-nero/`, a real-hardware embodiment that
  plugs the lab's dual Nero arm rig (CAN, 2x7 joints, 2 pika grippers, 3
  RealSense D405 cameras) into the framework as the embodiment name `nero`.
  ChatGPT GPT-6 Astra then drives it through the existing `agent` policy with
  zero new policy code: `--policy agent -P model=openai/gpt-6-astra --embodiment nero`.
- **Part B** upgrades `inspect-robots view LOG_DIR` from a flat run index into a
  task-grouped video library: task sidebar with search and policy filters,
  per-task pages with rollout tabs, and a playback-synced decision card
  (decision rationale plus per-arm command deltas) so evaluated Astra runs can
  be inspected the way the reference UI does.

The source of all hardware constants is the working bring-up in the
neo_manipulation RLtoken branch
(`scripts/eval/robot_itl_evaluation/run_ros2_real_robot_inference_rlt.sh`,
config `ros2_nero_inference.yaml`); only configuration values are reused, none
of its ROS 2 graph. Part A never imports from that repo; the plugin talks to
the vendor SDK and the cameras directly.

## Part A: the nero embodiment plugin

### Package layout

```
plugins/inspect-robots-nero/
  pyproject.toml                # entry point group inspect_robots.embodiments -> nero
  README.md                     # safety warning, bring-up, quickstart (ros plugin style)
  src/inspect_robots_nero/
    __init__.py                 # nero_embodiment(**kwargs) registry factory
    embodiment.py               # NeroEmbodiment(EmbodimentBase)
    _arm.py                     # pyAgxArm wrapper per arm (move_js, read_state, enable)
    _gripper.py                 # pika gripper width command + width readback
    _camera.py                  # D405 color stream, one background thread per camera
    _kinematics.py              # URDF model + per-arm IK + FK + rot6d conversion
    _config.py                  # defaults carried from ros2_nero_inference.yaml
    assets/dual_nero_pika.urdf  # copied from neo_manipulation assets (provenance noted)
    py.typed
  tests/                        # stub arm/gripper/camera fixtures, no hardware
```

Dependencies: `inspect-robots`, `numpy`, `opencv-python`, `pinocchio` (`pin`),
`scipy` (all lazily imported at module level of the hardware/kinematics modules
so `import inspect_robots_nero` stays cheap and testable). `pyagxarm` stays an
optional extra documented in the README (installed from the vendor checkout);
tests always run against stubs. Same workspace, ruff/mypy-strict,
per-plugin-test conventions as the other plugins; never counts toward the core
coverage gate.

### Spaces (the agent plugin's contract drives these)

Action box, shape `(20,)`, one `eef_abs_pose` vector over both arms:

| Segment | Dims | Meaning |
|---|---|---|
| left xyz | 3 | meters, base frame |
| left rot6d | 6 | two columns of the rotation matrix, entries in [-1, 1] |
| left gripper | 1 | opening width, 0 to 0.09 m |
| right segments | 10 | same layout |

`ActionSemantics(control_mode="eef_abs_pose", rotation_repr="rot6d",
gripper="continuous", frame="base", dim_labels=(left_x ... left_gripper,
right_x ... right_gripper), max_step=(...))`. The `max_step` declarations are
the safety rate limit in native units: position centimeter-scale, rotation and
gripper smaller. Position bounds default to a conservative workspace box
derived at construction by sampling FK over the joint limits and shrinking by
a margin; every bound stays constructor-overridable.

Observation space:

- cameras `left_rgbd`, `right_rgbd`, `chest_rgbd`, each `(480, 640)` RGB color;
  depth is out of scope for v1.
- state fields:
  - `eef_state` `(20,)`: current FK pose (rot6d) plus gripper width per arm,
    layout identical to the action vector. The agent plugin requires exactly
    one state field shaped like the action box to interpolate absolute targets
    against; this is that field.
  - `joint_pos` `(16,)`: 7 joints per arm plus 2 gripper widths, for logs and
    humans.

`control_hz=30` default (aligned with the bring-up action rate). One
`step()` tick runs IK once and commands `move_js` once; the SDK's internal
servo covers between-tick smoothing, so the neo controller's 200 Hz lookahead
logic is deliberately not ported. `capabilities={SELF_PACED, RESETTABLE}`.

### Hardware layer

- `_arm.NeroArm`: wraps `pyAgxArm.AgxArmFactory` per arm
  (channel `can_left`/`can_right`, socketcan, firmware v112). Exposes
  connect, enable, disable, `set_speed_percent(20)`, `move_js(joints)`,
  `read_state() -> joints`.
- `_gripper.NeroGripper`: width command (0 to 0.09 m, force 0.3) and width
  readback per arm via the SDK's gripper API.
- `_camera.D405Camera`: OpenCV V4L2 capture per camera, exactly like the
  proven bring-up (the neo stack reads the D405 color stream over V4L2 mmap,
  not pyrealsense2): `color_device` by-path node (defaults left
  `pci-0000:80:14.0-usb-0:11.2:1.0-video-index4`, right
  `pci-0000:80:14.0-usb-0:2.2:1.0-video-index4`, chest
  `pci-0000:00:0d.0-usb-0:2.2:1.0-video-index4`, all overridable), YUYV,
  640x480 at 30 fps, one daemon thread keeping the latest frame with a
  monotonic stamp. `reset()`/`step()` read the latest frame and reject
  stale frames past a timeout, following the ros plugin's staleness pattern.
  Cross-camera alignment is not reproduced in v1; the known millisecond-scale
  skew is documented in the README.

### Kinematics

`_kinematics.py` ports the working solver's core as a small class:
pinocchio model from the packaged dual URDF, per-arm forward kinematics and
Jacobian at `*_gripper_flange` with the TCP transform (translation
`[0, 0, 0.18]`, rpy `[0, -pi/2, 0]`), damped least squares iteration with
posture regularization seeded from the current joint angles, tolerances
1e-3 m / 1e-3 rad, per-iteration joint step cap 0.1 rad. The source solver's
ProxQP formulation stays with the neo controller: the embodiment clamps the
per-tick joint delta itself, so per-iteration QP limits would be redundant
here. rot6d to rotation matrix conversion lives here. All constants are
constructor-overridable.

### reset, step, close, and safety

- Construction builds spaces only (ros plugin pattern); CAN and cameras open
  lazily on the first `reset()`.
- `reset(scene)`: connect and enable arms, set speed percent, optional
  `operator_reset_confirm` (default on: arrange the scene, press Enter), drive
  both arms to `home_position` (left `[0.56, 0.92, -1.38, 1.90, -0.46, 0.04,
  0.20]`, right mirrored) blocking until the joint error settles or a timeout,
  open grippers to init width, return the first observation. The settle wait
  is bounded by a constructor-overridable timeout (default 10 s).
- `step(action)`: validate shape, per-arm IK (fault on non-convergence), clamp
  the joint delta to `max_joint_step`, `move_js`, gripper width command,
  assemble the observation from readback plus camera frames.
- Failure paths raise core error types (`EmbodimentFault`, timeouts) so the
  taxonomy applies; `close()` and the exception paths disable both arms.
- Core guardrails (Clamp + DeltaLimit) apply by default as on every run;
  first unsupervised runs should also lower `-P max_speed_frac` (documented in
  the README with the safety warning block).
- Constructor validation messages follow the ros plugin style: finiteness,
  bound consistency, device-node presence, with `-E`/`-P` fix hints.

### Policy integration (no code)

```bash
export OPENAI_API_KEY=...
uv run inspect-robots "put the cup on the pad with left arm" \
  --policy agent -P model=openai/gpt-6-astra \
  --embodiment nero
```

The agent plugin renders the three cameras plus `eef_state`, offers its
`move_to` tool (partial targets, only the dimensions the model names move),
interpolates at `control_hz` under its speed limits, and records transcripts
and token usage as on any embodiment. Single-arm tasks leave the other arm's
dimensions untouched, which is the "wire both arms, start single-arm" plan.
The exact Astra model id gets confirmed against the API at implementation time
(presumptive `gpt-6-astra`); proxies work via `-P base_url` + `-P api_key_env`.
v1 evaluation flow is an attended run with the default operator grader; task
registries and datasets stay out of scope.

## Part B: view task library

`inspect-robots view LOG_DIR` keeps its interface and gains a two-level
rendering on top of the existing per-run page (`_html.py`), with a new core
module `_library.py` for grouping and the new pages. Zero new dependencies,
still self-contained HTML plus `--serve` live refresh.

- **Library page** (the directory index): left sidebar of tasks, each entry
  showing the task title (registered name or instruction-derived slug),
  instruction snippet, and rollout count, with a client-side search filter.
  Filter chips per policy label (from log metadata). Aggregate score/success
  rate per task in the header. Newest-first within tasks.
- **Task page**: header (title, full instruction, score/success rate), rollout
  tabs, each tab hosting the existing per-trial player (composite MP4 with
  flipbook fallback, cameras labeled by their spec names), and the log meta
  line (score, seed, steps).
- **Decision card** (per-run player upgrade): synchronized with the playback
  position via the composite encoder's emitted-frame step timeline, showing
  decision index, control step index, elapsed time, the decision rationale
  from the policy transcript, and command details computed from the trial's
  `actions/` JSONL side-car: per-arm xyz deltas in cm, gripper opening, and
  executed-versus-requested steps. Reuses the shared transcript predicates and
  escape-once interpolation from `_html.py`.
- Out of scope for v1: same-seed side-by-side comparison (the seed is already
  in the log, the UX is deferred), multi-collection tabs (two libraries are
  two directories), and any hosted/server-backed deployment.

Part B is core: the 100% coverage gate, ruff, and mypy strict apply in full.

## Tests

Part A (plugin tests, stub-driven):

- Spaces and construction validation: label/bound/serial errors with fix hints;
  conformance via `check_embodiment`.
- `_kinematics`: FK-IK round trip on reachable poses generated by FK,
  rot6d-matrix round trip, unreachable pose and iteration-cap faults, TCP
  transform placement.
- Full loop against stub arm/gripper/camera: reset home sequence, multi-step
  `step()` issuing expected `move_js` deltas and gripper widths, stale-camera
  rejection, close-time disable, fault propagation.
- `_camera` thread: latest-frame keep, stamp monotonicity, stale timeout
  (clock injected).

Part B (core tests at 100%):

- `_library` grouping: logs across instructions and policy labels aggregate to
  tasks; slug stability; search/filter projections; success-rate arithmetic
  including errored trials.
- Task page and library page render: escape-once guarantees on instruction and
  rationale text, budget behavior, rollout tab ordering.
- Decision card: side-car parsing, delta computation (cm units, gripper
  opening), transcript alignment to the emitted-frame step timeline, missing
  side-car/transcript degradation.
- `view LOG_DIR` CLI paths updated for the new default outputs, including
  `--serve` rerender cadence unchanged.

## Milestones

Each milestone is independently verifiable, TDD throughout:

1. Plugin skeleton: spaces, factory, validation, conformance (no hardware, no
   IK).
2. `_kinematics.py`: URDF load, FK/IK, rot6d, unit tests only.
3. Hardware smoke on one arm: enable, home, guarded small `move_js` steps
   (bench, e-stop at hand, no load).
4. Cameras in: three D405 streams, observation assembly, staleness behavior.
5. End to end single-arm Astra task, attended, guardrails on.
6. Dual-arm tasks unlocked.
7. Part B library page and task page (independent of A: it develops against
   logs produced by any embodiment, including the mock; starts in parallel).
8. Part B decision card and command details; docs and CHANGELOG for both
   parts.

## Docs

- Plugin README (safety, bring-up, wiring table with the reused config
  values, quickstart); `docs/` gains a short nero guide page.
- `docs/guide/cli.md` view section: task library description.
- `CHANGELOG.md` under `## [Unreleased]`: plugin entry (`**Plugins:**`) and
  view library entry (`**Core:**`).

## Out of scope

- neo_manipulation imports or a ROS 2 runtime requirement for the plugin; the
  recorder/online-visualisation stack stays with the neo stack.
- Depth channels, fisheye cameras, and cross-camera alignment in the plugin.
- Registered nero tasks or datasets, VLA (non-agent) policy tuning, and
  automated scorers beyond the operator grader.
- Same-seed comparison UI, multi-collection tabs, and remote hosting in view.
