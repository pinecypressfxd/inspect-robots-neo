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
with this remedy. The URDF (`assets/dual_nero_pika.urdf`) ships with the
package, copied from the neo_manipulation repository (kinematics only; no mesh
assets are required).

## Quickstart (attended, single arm)

    export OPENAI_API_KEY=...
    inspect-robots "put the cup on the pad with left arm" \
      --policy agent -P model=openai/gpt-6-astra -P max_speed_frac=0.05 \
      --embodiment nero

The Astra model id above was current as of September 2026; confirm with
`curl -s https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY" | grep -i astra`
and record the chosen id here. Proxied endpoints work with
`-P base_url=... -P api_key_env=NAME`.

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

## Camera overrides

    -E cameras=left_rgbd=/dev/v4l/by-path/pci-0000:80:14.0-usb-0:11.2:1.0-video-index4

Cross-camera frame alignment is not reproduced in this adapter; v1 accepts the
millisecond-scale skew between the three streams (the bring-up stack's
alignment gate is not part of this plugin). Frames older than
`-E camera_max_age_s` (default 0.5) are rejected loudly.
