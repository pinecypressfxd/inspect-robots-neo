"""Operator-driven bench smoke for one Nero arm (spec milestone 3).

Run on the bench with the vendor SDK installed and the CAN link up, e.g.
``python plugins/inspect-robots-nero/scripts/bench_smoke.py --side left``.
Connects one arm, enables it at the default speed, enters follower mode,
homes it, then walks a few operator-confirmed 0.01 rad joint increments
while printing CAN feedback, and disables the arm on exit and on Ctrl-C.
This is a hardware-only path: it has no automated test by design.
"""

from __future__ import annotations

import argparse
import time

from inspect_robots_nero._arm import NeroArm
from inspect_robots_nero._config import (
    CAN_CHANNELS,
    HOME_LEFT,
    HOME_RIGHT,
    SPEED_PERCENT,
)

STEP_RAD = 0.01
FEEDBACK_SETTLE_S = 1.0


def _parse_args() -> argparse.Namespace:
    """Read side, optional channel override, and step count from the command line."""
    known = ", ".join(f"{side}={CAN_CHANNELS[side]}" for side in sorted(CAN_CHANNELS))
    parser = argparse.ArgumentParser(description="Nero single-arm bench smoke")
    parser.add_argument("--side", choices=sorted(CAN_CHANNELS), default="left")
    parser.add_argument(
        "--channel",
        default=None,
        help=f"CAN channel; defaults to the side's channel ({known})",
    )
    parser.add_argument("--steps", type=int, default=3, help="confirmed incremental moves")
    return parser.parse_args()


def _fmt(joints: tuple[float, ...]) -> str:
    """Format one joint-angle vector as a compact feedback line."""
    return " ".join(f"{value:+.3f}" for value in joints)


def _confirm(prompt: str) -> bool:
    """Ask the operator to confirm; Enter accepts, anything in n/no/q/quit or EOF stops."""
    try:
        answer = input(f"{prompt} [Enter = go, n = stop]: ").strip().lower()
    except EOFError:
        return False
    return answer not in ("n", "no", "q", "quit")


def main() -> int:
    """Enable, home, and nudge one Nero arm through operator-confirmed increments."""
    args = _parse_args()
    channel = args.channel or CAN_CHANNELS[args.side]
    home = HOME_LEFT if args.side == "left" else HOME_RIGHT
    print(f"[smoke] side={args.side} channel={channel} speed={SPEED_PERCENT}% steps={args.steps}")
    print(f"[smoke] home target (rad): {_fmt(home)}")
    if not _confirm(f"[smoke] about to ENABLE and HOME the {args.side} arm"):
        print("[smoke] operator stopped before enable")
        return 1
    arm = NeroArm(side=args.side, channel=channel)
    arm.connect()
    try:
        if not arm.enable():
            print("[smoke] ERROR: enable() failed; refusing to move")
            return 1
        arm.set_speed_percent(SPEED_PERCENT)
        arm.set_normal_mode()
        print("[smoke] enabled; streaming home target")
        arm.move_js(home)
        for step in range(1, args.steps + 1):
            before = arm.read_state()
            if before is None:
                print("[smoke] ERROR: no joint feedback on CAN; aborting")
                return 1
            print(f"[smoke] step {step}/{args.steps} measured (rad): {_fmt(before)}")
            if not _confirm(f"[smoke] move every joint +{STEP_RAD} rad"):
                print("[smoke] operator stopped; done")
                return 0
            target = tuple(value + STEP_RAD for value in before)
            arm.move_js(target)
            time.sleep(FEEDBACK_SETTLE_S)
            after = arm.read_state()
            if after is None:
                print("[smoke] ERROR: feedback lost after the move; aborting")
                return 1
            error = max(abs(a - t) for a, t in zip(after, target, strict=True))
            print(f"[smoke] step {step}/{args.steps} feedback (rad): {_fmt(after)}")
            print(f"[smoke] step {step}/{args.steps} max |commanded-measured| = {error:.3f} rad")
        print("[smoke] all steps done")
        return 0
    except KeyboardInterrupt:
        print("\n[smoke] interrupted (Ctrl-C)")
        return 130
    finally:
        print("[smoke] disabling and disconnecting")
        arm.disable()
        arm.close()


if __name__ == "__main__":
    raise SystemExit(main())
