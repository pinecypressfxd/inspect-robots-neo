"""Operator CLI for the Nero arms: home after power-on, or drop limp fast.

``home`` ports the bring-up checkout's ``test_move_j.py --stage move_j``:
enable, firmware normal mode, ``move_j`` to the config home poses, wait.
``disable`` ports ``disable_arms.py``: without ``--yes`` it only warns and
exits 2; with it, it disables and polls the per-joint enable flags until
every joint reports limp or ``--timeout`` expires (exit 1 on timeout).
"""

from __future__ import annotations

import argparse
import logging
import sys

from inspect_robots_nero._tools import build_arms, connect_all, disable_arms, home_arms

ARM_CHOICES = ("left", "right", "both")

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser shared by both subcommands."""
    parser = argparse.ArgumentParser(description="Nero arm operator tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    home = subparsers.add_parser("home", help="enable arms and move_j them to home")
    home.add_argument("--arm", choices=ARM_CHOICES, default="both")
    home.add_argument(
        "--wait", type=float, default=2.0, help="seconds to hold after the move_j (default 2.0)"
    )

    disable = subparsers.add_parser("disable", help="disable arms until all joints report limp")
    disable.add_argument("--arm", choices=ARM_CHOICES, default="both")
    disable.add_argument("--timeout", type=float, default=5.0)
    disable.add_argument("--yes", action="store_true")
    return parser


def _sides(choice: str) -> list[str]:
    """Expand an --arm choice into side names."""
    return ["left", "right"] if choice == "both" else [choice]


def main(argv: list[str] | None = None) -> int:
    """Run one operator command; return a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    args = build_parser().parse_args(argv)
    arms = build_arms(_sides(args.arm))
    connect_all(arms)
    try:
        if args.command == "home":
            logger.info("homing %s arm(s) with firmware move_j", args.arm)
            home_arms(arms, wait_s=args.wait)
            return 0
        if not args.yes:
            logger.warning(
                "Disabling %s arm(s). Arms will go limp. Re-run with --yes to confirm.",
                args.arm,
            )
            return 2
        logger.info("disabling %s arm(s) until all joints report limp", args.arm)
        return 0 if disable_arms(arms, timeout_s=args.timeout) else 1
    finally:
        for arm in reversed(list(arms.values())):
            try:
                arm.close()
            except Exception:
                logger.exception("[%s] failed to disconnect arm", arm.side)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
