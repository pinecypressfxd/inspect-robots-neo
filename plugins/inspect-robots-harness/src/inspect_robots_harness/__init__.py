"""HarnessVLA policy plugin.

Registers ``harness`` on ``inspect_robots.policies``. The policy implements
the core ``Policy`` contract by extending the vla plugin's hybrid executor:
analytic primitives and the VLA client stay on that interface, and this
package adds the fixed primitive library, trial memory, and restage-before-retry
rule.
"""

from inspect_robots_harness.policy import HarnessVlaPolicy, harness_policy

__all__ = ["HarnessVlaPolicy", "harness_policy"]
