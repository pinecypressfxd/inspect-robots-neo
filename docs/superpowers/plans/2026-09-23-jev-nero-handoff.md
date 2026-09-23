# Jev × Nero 双臂交接插件实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 inspect-robots 仓库新建 `plugins/inspect-robots-jev` 插件，把 jev-robot-control 的两段式决策协议移植为策略 `jev`，驱动 `NeroEmbodiment` 完成真机双臂交接任务。

**Architecture:** 策略每个控制周期做两次模型调用（1 个全局 phase 选择 + 8 个电机通道选择）；状态组装层把 `eef_state` 与示教声明的任务目标合并为 jev 风格结构化 state；执行器把方向选择换算为 20 维绝对 EE 位姿并施加 phase 硬门禁，产出 H=1 `ActionChunk` 交回框架。`NeroEmbodiment` 与 jev-robot-control 仓库零改动。

**Tech Stack:** Python ≥3.10、numpy、pytest；`scripts/declare_targets.py` 运行时才需要 `inspect_robots_nero`（测试不依赖它）。

**Spec:** `docs/superpowers/specs/2026-09-23-jev-nero-handoff-design.md`（执行者必须同时阅读 spec 与本计划）

## Global Constraints

- 不修改 `plugins/inspect-robots-nero`、`src/inspect_robots`、jev-robot-control 仓库的任何文件。
- 20 维动作布局（与 `NeroEmbodiment` 一致）：`[left_xyz(0:3), left_rot6d(3:9), left_width(9), right_xyz(10:13), right_rot6d(13:19), right_width(19)]`，单位米。
- 夹爪命令：`open`→0.09 m、`close`→0.0 m、`hold`→上次命令值。
- 默认阈值（spec §3.5/§3.6，全部进 `_config.py`）：held = 命令宽度 ≤ 0.005 m 且 实际−命令 > 0.015 m；secured = held 连续 ≥ 0.45 s（按 control_hz 换算步数，30 Hz → 14 步）；对齐容差 XY ≤ 6 mm / Z ±4 mm，`over_place` XY ≤ 8 mm，`at_release_height` Z ±5 mm；安全高度 = `travel_z_m`。
- 步长缩放：该轴距参考目标 <12 mm → 2 mm，<25 mm → 4 mm，否则 `max_step_mm`（默认 18）。
- 门禁被拒通道降级为 hold，不废整步，逐条记入 action `meta["rejections"]`。
- 模型调用/校验失败：`act()` 抛错结束 trial；无重试、无备用模型。
- 每个任务 TDD：先写失败测试 → 跑红 → 最小实现 → 跑绿 → 提交。测试命令从仓库根运行：`uv run pytest plugins/inspect-robots-jev/tests/<file> -v`。
- 提交信息 conventional commits，scope `jev`。

---

### Task 1: 插件脚手架、配置默认值与注册

**Files:**
- Create: `plugins/inspect-robots-jev/pyproject.toml`
- Create: `plugins/inspect-robots-jev/src/inspect_robots_jev/__init__.py`
- Create: `plugins/inspect-robots-jev/src/inspect_robots_jev/_config.py`
- Create: `plugins/inspect-robots-jev/tests/__init__.py`
- Test: `plugins/inspect-robots-jev/tests/test_registration.py`

**Interfaces:**
- Produces: 包 `inspect_robots_jev`；工厂 `jev_policy(**kwargs) -> JevHandoffPolicy`（Task 6 实现类，本任务先注册名字并 `raise NotImplementedError` 占位会破坏导入——因此本任务的 `jev_policy` 直接返回 None 也注册成功，但 Task 6 会替换实现。为避免中间态，本任务 `__init__.py` 只做注册与延迟导入，类在 Task 6 落地后工厂即刻可用）；`_config.py` 全部常量（后续任务按名引用，本任务定稿）。

- [ ] **Step 1: 写失败测试**

```python
"""plugins/inspect-robots-jev/tests/test_registration.py"""
from inspect_robots.registry import registered


def test_jev_policy_registered_on_import():
    import inspect_robots_jev  # noqa: F401

    factories = registered("policy")
    assert "jev" in factories
    assert callable(factories["jev"])


def test_config_defaults_match_spec():
    from inspect_robots_jev import _config as c

    assert c.GRIPPER_OPEN_M == 0.09 and c.GRIPPER_CLOSED_M == 0.0
    assert c.CLOSED_CMD_M == 0.005 and c.HELD_MARGIN_M == 0.015
    assert c.SECURED_S == 0.45
    assert c.XY_TOL_M == 0.006 and c.Z_TOL_M == 0.004
    assert c.OVER_PLACE_XY_TOL_M == 0.008 and c.RELEASE_Z_TOL_M == 0.005
    assert (c.NEAR_MM, c.MID_MM, c.FINE_STEP_MM, c.MID_STEP_MM) == (12.0, 25.0, 2.0, 4.0)
    assert c.MAX_STEP_MM == 18.0 and c.MIN_SEPARATION_M == 0.08
    assert c.STALL_STEPS == 20 and c.TIMEOUT_S == 90.0
    assert c.DEFAULT_MODEL == "typesafe/jev-1.13"
    assert c.API_KEY_ENV == "OPENROUTER_API_KEY"
```

- [ ] **Step 2: 跑红**

Run: `uv run pytest plugins/inspect-robots-jev/tests/test_registration.py -v`
Expected: FAIL（`ModuleNotFoundError: inspect_robots_jev`）

- [ ] **Step 3: 最小实现**

`pyproject.toml`（镜像 nero 插件结构，去掉硬件依赖）：

```toml
[build-system]
requires = ["hatchling>=1.18"]
build-backend = "hatchling.build"

[project]
name = "inspect-robots-jev"
version = "0.1.0"
description = "Jev two-stage decision policy for Inspect Robots: global handoff phase + per-arm motor channels, driving absolute-pose embodiments such as nero."
readme = "README.md"
requires-python = ">=3.10"
license = "MIT"
dependencies = ["inspect-robots>=0.2", "numpy>=1.24"]

[project.optional-dependencies]
dev = ["pytest>=8.0", "mypy>=1.11", "ruff>=0.6"]

# Entry-point discovery: an installed inspect-robots-jev appears in
# `inspect-robots list policies` without being imported first.
[project.entry-points."inspect_robots.policies"]
jev = "inspect_robots_jev:jev_policy"

[tool.uv.sources]
inspect-robots = { workspace = true }

[tool.hatch.build.targets.wheel]
packages = ["src/inspect_robots_jev"]

[tool.ruff]
extend = "../../pyproject.toml"

[tool.ruff.lint.isort]
known-first-party = ["inspect_robots", "inspect_robots_jev"]

[tool.mypy]
strict = true
```

`src/inspect_robots_jev/_config.py`：

```python
"""Defaults for the jev handoff policy; every value is a constructor override.

Mirrors the approved spec (docs/superpowers/specs/2026-09-23-jev-nero-handoff-design.md
§3.3/§3.5/§3.6/§4) and the constants jev-robot-control used in simulation.
"""
from __future__ import annotations

# Provider defaults (spec §3.3).
DEFAULT_MODEL = "typesafe/jev-1.13"
JEV_URL = "https://openrouter.ai/api/alpha/decisions"
LLM_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEY_ENV = "OPENROUTER_API_KEY"
TIMEOUT_S = 90.0

# Gripper commands in metres (nero pika width range is 0 to 0.09).
GRIPPER_OPEN_M = 0.09
GRIPPER_CLOSED_M = 0.0

# Held inference (spec §3.5): commanded near-closed while the reported width
# stays wider than commanded by more than the margin -> something is inside.
CLOSED_CMD_M = 0.005
HELD_MARGIN_M = 0.015
SECURED_S = 0.45  # held this long -> secured (steps = round(SECURED_S * control_hz))

# Alignment tolerances (spec §3.5), metres.
XY_TOL_M = 0.006
Z_TOL_M = 0.004
OVER_PLACE_XY_TOL_M = 0.008
RELEASE_Z_TOL_M = 0.005

# Increment scaling (spec §3.6, ported from the simulator's 18/4/2 mm tiers).
NEAR_MM = 12.0
MID_MM = 25.0
FINE_STEP_MM = 2.0
MID_STEP_MM = 4.0
MAX_STEP_MM = 18.0

# Executor guards.
MIN_SEPARATION_M = 0.08
STALL_STEPS = 20

# Phases where the two TCPs are expected to come close (collision guard exempt).
HANDOFF_PHASES = frozenset({"offer", "handoff_grasp", "handoff_release"})

AXES = ("x", "y", "z")
CHANNELS = (
    "left_x", "left_y", "left_z", "left_gripper",
    "right_x", "right_y", "right_z", "right_gripper",
)
PHASES_ORDER = (
    "approach_left", "grasp", "lift", "offer", "handoff_grasp",
    "handoff_release", "carry", "lower", "release", "withdraw", "finish",
)
```

`src/inspect_robots_jev/__init__.py`（本任务先注册工厂名，类 Task 6 落地）：

```python
"""Jev two-stage handoff policy plugin for Inspect Robots."""
from __future__ import annotations

from typing import Any

from inspect_robots.registry import register


@register("policy", "jev")
def jev_policy(**kwargs: Any) -> Any:  # pragma: no cover - replaced in Task 6
    """Registry-facing factory; the policy class lands with the wiring task."""
    from inspect_robots_jev.policy import JevHandoffPolicy

    return JevHandoffPolicy(**kwargs)


__all__ = ["jev_policy", "JevHandoffPolicy"]
```

`tests/__init__.py`：空文件。

注意：`jev_policy` 的延迟 import 在 Task 6 之前调用会 `ImportError`——注册测试只断言 `callable`，不调用它。

- [ ] **Step 4: 跑绿**

Run: `uv run pytest plugins/inspect-robots-jev/tests/test_registration.py -v`
Expected: 2 PASS。若 `ModuleNotFoundError` 仍在：先 `uv sync`（workspace `plugins/*` 自动纳入），再跑。

- [ ] **Step 5: 验证 entry point 发现**

Run: `uv sync && uv run inspect-robots list policies`
Expected: 输出包含 `jev`

- [ ] **Step 6: 提交**

```bash
git add plugins/inspect-robots-jev/pyproject.toml plugins/inspect-robots-jev/src plugins/inspect-robots-jev/tests
git commit -m "feat(jev): plugin scaffold, config defaults, policy registration"
```

---

### Task 2: phase 表与权限数据（`_phases.py`）

**Files:**
- Create: `plugins/inspect-robots-jev/src/inspect_robots_jev/_phases.py`
- Test: `plugins/inspect-robots-jev/tests/test_phases.py`

**Interfaces:**
- Produces:
  - `PhaseSpec` frozen dataclass：`condition: str`（choice criteria 文案）、`allowed_axes: Mapping[str, frozenset[str]]`（每臂可动 XYZ 轴，其余轴强制 hold）、`held_open_ok: Mapping[str, bool]`（持物臂是否允许 open）、`reference: str`（步长参考：`left_grasp|handoff_left|handoff_right|place|travel|current`）、`role: str`（两臂角色文案）。
  - `PHASES: dict[str, PhaseSpec]`——11 个 phase，键序与 `_config.PHASES_ORDER` 一致。
  - `PHASE_INSTRUCTIONS: str`、`MOTOR_INSTRUCTIONS: str`（提示词，双臂版本）。
- Consumes: `_config.PHASES_ORDER`。

- [ ] **Step 1: 写失败测试**

```python
"""plugins/inspect-robots-jev/tests/test_phases.py"""
from inspect_robots_jev import _config as c
from inspect_robots_jev._phases import MOTOR_INSTRUCTIONS, PHASES, PHASE_INSTRUCTIONS, PhaseSpec


def test_eleven_phases_in_spec_order():
    assert tuple(PHASES) == c.PHASES_ORDER
    assert all(isinstance(p, PhaseSpec) for p in PHASES.values())


def test_gate_permissions_match_spec():
    xyz = frozenset({"x", "y", "z"})
    # spec §3.6 gates 1-4, expressed as per-phase axis permissions.
    assert PHASES["grasp"].allowed_axes == {"left": frozenset(), "right": frozenset()}
    assert PHASES["release"].allowed_axes == {"left": frozenset(), "right": frozenset()}
    assert PHASES["handoff_release"].allowed_axes == {"left": frozenset(), "right": frozenset()}
    assert PHASES["lift"].allowed_axes == {"left": frozenset({"z"}), "right": frozenset()}
    assert PHASES["lower"].allowed_axes == {"left": frozenset(), "right": frozenset({"z"})}
    assert PHASES["withdraw"].allowed_axes == {"left": frozenset(), "right": frozenset({"z"})}
    assert PHASES["handoff_grasp"].allowed_axes == {"left": frozenset(), "right": xyz}
    assert PHASES["approach_left"].allowed_axes["left"] == xyz
    assert PHASES["approach_left"].allowed_axes["right"] == frozenset()
    assert PHASES["carry"].allowed_axes == {"left": frozenset(), "right": xyz}
    assert PHASES["finish"].allowed_axes == {"left": frozenset(), "right": frozenset()}


def test_held_open_whitelist():
    # spec §3.6 gate 5: a held arm may open only in these phases.
    open_ok = {name for name, p in PHASES.items()
               if p.held_open_ok["left"] or p.held_open_ok["right"]}
    assert open_ok == {"handoff_release", "release", "withdraw", "finish"}


def test_references_and_instructions_present():
    assert PHASES["approach_left"].reference == "left_grasp"
    assert PHASES["offer"].reference == "handoff_left"
    assert PHASES["handoff_grasp"].reference == "handoff_right"
    assert PHASES["carry"].reference == "place"
    assert PHASES["lift"].reference == "travel"
    assert all(p.condition and p.role for p in PHASES.values())
    assert "millimetres" in PHASE_INSTRUCTIONS
    assert "left" in MOTOR_INSTRUCTIONS and "right" in MOTOR_INSTRUCTIONS
```

- [ ] **Step 2: 跑红**

Run: `uv run pytest plugins/inspect-robots-jev/tests/test_phases.py -v`
Expected: FAIL（`No module named ... _phases`）

- [ ] **Step 3: 最小实现**

```python
"""Global handoff phase table: conditions for the model, permissions for the executor.

One phase drives both arms (spec §3.1), so conflicting arm intents are
structurally impossible; the executor turns the same table into hard gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from inspect_robots_jev._config import PHASES_ORDER

_XYZ = frozenset({"x", "y", "z"})
_Z = frozenset({"z"})
_NONE = frozenset()


def _axes(left: frozenset[str], right: frozenset[str]) -> Mapping[str, frozenset[str]]:
    return MappingProxyType({"left": left, "right": right})


def _open(left: bool, right: bool) -> Mapping[str, bool]:
    return MappingProxyType({"left": left, "right": right})


@dataclass(frozen=True)
class PhaseSpec:
    condition: str
    allowed_axes: Mapping[str, frozenset[str]]
    held_open_ok: Mapping[str, bool]
    reference: str
    role: str


PHASES: dict[str, PhaseSpec] = {
    "approach_left": PhaseSpec(
        condition=(
            "Left arm not holding the object and the object still at its start pose. "
            "Open the left gripper and align X/Y with the declared left grasp pose; "
            "descend Z only after X/Y alignment. Right arm stays put."),
        allowed_axes=_axes(_XYZ, _NONE), held_open_ok=_open(False, False),
        reference="left_grasp",
        role="LEFT: open gripper, correct X/Y then Z toward left grasp pose. RIGHT: hold."),
    "grasp": PhaseSpec(
        condition=(
            "Left arm not holding but left_grasp_reached is TRUE, or the left gripper "
            "closed on the object without left_secured yet. Hold all XYZ and keep "
            "closing until left_secured becomes TRUE."),
        allowed_axes=_axes(_NONE, _NONE), held_open_ok=_open(False, False),
        reference="current",
        role="LEFT: HOLD XYZ, CLOSE until secured. RIGHT: hold."),
    "lift": PhaseSpec(
        condition=(
            "left_secured is TRUE and the left TCP is below travel height. Keep fingers "
            "closed and lift vertically before any sideways motion."),
        allowed_axes=_axes(_Z, _NONE), held_open_ok=_open(False, False),
        reference="travel",
        role="LEFT: HOLD X/Y, fingers CLOSED, raise Z to travel_z_mm. RIGHT: hold."),
    "offer": PhaseSpec(
        condition=(
            "left_secured is TRUE and the left TCP is at travel height but not yet at "
            "the declared handoff pose. Carry the object horizontally to above the "
            "handoff pose, then descend to it."),
        allowed_axes=_axes(_XYZ, _XYZ), held_open_ok=_open(False, False),
        reference="handoff_left",
        role="LEFT: fingers CLOSED, move to handoff_left pose (X/Y first, then Z down). "
             "RIGHT: may pre-position toward handoff_right pose with gripper OPEN."),
    "handoff_grasp": PhaseSpec(
        condition=(
            "Left arm holds the object at the handoff pose (at_offer_pose TRUE) and the "
            "right arm does not hold it. Right gripper open, align with handoff_right "
            "pose and close. The left arm must not move."),
        allowed_axes=_axes(_NONE, _XYZ), held_open_ok=_open(False, False),
        reference="handoff_right",
        role="LEFT: HOLD everything, keep fingers CLOSED. RIGHT: gripper OPEN, align then CLOSE."),
    "handoff_release": PhaseSpec(
        condition=(
            "left_held AND right_secured are both TRUE. The object now has two holders; "
            "the left gripper may open. Both arms hold XYZ still."),
        allowed_axes=_axes(_NONE, _NONE), held_open_ok=_open(True, False),
        reference="current",
        role="LEFT: HOLD XYZ, OPEN fingers (only safe once right_secured). RIGHT: HOLD XYZ, keep CLOSED."),
    "carry": PhaseSpec(
        condition=(
            "right_secured is TRUE, the object is released by the left arm, and the "
            "right TCP X/Y error to the place point exceeds tolerance. Carry "
            "horizontally at travel height."),
        allowed_axes=_axes(_NONE, _XYZ), held_open_ok=_open(False, False),
        reference="place",
        role="LEFT: hold (or retreat if instructed). RIGHT: fingers CLOSED, correct X/Y "
             "toward place, keep Z at travel_z_mm."),
    "lower": PhaseSpec(
        condition=(
            "right_secured is TRUE and over_place is TRUE and the right TCP is above "
            "release height. Lower vertically with fingers closed; never lower away "
            "from the place point."),
        allowed_axes=_axes(_NONE, _Z), held_open_ok=_open(False, False),
        reference="place",
        role="LEFT: hold. RIGHT: fingers CLOSED, correct X/Y if needed, decrease Z to release height."),
    "release": PhaseSpec(
        condition=(
            "right_secured is TRUE and over_place is TRUE and at_release_height is TRUE. "
            "Open the right fingers; hold XYZ."),
        allowed_axes=_axes(_NONE, _NONE), held_open_ok=_open(False, True),
        reference="current",
        role="LEFT: hold. RIGHT: HOLD XYZ, OPEN fingers."),
    "withdraw": PhaseSpec(
        condition=(
            "Object already released at the place point, right gripper open, right TCP "
            "below safe height. Raise the open gripper vertically without regrasping."),
        allowed_axes=_axes(_NONE, _Z), held_open_ok=_open(False, True),
        reference="travel",
        role="LEFT: hold. RIGHT: fingers OPEN, raise Z to travel height."),
    "finish": PhaseSpec(
        condition=(
            "Object stably released at the place point (released_at_place TRUE), both "
            "grippers open, both TCPs above safe height. The task is complete."),
        allowed_axes=_axes(_NONE, _NONE), held_open_ok=_open(True, True),
        reference="current",
        role="Both arms: HOLD XYZ, keep grippers OPEN."),
}
assert tuple(PHASES) == PHASES_ORDER

PHASE_INSTRUCTIONS = """Choose the next global PHASE for this dual-arm handoff task from the actual geometry and gripper state.
One object is moved from its start pose to the place point: the LEFT arm picks it up, offers it at the handoff pose, the RIGHT arm takes over at handoff, then carries and releases it at the place point.
The phases form one shared state machine driving BOTH arms; they are a feedback policy, not an obligation to stay in the previous phase.
Alignment fields are measured geometry against operator-declared poses, not previous commands. Held/secured flags come from gripper width inference: a commanded-closed gripper whose reported width stays wide is holding something.
The handoff is the critical moment: handoff_release requires left_held AND right_secured; opening the left gripper earlier drops the object.
Relative errors are declared target MINUS current TCP, in millimetres. Conditions use absolute magnitudes.
No teleportation or scripted skill exists. Each update executes only small Cartesian increments per arm."""

MOTOR_INSTRUCTIONS = """Choose the next small motor increment per arm for the given phase and current geometry.
Channel labels refer to the robot base frame: positive increases that coordinate, negative decreases it, hold leaves it unchanged.
Read the signed target-minus-TCP errors for the arm this channel belongs to. A positive error needs positive motion; a negative error needs negative motion. Within 1 mm of that arm's current phase target, hold the axis. For travel and placement the tolerance is 3 mm.
Phase roles bind both arms:
""" + "\n".join(f"- {name.upper()}: {spec.role}" for name, spec in PHASES.items()) + """
The motion must follow the selected phase's role. Do not keep descending once the grasp or release height is reached.
Return the actual probability distribution for this motor channel."""
```

- [ ] **Step 4: 跑绿**

Run: `uv run pytest plugins/inspect-robots-jev/tests/test_phases.py -v`
Expected: 4 PASS

- [ ] **Step 5: 提交**

```bash
git add plugins/inspect-robots-jev/src/inspect_robots_jev/_phases.py plugins/inspect-robots-jev/tests/test_phases.py
git commit -m "feat(jev): global handoff phase table with per-arm permissions"
```

---

### Task 3: 决策客户端（`_decisions.py`，`validate_choice` 移植 + 双 provider）

**Files:**
- Create: `plugins/inspect-robots-jev/src/inspect_robots_jev/_decisions.py`
- Test: `plugins/inspect-robots-jev/tests/test_decisions.py`

**Interfaces:**
- Consumes: `_phases.PHASES/PHASE_INSTRUCTIONS/MOTOR_INSTRUCTIONS`、`_config.*`。
- Produces:
  - `validate_choice(answer: dict, options) -> dict`（移植 jev-robot-control `incremental_policy.validate_choice`，语义逐字一致）。
  - `Transport` 协议：`__call__(url: str, payload: dict, headers: dict, timeout: float) -> dict`（返回已解析 JSON）；默认 `_urlopen_transport`。
  - `DecisionsClient(provider="jev", model=None, base_url=None, api_key_env=API_KEY_ENV, timeout_s=TIMEOUT_S, transport=None)`：
    - `phase(state: dict) -> dict`（record）
    - `motor(state: dict, phase: str) -> dict`（record，8 问）
    - `ask(state, questions, stage) -> dict`
    - 属性 `records: list[dict]`、`calls: int`、`cost: float`、`wait_seconds: float`
  - record 结构：`{"stage", "provider", "model", "response_id", "answers": {name: {"choice", "probabilities"[, "confidence"]}}, "latency_seconds", "cost_usd", "usage", "observation", "questions"}`。

- [ ] **Step 1: 写失败测试**

```python
"""plugins/inspect-robots-jev/tests/test_decisions.py"""
import math

import pytest

from inspect_robots_jev._decisions import DecisionsClient, validate_choice


class FakeTransport:
    """Returns one scripted JSON response per call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, url, payload, headers, timeout):
        self.requests.append({"url": url, "payload": payload, "headers": headers, "timeout": timeout})
        return self.responses.pop(0)


def _jev_answers(name, choice, options):
    probs = {o: (1.0 if o == choice else 0.0) for o in options}
    return {name: {"choice": choice, "probabilities": probs}}


class TestValidateChoice:
    def test_accepts_valid_answer_with_confidence(self):
        answer = {"choice": "hold", "probabilities": {"negative": 0.1, "hold": 0.8, "positive": 0.1},
                  "confidence": 0.7}
        result = validate_choice(answer, ("negative", "hold", "positive"))
        assert result["choice"] == "hold" and result["confidence"] == 0.7

    def test_rejects_missing_label(self):
        answer = {"choice": "hold", "probabilities": {"hold": 1.0}}
        with pytest.raises(ValueError, match="labels"):
            validate_choice(answer, ("negative", "hold", "positive"))

    def test_rejects_unnormalized(self):
        answer = {"choice": "hold", "probabilities": {"negative": 0.5, "hold": 0.5, "positive": 0.5}}
        with pytest.raises(ValueError, match="probabilities"):
            validate_choice(answer, ("negative", "hold", "positive"))

    def test_rejects_choice_not_the_mode(self):
        answer = {"choice": "negative",
                  "probabilities": {"negative": 0.2, "hold": 0.6, "positive": 0.2}}
        with pytest.raises(ValueError, match="distribution"):
            validate_choice(answer, ("negative", "hold", "positive"))

    def test_rejects_nonfinite_probability(self):
        answer = {"choice": "hold",
                  "probabilities": {"negative": math.inf, "hold": 1.0, "positive": 0.0}}
        with pytest.raises(ValueError):
            validate_choice(answer, ("negative", "hold", "positive"))


def _phase_response(phase):
    return {"answers": _jev_answers("phase", phase, (
        "approach_left", "grasp", "lift", "offer", "handoff_grasp", "handoff_release",
        "carry", "lower", "release", "withdraw", "finish")),
        "id": "resp-1", "model": "typesafe/jev-1.13-20260917", "usage": {"cost": 0.0001}}


def _motor_response(choices):
    options_xyz = ("negative", "hold", "positive")
    options_grip = ("open", "hold", "close")
    answers = {}
    for channel, choice in choices.items():
        options = options_grip if channel.endswith("gripper") else options_xyz
        answers[channel] = {"choice": choice,
                            "probabilities": {o: (1.0 if o == choice else 0.0) for o in options}}
    return {"answers": answers, "id": "resp-2", "model": "typesafe/jev-1.13-20260917",
            "usage": {"cost": 0.0002}}


STATE = {
    "object": "cup",
    "left_tcp_mm": [320.0, -100.0, 200.0],
    "right_tcp_mm": [-320.0, 100.0, 200.0],
    "left_grasp_minus_left_tcp_mm": [10.0, 5.0, -30.0],
    "handoff_minus_right_tcp_mm": [200.0, 200.0, 0.0],
    "place_minus_right_tcp_mm": [300.0, 300.0, -100.0],
    "alignment": {"left_grasp_reached": False},
    "travel_z_mm": 200.0,
}


class TestDecisionsClientJev:
    def test_phase_then_motor_build_expected_payloads_and_records(self):
        transport = FakeTransport([
            _phase_response("approach_left"),
            _motor_response({"left_x": "positive", "left_y": "hold", "left_z": "hold",
                             "left_gripper": "open", "right_x": "hold", "right_y": "hold",
                             "right_z": "hold", "right_gripper": "hold"}),
        ])
        client = DecisionsClient(transport=transport)
        phase_record = client.phase(STATE)
        assert phase_record["answers"]["phase"]["choice"] == "approach_left"
        motor_record = client.motor(STATE, "approach_left")
        assert motor_record["answers"]["left_x"]["choice"] == "positive"

        assert client.calls == 2 and client.cost == pytest.approx(0.0003)
        client.reset()
        assert client.calls == 0 and client.records == []
        phase_request, motor_request = transport.requests
        assert phase_request["url"].endswith("/alpha/decisions")
        assert set(phase_request["payload"]["questions"]["phase"]["criteria"]) == {
            "approach_left", "grasp", "lift", "offer", "handoff_grasp", "handoff_release",
            "carry", "lower", "release", "withdraw", "finish"}
        assert set(motor_request["payload"]["questions"]) == {
            "left_x", "left_y", "left_z", "left_gripper",
            "right_x", "right_y", "right_z", "right_gripper"}
        # Per-axis instruction embeds that axis' numbers and the phase.
        assert "CURRENT PHASE: APPROACH_LEFT" in motor_request["payload"]["questions"]["left_x"]["instructions"]
        assert "left" in motor_request["payload"]["questions"]["left_x"]["instructions"]
        assert "right" in motor_request["payload"]["questions"]["right_x"]["instructions"]

    def test_missing_env_key_fails_fast(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        client = DecisionsClient(transport=FakeTransport([]))
        with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
            client.phase(STATE)

    def test_incomplete_answers_raise(self):
        transport = FakeTransport([{"answers": {}, "id": "r", "model": "m", "usage": {}}])
        client = DecisionsClient(transport=transport)
        with pytest.raises(ValueError, match="Incomplete"):
            client.phase(STATE)


class TestDecisionsClientLLM:
    def test_llm_payload_carries_strict_schema_and_parses_content(self):
        motor = _motor_response({"left_x": "hold", "left_y": "hold", "left_z": "hold",
                                 "left_gripper": "hold", "right_x": "hold",
                                 "right_y": "hold", "right_z": "hold",
                                 "right_gripper": "hold"})
        wrapped = {"choices": [{"message": {"content": __import__("json").dumps(motor)}}],
                   "id": "r", "model": "openai/gpt-4.1-mini", "usage": {"cost": 0.001}}
        transport = FakeTransport([_phase_response("grasp"), wrapped])
        client = DecisionsClient(provider="llm", model="openai/gpt-4.1-mini", transport=transport)
        client.phase(STATE)
        record = client.motor(STATE, "grasp")
        assert record["answers"]["left_gripper"]["choice"] == "hold"
        phase_request, motor_request = transport.requests
        assert motor_request["url"].endswith("/chat/completions")
        schema = motor_request["payload"]["response_format"]["json_schema"]["schema"]
        assert set(schema["properties"]["answers"]["properties"]) == set(
            motor["answers"])
        assert schema["properties"]["answers"]["properties"]["left_x"]["properties"][
            "choice"]["enum"] == ["negative", "hold", "positive"]
        assert motor_request["payload"]["max_tokens"] == 850
```

- [ ] **Step 2: 跑红**

Run: `uv run pytest plugins/inspect-robots-jev/tests/test_decisions.py -v`
Expected: FAIL（`No module named ... _decisions`）

- [ ] **Step 3: 最小实现**

```python
"""Two-stage decisions over the Jev endpoint or a plain chat model.

Ported from jev-robot-control incremental_policy.py: typed choice questions,
strict probability validation, one call for the global phase and one for the
eight per-arm motor channels.
"""
from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from collections.abc import Callable
from typing import Any, Protocol

from inspect_robots_jev import _config as c
from inspect_robots_jev._phases import MOTOR_INSTRUCTIONS, PHASES, PHASES_ORDER, PHASE_INSTRUCTIONS

XYZ_OPTIONS = ("negative", "hold", "positive")
GRIP_OPTIONS = ("open", "hold", "close")


def validate_choice(answer: dict[str, Any], options: tuple[str, ...] | list[str]) -> dict[str, Any]:
    """Port of jev-robot-control validate_choice, semantics unchanged."""
    probs = answer.get("probabilities", {})
    if set(probs) != set(options):
        raise ValueError("Unexpected model choice labels")
    if any(isinstance(p, bool) or not isinstance(p, (float, int)) or not math.isfinite(p)
           or not 0 <= p <= 1 for p in probs.values()):
        raise ValueError("Invalid model probabilities")
    if abs(sum(probs.values()) - 1) > 0.025:
        raise ValueError("Unnormalized model probabilities")
    choice = answer.get("choice")
    if choice not in options or probs[choice] + 1e-6 < max(probs.values()):
        raise ValueError("Choice does not match distribution")
    result: dict[str, Any] = {"choice": choice, "probabilities": probs}
    confidence = answer.get("confidence")
    if confidence is not None:
        if (not isinstance(confidence, (float, int)) or isinstance(confidence, bool)
                or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError("Invalid confidence")
        result["confidence"] = confidence
    return result


class Transport(Protocol):
    def __call__(self, url: str, payload: dict[str, Any], headers: dict[str, str],
                 timeout: float) -> dict[str, Any]: ...


def _urlopen_transport(url: str, payload: dict[str, Any], headers: dict[str, str],
                       timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url, json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = json.load(response)
    return raw if isinstance(raw, dict) else ValueError("provider response is not an object")


def _phase_question() -> dict[str, Any]:
    return {"type": "choice", "instructions": PHASE_INSTRUCTIONS,
            "criteria": {name: PHASES[name].condition for name in PHASES_ORDER}}


def _motor_questions(state: dict[str, Any], phase: str) -> dict[str, Any]:
    questions: dict[str, Any] = {}
    per_arm = {
        "left": {"tcp": state["left_tcp_mm"],
                 "grasp": state["left_grasp_minus_left_tcp_mm"],
                 "handoff": state["handoff_minus_right_tcp_mm"],  # owner: right; shown for context
                 "place": state["place_minus_right_tcp_mm"]},
        "right": {"tcp": state["right_tcp_mm"],
                  "grasp": state["left_grasp_minus_left_tcp_mm"],
                  "handoff": state["handoff_minus_right_tcp_mm"],
                  "place": state["place_minus_right_tcp_mm"]},
    }
    for arm in ("left", "right"):
        for i, axis in enumerate(c.AXES):
            channel = f"{arm}_{axis}"
            instruction = (
                MOTOR_INSTRUCTIONS
                + f"\nCURRENT PHASE: {phase.upper()}. This question controls ONLY the "
                f"{arm.upper()} arm's {axis.upper()} axis. "
                f"{arm.upper()} current {axis.upper()}={per_arm[arm]['tcp'][i]} mm. "
                f"{arm.upper()} grasp error on this axis={per_arm[arm]['grasp'][i]} mm. "
                f"{arm.upper()} handoff error on this axis={per_arm[arm]['handoff'][i]} mm. "
                f"{arm.upper()} place error on this axis={per_arm[arm]['place'][i]} mm. "
                "Use ONLY this axis and this arm; do not answer for another axis or arm.")
            questions[channel] = {
                "type": "choice", "instructions": instruction,
                "criteria": {
                    "negative": f"Move the {arm.upper()} arm {axis.upper()} toward a SMALLER coordinate.",
                    "hold": f"Do not move the {arm.upper()} arm on {axis.upper()}.",
                    "positive": f"Move the {arm.upper()} arm {axis.upper()} toward a LARGER coordinate."}}
        questions[f"{arm}_gripper"] = {
            "type": "choice",
            "instructions": MOTOR_INSTRUCTIONS + f" CURRENT PHASE: {phase.upper()}. "
                f"Decide only the {arm.upper()} gripper.",
            "criteria": {"open": f"Open the {arm.upper()} fingers.",
                         "hold": f"Keep the {arm.upper()} finger command.",
                         "close": f"Close the {arm.upper()} fingers."}}
    return questions


class DecisionsClient:
    """One phase question + eight motor questions per cycle, two providers."""

    def __init__(self, *, provider: str = "jev", model: str | None = None,
                 base_url: str | None = None, api_key_env: str = c.API_KEY_ENV,
                 timeout_s: float = c.TIMEOUT_S, transport: Transport | None = None) -> None:
        if provider not in ("jev", "llm"):
            raise ValueError(f"unknown provider {provider!r}; expected 'jev' or 'llm'")
        self.provider = provider
        self.model = model or c.DEFAULT_MODEL
        self.base_url = base_url or (c.JEV_URL if provider == "jev" else c.LLM_URL)
        self.api_key_env = api_key_env
        self.timeout_s = float(timeout_s)
        self._transport: Transport = transport or _urlopen_transport
        self.records: list[dict[str, Any]] = []
        self.calls = 0
        self.cost = 0.0
        self.wait_seconds = 0.0

    def reset(self) -> None:
        """Clear per-trial accumulators; the policy calls this on reset()."""
        self.records.clear()
        self.calls = 0
        self.cost = 0.0
        self.wait_seconds = 0.0

    def phase(self, state: dict[str, Any]) -> dict[str, Any]:
        return self.ask(state, {"phase": _phase_question()}, "phase")

    def motor(self, state: dict[str, Any], phase: str) -> dict[str, Any]:
        state = {**state, "selected_phase": phase}
        return self.ask(state, _motor_questions(state, phase), "motor")

    def ask(self, state: dict[str, Any], questions: dict[str, Any],
            stage: str) -> dict[str, Any]:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"{self.api_key_env} is not configured locally.")
        if self.provider == "jev":
            payload: dict[str, Any] = {"model": self.model, "state": state,
                                       "questions": questions}
        else:
            payload = {"model": self.model, "temperature": 0, "max_tokens": 850,
                       "response_format": {"type": "json_schema", "json_schema": {
                           "name": "robot_motor_decisions", "strict": True,
                           "schema": _answers_schema(questions)}},
                       "messages": [
                           {"role": "system", "content":
                            "Answer every typed choice question using its instructions and "
                            "criteria and the same supplied robot state. Return the specified "
                            "JSON answers. For each question, probabilities must include every "
                            "option, be finite numbers between 0 and 1, and sum to 1. Choose an "
                            "option with a highest probability. These are self-reported "
                            "assessments, not calibrated success probabilities. Do not invent "
                            "additional actions or call any external tools."},
                           {"role": "user",
                            "content": json.dumps({"state": state, "questions": questions})}]}
        started = time.perf_counter()
        raw = self._transport(self.base_url, payload,
                              {"Authorization": f"Bearer {key}",
                               "Content-Type": "application/json"}, self.timeout_s)
        latency = time.perf_counter() - started
        self.calls += 1
        self.wait_seconds += latency
        cost = raw.get("usage", {}).get("cost")
        if cost is not None and (not isinstance(cost, (int, float))
                                 or not math.isfinite(cost) or cost < 0):
            raise ValueError("Invalid provider cost")
        self.cost += cost or 0.0
        parsed = raw if self.provider == "jev" else json.loads(
            raw["choices"][0]["message"]["content"])
        if set(parsed.get("answers", {})) != set(questions):
            raise ValueError("Incomplete model response")
        answers = {name: validate_choice(parsed["answers"][name], q["criteria"])
                   for name, q in questions.items()}
        record = {"stage": stage, "provider": self.provider,
                  "model": raw.get("model", self.model), "response_id": raw.get("id"),
                  "answers": answers, "latency_seconds": round(latency, 4),
                  "cost_usd": cost, "usage": raw.get("usage", {}),
                  "observation": state, "questions": questions}
        self.records.append(record)
        return record


def _answers_schema(questions: dict[str, Any]) -> dict[str, Any]:
    answers_schema: dict[str, Any] = {}
    for name, question in questions.items():
        labels = list(question["criteria"])
        answers_schema[name] = {
            "type": "object",
            "properties": {
                "choice": {"type": "string", "enum": labels},
                "probabilities": {"type": "object",
                                  "properties": {k: {"type": "number"} for k in labels},
                                  "required": labels, "additionalProperties": False}},
            "required": ["choice", "probabilities"], "additionalProperties": False}
    return {"type": "object",
            "properties": {"answers": {"type": "object", "properties": answers_schema,
                                       "required": list(questions),
                                       "additionalProperties": False}},
            "required": ["answers"], "additionalProperties": False}
```

注意 `_urlopen_transport` 的最后一行是笔误防御说明——实际写为 `return raw`（若非 dict 由调用方 record 构建时报错）。实现时直接：

```python
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)
```

- [ ] **Step 4: 跑绿**

Run: `uv run pytest plugins/inspect-robots-jev/tests/test_decisions.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add plugins/inspect-robots-jev/src/inspect_robots_jev/_decisions.py plugins/inspect-robots-jev/tests/test_decisions.py
git commit -m "feat(jev): two-stage decisions client with jev/llm providers"
```

---

### Task 4: 目标载入与状态组装（`_state.py`）

**Files:**
- Create: `plugins/inspect-robots-jev/src/inspect_robots_jev/_state.py`
- Test: `plugins/inspect-robots-jev/tests/test_state.py`

**Interfaces:**
- Consumes: `_config.*`。
- Produces:
  - `Waypoint(tcp: tuple[float, float, float], rot6d: tuple[float, ...], width_m: float | None)` frozen dataclass。
  - `Targets(object_name: str, left_grasp: Waypoint, handoff_left: Waypoint, handoff_right: Waypoint, place: Waypoint, place_center: tuple[float, float, float], travel_z_m: float)` frozen dataclass。
  - `load_targets(path: str | Path) -> Targets`（缺失字段/非有限值/长度错误 → `ValueError`）。
  - `StateAssembler(targets, control_hz=30.0, secured_steps=None)`：
    - `assemble(observation, *, last_width_cmd: Mapping[str, float], feedback: str, operator_messages: list, previous_phase: str | None) -> dict`（state dict；内部推进 held/secured 步进计数与 `released_at_place`）
    - 属性 `held: {"left": bool, "right": bool}`、`secured: {"left": bool, "right": bool}`
    - `reset()` 清计数。

- [ ] **Step 1: 写失败测试**

```python
"""plugins/inspect-robots-jev/tests/test_state.py"""
import json

import numpy as np
import pytest

from inspect_robots_jev._state import StateAssembler, Targets, load_targets

TARGETS_DICT = {
    "object": "cup",
    "left_grasp": {"tcp_m": [0.30, -0.10, 0.15], "rot6d": [1, 0, 0, 0, 1, 0], "width_m": 0.05},
    "handoff_left": {"tcp_m": [0.10, 0.05, 0.20], "rot6d": [1, 0, 0, 0, 1, 0]},
    "handoff_right": {"tcp_m": [0.10, 0.05, 0.18], "rot6d": [1, 0, 0, 0, 1, 0], "width_m": 0.05},
    "place": {"tcp_m": [0.25, 0.25, 0.05], "rot6d": [1, 0, 0, 0, 1, 0],
              "center_m": [0.25, 0.25, 0.0]},
    "travel_z_m": 0.20,
}


@pytest.fixture
def targets_file(tmp_path):
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(TARGETS_DICT))
    return str(path)


def _eef(left_tcp, right_tcp, left_w=0.09, right_w=0.09):
    return np.array([*left_tcp, 1, 0, 0, 0, 1, 0, left_w,
                     *right_tcp, 1, 0, 0, 0, 1, 0, right_w], dtype=float)


def _observation(eef, instruction="hand off the cup"):
    from inspect_robots.types import Observation
    return Observation(state={"eef_state": eef, "joint_pos": np.zeros(16)},
                       instruction=instruction)


class TestLoadTargets:
    def test_round_trip(self, targets_file):
        targets = load_targets(targets_file)
        assert isinstance(targets, Targets)
        assert targets.left_grasp.tcp == (0.30, -0.10, 0.15)
        assert targets.left_grasp.width_m == 0.05
        assert targets.place_center == (0.25, 0.25, 0.0)
        assert targets.travel_z_m == 0.20

    def test_missing_field_rejected(self, tmp_path):
        broken = {k: v for k, v in TARGETS_DICT.items() if k != "place"}
        path = tmp_path / "broken.json"
        path.write_text(json.dumps(broken))
        with pytest.raises(ValueError, match="place"):
            load_targets(str(path))

    def test_nonfinite_rejected(self, tmp_path):
        bad = json.loads(json.dumps(TARGETS_DICT))
        bad["left_grasp"]["tcp_m"][0] = "oops"
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_targets(str(path))


class TestStateAssembly:
    def _assembler(self):
        return StateAssembler(load_targets_dict(), control_hz=30.0)

    def test_geometry_fields_and_alignment_flags(self, targets_file):
        assembler = StateAssembler(load_targets(targets_file), control_hz=30.0)
        state = assembler.assemble(
            _observation(_eef([0.305, -0.10, 0.20], [0.10, 0.05, 0.20])),
            last_width_cmd={"left": 0.09, "right": 0.09},
            feedback="Increment executed.", operator_messages=[], previous_phase=None)
        # target - current in mm, rounded to 0.1
        assert state["left_tcp_mm"] == [305.0, -100.0, 200.0]
        assert state["left_grasp_minus_left_tcp_mm"] == [-5.0, 0.0, -50.0]
        assert state["left_grasp_tcp_mm"] == [300.0, -100.0, 150.0]
        assert state["handoff_right_tcp_mm"] == [100.0, 50.0, 180.0]
        assert state["place_tcp_mm"] == [250.0, 250.0, 50.0]
        assert state["alignment"]["left_grasp_reached"] is False  # X off by 5mm > 6mm? no: 5<6 but Z 50mm off
        assert state["alignment"]["at_offer_pose"] is False
        assert state["alignment"]["right_at_travel_height"] is True  # 0.20 >= 0.20
        assert state["travel_z_mm"] == 200.0
        assert state["previous_phase"] is None
        assert state["feedback"] == "Increment executed."

    def test_grasp_reached_requires_xy_and_z(self, targets_file):
        assembler = StateAssembler(load_targets(targets_file), control_hz=30.0)
        near = _eef([0.302, -0.102, 0.152], [0.0, 0.0, 0.2])
        state = assembler.assemble(_observation(near), last_width_cmd={"left": 0.09, "right": 0.09},
                                   feedback="", operator_messages=[], previous_phase="approach_left")
        assert state["alignment"]["left_grasp_reached"] is True
        assert state["previous_phase"] == "approach_left"

    def test_held_inference_and_secured_steps(self, targets_file):
        assembler = StateAssembler(load_targets(targets_file), control_hz=30.0,
                                   secured_steps=3)
        obs = _observation(_eef([0.3, -0.1, 0.15], [0.0, 0.0, 0.2], left_w=0.05))
        state = assembler.assemble(obs, last_width_cmd={"left": 0.0, "right": 0.09},
                                   feedback="", operator_messages=[], previous_phase="grasp")
        assert state["left_held"] is True and state["left_secured"] is False
        for _ in range(2):
            state = assembler.assemble(obs, last_width_cmd={"left": 0.0, "right": 0.09},
                                       feedback="", operator_messages=[], previous_phase="grasp")
        assert state["left_secured"] is True
        assert state["right_held"] is False

    def test_open_gripper_not_held_and_operator_messages_merge(self, targets_file):
        assembler = StateAssembler(load_targets(targets_file), control_hz=30.0)
        state = assembler.assemble(
            _observation(_eef([0.3, -0.1, 0.15], [0.0, 0.0, 0.2], left_w=0.09)),
            last_width_cmd={"left": 0.09, "right": 0.09},
            feedback="hold", operator_messages=[{"t": 3, "text": "slow down", "source": "console"}],
            previous_phase="offer")
        assert state["left_held"] is False  # commanded open: not held
        assert "slow down" in state["feedback"]

    def test_released_at_place_after_release_and_retreat(self, targets_file):
        assembler = StateAssembler(load_targets(targets_file), control_hz=30.0)
        # right held at place, then opens and retreats above
        at_place = _observation(_eef([0.1, 0.05, 0.2], [0.25, 0.25, 0.05], right_w=0.05))
        assembler.assemble(at_place, last_width_cmd={"left": 0.09, "right": 0.0},
                           feedback="", operator_messages=[], previous_phase="lower")
        opened_retreated = _observation(_eef([0.1, 0.05, 0.2], [0.25, 0.25, 0.12], right_w=0.09))
        state = assembler.assemble(opened_retreated,
                                   last_width_cmd={"left": 0.09, "right": 0.09},
                                   feedback="", operator_messages=[], previous_phase="release")
        assert state["released_at_place"] is True
```

注：`test_geometry_fields_and_alignment_flags` 中 `load_targets_dict` 是笔误——用 `load_targets(targets_file)`（fixture 已在作用域）。`left_grasp_reached` 断言理由：X 差 5 mm ≤ 6 mm 但 Z 差 50 mm > 4 mm → False（代码注释里写明）。

- [ ] **Step 2: 跑红**

Run: `uv run pytest plugins/inspect-robots-jev/tests/test_state.py -v`
Expected: FAIL（`No module named ... _state`）

- [ ] **Step 3: 最小实现**

```python
"""Operator-declared targets -> per-step structured state for the decisions layer."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from inspect_robots.types import Observation

from inspect_robots_jev import _config as c

_WAYPOINTS = ("left_grasp", "handoff_left", "handoff_right", "place")


@dataclass(frozen=True)
class Waypoint:
    tcp: tuple[float, float, float]
    rot6d: tuple[float, ...]
    width_m: float | None = None


@dataclass(frozen=True)
class Targets:
    object_name: str
    left_grasp: Waypoint
    handoff_left: Waypoint
    handoff_right: Waypoint
    place: Waypoint
    place_center: tuple[float, float, float]
    travel_z_m: float


def _tcp3(value: Any, where: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or len(value) != 3 or not all(
            isinstance(v, (int, float)) and math.isfinite(v) for v in value):
        raise ValueError(f"targets {where}.tcp_m must hold three finite numbers")
    return (float(value[0]), float(value[1]), float(value[2]))


def load_targets(path: str | Path) -> Targets:
    data = json.loads(Path(path).read_text())
    missing = [key for key in (*_WAYPOINTS, "object", "place_center_missing_check") if key not in data]
    for key in _WAYPOINTS:
        if key not in data:
            raise ValueError(f"targets file is missing the {key!r} waypoint")
    if "travel_z_m" not in data:
        raise ValueError("targets file is missing 'travel_z_m'")
    waypoints = {}
    for key in _WAYPOINTS:
        entry = data[key]
        rot = entry.get("rot6d")
        if not isinstance(rot, Sequence) or len(rot) != 6 or not all(
                isinstance(v, (int, float)) and math.isfinite(v) for v in rot):
            raise ValueError(f"targets {key}.rot6d must hold six finite numbers")
        width = entry.get("width_m")
        if width is not None and (not isinstance(width, (int, float))
                                  or not math.isfinite(width)):
            raise ValueError(f"targets {key}.width_m must be a finite number or absent")
        waypoints[key] = Waypoint(_tcp3(entry["tcp_m"], key), tuple(float(v) for v in rot),
                                  None if width is None else float(width))
    center = data.get("place", {}).get("center_m", data.get("place_center"))
    if center is None:
        raise ValueError("targets place waypoint must carry center_m")
    travel_z = data["travel_z_m"]
    if not isinstance(travel_z, (int, float)) or not math.isfinite(travel_z) or travel_z <= 0:
        raise ValueError("targets travel_z_m must be a positive finite number")
    return Targets(object_name=str(data.get("object", "object")), **waypoints,
                   place_center=_tcp3(center, "place.center_m"), travel_z_m=float(travel_z))


class StateAssembler:
    """Fold eef_state and declared targets into the jev-style state dict.

    Held/secured are step-counter based (deterministic in tests): a held streak
    of `secured_steps` control steps counts as secured.
    """

    def __init__(self, targets: Targets, *, control_hz: float = 30.0,
                 secured_steps: int | None = None) -> None:
        self.targets = targets
        self.control_hz = float(control_hz)
        self._secured_steps = (round(c.SECURED_S * self.control_hz)
                               if secured_steps is None else int(secured_steps))
        self.reset()

    def reset(self) -> None:
        self._held_steps = {"left": 0, "right": 0}
        self._released_seen = False
        self._step = 0

    @property
    def held(self) -> dict[str, bool]:
        return {side: steps > 0 for side, steps in self._held_steps.items()}

    @property
    def secured(self) -> dict[str, bool]:
        return {side: steps >= self._secured_steps
                for side, steps in self._held_steps.items()}

    def _update_held(self, last_width_cmd: Mapping[str, float],
                     actual: Mapping[str, float]) -> None:
        for side in ("left", "right"):
            commanded = float(last_width_cmd.get(side, c.GRIPPER_OPEN_M))
            width = float(actual[side])
            is_held = commanded <= c.CLOSED_CMD_M and width - commanded > c.HELD_MARGIN_M
            self._held_steps[side] = self._held_steps[side] + 1 if is_held else 0

    def assemble(self, observation: Observation, *, last_width_cmd: Mapping[str, float],
                 feedback: str, operator_messages: list[dict[str, Any]],
                 previous_phase: str | None) -> dict[str, Any]:
        eef = np.asarray(observation.state["eef_state"], dtype=float)
        left_tcp, right_tcp = eef[0:3], eef[10:13]
        t = self.targets
        self._update_held(last_width_cmd, {"left": float(eef[9]), "right": float(eef[19])})
        left_held, right_held = self.held["left"], self.held["right"]
        # released_at_place: right gripper released on target and retreated.
        if (self._held_steps["right"] == 0 and not left_held
                and np.linalg.norm(right_tcp[:2] - np.asarray(t.place_center)[:2])
                <= c.OVER_PLACE_XY_TOL_M and right_tcp[2] > t.place.tcp[2] + 0.05):
            self._released_seen = True
        self._step += 1
        mm = lambda v: [round(float(x) * 1000.0, 1) for x in v]  # noqa: E731
        grasp_err = np.asarray(t.left_grasp.tcp) - left_tcp
        handoff_left_err = np.asarray(t.handoff_left.tcp) - left_tcp
        handoff_right_err = np.asarray(t.handoff_right.tcp) - right_tcp
        place_err = np.asarray(t.place.tcp) - right_tcp
        notes = [m["text"] for m in operator_messages if m.get("text")]
        feedback_text = feedback if not notes else feedback + " | operator: " + " / ".join(notes)
        return {
            "task": (f"Pick up the {t.object_name} with the left arm, hand it to the right "
                     f"arm at the handoff pose, and place it at the declared place point."),
            "source": "operator-declared targets + eef_state width inference; not camera vision",
            "units": "millimetres; robot base coordinates; +Z is up",
            "object": t.object_name,
            "left_tcp_mm": mm(left_tcp), "right_tcp_mm": mm(right_tcp),
            "left_grasp_tcp_mm": mm(t.left_grasp.tcp),
            "handoff_left_tcp_mm": mm(t.handoff_left.tcp),
            "handoff_right_tcp_mm": mm(t.handoff_right.tcp),
            "place_tcp_mm": mm(t.place.tcp),
            "left_grasp_minus_left_tcp_mm": mm(grasp_err),
            "handoff_left_minus_left_tcp_mm": mm(handoff_left_err),
            "handoff_minus_right_tcp_mm": mm(handoff_right_err),
            "place_minus_right_tcp_mm": mm(place_err),
            "alignment": {
                "left_grasp_reached": bool(
                    np.all(np.abs(grasp_err[:2]) <= c.XY_TOL_M)
                    and abs(grasp_err[2]) <= c.Z_TOL_M),
                "at_offer_pose": bool(
                    np.all(np.abs(handoff_left_err[:2]) <= c.XY_TOL_M)
                    and abs(handoff_left_err[2]) <= c.Z_TOL_M),
                "handoff_right_reached": bool(
                    np.all(np.abs(handoff_right_err[:2]) <= c.XY_TOL_M)
                    and abs(handoff_right_err[2]) <= c.Z_TOL_M),
                "over_place": bool(np.all(np.abs(right_tcp[:2]
                    - np.asarray(t.place_center)[:2]) <= c.OVER_PLACE_XY_TOL_M)),
                "left_at_travel_height": bool(left_tcp[2] >= t.travel_z_m - c.Z_TOL_M),
                "right_at_travel_height": bool(right_tcp[2] >= t.travel_z_m - c.Z_TOL_M),
                "at_release_height": bool(abs(place_err[2]) <= c.RELEASE_Z_TOL_M),
            },
            "travel_z_mm": round(t.travel_z_m * 1000.0, 1),
            "safe_height_mm": round(t.travel_z_m * 1000.0, 1),
            "left_gripper": {"command_m": round(float(last_width_cmd.get("left", c.GRIPPER_OPEN_M)), 4),
                             "actual_m": round(float(eef[9]), 4)},
            "right_gripper": {"command_m": round(float(last_width_cmd.get("right", c.GRIPPER_OPEN_M)), 4),
                              "actual_m": round(float(eef[19]), 4)},
            "left_held": left_held, "right_held": right_held,
            "left_secured": self.secured["left"], "right_secured": self.secured["right"],
            "held_by": [side for side, flag in (("left", left_held), ("right", right_held)) if flag],
            "released_at_place": self._released_seen,
            "previous_phase": previous_phase,
            "feedback": feedback_text,
            "step_index": self._step,
        }
```

实现说明（写代码时遵守）：
- `load_targets` 开头的 `missing` 行是笔误残留——实现时删掉该行，只保留逐 waypoint 的显式检查与 `travel_z_m` 检查（错误信息都要点名缺失字段）。
- `test_released_at_place_after_release_and_retreat` 里 `right_w=0.05 → 0.09` 且命令开：`_held_steps["right"]` 归零、`left_held` False、右 TCP 仍在 place 上方 → `_released_seen=True`。
- `lambda` 赋值仅此一处（局部简写）；ruff 若报 E731 就展开成局部函数 `_mm`。

- [ ] **Step 4: 跑绿**

Run: `uv run pytest plugins/inspect-robots-jev/tests/test_state.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add plugins/inspect-robots-jev/src/inspect_robots_jev/_state.py plugins/inspect-robots-jev/tests/test_state.py
git commit -m "feat(jev): operator-declared targets and dual-arm state assembly"
```

---

### Task 5: 执行器（`_executor.py`：方向→绝对位姿 + 7 条硬门禁）

**Files:**
- Create: `plugins/inspect-robots-jev/src/inspect_robots_jev/_executor.py`
- Test: `plugins/inspect-robots-jev/tests/test_executor.py`

**Interfaces:**
- Consumes: `_phases.PHASES/PhaseSpec`、`_config.*`、`_state.Targets`。
- Produces:
  - `class TaskStalled(RuntimeError)`
  - `Executor(box: Box, *, max_step_mm=c.MAX_STEP_MM, min_separation_m=c.MIN_SEPARATION_M, stall_steps=c.STALL_STEPS, dry_run=False)`
    - `build(*, phase: str, choices: Mapping[str, str], eef: np.ndarray, held: Mapping[str, bool], secured: Mapping[str, bool], targets: Targets) -> tuple[np.ndarray, dict]`（返回 20 维 action data 与 meta）
    - 属性 `last_widths: {"left": float, "right": float}`（初值 0.09）
    - `reset()` 清 stall 计数与 `last_widths`
  - meta 结构：`{"phase", "choices", "step_mm": {channel: float}, "rejections": [{"channel", "rule", "detail"}], "width_cmd_m": {"left","right"}, "dry_run": bool}`
  - 拒绝规则名（测试断言用）：`phase_axes`、`held_open`、`handoff_release_requires_right_secured`、`bounds`、`separation`、`dry_run`。

- [ ] **Step 1: 写失败测试**

```python
"""plugins/inspect-robots-jev/tests/test_executor.py"""
import numpy as np
import pytest

from inspect_robots.spaces import ActionSemantics, Box

from inspect_robots_jev._executor import Executor, TaskStalled
from inspect_robots_jev._state import Targets, Waypoint

LOW = np.array([-1.0] * 3 + [-1.0] * 6 + [0.0] + [-1.0] * 3 + [-1.0] * 6 + [0.0])
HIGH = np.array([1.0] * 3 + [1.0] * 6 + [0.09] + [1.0] * 3 + [1.0] * 6 + [0.09])
BOX = Box(shape=(20,), low=LOW, high=HIGH,
          semantics=ActionSemantics(control_mode="eef_abs_pose", rotation_repr="rot6d",
                                    gripper="continuous", frame="base"))

TARGETS = Targets(
    object_name="cup",
    left_grasp=Waypoint((0.30, -0.10, 0.15), (1, 0, 0, 0, 1, 0), 0.05),
    handoff_left=Waypoint((0.10, 0.05, 0.20), (1, 0, 0, 0, 1, 0)),
    handoff_right=Waypoint((0.10, 0.05, 0.18), (1, 0, 0, 0, 1, 0), 0.05),
    place=Waypoint((0.25, 0.25, 0.05), (1, 0, 0, 0, 1, 0)),
    place_center=(0.25, 0.25, 0.0), travel_z_m=0.20)

HELD_F = {"left": False, "right": False}
SEC_F = {"left": False, "right": False}


def _eef(left=(0.0, 0.0, 0.3), right=(0.5, 0.5, 0.3), lw=0.09, rw=0.09):
    return np.array([*left, 1, 0, 0, 0, 1, 0, lw, *right, 1, 0, 0, 0, 1, 0, rw], float)


def _choices(**over):
    base = {"left_x": "hold", "left_y": "hold", "left_z": "hold", "left_gripper": "hold",
            "right_x": "hold", "right_y": "hold", "right_z": "hold", "right_gripper": "hold"}
    base.update(over)
    return base


class TestScaling:
    def test_far_mid_near_tiers(self):
        ex = Executor(BOX)
        # left grasp x target 0.30; far -> 18 mm
        data, meta = ex.build(phase="approach_left", choices=_choices(left_x="positive"),
                              eef=_eef(left=(0.0, -0.10, 0.15)), held=HELD_F,
                              secured=SEC_F, targets=TARGETS)
        assert data[0] == pytest.approx(0.018)
        assert meta["step_mm"]["left_x"] == pytest.approx(18.0)
        # mid (20 mm away) -> 4 mm
        ex2 = Executor(BOX)
        data, _ = ex2.build(phase="approach_left", choices=_choices(left_x="positive"),
                            eef=_eef(left=(0.28, -0.10, 0.15)), held=HELD_F,
                            secured=SEC_F, targets=TARGETS)
        assert data[0] == pytest.approx(0.28 + 0.004)
        # near (5 mm away) -> 2 mm
        ex3 = Executor(BOX)
        data, _ = ex3.build(phase="approach_left", choices=_choices(left_x="positive"),
                            eef=_eef(left=(0.295, -0.10, 0.15)), held=HELD_F,
                            secured=SEC_F, targets=TARGETS)
        assert data[0] == pytest.approx(0.297)

    def test_max_step_mm_caps_far_tier(self):
        ex = Executor(BOX, max_step_mm=4.0)
        data, meta = ex.build(phase="approach_left", choices=_choices(left_x="positive"),
                              eef=_eef(left=(0.0, -0.10, 0.15)), held=HELD_F,
                              secured=SEC_F, targets=TARGETS)
        assert data[0] == pytest.approx(0.004)
        assert meta["step_mm"]["left_x"] == pytest.approx(4.0)

    def test_travel_reference_uses_current_xy(self):
        ex = Executor(BOX)
        data, _ = ex.build(phase="lift", choices=_choices(left_z="positive"),
                           eef=_eef(left=(0.33, -0.10, 0.15)), held=HELD_F,
                           secured=SEC_F, targets=TARGETS)
        # travel reference keeps X/Y; Z distance 50 mm -> far tier 18 mm
        assert data[2] == pytest.approx(0.15 + 0.018)
        assert data[0] == pytest.approx(0.33)


class TestGates:
    def test_grasp_forces_xyz_hold(self):
        ex = Executor(BOX)
        data, meta = ex.build(phase="grasp",
                              choices=_choices(left_x="positive", left_z="negative",
                                               left_gripper="close"),
                              eef=_eef(left=(0.30, -0.10, 0.15)), held=HELD_F,
                              secured=SEC_F, targets=TARGETS)
        assert data[0] == 0.30 and data[2] == 0.15  # unchanged
        assert data[9] == pytest.approx(0.0)  # close applied
        assert {r["channel"] for r in meta["rejections"]} == {"left_x", "left_z"}
        assert all(r["rule"] == "phase_axes" for r in meta["rejections"])

    def test_handoff_release_open_blocked_without_right_secured(self):
        ex = Executor(BOX)
        data, meta = ex.build(phase="handoff_release",
                              choices=_choices(left_gripper="open"),
                              eef=_eef(left=(0.10, 0.05, 0.20), right=(0.10, 0.05, 0.18),
                                       lw=0.05, rw=0.05),
                              held={"left": True, "right": True},
                              secured={"left": True, "right": False}, targets=TARGETS)
        assert data[9] == pytest.approx(0.0)  # still closed
        rejection = next(r for r in meta["rejections"] if r["channel"] == "left_gripper")
        assert rejection["rule"] == "handoff_release_requires_right_secured"

    def test_handoff_release_open_applied_with_right_secured(self):
        ex = Executor(BOX)
        data, meta = ex.build(phase="handoff_release",
                              choices=_choices(left_gripper="open"),
                              eef=_eef(left=(0.10, 0.05, 0.20), right=(0.10, 0.05, 0.18),
                                       lw=0.05, rw=0.05),
                              held={"left": True, "right": True},
                              secured={"left": True, "right": True}, targets=TARGETS)
        assert data[9] == pytest.approx(0.09)
        assert not any(r["channel"] == "left_gripper" for r in meta["rejections"])

    def test_held_open_rejected_outside_whitelist(self):
        ex = Executor(BOX)
        data, meta = ex.build(phase="carry", choices=_choices(right_gripper="open"),
                              eef=_eef(right=(0.25, 0.25, 0.20), rw=0.05),
                              held={"left": False, "right": True},
                              secured={"left": False, "right": True}, targets=TARGETS)
        assert data[19] == pytest.approx(0.0)
        assert meta["rejections"][0]["rule"] == "held_open"

    def test_bounds_reject_per_axis(self):
        ex = Executor(BOX, max_step_mm=18.0)
        data, meta = ex.build(phase="approach_left", choices=_choices(left_x="negative"),
                              eef=_eef(left=(-0.99, -0.10, 0.15)), held=HELD_F,
                              secured=SEC_F, targets=TARGETS)
        assert data[0] == pytest.approx(-0.99)  # candidate -1.008 < low -1.0 -> held
        assert meta["rejections"][0]["rule"] == "bounds"

    def test_separation_guard_active_outside_handoff_phases(self):
        ex = Executor(BOX)
        # left at 0.10, right at 0.175: 75 mm apart; left +x by 18 mm -> 57 mm < 80 mm
        data, meta = ex.build(phase="approach_left", choices=_choices(left_x="positive"),
                              eef=_eef(left=(0.10, 0.0, 0.3), right=(0.175, 0.0, 0.3)),
                              held=HELD_F, secured=SEC_F, targets=TARGETS)
        assert data[0] == pytest.approx(0.10)
        assert meta["rejections"][0]["rule"] == "separation"
        # same geometry inside a handoff phase is exempt (handoff_grasp: right free)
        ex2 = Executor(BOX)
        data, meta = ex2.build(phase="handoff_grasp", choices=_choices(right_x="negative"),
                               eef=_eef(left=(0.10, 0.0, 0.3), right=(0.175, 0.0, 0.3)),
                               held={"left": True, "right": False},
                               secured={"left": True, "right": False}, targets=TARGETS)
        assert data[10] == pytest.approx(0.175 - 0.018)
        assert not meta["rejections"]

    def test_gripper_hold_persists_last_command(self):
        ex = Executor(BOX)
        ex.build(phase="grasp", choices=_choices(left_gripper="close"), eef=_eef(),
                 held=HELD_F, secured=SEC_F, targets=TARGETS)
        data, _ = ex.build(phase="lift", choices=_choices(left_gripper="hold"), eef=_eef(),
                           held={"left": True, "right": False},
                           secured=SEC_F, targets=TARGETS)
        assert data[9] == pytest.approx(0.0)  # stays closed from last command


class TestDryRunAndStall:
    def test_dry_run_commands_full_hold(self):
        ex = Executor(BOX, dry_run=True)
        data, meta = ex.build(phase="approach_left",
                              choices=_choices(left_x="positive", left_gripper="close"),
                              eef=_eef(left=(0.0, -0.10, 0.15)), held=HELD_F,
                              secured=SEC_F, targets=TARGETS)
        assert data[0] == pytest.approx(0.0) and data[2] == pytest.approx(0.15)
        assert data[9] == pytest.approx(0.09) and data[19] == pytest.approx(0.09)
        assert meta["dry_run"] is True

    def test_stall_raises_after_stall_steps_of_no_motion(self):
        ex = Executor(BOX, stall_steps=3)
        eef = _eef(left=(0.0, -0.10, 0.15))
        for _ in range(3):
            ex.build(phase="approach_left", choices=_choices(), eef=eef, held=HELD_F,
                     secured=SEC_F, targets=TARGETS)
        with pytest.raises(TaskStalled):
            ex.build(phase="approach_left", choices=_choices(), eef=eef, held=HELD_F,
                     secured=SEC_F, targets=TARGETS)

    def test_dry_run_never_counts_stall(self):
        ex = Executor(BOX, dry_run=True, stall_steps=2)
        eef = _eef(left=(0.0, -0.10, 0.15))
        for _ in range(5):
            ex.build(phase="approach_left", choices=_choices(), eef=eef, held=HELD_F,
                     secured=SEC_F, targets=TARGETS)

    def test_reset_clears_widths_and_stall(self):
        ex = Executor(BOX, stall_steps=2)
        ex.build(phase="grasp", choices=_choices(left_gripper="close"), eef=_eef(),
                 held=HELD_F, secured=SEC_F, targets=TARGETS)
        ex.reset()
        assert ex.last_widths == {"left": 0.09, "right": 0.09}


class TestLayout:
    def test_rot6d_copied_and_indices(self):
        ex = Executor(BOX)
        eef = _eef(left=(0.30, -0.10, 0.15), right=(0.25, 0.25, 0.05))
        data, _ = ex.build(phase="finish", choices=_choices(), eef=eef, held=HELD_F,
                           secured=SEC_F, targets=TARGETS)
        np.testing.assert_allclose(data[3:9], eef[3:9])
        np.testing.assert_allclose(data[13:19], eef[13:19])
        np.testing.assert_allclose(data[0:3], eef[0:3])
        np.testing.assert_allclose(data[10:13], eef[10:13])
```

- [ ] **Step 2: 跑红**

Run: `uv run pytest plugins/inspect-robots-jev/tests/test_executor.py -v`
Expected: FAIL（`No module named ... _executor`）

- [ ] **Step 3: 最小实现**

```python
"""Direction choices -> one 20-dim absolute EE pose, behind seven hard gates.

Gate order per arm axis: phase permissions -> workspace bounds; per gripper:
held-open whitelist -> handoff safety; then cross-arm separation. Rejected
channels degrade to hold; the action still ships (spec §3.6).
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from inspect_robots.spaces import Box

from inspect_robots_jev import _config as c
from inspect_robots_jev._phases import PHASES
from inspect_robots_jev._state import Targets

_DIRECTION = {"negative": -1.0, "hold": 0.0, "positive": 1.0}
_GRIP = ("open", "hold", "close")
_ARM_OFFSET = {"left": 0, "right": 10}


class TaskStalled(RuntimeError):
    """No effective motion for stall_steps consecutive steps."""


class Executor:
    def __init__(self, box: Box, *, max_step_mm: float = c.MAX_STEP_MM,
                 min_separation_m: float = c.MIN_SEPARATION_M,
                 stall_steps: int = c.STALL_STEPS, dry_run: bool = False) -> None:
        self.box = box
        self.max_step_mm = float(max_step_mm)
        self.min_separation_m = float(min_separation_m)
        self.stall_steps = int(stall_steps)
        self.dry_run = bool(dry_run)
        self.last_widths: dict[str, float] = {"left": c.GRIPPER_OPEN_M,
                                              "right": c.GRIPPER_OPEN_M}
        self._stalled_steps = 0
        self._warned = False

    def reset(self) -> None:
        self.last_widths = {"left": c.GRIPPER_OPEN_M, "right": c.GRIPPER_OPEN_M}
        self._stalled_steps = 0
        self._warned = False

    def _reference(self, kind: str, arm: str, current: np.ndarray,
                   targets: Targets) -> np.ndarray:
        if kind == "current":
            return current
        if kind == "travel":
            return np.array([current[0], current[1], targets.travel_z_m])
        waypoint = {"left_grasp": targets.left_grasp, "handoff_left": targets.handoff_left,
                    "handoff_right": targets.handoff_right,
                    "place": targets.place}[kind]
        return np.asarray(waypoint.tcp, dtype=float)

    def _step_m(self, distance_m: float) -> float:
        if distance_m < c.NEAR_MM / 1000.0:
            return c.FINE_STEP_MM / 1000.0
        if distance_m < c.MID_MM / 1000.0:
            return c.MID_STEP_MM / 1000.0
        return self.max_step_mm / 1000.0

    def build(self, *, phase: str, choices: Mapping[str, str], eef: np.ndarray,
              held: Mapping[str, bool], secured: Mapping[str, bool],
              targets: Targets) -> tuple[np.ndarray, dict]:
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}")
        spec = PHASES[phase]
        rejections: list[dict[str, str]] = []
        step_mm: dict[str, float] = {}
        deltas: dict[str, np.ndarray] = {"left": np.zeros(3), "right": np.zeros(3)}
        width_cmd: dict[str, float] = {}
        current_xyz = {"left": np.asarray(eef[0:3], dtype=float).copy(),
                       "right": np.asarray(eef[10:13], dtype=float).copy()}

        for arm in ("left", "right"):
            offset = _ARM_OFFSET[arm]
            reference = self._reference(spec.reference, arm, current_xyz[arm], targets)
            for i, axis in enumerate(c.AXES):
                channel = f"{arm}_{axis}"
                choice = choices[channel]
                if choice not in _DIRECTION:
                    raise ValueError(f"Invalid XYZ option {choice!r} on {channel}")
                if choice == "hold":
                    continue
                if axis not in spec.allowed_axes[arm]:
                    rejections.append({"channel": channel, "rule": "phase_axes",
                                       "detail": f"{phase} holds {arm} {axis}"})
                    continue
                candidate = current_xyz[arm][i] + _DIRECTION[choice] * self._step_m(
                    abs(float(reference[i] - current_xyz[arm][i])))
                index = offset + i
                low = None if self.box.low is None else float(self.box.low[index])
                high = None if self.box.high is None else float(self.box.high[index])
                if (low is not None and candidate < low) or (high is not None and candidate > high):
                    rejections.append({"channel": channel, "rule": "bounds",
                                       "detail": f"target {candidate:.3f} outside the action box"})
                    continue
                deltas[arm][i] = candidate - current_xyz[arm][i]
                step_mm[channel] = abs(float(deltas[arm][i])) * 1000.0

        for arm in ("left", "right"):
            channel = f"{arm}_gripper"
            choice = choices[channel]
            if choice not in _GRIP:
                raise ValueError(f"Invalid gripper option {choice!r} on {channel}")
            width_cmd[arm] = self.last_widths[arm]
            if choice == "close":
                width_cmd[arm] = c.GRIPPER_CLOSED_M
            elif choice == "open":
                allowed = True
                if held.get(arm) and not spec.held_open_ok[arm]:
                    rejections.append({"channel": channel, "rule": "held_open",
                                       "detail": f"{arm} holds the object; {phase} forbids opening"})
                    allowed = False
                if allowed and phase == "handoff_release" and arm == "left" \
                        and not secured.get("right"):
                    rejections.append({"channel": channel,
                                       "rule": "handoff_release_requires_right_secured",
                                       "detail": "right arm has not secured the object yet"})
                    allowed = False
                if allowed:
                    width_cmd[arm] = c.GRIPPER_OPEN_M

        # Cross-arm separation outside handoff phases (spec §3.6 gate 7).
        if phase not in c.HANDOFF_PHASES:
            new_left = current_xyz["left"] + deltas["left"]
            new_right = current_xyz["right"] + deltas["right"]
            distance_now = float(np.linalg.norm(current_xyz["left"] - current_xyz["right"]))
            distance_left = float(np.linalg.norm(new_left - current_xyz["right"]))
            distance_both = float(np.linalg.norm(new_left - new_right))
            moved = {"left": bool(np.any(deltas["left"] != 0.0)),
                     "right": bool(np.any(deltas["right"] != 0.0))}
            for arm, other, distance in (("left", "right", distance_left),
                                          ("right", "left", distance_both)):
                if not moved[arm]:
                    continue
                if distance < self.min_separation_m and distance < distance_now:
                    for i, axis in enumerate(c.AXES):
                        if deltas[arm][i] != 0.0:
                            rejections.append({"channel": f"{arm}_{axis}", "rule": "separation",
                                               "detail": "would close the inter-arm distance below "
                                                         f"{self.min_separation_m} m"})
                            deltas[arm][i] = 0.0
                    break

        motion = bool(np.any(deltas["left"]) or np.any(deltas["right"])
                      or width_cmd["left"] != self.last_widths["left"]
                      or width_cmd["right"] != self.last_widths["right"])
        if self.dry_run:
            deltas = {"left": np.zeros(3), "right": np.zeros(3)}
            width_cmd = dict(self.last_widths)
            rejections.append({"channel": "all", "rule": "dry_run",
                               "detail": "dry run: commanded a full hold"})
            motion = False
        else:
            self._stalled_steps = 0 if motion else self._stalled_steps + 1
            if self._stalled_steps > self.stall_steps:
                raise TaskStalled(
                    f"no effective motion for {self._stalled_steps} consecutive steps")

        data = np.zeros(20, dtype=float)
        data[0:3] = current_xyz["left"] + deltas["left"]
        data[3:9] = eef[3:9]
        data[9] = width_cmd["left"]
        data[10:13] = current_xyz["right"] + deltas["right"]
        data[13:19] = eef[13:19]
        data[19] = width_cmd["right"]
        self.last_widths = dict(width_cmd)
        meta = {"phase": phase, "choices": dict(choices), "step_mm": step_mm,
                "rejections": rejections, "width_cmd_m": dict(width_cmd),
                "dry_run": self.dry_run}
        return data, meta
```

- [ ] **Step 4: 跑绿**

Run: `uv run pytest plugins/inspect-robots-jev/tests/test_executor.py -v`
Expected: 全部 PASS。若 `test_separation_guard_active_outside_handoff_phases` 因参考目标/距离不符失败：核对 `_reference` 在 `approach_left` 用 `left_grasp`、`handoff_grasp` 用 `handoff_right`，以及非交接 phase 的距离阈值 0.08。

- [ ] **Step 5: 提交**

```bash
git add plugins/inspect-robots-jev/src/inspect_robots_jev/_executor.py plugins/inspect-robots-jev/tests/test_executor.py
git commit -m "feat(jev): gated incremental executor for 20-dim absolute poses"
```

---

### Task 6: Policy 接线（`policy.py`）

**Files:**
- Create: `plugins/inspect-robots-jev/src/inspect_robots_jev/policy.py`
- Modify: `plugins/inspect-robots-jev/src/inspect_robots_jev/__init__.py`（工厂不再占位，直接导出类）
- Test: `plugins/inspect-robots-jev/tests/test_policy.py`

**Interfaces:**
- Consumes: Task 3 `DecisionsClient`（含 `reset()`——本任务若发现 Task 3 未实现 `reset()`，在 `_decisions.py` 补上：`records.clear()` 且 `calls/cost/wait_seconds` 归零，并给 Task 3 测试加一行断言）、Task 4 `load_targets/StateAssembler`、Task 5 `Executor/TaskStalled`、`_phases.PHASES`。
- Produces: `JevHandoffPolicy(PolicyBase)`，构造参数（全部可从 `-P k=v` 字符串强制转换）：`provider="jev"`、`model=None`、`base_url=None`、`api_key_env="OPENROUTER_API_KEY"`、`timeout_s=90.0`、`targets_file=None`、`dry_run=False`、`max_step_mm=18.0`、`min_separation_m=0.08`、`stall_steps=20`、`decisions=None`（注入测试替身）、`name="jev"`。方法：`bind(embodiment_info)`、`reset(scene)`、`act(observation) -> ActionChunk`、`transcript()`。

- [ ] **Step 1: 写失败测试**

```python
"""plugins/inspect-robots-jev/tests/test_policy.py"""
import json

import numpy as np
import pytest

from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.spaces import ActionSemantics, Box, ObservationSpace
from inspect_robots.types import Observation

from inspect_robots_jev._executor import TaskStalled
from inspect_robots_jev.policy import JevHandoffPolicy

from test_executor import BOX, TARGETS  # 复用兄弟测试模块的盒与目标

PHASES_ALL = ("approach_left", "grasp", "lift", "offer", "handoff_grasp",
              "handoff_release", "carry", "lower", "release", "withdraw", "finish")


def _info(box=BOX):
    return EmbodimentInfo(name="fake", action_space=box,
                          observation_space=ObservationSpace(), control_hz=30.0,
                          is_simulated=True, capabilities=frozenset(),
                          supported_setups=frozenset(), supported_target_kinds=frozenset(),
                          docs="fake nero-shaped embodiment for tests")


class FakeDecisions:
    """Scripted (phase, motor-choices) pairs; shape-compatible with DecisionsClient."""

    def __init__(self, script):
        self.script = list(script)
        self.records = []
        self.calls = 0
        self.cost = 0.0
        self.wait_seconds = 0.0

    def reset(self):
        self.records.clear()

    def phase(self, state):
        self.calls += 1
        entry = self.script.pop(0) if self.script else ("finish", None)
        record = {"stage": "phase", "model": "fake", "response_id": f"p{self.calls}",
                  "answers": {"phase": {"choice": entry[0], "probabilities": {}}},
                  "latency_seconds": 0.0, "cost_usd": 0.0, "usage": {},
                  "observation": state, "questions": {}}
        self.records.append(record)
        return record

    def motor(self, state, phase):
        self.calls += 1
        entry = self.script.pop(0) if self.script else ("finish", None)
        choices = entry[1] or {ch: "hold" for ch in
                               ("left_x", "left_y", "left_z", "left_gripper",
                                "right_x", "right_y", "right_z", "right_gripper")}
        record = {"stage": "motor", "model": "fake", "response_id": f"m{self.calls}",
                  "answers": {ch: {"choice": c, "probabilities": {}} for ch, c in choices.items()},
                  "latency_seconds": 0.0, "cost_usd": 0.0, "usage": {},
                  "observation": state, "questions": {}}
        self.records.append(record)
        return record


@pytest.fixture
def targets_file(tmp_path):
    path = tmp_path / "targets.json"
    path.write_text(json.dumps({
        "object": "cup",
        "left_grasp": {"tcp_m": list(TARGETS.left_grasp.tcp), "rot6d": [1, 0, 0, 0, 1, 0],
                       "width_m": 0.05},
        "handoff_left": {"tcp_m": list(TARGETS.handoff_left.tcp), "rot6d": [1, 0, 0, 0, 1, 0]},
        "handoff_right": {"tcp_m": list(TARGETS.handoff_right.tcp), "rot6d": [1, 0, 0, 0, 1, 0],
                          "width_m": 0.05},
        "place": {"tcp_m": list(TARGETS.place.tcp), "rot6d": [1, 0, 0, 0, 1, 0],
                  "center_m": list(TARGETS.place_center)},
        "travel_z_m": TARGETS.travel_z_m,
    }))
    return str(path)


def _observation(left=(0.30, -0.10, 0.20), right=(0.10, 0.05, 0.20), lw=0.09, rw=0.09,
                 messages=()):
    eef = np.array([*left, 1, 0, 0, 0, 1, 0, lw, *right, 1, 0, 0, 0, 1, 0, rw], float)
    return Observation(state={"eef_state": eef, "joint_pos": np.zeros(16)},
                       instruction="hand off the cup",
                       extra={"operator_messages": list(messages)})


def _policy(targets_file, script, **kwargs):
    policy = JevHandoffPolicy(targets_file=targets_file, decisions=FakeDecisions(script), **kwargs)
    policy.bind(_info())
    policy.reset(_scene())
    return policy


def _scene():
    from inspect_robots.scene import Scene
    return Scene(id="handoff-0", instruction="hand off the cup")


class TestBind:
    def test_rejects_wrong_shape(self, targets_file):
        from inspect_robots.spaces import Box as B
        bad = B(shape=(7,), low=np.zeros(7), high=np.ones(7),
                semantics=ActionSemantics(control_mode="joint_pos"))
        policy = JevHandoffPolicy(targets_file=targets_file)
        with pytest.raises(ValueError, match="20"):
            policy.bind(_info(box=bad))

    def test_rejects_non_absolute_pose_semantics(self, targets_file):
        bad = Box(shape=(20,), low=np.zeros(20), high=np.ones(20),
                  semantics=ActionSemantics(control_mode="joint_pos"))
        policy = JevHandoffPolicy(targets_file=targets_file)
        with pytest.raises(ValueError, match="eef_abs_pose"):
            policy.bind(_info(box=bad))

    def test_adopts_box_and_control_hz(self, targets_file):
        policy = JevHandoffPolicy(targets_file=targets_file)
        policy.bind(_info())
        assert policy.info.action_space is BOX
        assert policy.info.control_hz == 30.0
        assert policy.config.action_horizon == 1


class TestReset:
    def test_missing_targets_file_fails(self):
        policy = JevHandoffPolicy()
        policy.bind(_info())
        with pytest.raises(ValueError, match="targets_file"):
            policy.reset(_scene())

    def test_reset_clears_state(self, targets_file):
        policy = _policy(targets_file,
                         [("approach_left", {"left_x": "positive", "left_gripper": "open"})])
        policy.act(_observation())
        decisions = policy._decisions
        policy.reset(_scene())
        assert decisions.records == []
        assert policy._executor.last_widths == {"left": 0.09, "right": 0.09}


class TestAct:
    def test_happy_path_produces_one_bounded_action(self, targets_file):
        script = [("approach_left",
                   {"left_x": "positive", "left_y": "hold", "left_z": "hold",
                    "left_gripper": "open", "right_x": "hold", "right_y": "hold",
                    "right_z": "hold", "right_gripper": "hold"})]
        policy = _policy(targets_file, script)
        chunk = policy.act(_observation(left=(0.0, -0.10, 0.20)))
        assert len(chunk) == 1 and chunk.control_hz == 30.0
        action = chunk.actions[0]
        np.testing.assert_array_less(BOX.low - 1e-9, action.data)
        np.testing.assert_array_less(action.data, BOX.high + 1e-9)
        assert action.data[0] == pytest.approx(0.018)  # far tier toward left_grasp x=0.30
        assert action.data[9] == pytest.approx(0.09)   # open applied
        assert chunk.meta["phase"] == "approach_left"
        assert chunk.meta["dry_run"] is False

    def test_gates_flow_into_feedback_and_next_state(self, targets_file):
        script = [("handoff_release", {"left_gripper": "open"})]
        policy = _policy(targets_file, script)
        chunk = policy.act(_observation(left=(0.10, 0.05, 0.20), right=(0.10, 0.05, 0.18),
                                        lw=0.05, rw=0.05))
        assert chunk.actions[0].data[9] == pytest.approx(0.0)  # gated
        assert "handoff_release_requires_right_secured" in policy._last_feedback

    def test_dry_run_returns_full_hold(self, targets_file):
        script = [("approach_left",
                   {"left_x": "positive", "left_y": "hold", "left_z": "hold",
                    "left_gripper": "close", "right_x": "hold", "right_y": "hold",
                    "right_z": "hold", "right_gripper": "hold"})]
        policy = _policy(targets_file, script, dry_run=True)
        chunk = policy.act(_observation(left=(0.0, -0.10, 0.20)))
        assert chunk.actions[0].data[0] == pytest.approx(0.0)
        assert chunk.actions[0].data[9] == pytest.approx(0.09)

    def test_stall_terminates_act(self, targets_file):
        script = [("approach_left", None)] * 25  # all-hold motor answers
        policy = _policy(targets_file, script, stall_steps=5)
        with pytest.raises(TaskStalled):
            for _ in range(10):
                policy.act(_observation())

    def test_operator_messages_reach_state(self, targets_file):
        seen = {}

        class Spy(FakeDecisions):
            def phase(self, state):
                seen["state"] = state
                return super().phase(state)

        policy = JevHandoffPolicy(targets_file=targets_file,
                                  decisions=Spy([("approach_left", None)]))
        policy.bind(_info())
        policy.reset(_scene())
        policy.act(_observation(messages=[{"t": 1, "text": "gently", "source": "console"}]))
        assert "gently" in seen["state"]["feedback"]


class TestTranscript:
    def test_transcript_is_a_detached_copy(self, targets_file):
        policy = _policy(targets_file, [("approach_left", None)])
        policy.act(_observation())
        transcript = policy.transcript()
        assert transcript["records"], "transcript should carry decision records"
        transcript["records"].clear()
        assert policy._decisions.records, "mutating the transcript must not touch the client"
```

- [ ] **Step 2: 跑红**

Run: `uv run pytest plugins/inspect-robots-jev/tests/test_policy.py -v`
Expected: FAIL（`No module named ... policy`；`from test_executor import …` 需要 pytest rootdir 导入兄弟模块——在 `tests/` 放空 `__init__.py` 已满足）

- [ ] **Step 3: 最小实现**

```python
"""JevHandoffPolicy: two model calls per step, one gated absolute-pose action."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.policy import PolicyBase, PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.types import Action, ActionChunk, Observation

from inspect_robots_jev import _config as c
from inspect_robots_jev._decisions import DecisionsClient
from inspect_robots_jev._executor import Executor
from inspect_robots_jev._state import StateAssembler, load_targets


def _coerce_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes"):
        return True
    if text in ("0", "false", "no", ""):
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _coerce_float(value: Any, name: str) -> float:
    number = float(value)  # raises TypeError/ValueError for junk
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


class JevHandoffPolicy(PolicyBase):
    """Global handoff phase + eight motor channels -> one 20-dim absolute pose."""

    def __init__(self, *, provider: str = "jev", model: str | None = None,
                 base_url: str | None = None, api_key_env: str = c.API_KEY_ENV,
                 timeout_s: float = c.TIMEOUT_S, targets_file: str | None = None,
                 dry_run: Any = False, max_step_mm: Any = c.MAX_STEP_MM,
                 min_separation_m: Any = c.MIN_SEPARATION_M,
                 stall_steps: Any = c.STALL_STEPS,
                 decisions: DecisionsClient | None = None, name: str = "jev") -> None:
        self._provider = str(provider)
        self._model = model
        self._base_url = base_url
        self._api_key_env = str(api_key_env)
        self._timeout_s = _coerce_float(timeout_s, "timeout_s")
        self._targets_file = targets_file
        self._dry_run = _coerce_bool(dry_run, "dry_run")
        self._max_step_mm = _coerce_float(max_step_mm, "max_step_mm")
        self._min_separation_m = _coerce_float(min_separation_m, "min_separation_m")
        self._stall_steps = int(_coerce_float(stall_steps, "stall_steps"))
        self._decisions = decisions
        self._name = str(name)
        self.info = PolicyInfo(name=self._name, action_space=None)  # type: ignore[arg-type]
        self.config = PolicyConfig(action_horizon=1)
        self._executor: Executor | None = None
        self._assembler: StateAssembler | None = None
        self._targets = None
        self._control_hz: float | None = None
        self._last_feedback = "Scene initialized. No task actions executed."
        self._last_phase: str | None = None

    # -- lifecycle ---------------------------------------------------------

    def bind(self, embodiment_info: EmbodimentInfo) -> None:
        box = embodiment_info.action_space
        if box.shape != (20,):
            raise ValueError(
                f"jev policy needs a 20-dim dual-arm action space, got shape {box.shape}")
        semantics = box.semantics
        if semantics is None or semantics.control_mode != "eef_abs_pose" \
                or semantics.rotation_repr != "rot6d":
            raise ValueError(
                "jev policy needs eef_abs_pose + rot6d semantics "
                f"(nero-style), got {semantics and semantics.control_mode!r}")
        self._executor = Executor(box, max_step_mm=self._max_step_mm,
                                  min_separation_m=self._min_separation_m,
                                  stall_steps=self._stall_steps, dry_run=self._dry_run)
        self._control_hz = float(embodiment_info.control_hz)
        self.info = PolicyInfo(name=self._name, action_space=box,
                               control_hz=embodiment_info.control_hz)

    def reset(self, scene: Scene) -> None:
        del scene
        if self._executor is None or self._control_hz is None:
            raise RuntimeError("bind(embodiment_info) must run before reset()")
        if not self._targets_file:
            raise ValueError(
                "jev policy requires -P targets_file=<declared targets json> "
                "(produce one with scripts/declare_targets.py)")
        self._targets = load_targets(self._targets_file)
        self._assembler = StateAssembler(self._targets, control_hz=self._control_hz)
        self._executor.reset()
        if self._decisions is None:
            self._decisions = DecisionsClient(
                provider=self._provider, model=self._model, base_url=self._base_url,
                api_key_env=self._api_key_env, timeout_s=self._timeout_s)
        else:
            self._decisions.reset()
        self._last_feedback = "Scene initialized. No task actions executed."
        self._last_phase = None

    def transcript(self) -> dict[str, Any]:
        client = getattr(self, "_decisions", None)
        records = list(client.records) if client is not None else []
        return json.loads(json.dumps({"provider": self._provider, "records": records}))

    # -- inference ---------------------------------------------------------

    def act(self, observation: Observation) -> ActionChunk:
        if self._executor is None or self._assembler is None or self._decisions is None:
            raise RuntimeError("reset(scene) must run before act()")
        eef = np.asarray(observation.state["eef_state"], dtype=float)
        state = self._assembler.assemble(
            observation,
            last_width_cmd=self._executor.last_widths,
            feedback=self._last_feedback,
            operator_messages=list(observation.extra.get("operator_messages", [])),
            previous_phase=self._last_phase)
        phase_record = self._decisions.phase(state)
        phase = phase_record["answers"]["phase"]["choice"]
        motor_record = self._decisions.motor(state, phase)
        choices = {channel: answer["choice"]
                   for channel, answer in motor_record["answers"].items()}
        data, meta = self._executor.build(
            phase=phase, choices=choices, eef=eef, held=self._assembler.held,
            secured=self._assembler.secured, targets=self._targets)
        self._last_feedback = _feedback_text(meta)
        self._last_phase = phase
        return ActionChunk(actions=[Action(data=data)],
                           control_hz=self._control_hz, meta=meta)


def _feedback_text(meta: dict[str, Any]) -> str:
    if not meta["rejections"]:
        return "Increment executed. Inspect updated pose and gripper state."
    rejected = "; ".join(f"{r['channel']} ({r['rule']})" for r in meta["rejections"])
    return f"Held rejected channels: {rejected}. Reposition or change phase."
```

`__init__.py` 更新（工厂落地）：

```python
"""Jev two-stage handoff policy plugin for Inspect Robots."""
from __future__ import annotations

from typing import Any

from inspect_robots.registry import register


@register("policy", "jev")
def jev_policy(**kwargs: Any) -> "JevHandoffPolicy":
    """Registry-facing factory; -P k=v pairs arrive as keyword arguments."""
    from inspect_robots_jev.policy import JevHandoffPolicy

    return JevHandoffPolicy(**kwargs)


def __getattr__(name: str):  # lazy re-export keeps import cost off the registry path
    if name == "JevHandoffPolicy":
        from inspect_robots_jev.policy import JevHandoffPolicy
        return JevHandoffPolicy
    raise AttributeError(name)


__all__ = ["jev_policy", "JevHandoffPolicy"]
```

`PolicyInfo(name=…, action_space=None)` 会撞类型标注——实现时给 `action_space` 传一个占位 `Box(shape=(20,))`（无边界、无语义；`bind` 会整体替换）。相应把 `__init__` 里该行写成：

```python
        from inspect_robots.spaces import Box
        self.info = PolicyInfo(name=self._name, action_space=Box(shape=(20,)))
```

（`Box(shape=(20,))` 合法：low/high/semantics 均可缺省。）

- [ ] **Step 4: 跑绿**

Run: `uv run pytest plugins/inspect-robots-jev/tests/test_policy.py plugins/inspect-robots-jev/tests/test_registration.py -v`
Expected: 全部 PASS（注册测试此时也验证了工厂可导入）

- [ ] **Step 5: 提交**

```bash
git add plugins/inspect-robots-jev/src/inspect_robots_jev/policy.py plugins/inspect-robots-jev/src/inspect_robots_jev/__init__.py plugins/inspect-robots-jev/tests/test_policy.py plugins/inspect-robots-jev/src/inspect_robots_jev/_decisions.py plugins/inspect-robots-jev/tests/test_decisions.py
git commit -m "feat(jev): JevHandoffPolicy wiring two-stage decisions to the gated executor"
```

---

### Task 7: 契约级集成测试（fake embodiment + 剧本决策走完关键段）

**Files:**
- Test: `plugins/inspect-robots-jev/tests/test_integration.py`

**Interfaces:**
- Consumes: Task 4/5/6 全部产物（`JevHandoffPolicy`、`load_targets`）、`inspect_robots.types`。
- Produces: 无新生产代码；本任务建立「策略↔embodiment 契约」的回归测试（spec §5 集成项：fake embodiment + scripted decisions 走完 `approach_left → grasp` 并验证交接门禁）。

- [ ] **Step 1: 写测试（本任务只有测试，先写后跑红）**

```python
"""Contract-level integration: the policy loop against a nero-shaped fake rig.

The fake embodiment tracks commanded poses perfectly and scripts object
contact with grab()/release(); the scripted decisions walk the critical
approach -> grasp -> secure -> handoff_release path so the safety gates are
exercised end to end without hardware.
"""
import json

import numpy as np
import pytest

from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.scene import Scene
from inspect_robots.spaces import ActionSemantics, Box, ObservationSpace
from inspect_robots.types import Observation, StepResult

from inspect_robots_jev.policy import JevHandoffPolicy

LOW = np.array([-1.0] * 3 + [-1.0] * 6 + [0.0] + [-1.0] * 3 + [-1.0] * 6 + [0.0])
HIGH = np.array([1.0] * 3 + [1.0] * 6 + [0.09] + [1.0] * 3 + [1.0] * 6 + [0.09])


class FakeHandoffEmbodiment:
    """Nero-contract fake: perfect tracking, scripted object widths."""

    def __init__(self, left=(0.0, -0.10, 0.20), right=(0.10, 0.05, 0.20)):
        self.info = EmbodimentInfo(
            name="fake-nero",
            action_space=Box(shape=(20,), low=LOW, high=HIGH,
                             semantics=ActionSemantics(control_mode="eef_abs_pose",
                                                       rotation_repr="rot6d",
                                                       gripper="continuous", frame="base")),
            observation_space=ObservationSpace(), control_hz=30.0, is_simulated=True,
            capabilities=frozenset(), supported_setups=frozenset(),
            supported_target_kinds=frozenset(), docs="fake")
        self.eef = np.array([*left, 1, 0, 0, 0, 1, 0, 0.09,
                             *right, 1, 0, 0, 0, 1, 0, 0.09], float)
        self._floor = {"left": 0.0, "right": 0.0}
        self.actions: list[np.ndarray] = []
        self._instruction = None

    def grab(self, side, width=0.05):
        self._floor[side] = width

    def release(self, side):
        self._floor[side] = 0.0

    def _width(self, side, commanded):
        return max(commanded, self._floor[side])

    def _observe(self):
        return Observation(
            state={"eef_state": self.eef.copy(), "joint_pos": np.zeros(16)},
            instruction=self._instruction)

    def reset(self, scene, *, seed=None):
        self._instruction = scene.instruction
        return self._observe()

    def step(self, action):
        data = np.asarray(action.data, dtype=float)
        self.actions.append(data.copy())
        self.eef[0:3] = data[0:3]
        self.eef[10:13] = data[10:13]
        self.eef[9] = self._width("left", float(data[9]))
        self.eef[19] = self._width("right", float(data[19]))
        return StepResult(observation=self._observe())


class ScriptedDecisions:
    def __init__(self, script):
        self.script = list(script)
        self.records = []
        self.calls = 0
        self.cost = 0.0
        self.wait_seconds = 0.0
        self.seen_states = []

    def reset(self):
        self.records.clear()

    def _pop(self):
        return self.script.pop(0) if self.script else ("finish", None)

    def phase(self, state):
        self.seen_states.append(state)
        self.calls += 1
        phase, _ = self._pop()
        record = {"stage": "phase", "model": "scripted", "response_id": f"p{self.calls}",
                  "answers": {"phase": {"choice": phase, "probabilities": {}}},
                  "latency_seconds": 0.0, "cost_usd": 0.0, "usage": {},
                  "observation": state, "questions": {}}
        self.records.append(record)
        return record

    def motor(self, state, phase):
        self.calls += 1
        _, choices = self._pop()
        choices = choices or {}
        answers = {}
        for channel in ("left_x", "left_y", "left_z", "left_gripper",
                        "right_x", "right_y", "right_z", "right_gripper"):
            choice = choices.get(channel, "hold")
            answers[channel] = {"choice": choice, "probabilities": {}}
        record = {"stage": "motor", "model": "scripted", "response_id": f"m{self.calls}",
                  "answers": answers, "latency_seconds": 0.0, "cost_usd": 0.0,
                  "usage": {}, "observation": state, "questions": {}}
        self.records.append(record)
        return record


TARGETS_JSON = {
    "object": "cup",
    "left_grasp": {"tcp_m": [0.30, -0.10, 0.15], "rot6d": [1, 0, 0, 0, 1, 0], "width_m": 0.05},
    "handoff_left": {"tcp_m": [0.10, 0.05, 0.20], "rot6d": [1, 0, 0, 0, 1, 0]},
    "handoff_right": {"tcp_m": [0.10, 0.05, 0.18], "rot6d": [1, 0, 0, 0, 1, 0], "width_m": 0.05},
    "place": {"tcp_m": [0.25, 0.25, 0.05], "rot6d": [1, 0, 0, 0, 1, 0],
              "center_m": [0.25, 0.25, 0.0]},
    "travel_z_m": 0.20,
}


def _step_script():
    """(phase, motor-choices) per cycle; matches the fake rig's geometry."""
    script = []
    script += [("approach_left", {"left_x": "positive", "left_gripper": "open"})] * 3
    script += [("grasp", {"left_gripper": "close"})] * 2          # then test grabs left
    script += [("grasp", None)] * 14                              # secured at the 14th
    script += [("handoff_release", {"left_gripper": "open"})]     # gated: right not secured
    script += [("handoff_grasp", {"right_gripper": "close"})] * 2  # then test grabs right
    script += [("handoff_grasp", None)] * 14                      # right secured
    script += [("handoff_release", {"left_gripper": "open"})]     # applied
    return script


@pytest.fixture
def targets_file(tmp_path):
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(TARGETS_JSON))
    return str(path)


def test_handoff_critical_path_with_gates(targets_file):
    embodiment = FakeHandoffEmbodiment()
    decisions = ScriptedDecisions(_step_script())
    policy = JevHandoffPolicy(targets_file=targets_file, decisions=decisions,
                              stall_steps=50)
    policy.bind(embodiment.info)
    scene = Scene(id="handoff-0", instruction="hand off the cup")
    policy.reset(scene)
    observation = embodiment.reset(scene)

    chunks = []
    grabbed_left = grabbed_right = False
    for _ in range(len(_step_script())):
        chunk = policy.act(observation)
        chunks.append(chunk)
        result = embodiment.step(chunk.actions[0])
        observation = result.observation
        # Script object contact the moment a gripper is commanded closed in the
        # phase whose job is closing on the object (perfect tracking means the
        # reported width follows the command, so the floor is set here instead).
        if not grabbed_left and chunk.meta["phase"] == "grasp" \
                and chunk.meta["width_cmd_m"]["left"] == 0.0:
            embodiment.grab("left")
            grabbed_left = True
        if not grabbed_right and chunk.meta["phase"] == "handoff_grasp" \
                and chunk.meta["width_cmd_m"]["right"] == 0.0:
            embodiment.grab("right")
            grabbed_right = True

    # Contract: every action is one 20-dim vector inside the declared box.
    assert all(len(chunk) == 1 for chunk in chunks)
    assert all(chunk.control_hz == 30.0 for chunk in chunks)
    for chunk in chunks:
        data = chunk.actions[0].data
        np.testing.assert_array_less(LOW - 1e-9, data)
        np.testing.assert_array_less(data, HIGH + 1e-9)

    # Grasp phase: XYZ pinned while the gripper closes.
    grasp_actions = [c for c in chunks if c.meta["phase"] == "grasp"]
    xyz = np.array([c.actions[0].data[0:3] for c in grasp_actions])
    assert np.allclose(xyz, xyz[0])
    assert grasp_actions[0].actions[0].data[9] == pytest.approx(0.0)

    # The scripted state actually reached 'held' then 'secured' on both arms.
    assert decisions.seen_states[-1]["left_held"] is True
    assert decisions.seen_states[-1]["right_secured"] is True

    # Critical gate: the first handoff_release could not open the left gripper.
    handoff_chunks = [c for c in chunks if c.meta["phase"] == "handoff_release"]
    assert handoff_chunks[0].actions[0].data[9] == pytest.approx(0.0)
    assert any(r["rule"] == "handoff_release_requires_right_secured"
               for r in handoff_chunks[0].meta["rejections"])
    # Once the right arm secured, the same phase opens the left gripper.
    assert handoff_chunks[-1].actions[0].data[9] == pytest.approx(0.09)


def test_approach_moves_left_arm_only(targets_file):
    embodiment = FakeHandoffEmbodiment()
    decisions = ScriptedDecisions([("approach_left", {"left_x": "positive"})] * 2)
    policy = JevHandoffPolicy(targets_file=targets_file, decisions=decisions)
    policy.bind(embodiment.info)
    scene = Scene(id="s", instruction="hand off")
    policy.reset(scene)
    observation = embodiment.reset(scene)
    first = policy.act(observation)
    result = embodiment.step(first.actions[0])
    second = policy.act(result.observation)
    # Left advances 18 mm per far step; right stays exactly where it was.
    assert first.actions[0].data[0] == pytest.approx(0.018)
    assert second.actions[0].data[0] == pytest.approx(0.036)
    np.testing.assert_allclose(first.actions[0].data[10:13],
                                embodiment.eef[10:13] - 0.0)  # unchanged start pose
    np.testing.assert_allclose(second.actions[0].data[10:13], [0.10, 0.05, 0.20])
```

- [ ] **Step 2: 跑红→绿**

Run: `uv run pytest plugins/inspect-robots-jev/tests/test_integration.py -v`
Expected: 首跑可能已在 Task 6 后全绿（本测试纯消费）；若有断言失败，按 meta 与 state 调试——**不许改生产代码迁就剧本**，剧本与夹具必须服从 spec 语义（门禁拒绝=降级 hold、secured 步数=14）。全绿后进入下一步。

- [ ] **Step 3: 提交**

```bash
git add plugins/inspect-robots-jev/tests/test_integration.py
git commit -m "test(jev): contract-level handoff path with scripted decisions"
```

---

### Task 8: 示教工具（`scripts/declare_targets.py`）

**Files:**
- Create: `plugins/inspect-robots-jev/scripts/declare_targets.py`
- Test: `plugins/inspect-robots-jev/tests/test_declare_targets.py`

**Interfaces:**
- Consumes: `_state.load_targets`（回环校验）；运行时惰性 import `inspect_robots_nero`（`NeroArm`、`NeroKinematics`、`matrix_to_rot6d`、`_config.CAN_CHANNELS/FIRMWARE_VERSION`、`_arm.clamp_to_joint_envelope`）。
- Produces: 纯函数 `build_targets_document(poses: Mapping[str, Mapping[str, object]], *, place_center, travel_z_m, object_name="object") -> dict`（JSON 文档，可被 `load_targets` 读回）；CLI `python plugins/inspect-robots-jev/scripts/declare_targets.py -o targets.json [--force]`（交互式示教：`x+/x-/y+/y-/z+/z- <mm>` 笛卡尔 jog（双臂 IK，另一臂锁定当前关节）、`save`、`skip`、`quit`；抓取位示教时提示输入 `width_m`；`place` 额外提示 `center_m x y`；最后提示 `travel_z_m`，默认 0.20）。

- [ ] **Step 1: 写失败测试（纯函数部分）**

```python
"""plugins/inspect-robots-jev/tests/test_declare_targets.py"""
import json

import pytest

from inspect_robots_jev._state import load_targets
from inspect_robots_jev.scripts.declare_targets import build_targets_document

POSES = {
    "left_grasp": {"tcp": [0.30, -0.10, 0.15], "rot6d": [1, 0, 0, 0, 1, 0], "width_m": 0.05},
    "handoff_left": {"tcp": [0.10, 0.05, 0.20], "rot6d": [1, 0, 0, 0, 1, 0]},
    "handoff_right": {"tcp": [0.10, 0.05, 0.18], "rot6d": [1, 0, 0, 0, 1, 0], "width_m": 0.05},
    "place": {"tcp": [0.25, 0.25, 0.05], "rot6d": [1, 0, 0, 0, 1, 0]},
}


def test_document_round_trips_through_load_targets(tmp_path):
    document = build_targets_document(POSES, place_center=[0.25, 0.25, 0.0],
                                      travel_z_m=0.20, object_name="cup")
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(document))
    targets = load_targets(path)
    assert targets.object_name == "cup"
    assert targets.left_grasp.width_m == 0.05
    assert targets.place_center == (0.25, 0.25, 0.0)


def test_missing_pose_rejected():
    broken = {k: v for k, v in POSES.items() if k != "place"}
    with pytest.raises(ValueError, match="place"):
        build_targets_document(broken, place_center=[0.25, 0.25, 0.0], travel_z_m=0.20)


def test_refuses_to_overwrite(tmp_path):
    from inspect_robots_jev.scripts.declare_targets import refuse_overwrite
    existing = tmp_path / "targets.json"
    existing.write_text("{}")
    with pytest.raises(SystemExit):
        refuse_overwrite(str(existing), force=False)
    refuse_overwrite(str(tmp_path / "new.json"), force=False)  # no raise
```

- [ ] **Step 2: 跑红**

Run: `uv run pytest plugins/inspect-robots-jev/tests/test_declare_targets.py -v`
Expected: FAIL（`No module named ... declare_targets`）

- [ ] **Step 3: 实现**

```python
"""Teach the four handoff waypoints on the real arms; write a targets JSON.

Operator flow per waypoint: jog with confirmed Cartesian increments (the
other arm is held at its current joints), then `save`. Requires the nero
plugin and the vendor SDK; nothing here is imported by the policy itself.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

WAYPOINTS = ("left_grasp", "handoff_left", "handoff_right", "place")
JOINT_STEP_CAP_RAD = 0.05


def build_targets_document(poses: Mapping[str, Mapping[str, Any]], *, place_center,
                           travel_z_m: float, object_name: str = "object") -> dict[str, Any]:
    missing = [name for name in WAYPOINTS if name not in poses]
    if missing:
        raise ValueError(f"missing waypoint(s): {', '.join(missing)}")
    document: dict[str, Any] = {"object": object_name, "travel_z_m": float(travel_z_m)}
    for name in WAYPOINTS:
        pose = poses[name]
        document[name] = {"tcp_m": [round(float(v), 4) for v in pose["tcp"]],
                          "rot6d": [round(float(v), 6) for v in pose["rot6d"]]}
        if pose.get("width_m") is not None:
            document[name]["width_m"] = float(pose["width_m"])
    document["place"]["center_m"] = [round(float(v), 4) for v in place_center]
    return document


def refuse_overwrite(path: str, *, force: bool) -> None:
    if Path(path).exists() and not force:
        print(f"refusing to overwrite {path} (pass --force)", file=sys.stderr)
        raise SystemExit(2)


def _connect_arms() -> tuple[Any, Any, Any]:
    from inspect_robots_nero._arm import NeroArm, clamp_to_joint_envelope
    from inspect_robots_nero._config import CAN_CHANNELS, FIRMWARE_VERSION
    from inspect_robots_nero._kinematics import NeroKinematics

    arms = {}
    for side in ("left", "right"):
        arm = NeroArm(side, CAN_CHANNELS[side], firmware=FIRMWARE_VERSION)
        arm.connect()
        arm.enable()
        arms[side] = arm
    kinematics = NeroKinematics(str(_urdf_path()))
    return arms, kinematics, clamp_to_joint_envelope


def _urdf_path() -> Path:
    import inspect_robots_nero as nero

    return Path(str(nero.__file__)).parent / "assets" / "dual_nero_pika.urdf"


def _cart_jog(arms, kinematics, clamp, side: str, axis: int, mm: float) -> None:
    from inspect_robots_nero._kinematics import NeroKinematicsError

    joints = {"left": np.asarray(arms["left"].read_state(), float),
              "right": np.asarray(arms["right"].read_state(), float)}
    poses = kinematics.fk(kinematics.q_from(joints["left"], joints["right"]))
    target = poses[side].copy()
    target[axis, 3] += mm / 1000.0
    seed = kinematics.q_from(joints["left"], joints["right"])
    try:
        solution = kinematics.solve(
            target if side == "left" else poses["left"],
            target if side == "right" else poses["right"], seed)
    except NeroKinematicsError as exc:
        print(f"IK refused the jog ({exc}); not moving.", file=sys.stderr)
        return
    for name, current in joints.items():
        command = np.clip(solution["left" == name and 0 or (7 if name == "right" else 0):][:7]
                          if False else
                          (solution[:7] if name == "left" else solution[7:]),
                          current - JOINT_STEP_CAP_RAD, current + JOINT_STEP_CAP_RAD)
        arms[name].move_js(clamp(command).tolist())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    refuse_overwrite(args.output, force=args.force)

    print("WARNING: this tool moves the real arms. Keep the e-stop within reach.")
    arms, kinematics, clamp = _connect_arms()
    poses: dict[str, dict[str, Any]] = {}
    axis_index = {"x": 0, "y": 1, "z": 2}
    for name in WAYPOINTS:
        print(f"\n== teach {name} (arm jog: x+/x-/y+/y-/z+/z- <mm>; save; skip; quit)")
        while True:
            command = input(f"[{name}]> ").strip()
            parts = command.split()
            if not parts:
                continue
            if parts[0] == "save":
                side = "left" if name != "handoff_right" else "right"
                joints = {"left": np.asarray(arms["left"].read_state(), float),
                          "right": np.asarray(arms["right"].read_state(), float)}
                pose = kinematics.fk(kinematics.q_from(joints["left"], joints["right"]))[side]
                entry: dict[str, Any] = {
                    "tcp": pose[:3, 3].tolist(),
                    "rot6d": _rot6d(pose).tolist()}
                if name in ("left_grasp", "handoff_right"):
                    entry["width_m"] = float(input("gripper width in m [0.05]: ") or 0.05)
                poses[name] = entry
                break
            if parts[0] == "skip":
                break
            if parts[0] == "quit":
                print("aborted; nothing written")
                return 1
            if parts[0] in axis_index and len(parts) == 2:
                side = "left" if name != "handoff_right" else "right"
                mm = float(parts[1])
                input(f"about to jog {side} {parts[0]} {mm:+.1f} mm — Enter to move, "
                      "Ctrl-C to abort")
                _cart_jog(arms, kinematics, clamp, side, axis_index[parts[0]], mm)
                continue
            print("unknown command")
    place_center = json.loads(input(
        "place center x,y,z in m (default = place tcp): ") or "null") \
        if "place" in poses else None
    if place_center is None and "place" in poses:
        place_center = poses["place"]["tcp"]
    travel_z = float(input("travel height z in m [0.20]: ") or 0.20)
    object_name = input("object name [cup]: ") or "cup"
    document = build_targets_document(poses, place_center=place_center,
                                      travel_z_m=travel_z, object_name=object_name)
    Path(args.output).write_text(json.dumps(document, indent=2))
    from inspect_robots_jev._state import load_targets
    load_targets(args.output)  # round-trip validation before we trust the file
    print(f"wrote {args.output}")
    return 0


def _rot6d(pose: np.ndarray) -> np.ndarray:
    from inspect_robots_nero._kinematics import matrix_to_rot6d

    return np.asarray(matrix_to_rot6d(pose[:3, :3]), float)


if __name__ == "__main__":
    raise SystemExit(main())
```

实现注意：
- `_cart_jog` 中 `solution[...]` 那段三元嵌套是排版错误——实现时直接写：
  ```python
  for name, current in joints.items():
      commanded = solution[:7] if name == "left" else solution[7:]
      command = np.clip(commanded, current - JOINT_STEP_CAP_RAD,
                        current + JOINT_STEP_CAP_RAD)
      arms[name].move_js(clamp(command).tolist())
  ```
- 测试只覆盖纯函数；交互与硬件路径由 M3 首步人工验证（spec §5）。
- `scripts/` 需要能被测试导入：`pyproject.toml` 已把 wheel 限定为 `src/…`，测试导入路径用 `plugins/inspect-robots-jev/scripts` 不可行——把测试改为直接按文件导入：
  ```python
  import importlib.util
  from pathlib import Path

  _MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "declare_targets.py"
  _SPEC = importlib.util.spec_from_file_location("declare_targets", _MODULE_PATH)
  declare_targets = importlib.util.module_from_spec(_SPEC)
  _SPEC.loader.exec_module(declare_targets)
  build_targets_document = declare_targets.build_targets_document
  refuse_overwrite = declare_targets.refuse_overwrite
  ```
  （替代直接 `from inspect_robots_jev.scripts...` 导入——`scripts` 不是包的一部分。）

- [ ] **Step 4: 跑绿**

Run: `uv run pytest plugins/inspect-robots-jev/tests/test_declare_targets.py -v`
Expected: 3 PASS

- [ ] **Step 5: 提交**

```bash
git add plugins/inspect-robots-jev/scripts/declare_targets.py plugins/inspect-robots-jev/tests/test_declare_targets.py
git commit -m "feat(jev): operator teaching tool for handoff target declaration"
```

---

### Task 9: README、全量验证与收尾

**Files:**
- Create: `plugins/inspect-robots-jev/README.md`
- Modify: 仓库根 `README.md`（若存在插件一览，加一行；否则跳过此修改）

**Interfaces:**
- Consumes: 全部前序任务。
- Produces: 文档 + 全绿的验证记录（M1/M2 代码侧完成；M3 真机四步为操作员流程，写进 README）。

- [ ] **Step 1: 写 README**

```markdown
# inspect-robots-jev

## Safety

> [!WARNING]
> This policy drives the physical dual Nero arms. Keep a tested emergency
> stop within reach, keep the bench clear, and never run unsupervised.
> First runs: `-P dry_run=true`, then real motion with
> `-P max_step_mm=4` while watching the console Stop button.

- The executor gates every channel against the global phase (a held gripper
  only opens in release/handoff_release/withdraw/finish, and handoff_release
  additionally requires the other arm to be secured), but gates degrade to
  hold — they do not replace the embodiment guardrails or the operator.
- Held/secured come from gripper-width inference (commanded closed while the
  reported width stays wide). Soft objects or width drift can fool them;
  treat `secured` as advisory and keep watching the arms.

## What it is

The jev two-stage decision protocol (from the open-source jev-robot-control
simulation study) as an Inspect Robots policy for the nero embodiment: one
model call picks the global handoff phase, one call picks eight per-arm
motor channels (left/right × x/y/z/gripper), and the gated executor turns
them into one 20-dim absolute end-effector pose per control tick.

## Install

The plugin is a workspace member; `uv sync` from the repo root installs it.
The policy itself needs only `inspect-robots` and `numpy`; target teaching
(`scripts/declare_targets.py`) additionally needs the nero plugin and the
vendor SDK (`pip install -e <neo_manipulation>/third_party/pyAgxArm`).

## First run ladder (attended)

1. Teach targets: `python plugins/inspect-robots-jev/scripts/declare_targets.py -o targets.json`
2. Dry run (arms hold still, decisions and gates run live):
   `uv run inspect-robots "hand off the cup" --policy jev --embodiment nero -P targets_file=targets.json -P dry_run=true`
3. Slow no-object run: `-P max_step_mm=4` (verify directions and step sizes)
4. Slow full handoff with the object, operator verdict at the end.

## Parameters (`-P`)

| Name | Default | Meaning |
|---|---|---|
| `provider` | `jev` | `jev` (native decisions endpoint) or `llm` (chat + strict JSON schema) |
| `model` | `typesafe/jev-1.13` | model id; `base_url` passes through for proxies |
| `base_url` | OpenRouter endpoint | override for proxied endpoints |
| `api_key_env` | `OPENROUTER_API_KEY` | env var holding the key |
| `timeout_s` | `90` | HTTP timeout per call |
| `targets_file` | — | required; targets JSON from `declare_targets.py` |
| `dry_run` | `false` | full-hold actions; nothing moves |
| `max_step_mm` | `18` | far-tier step cap (first runs: 4) |
| `min_separation_m` | `0.08` | inter-TCP distance guard outside handoff phases |
| `stall_steps` | `20` | consecutive no-motion steps before TaskStalled |

## Known limitations

- Task geometry comes from operator-declared poses; nothing tracks the
  object after it moves. `released_at_place` is geometric inference.
- Phase-machine tolerances (mm) are in `_config.py`; loosen them if the
  rig's repeatability demands it.
```

- [ ] **Step 2: 全量验证**

Run（每条都要实际执行并确认输出）：

```bash
uv run pytest plugins/inspect-robots-jev -v          # 全部测试 PASS
uv run ruff check plugins/inspect-robots-jev         # 无告警（修复后重跑）
uv run mypy plugins/inspect-robots-jev/src           # strict 通过
uv run inspect-robots list policies                  # 输出包含 jev
```

- [ ] **Step 3: 根 README 插件一览（如有）**

检查根 `README.md` 是否有插件列表（nero 在列则加一行）：

```markdown
- [`inspect-robots-jev`](plugins/inspect-robots-jev/) — Jev two-stage handoff policy for the nero dual arms.
```

- [ ] **Step 4: 提交**

```bash
git add plugins/inspect-robots-jev/README.md README.md
git commit -m "docs(jev): README with safety ladder, parameters, limitations"
```

---

## 里程碑对照（spec §7）

- **M1** = Task 1–6 全绿 + Task 7 集成测试（零硬件）。
- **M2** = Task 9 的 `list policies` 验证 + README 阶梯第 2 步命令可用；真机 `dry_run=true` episode 由操作员按 README 执行。
- **M3** = README 阶梯 1→4，操作员流程，不在本计划的自动化范围。

## Self-Review 记录

- Spec 覆盖：§2 架构（Task 1/6）、§3.1 phase 机（Task 2）、§3.2 两段式与校验（Task 3）、§3.3 provider（Task 3）、§3.4 Policy 类（Task 6）、§3.5 示教/持有推断/字段映射（Task 4/8）、§3.6 执行器与门禁（Task 5）、§4 错误处理（Task 3 失败即抛、Task 5 stall、Task 6 dry_run）、§5 测试（Task 3–7、9）、§6 目录与参数（Task 1/9）。无缺口。
- 类型一致性：`DecisionsClient.reset()`（Task 3 定义、Task 6 调用）、`Executor.last_widths`（Task 5 定义、Task 4/6 经参数消费）、`load_targets`（Task 4 定义、Task 6/8 消费）、规则名 `phase_axes/held_open/handoff_release_requires_right_secured/bounds/separation/dry_run`（Task 5 定义、Task 5/7 断言）已核对一致。
- 无占位符；两处「实现注意」给出了精确替换代码。



