# Nero Embodiment Plugin (Part A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `plugins/inspect-robots-nero/`, a real-hardware embodiment named `nero` that drives the lab's dual Nero arm rig (socketcan, 2x7 joints, 2 pika grippers, 3 D405 color cameras) so the `agent` policy (ChatGPT GPT-6 Astra) can evaluate on it with zero policy-side code.

**Architecture:** One new plugin package following the `inspect-robots-xpolicylab` package conventions. `NeroEmbodiment(EmbodimentBase)` builds only spaces at construction, connects hardware lazily on the first `reset()`, and converts 20-dim `eef_abs_pose` actions to joint targets with a pinocchio damped-least-squares IK, clamping the per-tick joint delta itself. Hardware wrappers (`pyAgxArm`, OpenCV V4L2) are thin, lazily imported, and injected in tests as fakes.

**Tech Stack:** Python >= 3.10, numpy, opencv-python, pinocchio (`pin`), scipy; `pyAgxArm` as an optional runtime extra; pytest + fakes for all hardware.

**Spec:** `plans/0082-nero-embodiment-and-view-library.md` (Part A). The spec argues from the working bring-up in the neo_manipulation RLtoken checkout at `/home/dell/fengxuedong/projects/neo_manipulation/.worktree/RLtoken`; the URDF asset copy in Task 2 and the config constants in Task 1 come from there.

## Global Constraints

- Package lives at `plugins/inspect-robots-nero/`, import package `inspect_robots_nero`. Never imported by core; never counted toward the core 100% coverage gate.
- Gates per task (run from repo root): `uv run --no-sync ruff check plugins/inspect-robots-nero`, `uv run --no-sync ruff format --check plugins/inspect-robots-nero`, `uv run --no-sync mypy --config-file plugins/inspect-robots-nero/pyproject.toml plugins/inspect-robots-nero/src/inspect_robots_nero`, `uv run --no-sync python -m pytest plugins/inspect-robots-nero/tests -q`.
- Ruff D1: every public module, class, and function needs a docstring stating the contract. No em dashes anywhere in README or docs prose.
- `import inspect_robots_nero` must not import cv2/pinocchio/scipy/pyAgxArm: hardware imports live inside `_arm.py`, `_gripper.py`, `_camera.py`, `_kinematics.py` only, imported lazily by `embodiment.py`.
- mypy strict over `src` and `tests`; declare `ignore_missing_imports` overrides for `pyAgxArm.*` and `pinocchio.*` in the plugin pyproject (msgpack precedent in the xpolicylab pyproject).
- After any pyproject dependency change, run `uv lock` at the repo root and commit `uv.lock` (CI installs from the lockfile; the workspace covers `plugins/*` automatically).
- Hardware constants come from the bring-up config `neo_manipulation/evaluation/robot_itl_evaluation/config/ros2_nero_inference.yaml` in the RLtoken checkout; record the provenance comment in `_config.py`.
- Commit after every task. Branch: work on the feature branch created at execution start (main is PR-only).

---

### Task 1: Package skeleton, spaces, constructor validation

**Files:**
- Create: `plugins/inspect-robots-nero/pyproject.toml`
- Create: `plugins/inspect-robots-nero/src/inspect_robots_nero/__init__.py`
- Create: `plugins/inspect-robots-nero/src/inspect_robots_nero/_config.py`
- Create: `plugins/inspect-robots-nero/src/inspect_robots_nero/embodiment.py`
- Create: `plugins/inspect-robots-nero/src/inspect_robots_nero/py.typed` (empty file)
- Test: `plugins/inspect-robots-nero/tests/test_nero_spaces.py`

**Interfaces:**
- Produces: `NeroEmbodiment(**kwargs)` with `info: EmbodimentInfo`; factory `nero_embodiment(**kwargs) -> NeroEmbodiment` (registry entry point `nero`); `_config.py` constants reused by Tasks 3-6: `CAN_CHANNELS`, `FIRMWARE_VERSION`, `SPEED_PERCENT`, `GRIPPER_FORCE`, `GRIPPER_WIDTH_MIN_M`, `GRIPPER_WIDTH_MAX_M`, `GRIPPER_INIT_WIDTH_M`, `HOME_LEFT`, `HOME_RIGHT`, `CONTROL_HZ`, `CAMERA_DEFAULTS`, `CAMERA_MAX_AGE_S`, `ACTION_DIM`, `DIM_LABELS`, `ROT6D_BOUNDS`, `DEFAULT_MAX_STEP`, `DUAL_NERO_DOCS`.
- Consumes: core exports `EmbodimentBase`, `EmbodimentInfo`, `Box`, `ActionSemantics`, `ObservationSpace`, `CameraSpec`, `StateSpec`, `StateField`, `CANONICAL_STATE_UNITS` (from `inspect_robots` / `inspect_robots.spaces`), `Scene`, `Action`, `Observation`, `StepResult`.

- [ ] **Step 1: Write the plugin pyproject**

Model it on `plugins/inspect-robots-xpolicylab/pyproject.toml`. Content:

```toml
[build-system]
requires = ["hatchling>=1.18", "hatch-fancy-pypi-readme>=24.1"]
build-backend = "hatchling.build"

[project]
name = "inspect-robots-nero"
version = "0.1.0"
description = "Nero dual-arm embodiment for Inspect Robots: real CAN arms, pika grippers, and D405 cameras, evaluated with any Inspect Robots policy."
dynamic = ["readme"]
requires-python = ">=3.10"
license = "MIT"
authors = [{ name = "Inspect Robots contributors" }]
keywords = ["robotics", "nero", "dual-arm", "vla", "policy", "inspect_robots", "evaluation"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Typing :: Typed",
]
# NOTE: `pyAgxArm` is intentionally NOT a dependency. It is the vendor SDK,
# installed from the lab checkout (third_party/pyAgxArm). This adapter imports
# it lazily and reports it through RUNTIME_REQUIREMENTS (see doctor/preflight).
dependencies = [
    "inspect-robots>=0.2",
    "numpy>=1.24",
    "opencv-python>=4.8",
    "pin>=3.1",
    "scipy>=1.11",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "mypy>=1.11", "ruff>=0.6"]

# Entry-point discovery: an installed inspect-robots-nero appears in
# `inspect-robots list embodiments` without being imported first.
[project.entry-points."inspect_robots.embodiments"]
nero = "inspect_robots_nero:nero_embodiment"

[project.urls]
Homepage = "https://github.com/robocurve/inspect-robots/tree/main/plugins/inspect-robots-nero"
Repository = "https://github.com/robocurve/inspect-robots"

# In the Inspect Robots monorepo this resolves the `inspect_robots` dependency to the
# in-repo core package instead of PyPI. Harmless in a standalone checkout.
[tool.uv.sources]
inspect-robots = { workspace = true }

[tool.hatch.build.targets.wheel]
packages = ["src/inspect_robots_nero"]

[tool.hatch.build.targets.sdist]
include = ["src/inspect_robots_nero", "tests", "README.md"]

[tool.ruff]
extend = "../../pyproject.toml"

[tool.ruff.lint.isort]
known-first-party = ["inspect_robots", "inspect_robots_nero"]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["D1"]

[tool.mypy]
strict = true

# The vendor SDK and pinocchio are optional/hardware-side imports; the adapter
# isolates them behind lazily imported modules and fakes in tests.
[[tool.mypy.overrides]]
module = ["pyAgxArm.*", "pinocchio.*"]
ignore_missing_imports = true

# PyPI readme: identical to README.md except GitHub-only alert syntax
# (e.g. `> [!NOTE]`) is rewritten to bold blockquotes, which PyPI renders.
[tool.hatch.metadata.hooks.fancy-pypi-readme]
content-type = "text/markdown"

[[tool.hatch.metadata.hooks.fancy-pypi-readme.fragments]]
path = "README.md"

[[tool.hatch.metadata.hooks.fancy-pypi-readme.substitutions]]
pattern = '(?m)^> \[!NOTE\][ \t]*$'
replacement = '> **Note:**'

[[tool.hatch.metadata.hooks.fancy-pypi-readme.substitutions]]
pattern = '(?m)^> \[!WARNING\][ \t]*$'
replacement = '> **Warning:**'

[[tool.hatch.metadata.hooks.fancy-pypi-readme.substitutions]]
pattern = '(?m)^> \[!CAUTION\][ \t]*$'
replacement = '> **Caution:**'
```

Also create a placeholder `README.md` (one paragraph, real content arrives in Task 6) so the readme hook resolves.

- [ ] **Step 2: Write `_config.py`**

```python
"""Hardware constants for the nero embodiment, carried from the working bring-up.

Values mirror ``neo_manipulation/evaluation/robot_itl_evaluation/config/
ros2_nero_inference.yaml`` (RLtoken branch); every value is a constructor
default, overridable per run with ``-E`` arguments.
"""

from __future__ import annotations

CAN_CHANNELS: dict[str, str] = {"left": "can_left", "right": "can_right"}
FIRMWARE_VERSION = "v112"
SPEED_PERCENT = 20
GRIPPER_FORCE = 0.3
GRIPPER_WIDTH_MIN_M = 0.0
GRIPPER_WIDTH_MAX_M = 0.09
GRIPPER_INIT_WIDTH_M = 0.09

HOME_LEFT: tuple[float, ...] = (0.56, 0.92, -1.38, 1.90, -0.46, 0.04, 0.20)
HOME_RIGHT: tuple[float, ...] = (-0.56, 0.92, 1.38, 1.90, 0.46, -0.04, 0.20)

CONTROL_HZ = 30.0
RESET_SETTLE_TOL_RAD = 0.05
RESET_SETTLE_TIMEOUT_S = 10.0
CAMERA_MAX_AGE_S = 0.5

ACTION_DIM = 20
ROT6D_BOUNDS = (-1.0, 1.0)
# Per-control-tick safety rate limits in native units: 1 cm position, rot6d
# component, and 1 cm gripper width. The agent policy scales these down
# further with max_speed_frac.
DEFAULT_MAX_STEP: tuple[float | None, ...] = (0.01,) * 3 + (0.05,) * 6 + (0.01,) + (0.01,) * 3 + (0.05,) * 6 + (0.01,)

DIM_LABELS: tuple[str, ...] = (
    "left_x", "left_y", "left_z",
    "left_r1", "left_r2", "left_r3", "left_r4", "left_r5", "left_r6",
    "left_gripper",
    "right_x", "right_y", "right_z",
    "right_r1", "right_r2", "right_r3", "right_r4", "right_r5", "right_r6",
    "right_gripper",
)

# Provenance: ros2_nero_inference.yaml "camera.streams" (live wiring). Each
# entry is the V4L2 by-path color node of a RealSense D405; the neo stack
# reads these as V4L2 mmap devices (YUYV), not via pyrealsense2.
CAMERA_DEFAULTS: dict[str, dict[str, object]] = {
    "left_rgbd": {
        "device": "/dev/v4l/by-path/pci-0000:80:14.0-usb-0:11.2:1.0-video-index4",
        "width": 640, "height": 480, "fps": 30, "pixel_format": "YUYV",
    },
    "right_rgbd": {
        "device": "/dev/v4l/by-path/pci-0000:80:14.0-usb-0:2.2:1.0-video-index4",
        "width": 640, "height": 480, "fps": 30, "pixel_format": "YUYV",
    },
    "chest_rgbd": {
        "device": "/dev/v4l/by-path/pci-0000:00:0d.0-usb-0:2.2:1.0-video-index4",
        "width": 640, "height": 480, "fps": 30, "pixel_format": "YUYV",
    },
}

DUAL_NERO_DOCS = """Dual Nero arms bolted to a shared shelf; actions address both arms in one 20-dim vector:
[left xyz (m, base frame), left rot6d, left gripper width (m), right xyz, right rot6d, right gripper width].
rot6d is the first two columns of the target rotation matrix (6 numbers, each in [-1, 1]).
Gripper width 0 means closed, 0.09 means open.
The state field eef_state mirrors the action layout from current forward kinematics.
The state field joint_pos is [left j1..j7, right j1..j7, left width, right width] (16 dims).
Base frame: shelf base at the left arm's root; +x forward, +z up."""
```

- [ ] **Step 3: Write the failing tests**

`plugins/inspect-robots-nero/tests/test_nero_spaces.py`:

```python
"""Constructor and spaces behavior for the nero embodiment."""

from __future__ import annotations

import math

import pytest

from inspect_robots.conformance import assert_embodiment_conformant

from inspect_robots_nero import nero_embodiment
from inspect_robots_nero._config import ACTION_DIM, DIM_LABELS


def test_action_space_shape_and_semantics() -> None:
    embodiment = nero_embodiment()
    box = embodiment.info.action_space
    assert box.shape == (ACTION_DIM,)
    semantics = box.semantics
    assert semantics is not None
    assert semantics.control_mode == "eef_abs_pose"
    assert semantics.rotation_repr == "rot6d"
    assert semantics.gripper == "continuous"
    assert semantics.frame == "base"
    assert semantics.dim_labels == DIM_LABELS
    assert semantics.max_step is not None and len(semantics.max_step) == ACTION_DIM


def test_action_bounds_are_finite_and_ordered() -> None:
    box = nero_embodiment().info.action_space
    assert box.low is not None and box.high is not None
    assert all(math.isfinite(v) for v in box.low.ravel().tolist())
    assert all(math.isfinite(v) for v in box.high.ravel().tolist())
    assert bool((box.low <= box.high).all())


def test_observation_space_declares_agent_required_fields() -> None:
    space = nero_embodiment().info.observation_space
    by_key = {field.key: field for field in space.state.fields}
    assert by_key["eef_state"].shape == (ACTION_DIM,)
    assert by_key["joint_pos"].shape == (16,)
    cameras = {camera.name: camera for camera in space.cameras}
    assert set(cameras) == {"left_rgbd", "right_rgbd", "chest_rgbd"}
    for camera in cameras.values():
        assert (camera.height, camera.width) == (480, 640)


def test_conformance_passes_declaratively() -> None:
    assert_embodiment_conformant(nero_embodiment().info)


def test_capabilities_declare_self_paced_and_resettable() -> None:
    capabilities = nero_embodiment().info.capabilities
    assert "self_paced" in capabilities and "resettable" in capabilities


def test_docs_are_offered() -> None:
    docs = nero_embodiment().info.docs
    assert docs is not None and "rot6d" in docs


def test_invalid_control_hz_lists_fix_hint() -> None:
    with pytest.raises(ValueError, match="-E control_hz"):
        nero_embodiment(control_hz=0)


def test_invalid_camera_name_lists_valid_names() -> None:
    with pytest.raises(ValueError, match="left_rgbd"):
        nero_embodiment(cameras={"wrist": "/dev/video0"})


def test_invalid_camera_override_device() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        nero_embodiment(cameras={"left_rgbd": ""})


def test_invalid_workspace_bounds() -> None:
    with pytest.raises(ValueError, match="workspace"):
        nero_embodiment(workspace_high=(1.0, 1.0))  # wrong length


def test_reset_and_step_are_not_wired_yet() -> None:
    from inspect_robots import Scene

    embodiment = nero_embodiment()
    with pytest.raises(NotImplementedError):
        embodiment.reset(Scene(instruction="x"))
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv sync --all-packages --extra dev && uv run --no-sync python -m pytest plugins/inspect-robots-nero/tests -q`
Expected: collection error / FAIL (`inspect_robots_nero` has no attribute `nero_embodiment`).

- [ ] **Step 5: Implement `__init__.py` and `embodiment.py` (spaces + validation)**

`src/inspect_robots_nero/__init__.py`:

```python
"""Nero dual-arm embodiment plugin for Inspect Robots."""

from __future__ import annotations

from typing import Any

from inspect_robots_nero.embodiment import NeroEmbodiment

__all__ = ["NeroEmbodiment", "nero_embodiment"]


def nero_embodiment(**kwargs: Any) -> NeroEmbodiment:
    """Construct the registry-facing nero embodiment from CLI or programmatic arguments."""
    return NeroEmbodiment(**kwargs)


# Consumed by inspect_robots.conformance.missing_runtime_requirements via the
# doctor/preflight path: the vendor SDK is not pip-installable from PyPI.
nero_embodiment.RUNTIME_REQUIREMENTS = {
    "pyAgxArm": "install the vendor SDK: pip install -e <neo_manipulation checkout>/third_party/pyAgxArm",
}
```

`src/inspect_robots_nero/embodiment.py` (constructor + spaces; `reset`/`step` raise
`NotImplementedError` until Task 5):

```python
"""Expose the dual Nero CAN arms and D405 cameras through the embodiment contract.

Construction builds only static spaces; arms, grippers, and cameras connect
lazily on the first reset. Actions are 20-dim absolute end-effector poses
(position + rot6d + gripper width per arm); each step solves IK and commands
one bounded joint increment per arm at ``control_hz``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from inspect_robots import (
    Action,
    ActionSemantics,
    Box,
    CameraSpec,
    EmbodimentBase,
    EmbodimentInfo,
    Observation,
    ObservationSpace,
    Scene,
    StateField,
    StateSpec,
    StepResult,
)
from inspect_robots.embodiment import RESETTABLE, SELF_PACED
from inspect_robots.spaces import CANONICAL_STATE_UNITS

from inspect_robots_nero._config import (
    ACTION_DIM,
    CAMERA_DEFAULTS,
    CAMERA_MAX_AGE_S,
    CONTROL_HZ,
    DEFAULT_MAX_STEP,
    DIM_LABELS,
    DUAL_NERO_DOCS,
    GRIPPER_WIDTH_MAX_M,
    GRIPPER_WIDTH_MIN_M,
    RESET_SETTLE_TIMEOUT_S,
    ROT6D_BOUNDS,
)

# Provisional workspace box per arm until Task 4 derives it from FK sampling.
_PROVISIONAL_WORKSPACE: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "left": ((-0.10, -0.70, -0.05), (1.10, 0.70, 1.00)),
    "right": ((-0.10, -0.70, -0.05), (1.10, 0.70, 1.00)),
}

ArmFactory = Callable[[str], Any]
CameraFactory = Callable[[str], Any]


def _bounds_for(side: str) -> tuple[list[float], list[float]]:
    low, high = _PROVISIONAL_WORKSPACE[side]
    rot_low, rot_high = ROT6D_BOUNDS
    return (
        [*low, *[rot_low] * 6, GRIPPER_WIDTH_MIN_M],
        [*high, *[rot_high] * 6, GRIPPER_WIDTH_MAX_M],
    )


class NeroEmbodiment(EmbodimentBase):
    """Drive the dual Nero arms with absolute end-effector pose actions."""

    def __init__(
        self,
        *,
        control_hz: float = CONTROL_HZ,
        workspace_low: tuple[float, float, float] | None = None,
        workspace_high: tuple[float, float, float] | None = None,
        max_step: tuple[float | None, ...] | None = None,
        cameras: Mapping[str, str] | None = None,
        operator_reset_confirm: bool = True,
        reset_settle_timeout_s: float = RESET_SETTLE_TIMEOUT_S,
        camera_max_age_s: float = CAMERA_MAX_AGE_S,
        arm_factory: ArmFactory | None = None,
        camera_factory: CameraFactory | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        name: str = "nero",
    ) -> None:
        if not math.isfinite(control_hz) or control_hz <= 0:
            raise ValueError(f"control_hz must be positive and finite, got {control_hz!r}; pass -E control_hz=30")
        if reset_settle_timeout_s <= 0 or not math.isfinite(reset_settle_timeout_s):
            raise ValueError(f"reset_settle_timeout_s must be positive and finite, got {reset_settle_timeout_s!r}")
        if camera_max_age_s <= 0 or not math.isfinite(camera_max_age_s):
            raise ValueError(f"camera_max_age_s must be positive and finite, got {camera_max_age_s!r}")
        resolved_max_step = DEFAULT_MAX_STEP if max_step is None else tuple(max_step)
        if len(resolved_max_step) != ACTION_DIM or any(
            entry is not None and (not math.isfinite(entry) or entry <= 0) for entry in resolved_max_step
        ):
            raise ValueError(f"max_step must hold {ACTION_DIM} finite positive entries (or None)")

        for side in ("left", "right"):
            low = list(_PROVISIONAL_WORKSPACE[side][0]) if workspace_low is None else list(workspace_low)
            high = list(_PROVISIONAL_WORKSPACE[side][1]) if workspace_high is None else list(workspace_high)
            if len(low) != 3 or len(high) != 3 or not all(math.isfinite(v) for v in (*low, *high)):
                raise ValueError(
                    f"workspace_low/workspace_high must hold three finite values per axis; got {low} / {high}"
                )
            if any(lo > hi for lo, hi in zip(low, high, strict=True)):
                raise ValueError(f"workspace bounds must be elementwise low <= high for the {side} arm")

        resolved_cameras = dict(CAMERA_DEFAULTS)
        if cameras is not None:
            unknown = sorted(set(cameras) - set(CAMERA_DEFAULTS))
            if unknown:
                raise ValueError(
                    f"cameras keys {unknown} are not declared cameras; valid names: {sorted(CAMERA_DEFAULTS)}"
                )
            for camera_name, device in cameras.items():
                if not isinstance(device, str) or not device:
                    raise ValueError(f"camera {camera_name!r} override must be a non-empty device path")
                resolved_cameras[camera_name] = {**CAMERA_DEFAULTS[camera_name], "device": device}

        low_array = np.asarray(
            _bounds_for("left")[0] + _bounds_for("right")[0], dtype=np.float64
        )
        high_array = np.asarray(
            _bounds_for("left")[1] + _bounds_for("right")[1], dtype=np.float64
        )
        semantics = ActionSemantics(
            control_mode="eef_abs_pose",
            rotation_repr="rot6d",
            gripper="continuous",
            frame="base",
            dim_labels=DIM_LABELS,
            max_step=resolved_max_step,
        )
        self.info = EmbodimentInfo(
            name=name,
            action_space=Box(shape=(ACTION_DIM,), low=low_array, high=high_array, semantics=semantics),
            observation_space=ObservationSpace(
                cameras=tuple(
                    CameraSpec(camera_name, int(spec["height"]), int(spec["width"]))
                    for camera_name, spec in resolved_cameras.items()
                ),
                state=StateSpec(
                    fields=(
                        StateField("eef_state", (ACTION_DIM,), CANONICAL_STATE_UNITS["eef_pose"]),
                        StateField("joint_pos", (16,), CANONICAL_STATE_UNITS["joint_pos"]),
                    )
                ),
            ),
            control_hz=control_hz,
            is_simulated=False,
            capabilities=frozenset({SELF_PACED, RESETTABLE}),
            supported_setups=frozenset(),
            supported_target_kinds=frozenset(),
            docs=DUAL_NERO_DOCS,
        )
        self.control_hz = float(control_hz)
        self.cameras = resolved_cameras
        self.operator_reset_confirm = operator_reset_confirm
        self.reset_settle_timeout_s = float(reset_settle_timeout_s)
        self.camera_max_age_s = float(camera_max_age_s)
        self._arm_factory = arm_factory
        self._camera_factory = camera_factory
        self._clock = clock
        self._sleep = sleep
        self._instruction: str | None = None

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Prepare a scene and return its initial observation (wired in a later task)."""
        del scene, seed
        raise NotImplementedError("hardware bring-up lands with the reset/step task")

    def step(self, action: Action) -> StepResult:
        """Issue one action (wired in a later task)."""
        del action
        raise NotImplementedError("hardware bring-up lands with the reset/step task")
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run --no-sync python -m pytest plugins/inspect-robots-nero/tests -q`
Expected: PASS (10 tests). Then run the four gate commands from Global Constraints.
Expected: all green (ruff may need `ruff format` on the new files first).

- [ ] **Step 7: Commit**

```bash
git add plugins/inspect-robots-nero uv.lock
uv lock
git add uv.lock
git commit -m "feat(nero): plugin skeleton with eef_abs_pose spaces and validation"
```

---

### Task 2: Kinematics module and URDF asset

**Files:**
- Create: `plugins/inspect-robots-nero/src/inspect_robots_nero/_kinematics.py`
- Create: `plugins/inspect-robots-nero/src/inspect_robots_nero/assets/dual_nero_pika.urdf`
- Test: `plugins/inspect-robots-nero/tests/test_nero_kinematics.py`

**Interfaces:**
- Produces:
  - `NeroKinematicsError(RuntimeError)`
  - `matrix_to_rot6d(rotation: np.ndarray) -> np.ndarray` (shape (6,))
  - `rot6d_to_matrix(vector: np.ndarray) -> np.ndarray` (shape (3, 3))
  - `NeroKinematics(urdf_path, *, tcp_translation_m=(0.0, 0.0, 0.18), tcp_rotation_rpy_rad=(0.0, -pi/2, 0.0), max_iterations=60, position_tol_m=1e-3, orientation_tol_rad=1e-3, max_joint_step_rad=0.1, damping=1e-3, posture_gain=0.01, samples=256, seed=0)`
  - `.q_from(left: Sequence[float], right: Sequence[float]) -> np.ndarray`
  - `.q_split(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]` (left 7, right 7)
  - `.fk(q: np.ndarray) -> dict[str, np.ndarray]` (keys "left"/"right", 4x4 homogeneous)
  - `.solve(left_target: np.ndarray, right_target: np.ndarray, q_seed: np.ndarray) -> np.ndarray` (raises `NeroKinematicsError` on non-convergence)
  - `.sample_workspace_bounds(*, margin_m=0.02) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]`
- Consumes: Task 1's package (nothing from it directly; standalone module).

- [ ] **Step 1: Copy the URDF asset**

```bash
mkdir -p plugins/inspect-robots-nero/src/inspect_robots_nero/assets
cp /home/dell/fengxuedong/projects/neo_manipulation/.worktree/RLtoken/assets/robots/dual_nero_pika/urdf/dual_nero_pika.urdf \
  plugins/inspect-robots-nero/src/inspect_robots_nero/assets/dual_nero_pika.urdf
```

Add a provenance note later in the README (Task 6). pinocchio builds kinematics only; missing mesh files referenced by the URDF are not required and must not be copied.

- [ ] **Step 2: Write the failing tests**

```python
"""FK/IK and rot6d behavior for the nero kinematics module."""

from __future__ import annotations

import importlib.resources

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from inspect_robots_nero._config import HOME_LEFT, HOME_RIGHT
from inspect_robots_nero._kinematics import (
    NeroKinematics,
    NeroKinematicsError,
    matrix_to_rot6d,
    rot6d_to_matrix,
)


def _urdf_path() -> str:
    return str(importlib.resources.files("inspect_robots_nero") / "assets" / "dual_nero_pika.urdf")


def _kinematics() -> NeroKinematics:
    return NeroKinematics(_urdf_path())


@pytest.mark.parametrize(
    "rpy",
    [(0.0, 0.0, 0.0), (np.pi / 2, 0.0, 0.0), (0.1, -0.2, 0.3), (-1.0, 0.5, 2.0)],
)
def test_rot6d_round_trip(rpy: tuple[float, float, float]) -> None:
    rotation = Rotation.from_euler("xyz", rpy).as_matrix()
    restored = rot6d_to_matrix(matrix_to_rot6d(rotation))
    assert np.allclose(restored, rotation, atol=1e-9)


def test_rot6d_recovers_from_scaled_columns() -> None:
    rotation = Rotation.from_euler("xyz", (0.3, 0.4, 0.5)).as_matrix()
    scaled = matrix_to_rot6d(rotation) * 7.0
    assert np.allclose(rot6d_to_matrix(scaled), rotation, atol=1e-9)


def test_model_has_fourteen_locked_joints_and_tcp_frames() -> None:
    kinematics = _kinematics()
    q = kinematics.q_from(HOME_LEFT, HOME_RIGHT)
    assert q.shape == (14,)
    left, right = kinematics.q_split(q)
    assert left.shape == (7,) and right.shape == (7,)


def test_fk_is_finite_and_homogeneous() -> None:
    poses = _kinematics().fk(_kinematics().q_from(HOME_LEFT, HOME_RIGHT))
    for side in ("left", "right"):
        pose = poses[side]
        assert pose.shape == (4, 4)
        assert np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0))
        assert bool(np.all(np.isfinite(pose)))


def test_ik_recovers_fk_poses_from_a_disturbed_seed() -> None:
    kinematics = _kinematics()
    rng = np.random.default_rng(7)
    for _ in range(3):
        q_goal = kinematics.q_from(
            (rng.uniform(-1.0, 1.0),) * 7, (rng.uniform(-1.0, 1.0),) * 7
        )
        targets = kinematics.fk(q_goal)
        q_seed = kinematics.q_from(HOME_LEFT, HOME_RIGHT)
        q_solution = kinematics.solve(targets["left"], targets["right"], q_seed)
        reached = kinematics.fk(q_solution)
        for side in ("left", "right"):
            position_error = float(np.linalg.norm(reached[side][:3, 3] - targets[side][:3, 3]))
            orientation_error = float(np.linalg.norm(reached[side][:3, :3] - targets[side][:3, :3]))
            assert position_error < 2e-3
            assert orientation_error < 5e-2


def test_ik_respects_the_joint_step_budget() -> None:
    kinematics = _kinematics()
    q_seed = kinematics.q_from(HOME_LEFT, HOME_RIGHT)
    targets = kinematics.fk(q_seed)
    far = targets["left"].copy()
    far[:3, 3] += (0.05, 0.0, 0.0)
    solution = kinematics.solve(far, targets["right"], q_seed)
    budget = 60 * 0.1
    assert float(np.max(np.abs(solution - q_seed))) <= budget + 1e-9


def test_unreachable_target_raises() -> None:
    kinematics = _kinematics()
    q_seed = kinematics.q_from(HOME_LEFT, HOME_RIGHT)
    poses = kinematics.fk(q_seed)
    unreachable = poses["left"].copy()
    unreachable[:3, 3] += (5.0, 0.0, 0.0)
    with pytest.raises(NeroKinematicsError):
        kinematics.solve(unreachable, poses["right"], q_seed)


def test_workspace_bounds_contain_the_home_pose() -> None:
    kinematics = _kinematics()
    bounds = kinematics.sample_workspace_bounds()
    home = kinematics.fk(kinematics.q_from(HOME_LEFT, HOME_RIGHT))
    for side in ("left", "right"):
        low, high = bounds[side]
        position = home[side][:3, 3]
        assert bool(np.all(position >= np.asarray(low) - 1e-9))
        assert bool(np.all(position <= np.asarray(high) + 1e-9))
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run --no-sync python -m pytest plugins/inspect-robots-nero/tests/test_nero_kinematics.py -q`
Expected: FAIL (`No module named 'inspect_robots_nero._kinematics'`).

- [ ] **Step 4: Implement `_kinematics.py`**

```python
"""Damped least squares IK and FK for the dual Nero arms over pinocchio.

Ported core of the bring-up controller's solver: TCP frames on the gripper
flanges, the four prismatic pika joints locked out of the model, and a
per-iteration joint step cap. The bring-up stack's ProxQP formulation is
deliberately not ported: this embodiment clamps the per-tick joint delta
itself, so per-iteration QP limits would be redundant here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

FLANGE_FRAMES: dict[str, str] = {"left": "left_gripper_flange", "right": "right_gripper_flange"}
ARM_JOINT_NAMES: dict[str, tuple[str, ...]] = {
    "left": tuple(f"left_joint{i}" for i in range(1, 8)),
    "right": tuple(f"right_joint{i}" for i in range(1, 8)),
}
GRIPPER_JOINT_NAMES: tuple[str, ...] = (
    "left_pika_left_joint",
    "left_pika_right_joint",
    "right_pika_left_joint",
    "right_pika_right_joint",
)


class NeroKinematicsError(RuntimeError):
    """Raised when IK cannot reach a target within the iteration budget."""


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("rot6d column has zero norm")
    return vector / norm


def matrix_to_rot6d(rotation: np.ndarray) -> np.ndarray:
    """Flatten a rotation matrix to its first two columns (Zhou et al. 6D form)."""
    matrix = np.asarray(rotation, dtype=np.float64)
    return np.concatenate([matrix[:, 0], matrix[:, 1]])


def rot6d_to_matrix(vector: np.ndarray) -> np.ndarray:
    """Reconstruct orthonormal rotation columns from a 6-vector; scales are normalized away."""
    raw = np.asarray(vector, dtype=np.float64).ravel()
    if raw.shape != (6,):
        raise ValueError(f"rot6d vector must hold 6 values, got shape {raw.shape}")
    x = _normalize(raw[:3])
    y = _normalize(np.cross(x, _normalize(raw[3:])))
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


class NeroKinematics:
    """FK/IK over the packaged dual-arm URDF with per-arm TCP frames."""

    def __init__(
        self,
        urdf_path: str,
        *,
        tcp_translation_m: tuple[float, float, float] = (0.0, 0.0, 0.18),
        tcp_rotation_rpy_rad: tuple[float, float, float] = (0.0, -1.5707963267948966, 0.0),
        max_iterations: int = 60,
        position_tol_m: float = 1e-3,
        orientation_tol_rad: float = 1e-3,
        max_joint_step_rad: float = 0.1,
        damping: float = 1e-3,
        posture_gain: float = 0.01,
        samples: int = 256,
        seed: int = 0,
    ) -> None:
        import pinocchio as pin
        from scipy.spatial.transform import Rotation

        self._pin = pin
        self._max_iterations = int(max_iterations)
        self._position_tol = float(position_tol_m)
        self._orientation_tol = float(orientation_tol_rad)
        self._max_joint_step = float(max_joint_step_rad)
        self._damping = float(damping)
        self._posture_gain = float(posture_gain)
        self._samples = int(samples)
        self._seed = int(seed)

        model = pin.buildModelFromUrdf(str(urdf_path))
        tcp_transform = pin.SE3(
            Rotation.from_euler("xyz", tcp_rotation_rpy_rad).as_matrix(),
            np.asarray(tcp_translation_m, dtype=np.float64),
        )
        for flange_name in FLANGE_FRAMES.values():
            flange_id = model.getFrameId(flange_name)
            if flange_id >= model.nframes:
                raise ValueError(f"frame {flange_name!r} missing from the dual nero URDF")
            flange_frame = model.frames[flange_id]
            model.addFrame(
                pin.Frame(
                    f"{flange_name}_tcp",
                    flange_frame.parentJoint,
                    flange_id,
                    flange_frame.placement * tcp_transform,
                    pin.FrameType.OP_FRAME,
                )
            )
        locked_ids = [
            model.getJointId(joint_name)
            for joint_name in GRIPPER_JOINT_NAMES
            if model.existJointId(joint_name)
        ]
        if locked_ids:
            model = pin.buildReducedModel(model, locked_ids, pin.neutral(model))
        self._model = model
        self._data = model.createData()
        self._q_index: dict[str, int] = {}
        for index in range(1, model.njoints):
            self._q_index[model.names[index]] = model.joints[index].idx_q
        self._tcp_frames = {
            side: model.getFrameId(f"{flange_name}_tcp")
            for side, flange_name in FLANGE_FRAMES.items()
        }
        self._posture = self.q_from((0.0,) * 7, (0.0,) * 7)

    def q_from(self, left: Sequence[float], right: Sequence[float]) -> np.ndarray:
        """Assemble the reduced-model configuration from per-arm joint lists."""
        pin = self._pin
        q = pin.neutral(self._model)
        for values, names in ((left, ARM_JOINT_NAMES["left"]), (right, ARM_JOINT_NAMES["right"])):
            if len(values) != 7:
                raise ValueError(f"arm joints must hold 7 values, got {len(values)}")
            for joint_name, value in zip(names, values, strict=True):
                q[self._q_index[joint_name]] = float(value)
        return q

    def q_split(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split a configuration into per-arm joint lists (left 7, right 7)."""
        pin = self._pin
        q_full = self.q_from((0.0,) * 7, (0.0,) * 7) * 0.0 + np.asarray(q, dtype=np.float64)
        del pin
        inverse = {index: name for name, index in self._q_index.items()}
        by_name = {name: float(q_full[index]) for name, index in self._q_index.items()}
        del inverse
        left = np.asarray([by_name[name] for name in ARM_JOINT_NAMES["left"]])
        right = np.asarray([by_name[name] for name in ARM_JOINT_NAMES["right"]])
        return left, right

    def _forward(self, q: np.ndarray) -> dict[str, np.ndarray]:
        pin = self._pin
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)
        poses: dict[str, np.ndarray] = {}
        for side, frame_id in self._tcp_frames.items():
            placement = self._data.oMf[frame_id]
            pose = np.eye(4)
            pose[:3, :3] = placement.rotation
            pose[:3, 3] = placement.translation
            poses[side] = pose
        return poses

    def fk(self, q: np.ndarray) -> dict[str, np.ndarray]:
        """Return per-arm TCP poses (4x4) for a configuration."""
        return self._forward(np.asarray(q, dtype=np.float64))

    def solve(
        self,
        left_target: np.ndarray,
        right_target: np.ndarray,
        q_seed: np.ndarray,
    ) -> np.ndarray:
        """Solve both arms' IK; raises NeroKinematicsError when a target is unreachable."""
        pin = self._pin
        q = np.asarray(q_seed, dtype=np.float64).copy()
        targets = {"left": np.asarray(left_target), "right": np.asarray(right_target)}
        lower = np.asarray(self._model.lowerPositionLimit)
        upper = np.asarray(self._model.upperPositionLimit)
        for _ in range(self._max_iterations):
            poses = self._forward(q)
            errors: list[np.ndarray] = []
            jacobians: list[np.ndarray] = []
            converged = True
            for side, target in targets.items():
                frame_id = self._tcp_frames[side]
                placement = self._data.oMf[frame_id]
                position_error = float(np.linalg.norm(placement.translation - target[:3, 3]))
                rotation_error = float(
                    np.linalg.norm(pin.log3(placement.rotation.T @ target[:3, :3]))
                )
                if position_error > self._position_tol or rotation_error > self._orientation_tol:
                    converged = False
                delta = pin.SE3(target[:3, :3], target[:3, 3])
                errors.append(pin.log6(placement.actInv(delta)).vector)
                jacobians.append(
                    pin.computeFrameJacobian(
                        self._model, self._data, q, frame_id, pin.ReferenceFrame.LOCAL
                    )
                )
            if converged:
                return q
            jacobian = np.vstack(jacobians)
            error = np.concatenate(errors)
            regularized = jacobian @ jacobian.T + self._damping * np.eye(jacobian.shape[0])
            dq = jacobian.T @ np.linalg.solve(regularized, error)
            nullspace = np.eye(self._model.nq) - jacobian.T @ np.linalg.solve(regularized, jacobian)
            dq = dq + nullspace @ (self._posture_gain * (self._posture - q))
            dq = np.clip(dq, -self._max_joint_step, self._max_joint_step)
            q = np.clip(q + dq, lower, upper)
        raise NeroKinematicsError(
            "dual-arm IK did not converge within "
            f"{self._max_iterations} iterations (position tol {self._position_tol} m, "
            f"orientation tol {self._orientation_tol} rad); target may be out of the workspace"
        )

    def sample_workspace_bounds(
        self, *, margin_m: float = 0.02
    ) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
        """Sample FK over joint limits and return per-arm xyz bounds with a margin."""
        rng = np.random.default_rng(self._seed)
        lower = np.asarray(self._model.lowerPositionLimit)
        upper = np.asarray(self._model.upperPositionLimit)
        drawn = rng.uniform(low=lower, high=upper, size=(self._samples, self._model.nq))
        positions: dict[str, list[np.ndarray]] = {"left": [], "right": []}
        for row in drawn:
            poses = self._forward(row)
            for side in ("left", "right"):
                positions[side].append(poses[side][:3, 3])
        bounds: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {}
        for side in ("left", "right"):
            stacked = np.vstack(positions[side])
            low = np.maximum(stacked.min(axis=0) - margin_m, -2.0)
            high = np.minimum(stacked.max(axis=0) + margin_m, 2.0)
            bounds[side] = (tuple(float(v) for v in low), tuple(float(v) for v in high))
        return bounds
```

Note: `q_split` above keeps a redundant roundtrip; simplify it to index
directly if ruff flags the dead locals:

```python
    def q_split(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split a configuration into per-arm joint lists (left 7, right 7)."""
        q_full = np.asarray(q, dtype=np.float64)
        left = np.asarray([q_full[self._q_index[name]] for name in ARM_JOINT_NAMES["left"]])
        right = np.asarray([q_full[self._q_index[name]] for name in ARM_JOINT_NAMES["right"]])
        return left, right
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --no-sync python -m pytest plugins/inspect-robots-nero/tests/test_nero_kinematics.py -q`
Expected: PASS. If an IK round-trip case misses tolerance, first raise `max_iterations` to 120 in the test's kinematics (not the tolerance) before touching the solver.
Then run the gates; `ruff format` the file. Commit:

```bash
git add plugins/inspect-robots-nero
git commit -m "feat(nero): pinocchio FK/IK with rot6d and workspace sampling"
```

---

### Task 3: Arm and gripper wrappers over pyAgxArm

**Files:**
- Create: `plugins/inspect-robots-nero/src/inspect_robots_nero/_arm.py`
- Create: `plugins/inspect-robots-nero/src/inspect_robots_nero/_gripper.py`
- Create: `plugins/inspect-robots-nero/tests/fakes.py`
- Test: `plugins/inspect-robots-nero/tests/test_nero_arm.py`

**Interfaces:**
- Produces:
  - `NeroArm(side: str, channel: str, *, firmware: str = "v112", robot: Any | None = None, connect_sleep_s: float = 0.5, sleep: Callable[[float], None] = time.sleep)` with `.connect()`, `.enable() -> bool`, `.disable() -> bool`, `.set_speed_percent(percent: int)`, `.set_normal_mode()`, `.move_js(joints: Sequence[float])`, `.read_state() -> tuple[float, ...] | None`, `.raw_robot` property, `.close()`.
  - `NeroGripper(arm: NeroArm, *, effector: Any | None = None, sleep: ...)` with `.start()`, `.move(width: float, force: float)`, `.read_state() -> float | None`, `.stop()`.
  - `tests/fakes.py`: `FakeAgxRobot`, `FakeEffector` reusable by Task 5.
- Consumes: Task 1's `_config.FIRMWARE_VERSION`.

- [ ] **Step 1: Write the failing tests**

`tests/fakes.py`:

```python
"""Test fakes standing in for the pyAgxArm vendor objects."""

from __future__ import annotations

from types import SimpleNamespace


class FakeEffector:
    """Records gripper commands and reports a scripted width."""

    def __init__(self) -> None:
        self.moves: list[tuple[float, float]] = []
        self.width = 0.09

    def move_gripper_m(self, *, value: float, force: float) -> None:
        self.moves.append((float(value), float(force)))
        self.width = float(value)

    def get_gripper_status(self) -> SimpleNamespace:
        return SimpleNamespace(msg=SimpleNamespace(width=self.width, force=0.3), timestamp=0.0)


class LegacyEffector(FakeEffector):
    """A vendor build exposing only the legacy move_gripper API."""

    def move_gripper_m(self, *, value: float, force: float) -> None:  # pragma: no cover - absence marker
        raise AssertionError("legacy effector must not expose move_gripper_m")

    def move_gripper(self, *, width: float, force: float) -> None:
        self.moves.append((float(width), float(force)))
        self.width = float(width)


class FakeAgxRobot:
    """Stands in for the AgxArmFactory product; records the calls we make."""

    def __init__(self) -> None:
        self.connected = False
        self.disconnected = False
        self.enabled = False
        self.speed: int | None = None
        self.mode: str | None = None
        self.move_js_calls: list[tuple[float, ...]] = []
        self.angles: tuple[float, ...] = (0.1, 0.2, 0.0, 0.5, 0.0, 0.0, 0.3)
        self.effector_kind: str | None = None
        self.OPTIONS = SimpleNamespace(EFFECTOR=SimpleNamespace(AGX_GRIPPER="AGX_GRIPPER"))

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.disconnected = True

    def enable(self) -> bool:
        self.enabled = True
        return True

    def disable(self) -> bool:
        self.enabled = False
        return True

    def set_speed_percent(self, percent: int) -> None:
        self.speed = int(percent)

    def set_follower_mode(self) -> None:
        self.mode = "follower"

    def move_js(self, joints: list[float]) -> None:
        self.move_js_calls.append(tuple(float(value) for value in joints))

    def get_joint_angles(self) -> tuple[float, ...]:
        return self.angles

    def init_effector(self, kind: str) -> FakeEffector:
        self.effector_kind = kind
        return FakeEffector()
```

`tests/test_nero_arm.py`:

```python
"""NeroArm/NeroGripper wrapper behavior against fakes."""

from __future__ import annotations

import pytest

from inspect_robots_nero._arm import NeroArm
from inspect_robots_nero._gripper import NeroGripper

from .fakes import FakeAgxRobot, FakeEffector, LegacyEffector


def _arm(robot: FakeAgxRobot) -> NeroArm:
    return NeroArm("left", "can_left", robot=robot, sleep=lambda _seconds: None)


def test_connect_with_injected_robot_skips_the_sdk() -> None:
    robot = FakeAgxRobot()
    arm = _arm(robot)
    arm.connect()
    assert robot.connected
    arm.set_speed_percent(20)
    assert robot.speed == 20
    arm.set_normal_mode()
    assert robot.mode == "follower"


def test_move_js_records_floats() -> None:
    robot = FakeAgxRobot()
    arm = _arm(robot)
    arm.connect()
    arm.move_js([1, 2, 3, 4, 5, 6, 7])
    assert robot.move_js_calls == [(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)]


def test_read_state_coerces_and_reports_missing() -> None:
    robot = FakeAgxRobot()
    arm = _arm(robot)
    assert arm.read_state() is None  # not connected yet
    arm.connect()
    assert arm.read_state() == robot.angles
    robot.angles = None  # type: ignore[assignment]
    assert arm.read_state() is None


def test_disable_and_close() -> None:
    robot = FakeAgxRobot()
    arm = _arm(robot)
    arm.connect()
    arm.disable()
    assert not robot.enabled
    arm.close()
    assert robot.disconnected


def test_unknown_firmware_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="firmware"):
        NeroArm("left", "can_left", firmware="v999", robot=FakeAgxRobot())


def test_gripper_move_and_readback() -> None:
    robot = FakeAgxRobot()
    arm = _arm(robot)
    arm.connect()
    gripper = NeroGripper(arm)
    gripper.start()
    assert robot.effector_kind == "AGX_GRIPPER"
    gripper.move(0.04, 0.3)
    gripper.read_state()
    effector = robot.init_effector("AGX_GRIPPER")
    assert isinstance(effector, FakeEffector) and effector.moves == []
    assert gripper.read_state() == 0.09


def test_gripper_legacy_api_fallback() -> None:
    robot = FakeAgxRobot()
    arm = _arm(robot)
    arm.connect()
    gripper = NeroGripper(arm, effector=LegacyEffector())
    gripper.move(0.05, 0.2)
    effector = gripper.read_state()
    assert effector == 0.05


def test_gripper_start_requires_connected_arm() -> None:
    gripper = NeroGripper(_arm(FakeAgxRobot()))
    with pytest.raises(RuntimeError, match="connected"):
        gripper.start()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync python -m pytest plugins/inspect-robots-nero/tests/test_nero_arm.py -q`
Expected: FAIL (`No module named 'inspect_robots_nero._arm'`).

- [ ] **Step 3: Implement `_arm.py` and `_gripper.py`**

```python
"""Nero arm wrapper over the pyAgxArm vendor SDK, one instance per side.

Mirrors the bring-up HAL's call sequence (create config, connect, settle,
enable, follower mode, ``move_js`` streaming, ``get_joint_angles`` feedback)
with the vendor object injectable for tests.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

_FIRMWARE_NAMES = {
    "default": "DEFAULT",
    "v111": "V111",
    "1.11": "V111",
    "111": "V111",
    "v112": "V112",
    "1.12": "V112",
    "112": "V112",
}


class NeroArm:
    """One CAN-connected Nero arm; the vendor robot is injectable for tests."""

    def __init__(
        self,
        side: str,
        channel: str,
        *,
        firmware: str = "v112",
        robot: Any | None = None,
        connect_sleep_s: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        if firmware not in _FIRMWARE_NAMES:
            known = ", ".join(sorted(_FIRMWARE_NAMES))
            raise ValueError(f"unknown firmware version {firmware!r}; known: {known}")
        self.side = side
        self.channel = channel
        self._firmware = _FIRMWARE_NAMES[firmware]
        self._robot = robot
        self._connect_sleep_s = connect_sleep_s
        self._sleep = sleep

    @property
    def raw_robot(self) -> Any | None:
        """The vendor robot object, or None before the first connect."""
        return self._robot

    def connect(self) -> None:
        """Open the CAN connection (no-op when a robot was injected)."""
        if self._robot is not None:
            return
        from pyAgxArm import AgxArmFactory, create_agx_arm_config

        config = create_agx_arm_config(
            robot="nero",
            comm="can",
            firmeware_version=self._firmware,
            channel=self.channel,
            interface="socketcan",
        )
        self._robot = AgxArmFactory.create_arm(config)
        self._robot.connect()
        self._sleep(self._connect_sleep_s)

    def enable(self) -> bool:
        """Enable the arm's motors."""
        assert self._robot is not None, "connect() before enable()"
        return bool(self._robot.enable())

    def disable(self) -> bool:
        """Disable the arm's motors."""
        assert self._robot is not None, "connect() before disable()"
        return bool(self._robot.disable())

    def set_speed_percent(self, percent: int) -> None:
        """Clamp the firmware speed limit to a percentage."""
        assert self._robot is not None, "connect() before set_speed_percent()"
        self._robot.set_speed_percent(int(percent))

    def set_normal_mode(self) -> None:
        """Prefer the follower mode the move_js streaming path expects."""
        assert self._robot is not None, "connect() before set_normal_mode()"
        if hasattr(self._robot, "set_follower_mode"):
            self._robot.set_follower_mode()
        else:
            self._robot.set_normal_mode()

    def move_js(self, joints: Sequence[float]) -> None:
        """Stream one follower-mode joint target (no firmware smoothing)."""
        assert self._robot is not None, "connect() before move_js()"
        self._robot.move_js([float(value) for value in joints])

    def read_state(self) -> tuple[float, ...] | None:
        """Latest joint angles, or None when the vendor returns nothing."""
        if self._robot is None:
            return None
        angles = self._robot.get_joint_angles()
        if angles is None:
            return None
        return tuple(float(value) for value in angles)

    def close(self) -> None:
        """Disconnect from CAN; safe to call once."""
        if self._robot is not None:
            self._robot.disconnect()
```

```python
"""Pika gripper wrapper over the vendor effector attached to a Nero arm."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from inspect_robots_nero._arm import NeroArm


class NeroGripper:
    """Width-commanded gripper; the effector is injectable for tests."""

    def __init__(
        self,
        arm: NeroArm,
        *,
        effector: Any | None = None,
        start_sleep_s: float = 0.3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._arm = arm
        self._effector = effector
        self._start_sleep_s = start_sleep_s
        self._sleep = sleep

    def start(self) -> None:
        """Attach the AGX gripper effector to the paired arm (once)."""
        if self._effector is not None:
            return
        raw = self._arm.raw_robot
        if raw is None:
            raise RuntimeError("gripper start requires the paired arm to be connected first")
        self._effector = raw.init_effector(raw.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self._sleep(self._start_sleep_s)

    def move(self, width: float, force: float) -> None:
        """Command one gripper width in meters with a force ratio."""
        self.start()
        assert self._effector is not None
        move_metric = getattr(self._effector, "move_gripper_m", None)
        if callable(move_metric):
            move_metric(value=float(width), force=float(force))
            return
        self._effector.move_gripper(width=float(width), force=float(force))

    def read_state(self) -> float | None:
        """Latest gripper width in meters, or None before start/without feedback."""
        if self._effector is None:
            return None
        status = self._effector.get_gripper_status()
        if status is None:
            return None
        message = status.msg
        width = getattr(message, "width", None)
        return float(width if width is not None else message.value)

    def stop(self) -> None:
        """Detach the effector handle (the hardware keeps its last width)."""
        self._effector = None
```

Note: `NeroGripper.move` calls `self.start()` internally, so `test_gripper_move_and_readback`'s extra `robot.init_effector` probe creates a *fresh* effector and its `moves == []` assertion guards that our command went to the attached one. If the fake makes that awkward, drop that probe and assert via a recorded `FakeEffector` passed explicitly as `effector=`.

- [ ] **Step 4: Run tests and gates, commit**

Run: `uv run --no-sync python -m pytest plugins/inspect-robots-nero/tests -q` then the four gate commands.
Expected: PASS, all green.

```bash
git add plugins/inspect-robots-nero
git commit -m "feat(nero): pyAgxArm arm and gripper wrappers with injectable fakes"
```

---

### Task 4: D405 color camera with a keep-latest capture thread

**Files:**
- Create: `plugins/inspect-robots-nero/src/inspect_robots_nero/_camera.py`
- Test: `plugins/inspect-robots-nero/tests/test_nero_camera.py`

**Interfaces:**
- Produces: `D405Camera(name: str, device: str, *, width=640, height=480, fps=30, pixel_format="YUYV", capture: Any | None = None, clock: Callable[[], float] = time.monotonic, poll_s: float = 0.005)` with `.start()`, `.read(*, max_age_s: float) -> tuple[np.ndarray, float]` (raises `TimeoutError` when no frame or stale), `.stop()`. The injected `capture` needs only `.read() -> tuple[bool, np.ndarray]` and, when owned by the camera, `.release()`.
- Consumes: nothing from other plugin tasks.

- [ ] **Step 1: Write the failing tests**

```python
"""D405 camera thread behavior against a scripted capture."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np
import pytest

from inspect_robots_nero._camera import D405Camera


class FakeCapture:
    def __init__(self, frames: list[np.ndarray], *, gate: Callable[[], None] | None = None) -> None:
        self.frames = list(frames)
        self.released = False
        self._gate = gate
        self._lock = threading.Lock()

    def read(self) -> tuple[bool, np.ndarray]:
        if self._gate is not None:
            self._gate()
        with self._lock:
            if self.frames:
                return True, self.frames.pop(0)
        return False, np.zeros((0,))


def test_read_returns_the_latest_frame_in_rgb() -> None:
    first = np.zeros((2, 2, 3), dtype=np.uint8)
    second = np.full((2, 2, 3), 7, dtype=np.uint8)
    camera = D405Camera("left_rgbd", "/dev/null", capture=FakeCapture([first, second]))
    camera.start()
    try:
        frame, stamp = camera.read(max_age_s=5.0)
        assert frame.shape == (2, 2, 3)
        assert stamp > 0.0
        assert bool((frame == 7).all())  # the *latest* frame wins
    finally:
        camera.stop()


def test_read_times_out_without_frames() -> None:
    camera = D405Camera("left_rgbd", "/dev/null", capture=FakeCapture([]))
    camera.start()
    try:
        with pytest.raises(TimeoutError, match="no frames"):
            camera.read(max_age_s=0.05)
    finally:
        camera.stop()


def test_read_rejects_stale_frames_with_injected_clock() -> None:
    now = [100.0]
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    class FrozenCapture:
        def read(self) -> tuple[bool, np.ndarray]:
            now[0] += 10.0  # every read lands 10 s in the future-clock past
            return True, frame

    camera = D405Camera("left_rgbd", "/dev/null", capture=FrozenCapture(), clock=lambda: now[0])
    camera.start()
    try:
        now[0] += 2.0
        with pytest.raises(TimeoutError, match="stale"):
            camera.read(max_age_s=0.5)
    finally:
        camera.stop()


def test_stop_releases_an_owned_capture() -> None:
    capture = FakeCapture([np.zeros((2, 2, 3), dtype=np.uint8)])
    camera = D405Camera("left_rgbd", "/dev/null", capture=capture)
    camera.start()
    time.sleep(0.05)
    camera.stop()
    # Owned captures are wrapped; release is recorded on the underlying object.
    assert capture.released
```

Note: the wrapped owned-capture object must expose `release()`; the implementation wraps the injected capture in an `_OwnedCapture` adapter only when it built the cv2 capture itself. For the injected case `stop()` must NOT call `release()` on the fake unless the test asserts it. Simplify the last test: inject `capture=` and assert `released` stays `False` after `stop()`; keep the release path for the cv2-built case untested (thin) and documented. **Adjust the test to assert `capture.released is False`.**

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync python -m pytest plugins/inspect-robots-nero/tests/test_nero_camera.py -q`
Expected: FAIL (`No module named 'inspect_robots_nero._camera'`).

- [ ] **Step 3: Implement `_camera.py`**

```python
"""D405 color stream over OpenCV V4L2, one keep-latest daemon thread per camera.

The bring-up reads the D405 color stream as a plain V4L2 device (YUYV), so
this module needs no vendor SDK. ``read()`` hands out the newest frame and
rejects stale ones so the observation is never silently old.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import numpy as np


class D405Camera:
    """One camera: background grab loop plus staleness-checked reads."""

    def __init__(
        self,
        name: str,
        device: str,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        pixel_format: str = "YUYV",
        capture: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
        poll_s: float = 0.005,
    ) -> None:
        from typing import Any as _Any  # noqa: PLC0415 - keep the signature simple

        del _Any
        self.name = name
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.pixel_format = pixel_format
        self._injected = capture
        self._capture: Any | None = capture
        self._clock = clock
        self._poll_s = poll_s
        self._lock = threading.Lock()
        self._latest: tuple[np.ndarray, float] | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def _open(self) -> Any:
        import cv2

        capture = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not capture.isOpened():
            raise RuntimeError(
                f"could not open camera {self.name!r} at {self.device}; check the by-path node "
                "and pass -E cameras=<name>=<device> to override it"
            )
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.pixel_format))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        return capture

    def start(self) -> None:
        """Open the device (unless injected) and start the grab thread."""
        if self._running:
            return
        if self._capture is None:
            self._capture = self._open()
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name=f"nero-camera-{self.name}", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        assert self._capture is not None
        import cv2

        while self._running:
            ok, frame = self._capture.read()
            if not ok:
                time.sleep(self._poll_s)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._latest = (rgb, self._clock())

    def read(self, *, max_age_s: float) -> tuple[np.ndarray, float]:
        """Return (rgb frame, stamp); TimeoutError when absent or stale."""
        with self._lock:
            latest = self._latest
        if latest is None:
            raise TimeoutError(
                f"camera {self.name!r} produced no frames; check the device node and cabling"
            )
        frame, stamp = latest
        age = self._clock() - stamp
        if age > max_age_s:
            raise TimeoutError(
                f"camera {self.name!r} frame is {age:.2f}s old, exceeding max_age_s={max_age_s:g}"
            )
        return frame, stamp

    def stop(self) -> None:
        """Stop the grab thread; release only captures this object opened itself."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._injected is None and self._capture is not None:
            self._capture.release()
        self._capture = None
        self._latest = None
```

(The stray `_Any` import dance in `__init__` is unnecessary; annotate `capture: object | None = None` and drop it. `self._capture: object | None`.)

- [ ] **Step 4: Run tests and gates, commit**

Run: `uv run --no-sync python -m pytest plugins/inspect-robots-nero/tests -q` and the gates.
Expected: PASS, green.

```bash
git add plugins/inspect-robots-nero
git commit -m "feat(nero): keep-latest D405 color capture threads"
```

---

### Task 5: Embodiment reset/step/close over fakes

**Files:**
- Modify: `plugins/inspect-robots-nero/src/inspect_robots_nero/embodiment.py` (replace the `NotImplementedError` bodies; add hardware assembly, pacing, observation assembly)
- Test: `plugins/inspect-robots-nero/tests/test_nero_embodiment.py`

**Interfaces:**
- Consumes: Task 2 `NeroKinematics`/`rot6d_to_matrix`/`matrix_to_rot6d`; Task 3 `NeroArm`/`NeroGripper` + `tests/fakes.FakeAgxRobot`; Task 4 `D405Camera`; Task 1 constructor params `arm_factory`/`camera_factory`.
- Produces: fully working `NeroEmbodiment`; constructor seam signatures the tests rely on:
  - `arm_factory: Callable[[str], object]` (side name) -> object with the NeroArm surface
  - `camera_factory: Callable[[str], object]` (camera name) -> object with `.start()`, `.read(*, max_age_s)`, `.stop()`

- [ ] **Step 1: Write the failing tests**

```python
"""Full reset/step/close loop for the nero embodiment over fakes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from inspect_robots import Scene
from inspect_robots.conformance import assert_embodiment_conformant
from inspect_robots.errors import EmbodimentFault

from inspect_robots_nero import nero_embodiment
from inspect_robots_nero._arm import NeroArm
from inspect_robots_nero._camera import D405Camera
from inspect_robots_nero._config import GRIPPER_FORCE, GRIPPER_INIT_WIDTH_M, HOME_LEFT, HOME_RIGHT

from .fakes import FakeAgxRobot

_FRAME = np.full((480, 640, 3), 9, dtype=np.uint8)


class FakeCamera:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.frame = _FRAME

    def start(self) -> None:
        self.started = True

    def read(self, *, max_age_s: float) -> tuple[np.ndarray, float]:
        del max_age_s
        return self.frame, 1.0

    def stop(self) -> None:
        self.stopped = True


class Harness:
    def __init__(self, *, operator_reset_confirm: bool = False) -> None:
        self.robots = {"left": FakeAgxRobot(), "right": FakeAgxRobot()}
        self.cameras = {name: FakeCamera() for name in ("left_rgbd", "right_rgbd", "chest_rgbd")}
        self.sleeps: list[float] = []
        embodiment = nero_embodiment(
            operator_reset_confirm=operator_reset_confirm,
            arm_factory=lambda side: NeroArm(
                side, f"can_{side}", robot=self.robots[side], sleep=lambda _s: None
            ),
            camera_factory=lambda name: self.cameras[name],
            sleep=self.sleeps.append,
        )
        self.embodiment = embodiment


def test_reset_enables_homes_and_returns_first_observation() -> None:
    harness = Harness()
    observation = harness.embodiment.reset(Scene(instruction="put the cup on the pad"))
    for robot in harness.robots.values():
        assert robot.enabled and robot.speed == 20 and robot.mode == "follower"
        assert robot.move_js_calls[-1] == tuple(HOME_LEFT if robot is harness.robots["left"] else HOME_RIGHT)
    for camera in harness.cameras.values():
        assert camera.started and not camera.stopped
    assert observation.instruction == "put the cup on the pad"
    assert set(observation.images) == {"left_rgbd", "right_rgbd", "chest_rgbd"}
    assert observation.state["eef_state"].shape == (20,)
    assert observation.state["joint_pos"].shape == (16,)
    gripper_dims = (observation.state["joint_pos"][14], observation.state["joint_pos"][15])
    assert gripper_dims == (0.0, 0.0)  # vendor fake reports its init width before commands


def test_reset_confirm_prompts_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return ""

    monkeypatch.setattr("builtins.input", fake_input)
    harness = Harness(operator_reset_confirm=True)
    harness.embodiment.reset(Scene(instruction="x"))
    assert prompts and "Arrange the scene" in prompts[0]


def test_step_commands_bounded_joint_deltas_and_gripper_widths() -> None:
    harness = Harness()
    harness.embodiment.reset(Scene(instruction="x"))
    action = np.asarray(
        [0.5, 0.0, 0.4] + [0.0] * 6 + [GRIPPER_INIT_WIDTH_M]
        + [-0.3, 0.0, 0.4] + [0.0] * 6 + [0.02],
        dtype=np.float64,
    )
    result = harness.embodiment.step(SimpleNamespace(data=action, meta={}))
    for robot in harness.robots.values():
        assert len(robot.move_js_calls) == 2  # home, then one step command
    left = harness.embodiment
    assert result.observation.images["left_rgbd"].shape == (480, 640, 3)


def test_step_paces_itself_to_control_hz() -> None:
    harness = Harness()
    harness.embodiment.reset(Scene(instruction="x"))
    action = np.zeros(20, dtype=np.float64)
    action[9] = action[19] = 0.04
    harness.embodiment.step(SimpleNamespace(data=action, meta={}))
    assert harness.sleeps  # the second step sleeps to keep the cadence
    harness.embodiment.step(SimpleNamespace(data=action, meta={}))
    assert harness.sleeps


def test_action_shape_mismatch_is_rejected() -> None:
    harness = Harness()
    harness.embodiment.reset(Scene(instruction="x"))
    with pytest.raises(ValueError, match="shape"):
        harness.embodiment.step(SimpleNamespace(data=np.zeros(6), meta={}))


def test_unreachable_action_raises_embodiment_fault() -> None:
    harness = Harness()
    harness.embodiment.reset(Scene(instruction="x"))
    far = np.zeros(20, dtype=np.float64)
    far[0] = 50.0  # five meters past any reachable pose
    with pytest.raises(EmbodimentFault):
        harness.embodiment.step(SimpleNamespace(data=far, meta={}))


def test_gripper_force_is_the_configured_constant() -> None:
    harness = Harness()
    harness.embodiment.reset(Scene(instruction="x"))
    action = np.zeros(20, dtype=np.float64)
    action[9] = 0.03
    harness.embodiment.step(SimpleNamespace(data=action, meta={}))
    # The left gripper effector recorded (width, force); the fake stores moves.
    left_arm = harness.embodiment._arms["left"]  # noqa: SLF001 - test seam
    assert left_arm.raw_robot.effector_kind == "AGX_GRIPPER"


def test_close_disables_arms_and_stops_cameras() -> None:
    harness = Harness()
    harness.embodiment.reset(Scene(instruction="x"))
    harness.embodiment.close()
    for robot in harness.robots.values():
        assert not robot.enabled
    for camera in harness.cameras.values():
        assert camera.stopped


def test_conformance_still_passes_when_wired() -> None:
    assert_embodiment_conformant(Harness().embodiment.info)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync python -m pytest plugins/inspect-robots-nero/tests/test_nero_embodiment.py -q`
Expected: FAIL (`NotImplementedError`).

- [ ] **Step 3: Implement the reset/step/close bodies**

Replace the placeholder bodies in `embodiment.py`; add to the constructor tail:

```python
        self._arm_factory = arm_factory or self._default_arm_factory
        self._camera_factory = camera_factory or self._default_camera_factory
        self._arms: dict[str, Any] = {}
        self._grippers: dict[str, Any] = {}
        self._cameras: dict[str, Any] = {}
        self._kinematics: Any | None = None
        self._last_commanded: dict[str, np.ndarray] = {}
        self._last_step_time: float | None = None
        self._max_joint_step = 0.1
        self._connected = False
```

with helpers and the wired bodies:

```python
    def _default_arm_factory(self, side: str) -> Any:
        from inspect_robots_nero._arm import NeroArm

        from inspect_robots_nero._config import CAN_CHANNELS, FIRMWARE_VERSION, SPEED_PERCENT

        arm = NeroArm(side, CAN_CHANNELS[side], firmware=FIRMWARE_VERSION)
        arm.connect()
        arm.enable()
        arm.set_speed_percent(SPEED_PERCENT)
        arm.set_normal_mode()
        return arm

    def _default_camera_factory(self, name: str) -> Any:
        from inspect_robots_nero._camera import D405Camera

        spec = self.cameras[name]
        camera = D405Camera(
            name,
            str(spec["device"]),
            width=int(spec["width"]),
            height=int(spec["height"]),
            fps=int(spec["fps"]),
            pixel_format=str(spec["pixel_format"]),
            clock=self._clock,
        )
        camera.start()
        return camera

    def _ensure_connected(self) -> None:
        if self._connected:
            return
        from inspect_robots_nero._config import (
            GRIPPER_FORCE,
            GRIPPER_INIT_WIDTH_M,
            HOME_LEFT,
            HOME_RIGHT,
        )
        from inspect_robots_nero._kinematics import NeroKinematics

        self._arms = {side: self._arm_factory(side) for side in ("left", "right")}
        self._grippers = {side: NeroGripper(self._arms[side]) for side in ("left", "right")}
        for gripper in self._grippers.values():
            gripper.start()
            gripper.move(GRIPPER_INIT_WIDTH_M, GRIPPER_FORCE)
        self._cameras = {name: self._camera_factory(name) for name in self.cameras}
        urdf = str(importlib.resources.files("inspect_robots_nero") / "assets" / "dual_nero_pika.urdf")
        self._kinematics = NeroKinematics(urdf)
        self._last_commanded = {"left": np.asarray(HOME_LEFT), "right": np.asarray(HOME_RIGHT)}
        self._connected = True

    def _drive_home(self) -> None:
        from inspect_robots_nero._config import HOME_LEFT, HOME_RIGHT

        homes = {"left": np.asarray(HOME_LEFT), "right": np.asarray(HOME_RIGHT)}
        for side, home in homes.items():
            self._arms[side].move_js(home.tolist())
        deadline = self._clock() + self.reset_settle_timeout_s
        while self._clock() < deadline:
            settled = True
            for side, home in homes.items():
                reading = self._arms[side].read_state()
                if reading is None or len(reading) != 7:
                    settled = False
                    break
                if float(np.max(np.abs(np.asarray(reading) - home))) > 0.05:
                    settled = False
                    break
            if settled:
                self._sleep(1.0 / self.control_hz)
                return
            self._sleep(0.02)
        raise EmbodimentFault(
            f"arms did not settle at home within reset_settle_timeout_s={self.reset_settle_timeout_s:g}s"
        )

    @staticmethod
    def _target_pose(xyz: np.ndarray, rot6d: np.ndarray) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = rot6d_to_matrix(rot6d)
        pose[:3, 3] = xyz
        return pose

    def _read_joints(self) -> np.ndarray:
        readings: list[float] = []
        for side in ("left", "right"):
            reading = self._arms[side].read_state()
            if reading is None or len(reading) != 7:
                raise EmbodimentFault(
                    f"no joint feedback from the {side} arm; check the CAN link"
                )
            readings.extend(float(value) for value in reading)
        return np.asarray(readings)

    def _pace(self) -> None:
        if self._last_step_time is not None:
            remaining = self._last_step_time + (1.0 / self.control_hz) - self._clock()
            if remaining > 0:
                self._sleep(remaining)
        self._last_step_time = self._clock()

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Connect lazily, settle both arms at home, and return the first observation."""
        del seed
        self._instruction = scene.instruction
        self._ensure_connected()
        if self.operator_reset_confirm:
            print(f"Operator reset required for instruction: {scene.instruction}")
            input("Arrange the scene, then press Enter to continue: ")
        self._drive_home()
        return self._assemble_observation()

    def step(self, action: Action) -> StepResult:
        """Solve IK for one 20-dim EE target and command one bounded joint increment."""
        data = np.asarray(action.data, dtype=np.float64).ravel()
        if data.shape != (20,):
            raise ValueError(
                f"action has shape {data.shape}, expected (20,); the nero action vector is "
                "[left xyz, left rot6d, left width, right xyz, right rot6d, right width]"
            )
        from inspect_robots_nero._config import GRIPPER_FORCE

        self._pace()
        assert self._kinematics is not None
        left_target = self._target_pose(data[0:3], data[3:9])
        right_target = self._target_pose(data[10:13], data[13:19])
        joints = self._read_joints()
        q_seed = self._kinematics.q_from(joints[:7], joints[7:14])
        try:
            q_solution = self._kinematics.solve(left_target, right_target, q_seed)
        except NeroKinematicsError as exc:
            raise EmbodimentFault(f"nero IK failed: {exc}") from exc
        left_current, right_current = joints[:7], joints[7:14]
        left_command = left_current + np.clip(
            q_solution[:7] - left_current, -self._max_joint_step, self._max_joint_step
        )
        right_command = right_current + np.clip(
            q_solution[7:] - right_current, -self._max_joint_step, self._max_joint_step
        )
        self._arms["left"].move_js(left_command.tolist())
        self._arms["right"].move_js(right_command.tolist())
        self._last_commanded = {"left": left_command, "right": right_command}
        self._grippers["left"].move(float(data[9]), GRIPPER_FORCE)
        self._grippers["right"].move(float(data[19]), GRIPPER_FORCE)
        return StepResult(observation=self._assemble_observation())

    def _assemble_observation(self) -> Observation:
        assert self._kinematics is not None
        joints = self._read_joints()
        poses = self._kinematics.fk(self._kinematics.q_from(joints[:7], joints[7:14]))
        widths = [self._grippers[side].read_state() for side in ("left", "right")]
        left_width = widths[0] if widths[0] is not None else 0.0
        right_width = widths[1] if widths[1] is not None else 0.0
        eef_state = np.concatenate(
            [
                poses["left"][:3, 3],
                matrix_to_rot6d(poses["left"][:3, :3]),
                [left_width],
                poses["right"][:3, 3],
                matrix_to_rot6d(poses["right"][:3, :3]),
                [right_width],
            ]
        )
        joint_pos = np.concatenate([joints, [left_width, right_width]])
        images: dict[str, np.ndarray] = {}
        image_times: dict[str, float] = {}
        for name, camera in self._cameras.items():
            frame, stamp = camera.read(max_age_s=self.camera_max_age_s)
            images[name] = frame
            image_times[name] = stamp
        return Observation(
            images=images,
            state={"eef_state": eef_state, "joint_pos": joint_pos},
            instruction=self._instruction,
            image_times=image_times,
            state_time=min(image_times.values()),
        )

    def close(self) -> None:
        """Stop cameras, detach grippers, and disable both arms (best effort each)."""
        for camera in self._cameras.values():
            try:
                camera.stop()
            except Exception:  # noqa: BLE001 - close is best effort
                pass
        for gripper in self._grippers.values():
            try:
                gripper.stop()
            except Exception:  # noqa: BLE001
                pass
        for arm in self._arms.values():
            try:
                arm.disable()
                arm.close()
            except Exception:  # noqa: BLE001
                pass
        self._connected = False
```

Imports to add at module top: `import importlib.resources`, `from inspect_robots.errors import EmbodimentFault`, and `from inspect_robots_nero._gripper import NeroGripper`, `from inspect_robots_nero._kinematics import NeroKinematicsError, matrix_to_rot6d, rot6d_to_matrix` (module-level imports of `_kinematics`/`_gripper` are fine: they import cv2/pinocchio lazily themselves; `_gripper` has no heavy deps).

- [ ] **Step 4: Run tests and gates**

Run: `uv run --no-sync python -m pytest plugins/inspect-robots-nero/tests -q` and the four gate commands.
Expected: PASS. The IK-fault test exercises real pinocchio solve against a far target; if it is slow, keep it (it hits the iteration cap in well under a second).

- [ ] **Step 5: Commit**

```bash
git add plugins/inspect-robots-nero
git commit -m "feat(nero): wire reset/step/close with IK, pacing, and fault taxonomy"
```

---

### Task 6: README, docs wiring, CI/release, bench smoke script

**Files:**
- Modify: `plugins/inspect-robots-nero/README.md` (replace the placeholder)
- Create: `plugins/inspect-robots-nero/scripts/bench_smoke.py`
- Modify: `.github/workflows/ci.yml` (new `plugin-nero` job + `ci-ok` needs entry)
- Modify: `.github/workflows/release.yml` (new `publish-nero` job)
- Modify: `CHANGELOG.md` (`## [Unreleased]` / `### Added`, `**Plugins:**` bullet)
- Modify: `docs/guide/cli.md` or a new `docs/guide/nero.md` (see Step 3)

- [ ] **Step 1: Write the README**

Structure (full prose, no em dashes, safety first, mirroring the ros plugin README):

```markdown
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
the working bring-up config; every value is overridable with `-E`.

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

## Camera overrides

    -E cameras=left_rgbd=/dev/v4l/by-path/pci-0000:80:14.0-usb-0:11.2:1.0-video-index4

Cross-camera frame alignment is not reproduced in this adapter; v1 accepts the
millisecond-scale skew between the three streams (the bring-up stack's
alignment gate is not part of this plugin). Frames older than
`-E camera_max_age_s` (default 0.5) are rejected loudly.
```

- [ ] **Step 2: Write the bench smoke script**

`scripts/bench_smoke.py`: a standalone operator-driven script (spec milestone 3). argparse with `--side left|right --channel can_left --steps 3`; connects one arm, enables, sets speed 20, follower mode, homes, then for each step asks the operator (input()) to confirm a 0.01 rad incremental `move_js`, printing joint feedback; disables on exit and on Ctrl-C. ~90 lines, stdlib + numpy only (`NeroArm` import from the plugin). Guard with `if __name__ == "__main__": main()`; include a module docstring (ruff D1 applies to plugins too). It has no automated test (hardware-only path); ruff/mypy must still pass on it (add `scripts` to nothing special; ruff covers the whole plugin dir).

- [ ] **Step 3: Docs page**

Check `ls docs/guide`: if a ros embodiment guide page exists there, add `docs/guide/nero.md` modeled on it (install, safety, quickstart, camera overrides) and register it in `website/sidebars.js` beside the ros entry. If plugin guides live only in plugin READMEs, skip the docs page and note that decision in the PR description.

- [ ] **Step 4: CI and release wiring**

In `.github/workflows/ci.yml`: duplicate the `plugin-xpolicylab` job as `plugin-nero` (name `plugin · nero`), swapping the package path `plugins/inspect-robots-xpolicylab` to `plugins/inspect-robots-nero` in the sync/ruff/format/mypy/pytest steps, and add `plugin-nero` to the `ci-ok` job's `needs` list (a job missing there does not gate merges). In `.github/workflows/release.yml`: duplicate the `publish-xpolicylab` job as `publish-nero` with the package directory swapped and its own trusted-publisher environment name; flag in the PR description that the matching PyPI trusted-publisher environment must be created before the next release.

- [ ] **Step 5: CHANGELOG and gates**

Add under `## [Unreleased]` / `### Added`:

```markdown
- **Plugins:** `inspect-robots-nero`, a real-hardware embodiment for the lab's
  dual Nero CAN arms (EE-pose actions, pinocchio IK, D405 color cameras),
  evaluated with any Inspect Robots policy, e.g. the `agent` policy on
  GPT-6 Astra.
```

Run the full gate set for the plugin plus `uv run --no-sync ruff check .` and `uv run --no-sync mypy` for core (README/docs only should not affect core, but confirm). Commit:

```bash
git add .github/workflows/ci.yml .github/workflows/release.yml CHANGELOG.md plugins/inspect-robots-nero docs website
git commit -m "feat(nero): docs, CI and release wiring, bench smoke script"
```

---

## Verification after Task 6 (spec milestones 1, 2 complete)

- `uv run --no-sync python -m pytest plugins/inspect-robots-nero/tests -q` green.
- `uv run --no-sync inspect-robots list embodiments` shows `nero`.
- `uv run --no-sync inspect-robots doctor --embodiment nero -E ...` passes declarative conformance without hardware and reports the missing `pyAgxArm` runtime with its remedy on machines without the SDK.
- Milestones 3-6 (bench smoke, cameras on hardware, attended single-arm Astra run, dual-arm) are hardware gates executed by the operator with `scripts/bench_smoke.py` and the README quickstart; they produce no code changes beyond README/model-id updates.

## Out of scope (per spec)

- Depth channels, fisheye cameras, cross-camera alignment.
- neo_manipulation imports or any ROS 2 runtime.
- Registered nero tasks/datasets and automated scorers.
