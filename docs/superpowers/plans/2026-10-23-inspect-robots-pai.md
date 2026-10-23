# inspect-robots-pai Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 plugins/inspect-robots-pai 里做出与 evo 无关的 HarnessPAI 框架：固定程序跑一次真机试验，试验之间演化，成功试验事后转成 LeRobot v2.1。

**Architecture:** 协议 Port 放在包根，只有 policy.py 可以 import domains/nero。execution 加载任务包的 run(port, trial_dir) 并写 steps.jsonl。evolution 只改运行目录里的候选，合同通过后才复制进 code_library。LeRobot 由单独的导出命令生成，不进控制环。

**Tech Stack:** Python 3.10+、inspect-robots、numpy。导出额外使用 pyarrow 与 ffmpeg。测试在 inspect-robots 仓库根执行。

## Global Constraints

- 新代码只在 plugins/inspect-robots-pai/。不改 evo、harness、vla、nero、agent。
- 不 import inspect_robots_evo、inspect_robots_harness、inspect_robots_vla、inspect_robots_nero、inspect_robots_agent。
- 除 src/inspect_robots_pai/policy.py 外，包源码不 import domains.nero。末端标签字符串只定义在 domains/nero/motion.py。
- 依赖只有 inspect-robots 与 numpy。pyarrow 仅作为可选 extra export。
- 操作员标签才写入 diagnostics.success。自动检查只写 auto_check.json。
- 单次试验默认墙钟 600 秒，move 与 set_gripper 合计 80 次。
- probe 默认最多 3 次。演化轮数默认最多 5，合同通过即停。
- trial_count 默认 1。运行根目录默认 runs/pai/任务/YYYYmmdd-HHMMSS/，环境变量 INSPECT_ROBOTS_PAI_RUNS 可换根。
- 真机不进自动测试。
- 规格：docs/superpowers/specs/2026-10-23-inspect-robots-pai-design.md。

测试命令一律在仓库根执行，前面加上 PYTHONPATH=plugins/inspect-robots-pai/src。

---

### Task 1: 插件骨架与依赖边界

**Files:**
- Create: plugins/inspect-robots-pai/pyproject.toml
- Create: plugins/inspect-robots-pai/README.md
- Create: plugins/inspect-robots-pai/.gitignore
- Create: plugins/inspect-robots-pai/configs/models.yaml
- Create: plugins/inspect-robots-pai/src/inspect_robots_pai/__init__.py
- Create: plugins/inspect-robots-pai/src/inspect_robots_pai/ports.py
- Test: plugins/inspect-robots-pai/tests/test_boundary.py

**Interfaces:**
- Consumes: 无
- Produces: Port 协议，方法 observation、move、set_gripper、auto_check。StepView 字段 instruction、images、camera_times、body_time、eef_state、joint_pos。

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

ROOT = Path("plugins/inspect-robots-pai/src/inspect_robots_pai")
FORBIDDEN = (
    "inspect_robots_evo",
    "inspect_robots_harness",
    "inspect_robots_vla",
    "inspect_robots_nero",
    "inspect_robots_agent",
)


def test_package_does_not_import_other_plugins():
    offenders = []
    for path in ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in FORBIDDEN:
            if name in text:
                offenders.append(str(path) + ":" + name)
        if "domains.nero" in text and path.name != "policy.py":
            offenders.append(str(path) + ":domains.nero")
    assert offenders == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: PYTHONPATH=plugins/inspect-robots-pai/src .venv/bin/python -m pytest plugins/inspect-robots-pai/tests/test_boundary.py -q
Expected: FAIL，包目录不存在。

- [ ] **Step 3: Write minimal implementation**

pyproject.toml 按 plugins/inspect-robots-harness/pyproject.toml 的 hatch 布局。包名 inspect-robots-pai，版本 0.1.0，描述 HarnessPAI framework for real Nero trials。依赖只有 inspect-robots>=0.44 与 numpy>=1.24。策略入口 pai = inspect_robots_pai:pai_policy。脚本入口 inspect-robots-pai = inspect_robots_pai.cli:main。optional-dependencies 的 export 与 dev 都包含 pyarrow>=15，dev 另含 pytest>=8.0。uv.sources 只有 inspect-robots workspace。known-first-party 只有 inspect_robots 与 inspect_robots_pai。

README.md 四句：策略名 pai；与 evo 无关；域控制在 domains/nero；成功试验用 export-lerobot 导出。

.gitignore 只含 configs/local.yaml。

configs/models.yaml 两段：coder.model 为 gpt-5-codex，diagnoser.model 为 gemini-2.5-pro。

ports.py 定义数据类 StepView 与协议 Port，方法签名与接口节一致。图像值是 numpy 数组，末端状态长度 20，关节长度 16。

__init__.py 只写模块文档字符串。pai_policy 到 Task 7 再导出。ports.py 的完整内容：

```python
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class StepView:
    instruction: str
    images: dict
    camera_times: dict
    body_time: float
    eef_state: np.ndarray
    joint_pos: np.ndarray


class Port(Protocol):
    def observation(self) -> StepView: ...

    def move(self, targets: dict) -> dict: ...

    def set_gripper(self, side: str, width: float) -> dict: ...

    def auto_check(self) -> dict: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: PYTHONPATH=plugins/inspect-robots-pai/src .venv/bin/python -m pytest plugins/inspect-robots-pai/tests/test_boundary.py -q
Expected: PASS

- [ ] **Step 5: Commit**

git add plugins/inspect-robots-pai
git commit -m "feat: add inspect-robots-pai package boundary"

---

### Task 2: Nero 插补

**Files:**
- Create: plugins/inspect-robots-pai/src/inspect_robots_pai/domains/nero/motion.py
- Create: plugins/inspect-robots-pai/src/inspect_robots_pai/domains/nero/__init__.py
- Test: plugins/inspect-robots-pai/tests/test_motion.py

**Interfaces:**
- Consumes: 无
- Produces: DIM_LABELS、interpolate(current, targets) 返回 (rows, steps)、reached(row, targets) 返回 bool

- [ ] **Step 1: Write the failing test**

```python
from inspect_robots_pai.domains.nero.motion import DIM_LABELS, interpolate, reached


def test_labels_and_step():
    assert DIM_LABELS[0] == "left_x"
    assert DIM_LABELS[9] == "left_gripper"
    assert len(DIM_LABELS) == 20
    current = dict.fromkeys(DIM_LABELS, 0.0)
    rows, steps = interpolate(current, {"left_x": 0.02, "left_r1": 0.1})
    assert steps == 2
    assert abs(rows[-1][0] - 0.02) < 1e-9
    assert reached(rows[-1], {"left_x": 0.02})
```

- [ ] **Step 2: Run test to verify it fails**

Run: PYTHONPATH=plugins/inspect-robots-pai/src .venv/bin/python -m pytest plugins/inspect-robots-pai/tests/test_motion.py -q
Expected: FAIL，ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

domains/nero/__init__.py 为空。motion.py 是 plugins/inspect-robots-evo/src/inspect_robots_evo/motion.py 的逐行拷贝，模块文档改为「拷贝自 evo 的末端插补，禁止 import evo」。DIM_LABELS、步长 0.01 与 0.05、interpolate、reached 保持原算法。不要在别的源文件写这些标签字符串。

- [ ] **Step 4: Run test to verify it passes**

Run: PYTHONPATH=plugins/inspect-robots-pai/src .venv/bin/python -m pytest plugins/inspect-robots-pai/tests/test_motion.py plugins/inspect-robots-pai/tests/test_boundary.py -q
Expected: PASS

- [ ] **Step 5: Commit**

git add plugins/inspect-robots-pai/src/inspect_robots_pai/domains plugins/inspect-robots-pai/tests/test_motion.py
git commit -m "feat: copy Nero end-effector interpolation into pai"

---

### Task 3: 一次试验、逐步记录与标注

**Files:**
- Create: plugins/inspect-robots-pai/src/inspect_robots_pai/execution/__init__.py
- Create: plugins/inspect-robots-pai/src/inspect_robots_pai/execution/trial.py
- Test: plugins/inspect-robots-pai/tests/test_trial.py

**Interfaces:**
- Consumes: Port、StepView
- Produces: run_trial(program, trial_dir, port, wall_s=600, max_actions=80, abort=None) 返回 TrialResult。字段 process_ok、termination、diagnostics、auto_check。label_trial(trial_dir, success)。

- [ ] **Step 1: Write the failing test**

tests/test_trial.py 使用一个 FakePort。observation 返回 2x2 uint8 三路图、20 维零末端、16 维零关节。move 把计数加一并返回 held False、steps 1。auto_check 返回 ok False。

程序目录里的 __init__.py 定义 run(port, trial_dir)，调用 port.move 一次，参数是字典 left_x 等于 0.1。

test_run_trial_does_not_set_success：process_ok 为真；diagnostics 第一行没有 success；auto_check.json 的 ok 为假；label_trial(trial, True) 之后 success 为真；steps.jsonl 第一行 eef_state 长度 20，targets["left_x"] 为 0.1；frames/left/000000.png 存在。

test_81st_action_fails：run 循环调用 move 81 次，max_actions 为 80，process_ok 为假，termination 为 action_budget。

test_wall_clock：wall_s 为 0，run 调用一次 move。process_ok 为假，termination 为 wall_clock。实现里用 time.monotonic() - start >= wall_s。

- [ ] **Step 2: Run test to verify it fails**

Run: PYTHONPATH=plugins/inspect-robots-pai/src .venv/bin/python -m pytest plugins/inspect-robots-pai/tests/test_trial.py -q
Expected: FAIL，run_trial 未定义

- [ ] **Step 3: Write minimal implementation**

execution/__init__.py 为空。trial.py：

```python
from __future__ import annotations

import importlib.util
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from inspect_robots._pngenc import encode_png


class _Budget(Exception):
    def __init__(self, termination: str) -> None:
        self.termination = termination


@dataclass
class TrialResult:
    process_ok: bool
    termination: str
    diagnostics: Path
    auto_check: Path


def _load_run(program: Path):
    spec = importlib.util.spec_from_file_location(
        "pai_task", program / "__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run


def _write_step(trial: Path, view, targets: dict, frame: int) -> None:
    trial.mkdir(parents=True, exist_ok=True)
    for camera, image in view.images.items():
        dest = trial / "frames" / camera / f"{frame:06d}.png"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(encode_png(image))
    row = {
        "instruction": view.instruction,
        "camera_times": view.camera_times,
        "body_time": view.body_time,
        "eef_state": view.eef_state.tolist(),
        "joint_pos": view.joint_pos.tolist(),
        "targets": targets,
        "frame": frame,
    }
    with (trial / "steps.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def run_trial(program, trial_dir, port, wall_s=600, max_actions=80, abort=None):
    trial = Path(trial_dir)
    trial.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    count = 0
    frame = 0

    def _gate():
        nonlocal count
        if abort is not None and abort():
            raise _Budget("aborted")
        if time.monotonic() - start >= wall_s:
            raise _Budget("wall_clock")
        if count >= max_actions:
            raise _Budget("action_budget")
        count += 1

    class _Counted:
        def observation(self):
            return port.observation()

        def move(self, targets):
            nonlocal frame
            _gate()
            _write_step(trial, port.observation(), dict(targets), frame)
            frame += 1
            return port.move(targets)

        def set_gripper(self, side, width):
            _gate()
            return port.set_gripper(side, width)

        def auto_check(self):
            return port.auto_check()

    termination = "returned"
    ok = True
    try:
        _load_run(Path(program))(_Counted(), trial)
    except _Budget as exc:
        ok = False
        termination = exc.termination
    except Exception:
        ok = False
        termination = "error"
        (trial / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    check = port.auto_check()
    check_path = trial / "auto_check.json"
    check_path.write_text(json.dumps(check), encoding="utf-8")
    diag = trial / "diagnostics.jsonl"
    diag.write_text(
        json.dumps({"process_ok": ok, "termination": termination}) + "\n",
        encoding="utf-8",
    )
    return TrialResult(ok, termination, diag, check_path)


def label_trial(trial_dir, success: bool) -> None:
    path = Path(trial_dir) / "diagnostics.jsonl"
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    row["success"] = bool(success)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: PYTHONPATH=plugins/inspect-robots-pai/src .venv/bin/python -m pytest plugins/inspect-robots-pai/tests/test_trial.py -q
Expected: PASS

- [ ] **Step 5: Commit**

git add plugins/inspect-robots-pai/src/inspect_robots_pai/execution plugins/inspect-robots-pai/tests/test_trial.py
git commit -m "feat: run one fixed program and record trial steps"

---

### Task 4: 感知与动作客户端

**Files:**
- Create: plugins/inspect-robots-pai/src/inspect_robots_pai/primitives/__init__.py
- Create: plugins/inspect-robots-pai/src/inspect_robots_pai/primitives/clients.py
- Test: plugins/inspect-robots-pai/tests/test_clients.py

**Interfaces:**
- Consumes: 无
- Produces: sam、molmo、grasp、vla。每个函数参数含 trial_dir，失败返回空列表并把错误追加到 trial_dir/client_errors.jsonl。环境变量 SAM3_SERVICE_URL、MOLMO_SERVICE_URL、GRASPNET_SERVICE_URL、VLA_SERVICE_URL。

- [ ] **Step 1: Write the failing test**

```python
from inspect_robots_pai.primitives.clients import sam


def test_unreachable_service_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SAM3_SERVICE_URL", "http://127.0.0.1:9")
    assert sam(b"img", "cup", tmp_path) == []
    assert (tmp_path / "client_errors.jsonl").is_file()
```

- [ ] **Step 2: Run test to verify it fails**

Run: PYTHONPATH=plugins/inspect-robots-pai/src .venv/bin/python -m pytest plugins/inspect-robots-pai/tests/test_clients.py -q
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

四个函数共用 _post(env_name, payload, trial_dir)。urllib.request.urlopen 超时 2 秒。异常或返回值不是 JSON 列表时，向 client_errors.jsonl 追加 name 与 error，返回 []。不重试。

- [ ] **Step 4: Run test to verify it passes**

Run: PYTHONPATH=plugins/inspect-robots-pai/src .venv/bin/python -m pytest plugins/inspect-robots-pai/tests/test_clients.py -q
Expected: PASS

- [ ] **Step 5: Commit**

git add plugins/inspect-robots-pai/src/inspect_robots_pai/primitives plugins/inspect-robots-pai/tests/test_clients.py
git commit -m "feat: add perception and action clients that fail empty"

---

### Task 5: 记忆、脚手架、合同与晋升

**Files:**
- Create: plugins/inspect-robots-pai/src/inspect_robots_pai/memory/store.py
- Create: plugins/inspect-robots-pai/src/inspect_robots_pai/memory/scaffold.py
- Create: plugins/inspect-robots-pai/src/inspect_robots_pai/evolution/contract.py
- Test: plugins/inspect-robots-pai/tests/test_promote.py

**Interfaces:**
- Consumes: diagnostics.jsonl 的 process_ok 与 success
- Produces: scaffold(dest, task)、passed(run_dir, trial_count)、promote(run_dir, candidate, library_root, task, memory_root, description, failure, repair)、lookup_task(memory_root, description)、write_skill、write_task

- [ ] **Step 1: Write the failing test**

test_unlabeled_does_not_pass：试验 0 的 diagnostics 只有 process_ok true，另写 auto_check.json ok true，以及一个 note.txt 内容含 CONVERGED。passed(run, 1) 为假。promote 不被调用时 library 不存在。

test_failing_promote_does_not_copy：success 为 false 时调用 promote，library/task 不存在。

test_success_promotes：scaffold(candidate, "bottle") 后，diagnostics success 为 true。promote 之后 library/bottle/__init__.py 存在，memory/skills/approach.json 第一条 repair 为 fix，memory/tasks.json 把描述映射到该程序。再改 candidate/__init__.py，library 里的文件不变。

- [ ] **Step 2: Run test to verify it fails**

Run: PYTHONPATH=plugins/inspect-robots-pai/src .venv/bin/python -m pytest plugins/inspect-robots-pai/tests/test_promote.py -q
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

scaffold 写 dest/__init__.py。文件包含 TASK_NAME = 参数 task，以及函数 run(port, trial_dir)，函数体只把 phases.json 写成一条 name 为 scaffold 的记录，不调用 move。

passed 检查 trials/0 到 trials/trial_count-1。每一份必须有 diagnostics，process_ok 为真且 success 为真。缺文件、缺 success、success 为假都返回假。不读 auto_check，不读其他文本。

promote 在 passed 为假时返回。为真时 shutil.copytree(candidate, library_root / task, dirs_exist_ok=True)。write_task 把 description 写进 memory_root/tasks.json。write_skill 向 memory_root/skills/approach.json 追加 failure、repair、code_ref。code_ref 为 code_library/nero/任务名。lookup_task 按描述返回 Path 或 None。

- [ ] **Step 4: Run test to verify it passes**

Run: PYTHONPATH=plugins/inspect-robots-pai/src .venv/bin/python -m pytest plugins/inspect-robots-pai/tests/test_promote.py -q
Expected: PASS

- [ ] **Step 5: Commit**

git add plugins/inspect-robots-pai/src/inspect_robots_pai/memory plugins/inspect-robots-pai/src/inspect_robots_pai/evolution plugins/inspect-robots-pai/tests/test_promote.py
git commit -m "feat: promote only operator-labeled successful trials"

---

### Task 6: 演化循环

**Files:**
- Create: plugins/inspect-robots-pai/src/inspect_robots_pai/evolution/loop.py
- Create: plugins/inspect-robots-pai/src/inspect_robots_pai/agents/codex.py
- Create: plugins/inspect-robots-pai/src/inspect_robots_pai/agents/gemini.py
- Test: plugins/inspect-robots-pai/tests/test_evolve.py

**Interfaces:**
- Consumes: scaffold、passed、promote、lookup_task
- Produces: evolve(...) 返回 "promoted" 或 "stopped"。参数 task、run_dir、library_root、memory_root、trial_count=1、max_rounds=5、probe_max=3、writer、diagnoser、synthesizer、executor。AgentError 表示智能体调用失败。

- [ ] **Step 1: Write the failing test**

test_preflight_failure_does_not_create_trials：writer 把 candidate/__init__.py 写成 `def run(:` 这种语法错误。executor 被调用时在 tmp 写 executed。evolve 返回 stopped，run_dir/trials 不存在。

test_probe_stops_after_three_failures：executor 写 process_ok false 且 success false。记录调用次数。次数等于 3。code library 没有该任务目录。

test_success_promotes_on_first_round：executor 写 process_ok true 并调用 label_trial True。返回 promoted。library 出现任务目录。轮数不需要凑满 5。

test_agent_error_stops：writer 抛 AgentError。返回 stopped，library 不变。

- [ ] **Step 2: Run test to verify it fails**

Run: PYTHONPATH=plugins/inspect-robots-pai/src .venv/bin/python -m pytest plugins/inspect-robots-pai/tests/test_evolve.py -q
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

preflight：importlib 加载 candidate/__init__.py。加载失败，或模块没有 TASK_NAME，或 TASK_NAME 不等于 task，都算失败。失败时再调用一次 writer 然后重试；第二次仍失败返回 stopped，不调用 executor。

evolve 正文：

1. candidate 不存在时，lookup_task 若无结果则 scaffold(candidate, task)。
2. writer(candidate, instruction)。捕获 AgentError 返回 stopped。
3. preflight。
4. 若上一轮失败序号非空：对每个失败序号最多重复 probe_max 次 executor。某次之后该序号 diagnostics 的 process_ok 与 success 都为真，则把它移出失败集。probe_max 用尽仍有失败则 diagnoser 加 synthesizer，轮数加一，不跑其余试验。executor 调用次数在「只失败 1 个试验、trial_count 为 1」时等于 3。
5. 否则对 range(trial_count) 调用 executor。若任一 diagnostics 没有 success 键，返回 stopped。
6. passed 为真则 promote 并返回 promoted。
7. 否则收集 process_ok 或 success 不为真的序号，diagnoser、synthesizer 得到下一轮 instruction。轮数达到 max_rounds 返回 stopped。

agents/codex.py 定义 AgentError 与 CodexWriter。CodexWriter 读 configs/models.yaml 的 coder.model，然后抛 AgentError。CodexSynthesizer 同样抛 AgentError，且以只读方式打开候选目录（不写文件）。agents/gemini.py 的 GeminiDiagnoser 抛 AgentError，不写 success。

- [ ] **Step 4: Run test to verify it passes**

Run: PYTHONPATH=plugins/inspect-robots-pai/src .venv/bin/python -m pytest plugins/inspect-robots-pai/tests/test_evolve.py plugins/inspect-robots-pai/tests/test_promote.py -q
Expected: PASS

- [ ] **Step 5: Commit**

git add plugins/inspect-robots-pai/src/inspect_robots_pai/evolution plugins/inspect-robots-pai/src/inspect_robots_pai/agents plugins/inspect-robots-pai/tests/test_evolve.py
git commit -m "feat: evolve candidates between trials without touching the library"

---

### Task 7: pai 策略

**Files:**
- Create: plugins/inspect-robots-pai/src/inspect_robots_pai/domains/nero/live.py
- Create: plugins/inspect-robots-pai/src/inspect_robots_pai/policy.py
- Modify: plugins/inspect-robots-pai/src/inspect_robots_pai/__init__.py
- Test: plugins/inspect-robots-pai/tests/test_policy.py

**Interfaces:**
- Consumes: run_trial、DIM_LABELS、interpolate、StepView
- Produces: pai_policy 必填参数 program 与 trial_dir，返回 PaiPolicy。act(observation) 返回 ActionChunk。

- [ ] **Step 1: Write the failing test**

构造 Observation：eef_state 与 joint_pos 全 0，images 含 left_rgbd、right_rgbd、chest_rgbd，各是 2x2 uint8。程序 run 调用 port.move 一次，targets 为 left_x = 0.02，然后返回。

第一次 act 返回的最后一拍，下标 0 为 0.02。steps.jsonl 至少一行。从 sys.modules 移除 inspect_robots_pai.agents 后再跑，仍然成功。试验结束后再 act，该动作 meta["request_stop"] 为真。

- [ ] **Step 2: Run test to verify it fails**

Run: PYTHONPATH=plugins/inspect-robots-pai/src .venv/bin/python -m pytest plugins/inspect-robots-pai/tests/test_policy.py -q
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

domains/nero/live.py 的 LivePort 保存最新 Observation。observation 把 left_rgbd 放进 images["left"]，right_rgbd 到 right，chest_rgbd 到 chest。camera_times 来自 observation.image_times，缺的键写 0。body_time 来自 observation.state_time。move 用 interpolate 得到多行，放进队列，阻塞直到 act 把队列取空，返回 held False 与 steps 等于行数。set_gripper 改左或右夹爪维后调用 move。auto_check 返回 ok true、detail 为 no geometric check。

policy.py 是唯一 import domains.nero 的模块。它 import live.LivePort 与 motion.DIM_LABELS。PaiPolicy.act 把观察交给 LivePort。第一次 act 启动守护线程调用 run_trial。队列非空时把每一行打成 Action，data 为长度 20 的 float64。队列空且线程已结束时返回一个 Action，meta request_stop 为真，stop_reason 为 termination。__init__.py 从 policy 导出 pai_policy。缺 program 或 trial_dir 时 pai_policy 抛 ValueError。

- [ ] **Step 4: Run test to verify it passes**

Run: PYTHONPATH=plugins/inspect-robots-pai/src .venv/bin/python -m pytest plugins/inspect-robots-pai/tests/test_policy.py plugins/inspect-robots-pai/tests/test_boundary.py -q
Expected: PASS

- [ ] **Step 5: Commit**

git add plugins/inspect-robots-pai/src/inspect_robots_pai/policy.py plugins/inspect-robots-pai/src/inspect_robots_pai/domains/nero/live.py plugins/inspect-robots-pai/src/inspect_robots_pai/__init__.py plugins/inspect-robots-pai/tests/test_policy.py
git commit -m "feat: register the pai policy on the Nero port"

---

### Task 8: 命令行与 LeRobot 导出

**Files:**
- Create: plugins/inspect-robots-pai/src/inspect_robots_pai/cli.py
- Create: plugins/inspect-robots-pai/src/inspect_robots_pai/export/lerobot.py
- Create: plugins/inspect-robots-pai/src/inspect_robots_pai/export/__init__.py
- Test: plugins/inspect-robots-pai/tests/test_export.py
- Test: plugins/inspect-robots-pai/tests/test_cli.py

**Interfaces:**
- Consumes: steps.jsonl 的 eef_state、targets、joint_pos、camera_times、body_time、frame；diagnostics.success；contract.json 的 task；DIM_LABELS
- Produces: export_lerobot(run_dir, out_dir) 返回写出的 episode 数。CLI 子命令 label、evolve、export-lerobot。

- [ ] **Step 1: Write the failing test**

运行目录 contract.json 的 task 为 "put the water bottle on the black pad."。试验 0 diagnostics success 为 true。steps.jsonl 两行。eef_state 长度 20，下标 9 为 0.09，其余 0。targets 为 left_x 等于 0.1，left_gripper 等于 0.09。joint_pos 长度 16，下标 14 为 0.09，其余 0。三路 frames/left、right、chest 的 000000.png 存在，两行的 frame 都是 0。试验 1 success 为 false，即使有 steps 也不导出。

export_lerobot 返回 1。meta/info.json 的 codebase_version 为 v2.1，fps 为 30，auxiliary_action_joints_source 为 measured_joint_pos。parquet 第一行 observation.state 长度 16，下标 0 为 0，下标 7 为 1。action 下标 0 为 0.1。auxiliary.observation.joints 下标 7 为 1。auxiliary.action.joints 与 observation.joints 相同。next.done 只在第二行为真。第二行 timestamp 为 1/30。

test_cli：label 子命令把 success 写成 false。没有成功试验时 export_lerobot 返回 0。

- [ ] **Step 2: Run test to verify it fails**

Run: PYTHONPATH=plugins/inspect-robots-pai/src .venv/bin/python -m pytest plugins/inspect-robots-pai/tests/test_export.py plugins/inspect-robots-pai/tests/test_cli.py -q
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

export/lerobot.py 从 domains.nero.motion 读取 DIM_LABELS 会破坏边界，所以标签顺序在导出模块里再列一份是违规的。改为导出函数接受已经排好的 20 维向量：steps 里的 targets 是名字字典时，由 cli 层调用一个只存在于 policy.py 的帮助函数不行，因为导出测试会直接调用 export_lerobot。

把名字到下标的映射放在 ports.py 不可行，标签字符串只能在 domains/nero/motion.py。export 测试位于 tests/，允许 import domains.nero.motion.DIM_LABELS。export/lerobot.py 位于 src，不允许 import domains.nero。

因此 steps.jsonl 在 Task 7 的 LivePort.move 里除了 targets 字典，再写 target_vector：按 DIM_LABELS 排好的 20 个数。Task 3 的 FakePort 路径仍只写 targets 字典。export 优先读 target_vector；没有该键时，测试在写 steps 时自己写入 target_vector。测试数据直接含 target_vector，下标 0 为 0.1，下标 9 为 0.09。实现只读 target_vector 与 eef_state，不在 export 模块出现标签字符串。

转换函数：

```python
def _quat_wxyz(matrix):
    m = matrix
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = (trace + 1.0) ** 0.5 * 2
        return [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    return [1.0, 0.0, 0.0, 0.0]


def _pose16(vector):
    out = []
    for base in (0, 10):
        out.extend(vector[base:base + 3].tolist())
        rot6d = vector[base + 3:base + 9]
        x = rot6d[:3]
        x = x / max(np.linalg.norm(x), 1e-12)
        y = rot6d[3:] - np.dot(rot6d[3:], x) * x
        y = y / max(np.linalg.norm(y), 1e-12)
        z = np.cross(x, y)
        matrix = np.column_stack([x, y, z])
        out.extend(_quat_wxyz(matrix))
        width = float(vector[base + 9])
        out.append(float(np.clip(width / 0.09, 0.0, 1.0)))
    return out


def _joints16(joint_pos):
    left = joint_pos[:7]
    right = joint_pos[7:14]
    left_g = float(np.clip(joint_pos[14] / 0.09, 0.0, 1.0))
    right_g = float(np.clip(joint_pos[15] / 0.09, 0.0, 1.0))
    return [*left, left_g, *right, right_g]
```

零 rot6d 的范数为 0。测试的 eef 除夹爪外为 0，此时第一列用单位阵，四元数为 1,0,0,0。实现里范数低于 1e-12 时用单位阵，不要抛错。

只导出 success 为真且每一行 frame 的三路 png 都存在的试验。episode_index 按试验序号从小到大重新从 0 编号。parquet 用 pyarrow。缺 pyarrow 时抛 SystemExit 并提示安装 export extra。ffmpeg 不存在时仍写 parquet 与 meta，info.json 增加 videos_skipped true。存在时为每路相机写 videos/chunk-000/observation.images.相机/episode_NNNNNN.mp4，h264、yuv420p、30 fps。

info.json 还包含 data_path 与 video_path 模板，与瓶子数据集相同。timestamp 为 frame_index / 30。next.done 仅最后一拍为真。对齐列用 camera_times 与 body_time 乘 1e9。序列号用拍号。延迟写 0。相机 id 为 left_rgbd、right_rgbd、chest_rgbd。auxiliary.action.joints 复制 auxiliary.observation.joints。

cli.py：label 调用 label_trial。export-lerobot 调用 export_lerobot。evolve 使用 CodexWriter。writer 抛 AgentError 时进程退出码 2，打印候选未晋升。

另有函数 trial_command(task, program, trial_dir) 返回参数列表，依次包含 inspect-robots、任务句子、--policy、pai、--embodiment、nero、program 参数与 trial_dir 参数。测试只断言这个列表，不启动进程。evolve 在真机执行阶段用这个列表起子进程；子进程退出后若 diagnostics 没有 success，循环停住等待 label，直到字段出现或收到中断。中断返回 stopped，不调用 promote。自动测试不跑这个等待。

Task 7 补一行：LivePort.move 在调用记录时写入 target_vector，顺序就是 interpolate 使用的 DIM_LABELS。

- [ ] **Step 4: Run test to verify it passes**

Run: PYTHONPATH=plugins/inspect-robots-pai/src .venv/bin/python -m pytest plugins/inspect-robots-pai/tests -q
Expected: PASS

- [ ] **Step 5: Commit**

git add plugins/inspect-robots-pai
git commit -m "feat: export successful pai trials to LeRobot v2.1"






