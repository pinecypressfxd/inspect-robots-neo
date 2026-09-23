# Jev × Nero 真机双臂交接链路设计

日期：2026-09-23
状态：已与需求方逐节确认，待转入实现计划

## 1. 背景与目标

`jev-robot-control`（独立仓库）验证了一种两段式 LLM 控制协议：每个决策周期先选
intent，再为 X/Y/Z/夹爪各选一个方向通道，由共享执行器把方向选择转成小步笛卡尔
增量。该仓库只在 MuJoCo 仿真（单臂 xArm7）中运行。

本设计把该协议移植为 inspect-robots 的一个策略插件，直接驱动
`plugins/inspect-robots-nero` 的 `NeroEmbodiment`（真机双臂、CAN、pika 夹爪、
D405 相机），在真机上完成**双臂紧协同交接任务**：左臂从台上抓起物体举起，右臂
从左臂手中接管，再放到目标点。物体全程至少被一臂持有。

**不使用 MuJoCo**；真机是唯一执行环境，测试用注入的 fake 工厂。

### 目标

- 新插件 `plugins/inspect-robots-jev`，注册策略名 `jev`，命令行
  `inspect-robots "<任务指令>" --policy jev --embodiment nero` 可跑。
- 真机低速（attended）完成一次完整交接，全程无门禁违规下发、无 CAN fault，
  trial log 与 transcript 可回放审计。
- 复用框架现有能力：rollout、operator console、日志、verdict、LeRobot 导出、
  NeroEmbodiment 的全部安全机制。

### 非目标

- 不改 `NeroEmbodiment`、不改 jev-robot-control 仓库（它保持为仿真基准与逻辑
  参考源，本插件不 import 它）。
- 不做相机感知（目标几何来自操作员示教声明）。
- 不做双臂共抬/对接装配等其他协同形态（phase 机仅覆盖交接）。
- 不做模型端点的静默重试或备用模型回退。

## 2. 总体架构

```
CLI: inspect-robots "指令" --policy jev --embodiment nero [-P ...]

rollout 循环（框架现有）
  ├─ NeroEmbodiment（现有，零改动）
  │    reset(scene) → Observation
  │    step(20维绝对EE位姿) → StepResult
  └─ JevHandoffPolicy(PolicyBase)（新插件）
       act(observation) → ActionChunk(H=1)
         1. _state:    observation + 示教目标 → 结构化双臂 state dict
         2. _decisions: Jev 端点两段式调用（phase 1 问 + motor 8 问）
         3. _executor: 方向选择 + phase 硬门禁 → 20维绝对位姿 Action
```

每循环数据流：`step()` 的 Observation（`eef_state` 20 维、`joint_pos`、三路相
机、框架注入的 `extra.operator_messages`）→ `_state` 组装 jev 风格 state →
`_decisions` 两次模型调用 → `_executor` 产出 20 维 action → 框架交回
`NeroEmbodiment.step()`（解 IK → 限幅关节增量 → CAN 下发）→ 新 Observation。

### 边界与不变量

- 插件只消费 `NeroEmbodiment` 公开契约（action space `Box`、observation
  state、`control_hz`）；guardrails（关节增量限幅 + 关节包络）继续生效，插件
  侧安全规则是叠加而非替代。
- `act()` 每次恰好产出 1 个 action（H=1 反应式，等价原仓库每周期一次决策）。
- 双臂都在环内；非活跃臂（该 phase 无角色的一侧）hold 当前位，不回填 home
  （两臂归 home 只发生在 embodiment reset，由其增量限幅保证平滑）。

## 3. 组件设计

### 3.1 全局交接 phase 状态机（`_phases.py`）

方案对比后选定**全局联合 phase + 两臂 8 电机通道**（否决了双臂各自独立 intent
的方案：两臂 intent 可能不一致，交接安全只能靠提示词兜底，真机风险不可接受）。
单一 phase 驱动两臂，结构上不可能出现「左臂在右臂未持稳时松手」这类意图冲突，
且允许执行器把交接安全规则做成代码级硬门禁。

11 个 phase（条件为反馈策略语义——条件表，非强制序列；条件同时引用双臂几何
与持有推断）：

| Phase | 进入条件（示例） | 两臂角色 |
|---|---|---|
| `approach_left` | 左未持有 | 左臂开爪、XY 对齐物体抓取位后再降 Z；右臂 hold |
| `grasp` | 左未持有但已到位，或已接触未持稳 | 左臂 XYZ hold + close 至 `left_secured`；右臂 hold |
| `lift` | 左已持稳，低于运送高度 | 左臂垂直抬升至 `travel_z`；右臂 hold |
| `offer` | 左持稳且在运送高度 | 左臂水平移到交接位上方再降到位；右臂预备 |
| `handoff_grasp` | 左持物在交接位 且 右未持有 | 左臂全 hold；右臂开爪对齐交接抓取位后 close |
| `handoff_release` | 左持有 且 右已持稳 | 左臂 XYZ hold + open；右臂 hold 闭合 |
| `carry` | 右持稳，放置点 XY 误差超容差 | 右臂水平搬运保持 `travel_z`；左臂 hold |
| `lower` | 右持稳且在放置点正上方 | 右臂垂直下降至释放高度；左臂 hold |
| `release` | 右持稳且到位 | 右臂 XYZ hold + open |
| `withdraw` | 物体已释放且右臂低于安全高度 | 右臂开爪抬升 |
| `finish` | 物体已稳定释放、两爪全开、双臂高于安全高度 | 全 hold |

文案为通用物体表述（物体与放置点来自示教声明，不写死具体物体）。

### 3.2 两段式决策（`_decisions.py`，移植 `IncrementalPolicy`）

- **Phase 调用**：1 个 choice 问题，`criteria` = 11 个 phase 的条件，
  `instructions` 讲清反馈策略语义、毫米误差读法（目标−当前）、对齐标志是真值
  实测非上次命令。
- **Motor 调用**：8 个 choice 问题 `left_x/left_y/left_z/left_gripper/
  right_x/right_y/right_z/right_gripper`；每个问题的 instructions 内嵌该轴当
  前值、抓取/交接/放置目标误差（毫米）与「只答本轴」约束；criteria 同原版
  （XYZ: negative/hold/positive；夹爪: open/hold/close）。`MOTOR_INSTRUCTIONS`
  重写为按 phase 的双臂角色约束（如 `handoff_grasp` 期间左臂三轴必须 hold）。
- **校验**：逐字移植 `validate_choice`：概率须覆盖全部选项、有限 0–1、和
  ≈1（±0.025）、choice 须为概率众数；`confidence` 字段可选并校验。响应不合规
  → `act()` 抛错，trial 记 error 结束，不静默重试。
- 成本/延迟/调用数自累计进 transcript（对齐原仓库 records 结构）。

### 3.3 Provider 适配（`-P` 参数化）

| | `provider=jev`（默认） | `provider=llm`（对比实验） |
|---|---|---|
| 端点 | `POST {base_url}`，默认 `https://openrouter.ai/api/alpha/decisions` | `POST {base_url}`，默认 `https://openrouter.ai/api/v1/chat/completions` |
| 载荷 | `{model, state, questions}`，原生概率分布 | strict JSON schema（按问题动态构建 answers schema） |
| Key | `api_key_env`，默认 `OPENROUTER_API_KEY` | 同左 |
| 其他 | `timeout_s` 默认 90 | `max_tokens` 850 / GPT-6 类 `reasoning_effort` + `max_completion_tokens` 4096（同原版） |

`base_url` 直通设计兼容 experientiallabs proxy（nero 侧已验证模式），不写死
域名。key/配置缺失在启动时 fail-fast。

### 3.4 Policy 类（`policy.py`）

```python
@register("policy", "jev")
class JevHandoffPolicy(PolicyBase):
    def bind(embodiment_info)   # 校验 eef_abs_pose + rot6d + 20 维双臂；记录 control_hz
    def reset(scene)            # 清 trial 状态；载入 targets_file；重置 secured/步进计数
    def act(observation) -> ActionChunk   # H=1
    def transcript()            # 本次 trial 全部调用记录（请求/答案/延迟/成本/门禁拒绝）
```

框架注入的 `observation.extra["operator_messages"]` 并入 `state.feedback`。

### 3.5 状态组装（`_state.py`）

**示教式目标声明**：`scripts/declare_targets.py`（bench_smoke 风格，操作员逐
步确认的 jog 工具）把臂开到各位姿并读 `eef_state` 存档，产出 targets JSON：

```json
{
  "object": "cup",
  "left_grasp":   {"tcp_m": [x, y, z], "rot6d": "[6 个数的 rot6d]", "width_m": 0.05},
  "handoff_left": {"tcp_m": [x, y, z], "rot6d": "…"},
  "handoff_right":{"tcp_m": [x, y, z], "rot6d": "…", "width_m": 0.05},
  "place":        {"tcp_m": [x, y, z], "rot6d": "…", "center_m": [x, y, z]},
  "travel_z_m": 0.20
}
```

（`tcp_m` 单位米、基座系；`rot6d` 为示教时 `eef_state` 中的姿态六元组。）

运行时 `-P targets_file=targets.json` 载入。姿态（rot6d）随示教记录，执行器全
程回填——等价原版「固定姿态 IK」。

**持有推断**（无接触真值的替代）：`held = 命令宽度 ≤ 0.005 m 且 实际宽度 −
命令宽度 > 0.015 m`（命令闭合、实际停在物体宽度 → 夹到物体；两个阈值均在
`_config.py` 可调）；`secured = held 连续 ≥ 0.45 s`。已知局限（写入插件
README）：宽度漂移或软物体可能误判；`handoff_release` 门禁额外要求
`right_secured`，双保险。

**对齐容差默认值**（`_config.py` 可调）：`left_grasp_reached` / `at_offer_pose`
/ `handoff` 对齐 XY ≤ 6 mm、Z ±4 mm；`over_place` XY ≤ 8 mm；`at_release_height`
Z ±5 mm；`at_travel_height` / **安全高度**（`withdraw`/`finish` 条件中的「高于
安全高度」）= `travel_z_m`。真机首跑若达不到可放宽，写进 README。

**state 字段映射**（jev 单臂版 → 真机双臂版）：

| 原字段 | 真机版 |
|---|---|
| `tcp_mm` | `left_tcp_mm` / `right_tcp_mm`（`eef_state` 换算） |
| `grasp_tcp_mm` | `left_grasp_tcp_mm`、`handoff_right_tcp_mm`（按阶段给目标） |
| `release_tcp_mm` | `place_tcp_mm` |
| `*_minus_tcp_mm` | `left_grasp_minus_left_tcp_mm`、`handoff_minus_right_tcp_mm`、`place_minus_right_tcp_mm`（目标−当前，毫米） |
| `alignment.*` | `left_grasp_reached` / `at_offer_pose` / `over_place` / `at_travel_height` / `at_release_height`（几何实测） |
| `held` / `grasp_secured` | `left_held` / `right_held` / `left_secured` / `right_secured`（宽度推断） |
| `gripper_command` | 每臂命令宽度 + 实际宽度 |
| `finger_contacts` 等接触字段 | 无真值，不进 state |
| `fruit_in_plate` / `stable_released_in_plate` | 近似：`released_at_place`（曾到位释放且 TCP 已离开）+ 时间阈值 |
| `previous_intent` / `feedback` | `previous_phase` / 执行器结果 + `operator_messages` |

### 3.6 执行器（`_executor.py`）

**方向 → 绝对位姿**：

- `target_xyz = current_xyz + direction × step_size`；步长缩放移植原版——距该
  phase 参考目标 <12 mm → 2 mm，<25 mm → 4 mm，否则 18 mm（`max_step_mm` 可
  压低）。参考目标按 phase：`approach_left`→左抓取位；`offer`→交接左位；
  `handoff_grasp`→交接右位；`carry/lower/release`→放置位；`lift/withdraw`→
  当前 XY 保持 `travel_z`。
- rot6d 全程回填当前值（示教姿态即任务姿态）。
- 夹爪：`open`→0.09 m（`GRIPPER_WIDTH_MAX_M`），`close`→0 m，`hold`→上次命
  令值。

**Phase 硬门禁**（代码级，先于下发；被拒通道降级为 hold，不废整步，逐条落
`meta` 供审计）：

1. `grasp` / `release`：三轴强制 hold，仅夹爪可动；
2. `lift` / `lower` / `withdraw`：仅 Z 可动，XY hold；
3. `handoff_grasp`：左臂全部强制 hold，仅右臂可动；
4. `handoff_release`：左爪 `open` 仅当 `right_secured`，否则强制 hold 并写明
   原因（防摔核心）；
5. 通用：某臂 `held` 时，`open` 仅允许出现在 `release` / `handoff_release` /
   `withdraw` / `finish`；
6. 动作空间边界：目标越出 embodiment `Box` 的轴 → 该轴降级 hold；
7. 防碰撞：非交接 phase（`offer` / `handoff_*` 豁免——交接必然贴近）中，会缩
   小双臂 TCP 间距至 `min_separation_m`（默认 0.08）的方向被拒。

每步 action 的 `meta` 记录 phase、模型原始答案、被拒通道与原因（审计 + LeRobot
导出兼容）。

## 4. 错误处理与安全

| 故障层 | 行为 |
|---|---|
| 模型端点/校验失败 | `act()` 抛错 → 框架记 error 结束 trial；无静默重试、无备用模型 |
| IK 不可达 | embodiment 现有行为：hold + 警告 + 返回观测；策略下循环看到「没动」自行换目标 |
| CAN 掉线/反馈丢失 | `EmbodimentFault` → trial 结束（现有） |
| 物体滑落（held 突变 False） | 门禁 5 已禁 open；state 如实反映，模型重新决策；操作员可 Esc / console Stop |
| 门禁拒绝风暴 | 连续 `stall_steps`（默认 20）步无任何有效运动 → policy 抛 `TaskStalled` 结束 trial，operator 给 verdict |
| key/配置缺失 | 启动 fail-fast |

安全叠加项：

- `-P dry_run=True`：推理与门禁照常，但下发全 hold action——不动真机的全链路
  冒烟。
- 首跑规范（README 置顶，风格同 nero）：SPEED_PERCENT=20（embodiment 默认）、
  executor `max_step_mm` 压到 4、操作员 reset 确认 + Esc 急停 + console 两步
  Start。

## 5. 测试与验收

**单测（零硬件）**——fake `arm_factory` / `camera_factory`（仿 nero
`tests/fakes.py`）+ fake decisions transport：

- `_decisions`：双臂问题构建、8 通道 JSON schema、`validate_choice` 移植；
- `_state`：`eef_state`→state 映射、误差/对齐标志、held 与 secured 时间累计；
- `_executor`：步长缩放表、全 phase × 越权通道的门禁矩阵、边界/防碰撞拒绝、
  `handoff_release` 在 `right_secured=False` 时拒 open；
- `policy`：`bind` 拒绝非 20 维/非 eef_abs_pose、`reset` 载入 targets、`act`
  端到端；
- 集成：fake embodiment + scripted fake decisions 走完 `approach_left →
  grasp` 数步。

**硬件验收（attended 渐进四步）**：

1. `declare_targets.py` 示教产出 targets.json；
2. `dry_run=True` 完整 episode（两臂不动，查日志）；
3. 低速空载（无物体）验证方向与步长；
4. 低速带物体完整交接一次，operator verdict。

**成功标准**：真机完成一次双臂交接；全程无门禁违规下发、无 CAN fault；trial
log + transcript 可完整回放审计。

## 6. 目录结构与配置

```
plugins/inspect-robots-jev/
├─ pyproject.toml              # entry point: inspect_robots.policy → "jev"
├─ README.md                   # 安全警告置顶（同 nero 风格）
├─ src/inspect_robots_jev/
│  ├─ __init__.py              # @register("policy", "jev")
│  ├─ policy.py                # JevHandoffPolicy(PolicyBase)
│  ├─ _phases.py               # phase 表 + 提示词
│  ├─ _decisions.py            # provider 适配 + validate_choice
│  ├─ _state.py                # targets 载入 + state 组装 + held 推断
│  ├─ _executor.py             # 方向→绝对位姿 + 硬门禁
│  └─ _config.py               # 阈值/步长/端点默认
├─ scripts/declare_targets.py  # 示教式目标声明
└─ tests/                      # fakes.py + 单测
```

`-P` 参数总览：`provider` `model` `base_url` `api_key_env` `timeout_s`
`targets_file` `dry_run` `max_step_mm` `min_separation_m` `stall_steps`。

## 7. 里程碑

- **M1** 决策层 + 状态组装 + 执行器（单测全绿，零硬件）；
- **M2** dry-run 全链路（fake embodiment 全流程 + 真机 `dry_run=True` episode）；
- **M3** 真机低速验收（四步渐进至完整交接）。

## 8. 参考实现映射

| 本插件 | jev-robot-control 来源 |
|---|---|
| `_decisions.py` 两段式调用与校验 | `incremental_policy.py` `IncrementalPolicy.request/intent/motor`、`validate_choice` |
| `_executor.py` 步长缩放与拒绝语义 | `incremental_env.py` `IncrementalEnv.increment`（18/4/2 mm 缩放、边界拒绝、feedback） |
| `_state.py` 误差/对齐字段结构 | `incremental_env.py` `IncrementalEnv.observe` |
| phase 条件的反馈策略写法 | `incremental_policy.py` `INTENTS` / `INTENT_INSTRUCTIONS` |
