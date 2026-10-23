# inspect-robots-pai：真机 Nero 上的 HarnessPAI 框架

日期：2026-10-23
状态：已审阅，待实现

## 1. 背景与目标

HarnessPAI 把任务执行和程序改进分成两个时间尺度。一次试验内运行固定程序，程序负责感知、几何动作、动作原语和预设检查。多次试验之间，编程智能体根据反馈修改隔离候选，验证通过后才晋升进代码库。

本设计在 inspect-robots 里新增一个独立插件，把这套框架接到真机 Nero。不使用仿真。SAM、Molmo、GraspNet、VLA 由操作员准备服务，插件只提供客户端。

### 目标

- 新插件 `plugins/inspect-robots-pai`，注册策略名 `pai`，并提供命令 `inspect-robots-pai`。
- 与机器人无关的循环放在插件根包：`agents/`、`memory/`、`execution/`、`evolution/`、`primitives/`。
- Nero 的关节名、末端标签、相机和自动检查只放在 `domains/nero/`。
- 已晋升的任务程序放在 `code_library/nero/<task>/`。
- 一次试验内程序固定。试验之间走写候选、preflight、probe、执行、诊断、综合、晋升。
- 操作员给出的成败是晋升硬门。自动检查只辅助操作员判断。

### 非目标

- 不修改 `inspect-robots-evo`、`inspect-robots-harness`、`inspect-robots-vla`、`inspect-robots-nero`、`inspect-robots-agent`。
- 不 import 上述插件。需要的末端插补拷贝进 `domains/nero/`。
- 不引入 LIBERO、robosuite、RoboCasa 或 micromamba 环境。
- 第一版不实现示范视频抽工作流图。控制环里不直接写 LeRobot。成功试验在事后按第 7 节转成 LeRobot v2.1。
- 插件不操作 CAN，不实现 IK。关节包络和 IK 保持 Nero embodiment 的现有行为。
- 真机 Nero 不进自动测试。

## 2. 插件位置

代码位于 `inspect-robots/plugins/inspect-robots-pai/`，与 `inspect-robots-evo` 同级。仓库根 `pyproject.toml` 已有 `members = ["plugins/*"]`，根配置不用改。

```
inspect-robots/plugins/inspect-robots-pai/
├── pyproject.toml
├── README.md
├── configs/
│   ├── models.yaml
│   └── local.yaml          # 密钥，gitignored
├── tests/
└── src/inspect_robots_pai/
    ├── agents/
    ├── memory/
    ├── execution/
    ├── evolution/
    ├── primitives/
    ├── domains/nero/
    └── code_library/nero/
```

依赖只有 `inspect-robots` 核心和 `numpy`。策略入口：

```
[project.entry-points."inspect_robots.policies"]
pai = "inspect_robots_pai:pai_policy"

[project.scripts]
inspect-robots-pai = "inspect_robots_pai.cli:main"
```

`agents/`、`memory/`、`execution/`、`evolution/`、`primitives/` 不 import `domains/nero/`。域控制从 `domains/nero/` 调用通用模块。`code_library/nero/` 里的程序只通过 `NeroPort` 动机器人。

## 3. 运行目录与数据流

运行产物不进源码树。默认根目录是进程当前目录下的 `runs/pai/<task>/<YYYYmmdd-HHMMSS>/`，可用环境变量 `INSPECT_ROBOTS_PAI_RUNS` 更换。一次演化包含：

- `candidate/`：任务程序的隔离副本。Codex 只改这里。
- `trials/<index>/diagnostics.jsonl`：这次试验的过程记录。
- `trials/<index>/auto_check.json`：自动检查的辅助结论。
- `trials/<index>/steps.jsonl`：每一拍一行，供诊断和事后转换。
- `trials/<index>/frames/left|right|chest/`：该拍的三路画面。
- `contract.json`：本次演化的不可变成功合同。诊断用的视频由这三路帧编码得到，不另录一套无关视频。

`diagnostics.jsonl` 在操作员标注前不含 `success`。进程是否正常退出记在 `process_ok`。自动检查写入 `auto_check.json`，不写入 `diagnostics.success`。

### 一次试验

操作员或演化循环启动一次试验时，策略是 pai，本体是 nero。policy 参数 program 指向程序目录，trial_dir 指向 trials 下的序号目录。本体由 inspect-robots 运行时选择。pai 不 import Nero 插件。

PaiPolicy.act 只做适配：把观察交给 NeroPort，把末端目标插补成动作块交回运行时。execution.run_trial 加载该程序并调用程序入口 run。调用期间不改写程序，不调用 Codex 或 Gemini。

程序只能通过 NeroPort 动机器人，通过 primitives 访问感知和动作模型。每一次 act 返回的每个末端目标各写一行 steps.jsonl，并配上这次 act 看到的三路帧。一次插补若含多拍，这几行共用同一帧。一行包含：任务句子、三路相机时刻、本体时刻、20 维末端状态、16 维实测关节、该拍下发的 20 维末端目标。末端布局是左 xyz、左 rot6d、左夹爪宽度，右臂同样。关节布局是左 7 关节、右 7 关节、左夹爪宽度、右夹爪宽度。宽度单位是米。试验在程序返回、墙钟用尽、动作次数用尽、未捕获异常或操作员中止时结束。返回值写入 process_ok、termination 和阶段记录。run_trial 不设置 success。

操作员先看 auto_check.json，再运行 inspect-robots-pai label，传入试验目录，并选择成功或失败。该命令只写入这条 diagnostics 的 success 字段。

### 试验之间

inspect-robots-pai evolve 持有候选和合同。试验次数由参数 trial_count 决定，默认 1。合同写下序号 0 到 trial_count 减 1 的每一次试验。一轮顺序：

1. Write。Codex 按任务或上一轮指令修改 candidate。
2. Preflight。检查入口可导入、任务身份与合同一致。失败则回到 Write，不启动真机。
3. Probe。只重跑上一轮失败的试验。默认最多 3 次，可用参数提高。仍失败则进入诊断，不展开合同里的整批试验。
4. Execute。对合同列出的每次试验，演化进程用子进程启动第 3 节的 inspect-robots，策略 pai，本体 nero，并传入程序目录和试验目录，然后等待子进程退出。子进程里的 policy 调用 run_trial。演化进程不打开 CAN，不构造硬件。子进程退出后若还没有 success，循环停在这次试验，直到 label 写入或操作员中断演化。
5. Diagnose。Gemini 读取视频、diagnostics 和工作流图。没有图时只看视频和 diagnostics。Gemini 不写 success，也不宣布收敛。
6. Synthesize。Codex 只读候选、诊断和合同，返回下一轮修改指令。

合同内每一次试验都同时满足 process_ok 为真、存在 diagnostics、success 为真时，本轮晋升并停止。promote 把 candidate 复制到工作区源码目录 plugins/inspect-robots-pai/src/inspect_robots_pai/code_library/nero 下的对应任务目录，不写入 site-packages。同时把失败到修复的经验写入 memory，修复条目指向晋升后的实现。

下面任一情况都不晋升：只有进程正常退出而没有操作员成功标签、程序走到自己的阶段终点、自动检查通过、诊断文本写了已收敛、合同内仍有未标注或失败的试验、操作员中断演化。子进程退出码只说明 inspect-robots 进程是否正常结束。任务成败看 diagnostics 里的 process_ok 和 success。子进程异常退出时，该次试验的 process_ok 记为假。

轮数上限默认 5，是天花板。合同一旦通过就晋升并停止，不为了凑满 5 轮继续跑真机。5 轮用尽仍未通过时，候选留在运行目录，代码库保持原样。

## 4. 模块职责与接口

依赖方向：只有策略入口 pai_policy 可以 import domains/nero，并用它构造 NeroPort。execution、evolution、agents、memory、primitives 以及任务程序都不 import domains/nero。任务程序只调用传进来的 port。evolution 调用 execution、agents 和 memory。

### primitives

SAM、Molmo、GraspNet、VLA 各自一个构造函数。地址来自环境变量 SAM3_SERVICE_URL、MOLMO_SERVICE_URL、GRASPNET_SERVICE_URL、VLA_SERVICE_URL。输入是图像、文本或位姿，输出是掩码、点、抓取或动作块。客户端不发机器人动作。服务连不上时返回空结果，并把错误写入试验目录，不在客户端里重跑整次试验。

### domains/nero

唯一允许出现 Nero 关节名和末端标签的目录。末端标签与现有真机动作一致：left_x、left_y、left_z、left_r1 到 left_r6、left_gripper，右臂同样一组。

插补从 evo 的实现拷贝，不 import。位置和夹爪每拍步长 0.01，旋转每拍步长 0.05。未给出的维度保持当前值。位置单位是米。夹爪宽度 0 为闭合，0.09 为张开。

NeroPort 的方法：

- observation：当前三路画面、20 维末端状态、16 维实测关节，以及各路相机时刻和本体时刻。不发动作。
- move：阻塞到这段绝对末端目标执行完。IK 失败或关节步长被拒绝时臂保持原位，返回 held 为真、steps 为 0。成功返回 held 为假、steps 为 N。
- set_gripper：按侧和宽度开合指定夹爪。
- auto_check：返回辅助结论，并由执行器写入 auto_check.json。

相机名沿用 Nero 已有的左、右、胸前三路。

### code_library/nero 下的任务包

一份已晋升程序是一个可导入包。入口函数名是 run，参数是 port 和 trial_dir，无返回值。它组合 NeroPort 和 primitives，把阶段记录写入试验目录。它不给自己写 success。候选副本使用同一入口。

### execution

run_trial 的参数是程序目录、试验目录和 port。它加载固定程序并调用 run。返回进程是否正常退出、diagnostics 路径、自动检查路径。单次试验默认墙钟 600 秒，move 加 set_gripper 合计 80 次，两者都可通过 policy 参数配置。到达任一上限时试验以进程失败结束，手臂停在最后一次成功动作上。操作员中止同样记为进程失败。

pai policy 在 act 里把运行时的观察和动作接到 NeroPort。第一次 act 启动 run_trial 所在的线程。试验结束后 policy 请求停止。

### agents

- Codex 写模式：可修改给定的候选目录，返回是否写完。
- Codex 综合模式：只读，返回下一轮指令文本。
- Gemini：读取视频、diagnostics 和图，返回失败位置文本。

模型名和密钥放在 configs/models.yaml 与 gitignored 的 configs/local.yaml。调用失败时向演化循环抛出智能体错误，不改 success。

### memory

- 按任务描述找回已晋升程序的路径。找不到则演化在 candidate 里放一份脚手架：可导入的任务包，入口 run 只写一条阶段记录后返回，不调用 move。
- 按工作流节点保存失败、修复和代码引用。技能写在 src/inspect_robots_pai/memory/skills 下，每个节点一个 json。任务索引写在 src/inspect_robots_pai/memory/tasks.json。
- 只在晋升成功后写入。

### evolution

持有 contract.json 和候选目录。promote 的参数是候选目录和任务名，仅在第 3 节的合同全部通过时复制目录并写 memory。label 只改指定试验的 diagnostics.success。

## 5. 失败时停在哪里

- 程序异常、墙钟用尽、动作次数用尽、操作员中止：这次试验 process_ok 为假，不写 success。
- 感知或动作服务不可用：客户端记录空结果，程序结束当前阶段。execution 不自动重跑。
- 自动检查失败：只体现在 auto_check.json。试验继续，晋升门不读它。
- 未标注：演化停在该次试验，不进入诊断，不晋升。
- preflight 失败：同一轮回到 Codex，不启动子进程。
- probe 用尽仍失败：进入诊断，跳过整批 Execute。
- Codex 或 Gemini 失败：本轮停止，候选保留，代码库不变。
- 操作员中断演化：候选保留，代码库不变。

## 6. 测试

自动测试使用假的 NeroPort、假 Codex、假 Gemini 和假服务客户端，不启动真机，不访问网络。

- 晋升门：未标注、只有 process_ok、只有 auto_check.json 通过、或诊断文本含已收敛，都不复制 code_library。合同内每次试验都是进程成功、有 diagnostics、success 为真，才复制到 code_library/nero 下的任务目录，并写入一条技能。
- 隔离：修改 candidate 后，code_library/nero 的文件保持不变。晋升失败的路径也不写代码库。
- 依赖：src/inspect_robots_pai 里，除策略入口外，不 import domains/nero，也不 import inspect_robots_evo、inspect_robots_harness、inspect_robots_vla、inspect_robots_nero、inspect_robots_agent。末端标签字符串只定义在 domains/nero，测试通过该定义引用它们。
- 单次执行：假端口上的 run_trial 不调用智能体。label 选择成功之后，diagnostics 才有 success 为真。auto_check.json 与 success 分开存放。
- 预算：第 81 次动作或墙钟用尽时 process_ok 为假。preflight 失败时不产生 trials。probe 第 4 次不再启动试验。
- 转换：success 不为真的试验不进入数据集。一条假的 steps.jsonl 配上三路帧后，导出的 observation.state 与 action 是 16 维，夹爪由米换成 0 到 1，关节顺序变成每臂 7 关节后紧跟该臂夹爪。

真机验收由操作员在 Nero 上跑一次固定程序，标注成败，再跑一轮不会晋升的失败演化，确认代码库未被改写。自动测试不覆盖这一步。

## 7. 事后转换成 LeRobot v2.1

diagnostics、单独一条视频和操作员标签不能转成训练表。逐拍的 steps.jsonl 和三路帧可以。命令是 inspect-robots-pai export-lerobot，输入一次运行目录，输出目录的布局与 VR-2026-1020-inference-18-bottle 相同：codebase_version 为 v2.1，fps 为 30，data/chunk-000/episode_NNNNNN.parquet，videos 下 left、right、chest 三路，以及 meta 里的 info、tasks、episodes、episodes_stats、stats。

只导出 success 为真、并且每一拍都有三路帧的试验。失败、未标注或丢帧的试验留在运行目录，不进入数据集。一个运行目录里的多条成功试验按试验序号排成 episode_index。任务句子来自合同。

列的对应关系：

- observation.state 与 action：把 20 维 rot6d 末端还原成旋转矩阵，再写成 wxyz 四元数，得到 16 维。顺序是左 xyz、左四元数、左夹爪、右 xyz、右四元数、右夹爪。夹爪写入 clip(宽度米 / 0.09, 0, 1)。
- auxiliary.observation.joints：16 维。把记录里的「左 7 关节、右 7 关节、左宽度、右宽度」改成「左 7 关节、左夹爪、右 7 关节、右夹爪」。夹爪同样换成 0 到 1。
- auxiliary.action.joints：策略看不到 IK 发出的关节指令。这一列复制本拍的 auxiliary.observation.joints，并在 info.json 写明 auxiliary_action_joints_source 为 measured_joint_pos。训练若使用关节指令列，得到的是实测关节，不是电机指令。
- 视频键为 observation.images.left、right、chest，由对应帧按 30 fps 编成 h264，分辨率沿用该拍画面。Nero 默认画面是 480 乘 640。
- timestamp 为 frame_index / 30。next.done 只在该试验最后一拍为真。episode_index、index、task_index 按 LeRobot v2.1 填写。
- 对齐列用本拍已有的相机时刻和本体时刻填纳秒。没有硬件序列号时，序列号写拍号。算不出的延迟写 0。相机名写 left_rgbd、right_rgbd、chest_rgbd。

转换不参与晋升，也不改 candidate 和 code_library。









