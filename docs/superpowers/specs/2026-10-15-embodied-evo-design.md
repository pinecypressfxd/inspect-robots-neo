# EmbodiedEvo 在 inspect-robots 上的设计

日期：2026-10-15
状态：已与需求方逐节确认，待审阅规格

## 1. 背景与目标

EmbodiedEvo（arXiv 2610.18867）把可执行 Python 当作技能，把不可执行文本当作教训。Playground 里模型反复提交程序、根据真实执行结果改技能和教训；Freeze 只带走这两样；Test 用新的尝试和新的程序，技能与教训只读。

本设计在 inspect-robots 里增加策略 evo，在真机 Nero 上跑这一循环。一次 Playground 就是一次真实尝试。没有场景 reset()。物体掉了不能放回去重来。

### 目标

- 新插件 plugins/inspect-robots-evo，注册策略名 evo。
- 模型通过读文件、写文件、提交代码来改 submission/solution.py。提交在受限 Python 里执行，动作经已接上的 Nero embodiment 下发。
- Playground 结束后由操作员标成功或失败，模型再改一轮技能和教训，然后只冻结这两个目录。Test 使用冻住的副本，且不能再改它们。

### 非目标

- 不改现有 agent、harness、vla。
- 插件不操作 CAN，不实现 IK。关节包络和 IK 保持原位的行为留在 Nero embodiment。
- 技能不是 Markdown，也不是从视频里选一条 VLA 句子。
- 第一版没有跨任务技能库，也没有自动课程。操作员可以把上一份冻住的 skills/ 和 lessons/ 手工拷进新任务工作区。
- 真机 Nero 不进自动测试。

## 2. 总体架构

操作员在 mission console 选择 policy=evo，并选择 playground 或 test。

EvoPolicy 包含三块：一个任务一个工作区；Astra（mog-10-a，responses），工具为 read_file、write_file、submit_code、done、give_up；受限执行器运行 solution.py 和 skills/*.py。原语是 get_instruction、get_observation、move_ee。动作交给 NeroEmbodiment 现有末端路径。

Astra 不直接调用 move_to。一次 submit_code 可以在函数内部连续多次 move_ee，函数返回后才把结果和最新画面交回 Astra。E-STOP 打断这次执行，并结束整次任务。

### 边界

- 插件只消费 Nero embodiment 已有的末端动作和观察。IK 失败时臂保持原位，这个行为不在 evo 里重写。
- primitives.md 和 task.md 由插件在建工作区时写入，write_file 不能改它们。
- 现有策略插件和 Nero 的 CAN、IK 实现保持不动。

## 3. 工作区

路径：.inspect_traces/evo/<YYYYmmdd-HHMMSS>_<id>/

- task.md：插件写入操作员给出的任务句子。
- primitives.md：插件写入三个原语、维度标签和返回值。
- submission/solution.py：Astra 写入，必须定义 solve()，返回 str。
- skills/*.py：Playground 的 Astra 写入可执行技能函数。
- lessons/*.md：Playground 的 Astra 写入何时调用、失败方式和恢复。
- scratch/：Astra 的草稿，不进入冻结。
- submits/<nnn>/：执行器写入该次提交的 stdout、返回值、画面和手臂状态。
- outcome.json：控制台写入操作员的成功或失败标签。

Playground 的 write_file 只能写 submission/solution.py、skills/、lessons/、scratch/。Test 只能写 submission/solution.py 和 scratch/。

Freeze 只复制 skills/ 和 lessons/ 到 freeze/。Test 新建一个工作区，把 freeze/ 里的这两个目录拷进去。Test 的 write_file 拒绝再改它们。solution.py 在 Test 里从空文件开始，仍然可改。

## 4. 原语与一次提交

维度标签与 Nero 末端动作相同：left_x/y/z、left_r1 到 left_r6（rot6d）、left_gripper，右臂同样一组。未写出的维度保持当前值。位置单位是米。夹爪宽度 0 为闭合，0.09 为张开。

- get_instruction() 返回 task.md 的任务句子，不带动机器人。
- get_observation() 返回 left_rgbd、right_rgbd、chest_rgbd 和当前 eef_state，不带动机器人。
- move_ee(targets) 走到给出的绝对末端目标，走现有插补、IK 和限位。成功返回 held=false 和 steps=N。IK 失败或关节步长被拒绝时臂保持原位，返回 held=true 和 steps=0。
- 没有 reset()。

执行器在调用 solve() 之前载入 skills/ 里的函数，solution.py 可以直接按函数名调用。solve() 的返回值必须是字符串，作为这次提交的状态证据，不是分数。

一次提交在下列任一条件出现时停止，并把当时的 stdout、返回值或 traceback、以及最新观察写入 submits/<nnn>/，再交回 Astra：

- solve() 返回。返回值不是字符串时，按运行时异常交回。
- 本次 move_ee 次数达到 80。
- 墙钟达到 600 秒。
- 运行时异常。

skills/ 按文件名排序载入。两个文件定义同名函数时，提交前拒绝执行。中途异常、动作数用尽或墙钟用尽时，手臂停在最后一次成功的 move_ee 上。E-STOP 不交回 Astra，直接结束整次任务。

执行阶段最多 20 次 submit_code，最多 80 次模型调用，其中包含读、写和提交。提交用尽或模型调用用尽都强制 give_up。Playground 结束后的改文件回合另计，最多 10 次模型调用，不占用这 80 次。

## 5. 任务如何结束

done 和 give_up 都不带动机器人。done 表示 Astra 认为任务完成。give_up 表示它停止。20 次提交或 80 次模型调用用完时，执行器强制 give_up。

任务结束后，包括 E-STOP 结束的情况，mission console 问操作员一次：成功还是失败。答案写入 outcome.json。

Playground 在拿到这个标签后，再给 Astra 一轮只许 read_file 和 write_file 的回合，用来更新 skills/ 和 lessons/。这一轮不能 submit_code，最多 10 次模型调用。回合结束后把这两个目录复制到 freeze/。Test 没有这一轮，操作员标签只记入 outcome.json。

## 6. 执行限制

执行器只把 get_instruction、get_observation、move_ee，以及 skills/ 里的函数放进命名空间。solution.py 和 skill 允许 math、json。os、subprocess、socket、open 以及其他 import 在执行前被拒绝，代码碰不到 CAN、IK 和文件系统。

600 秒墙钟到点后，按异常交回 Astra，手臂停在最后一次成功动作上。E-STOP 优先于墙钟和动作计数：打断当前执行并结束整次任务。

## 7. 验证

自动测试使用假 embodiment，不碰 CAN。覆盖：

- 新建工作区包含第 3 节的文件和目录。
- Freeze 只复制 skills/ 和 lessons/。
- 一次提交会调用 solve()，move_ee 到 80 次即停。
- os、subprocess 在执行前被拒绝。
- 假 embodiment 报告 IK 失败时，move_ee 返回 held=true，且不再下发新目标。
- E-STOP 结束整次任务。
- done 和 give_up 不产生动作。
- Test 模式下写入 skills/ 或 lessons/ 被拒绝。
- 操作员标签写入之后，Playground 才进入只改文件的回合，且该回合不能提交。

真机验收由操作员在 Nero 上做一次 Playground，再做一次 Test。自动测试不启动真机。
