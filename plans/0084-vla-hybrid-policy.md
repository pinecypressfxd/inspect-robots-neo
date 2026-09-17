# 0084: Helix 式混合策略——Astra 规划 + VLA(10055)执行

设计对谈结论(2026-09-17,范式 1 起步):

- **分层**:Astra(GPT-6 Astra,复用 agent 插件 wire)把任务分解为子目标;
  每个子目标委托给 VLA(本机 10055,`umi_replay_http` 服务)执行一段动作块;
  Astra 在检查点看执行结果,决定 继续/重试/纠正/放弃(重查询循环,ICML'26
  Specialist VLA 报告该循环带来 62%→90% 的提升)。
- **VLA 是手,脑在 Astra**。Astra 保留 done/give_up 出口。

## 协议事实(2026-09-17 对**线上 10055 实测 + serve_rlt_inference 源码**核实)

- 10055 = `serve_rlt_inference`(RLT stage2 进程,用户手改版,勿以仓里
  umi_replay 名称为准)。
- **POST /submit**:NPZ 载荷,字段 `image0..image3`(CHW uint8)或
  `images` 列表、`state`(本体状态,维度由 checkpoint 的 state_dim 决定)、
  `task`(提示词)。垃圾输入直接断连——客户端必须自带容错与超时。
- **GET /result/latest?after_request_id=N** → NPZ{request_id int64,
  **actions float32 (20,14)**, action_format="xyz_rpy", status}。
  20 步 × 2 臂 ×(xyz3+rpy3+gripper1),**delta EE pose** 块,gripper 绝对。
- `state` 的确切维度/布局实现时从 checkpoint config(stage2.py 的
  state_dim)与 `.rlt_runtime` 调试捕获(`*_state.npy`)钉死。

## 组件(新插件 `plugins/inspect-robots-vla`,import `inspect_robots_vla`)

1. **`_client.py` `UmiReplayClient`**:httpx 说 submit/poll;打包
   exteroception(三相机帧编码按 umi_replay 服务期望的格式,实现时以
   `umi_replay_proprioception_adapter.py`/`decode_replay_proprioception_json`
   为准)、proprioception、last_action、时间戳;解析动作块;超时与重试。
2. **`_anchor.py`**:**逐块重锚积分**——块开始读 eef_state(绝对),块内
   `目标_t = 锚点 + Σdelta`(xyz 直加;rpy 转旋转向量积分后转 rot6d);
   每块重新锚定,漂移不跨块;gripper 直通。输出我们的 20 维绝对 ActionChunk。
   附 rpy↔rot6d 转换(scipy,插件依赖)。
3. **`policy.py` `VlaPolicy`**:纯 VLA 适配器(独立注册 `umi-replay`),
   逐观测请求动作块、经 `_anchor` 输出。用途:纯 VLA 基线对照 + hybrid 内层。
4. **`policy.py` `HybridPolicy`**(注册 `hybrid`):对 eval 暴露为单一 Policy;
   内部持有 VlaPolicy 逻辑 + Astra 复用 agent 插件的 `_llm` 客户端与工具集
   模式。Astra 工具集:`delegate_skill(subgoal, max_seconds)`、`done`、
   `give_up`。`delegate_skill` 进入 VLA 执行态:逐块(观测→submit→poll→
   `_anchor`→ActionChunk)直到子目标完成/超时/**跟踪中断**;每
   `CHECKPOINT_INTERVAL_S` 携带最新观测回调 Astra 决策;
   块内末端跟踪误差超阈值提前结束块交回 Astra。

## 配置(全部 `_config.py` 常量,`-P k=v` 可覆盖)

```python
VLA_BASE_URL = "http://127.0.0.1:10055"
VLA_POLL_TIMEOUT_S = 30.0
# 块内跟踪中断阈值(用户要求可配置,默认对谈共识值)
TRACKING_ABORT_POS_M = 0.03      # 3 cm
TRACKING_ABORT_ROT_DEG = 20.0
CHECKPOINT_INTERVAL_S = 5.0      # Astra 检查点周期
MAX_SKILL_SECONDS = 60.0         # delegate_skill 单次上限
```

## 安全(不变式)

VLA 积分出的绝对目标走与 move_to **完全相同**的四层闸门:工作空间 Box →
Clamp + DeltaLimit → 每 tick 关节增量 ±0.1 rad → 关节包络(固件限位-0.05)。
VLA 输出物理上不可能命令出危险动作。钳位导致的滞后由逐块重锚自愈,
超阈值时块中断回到 Astra(重查询)。

## 测试

- fake VLA 服务(fixture,回放录制的块序列):协议打包/解析/超时。
- `_anchor` 单测:delta 积分、rpy↔rot6d 往返、重锚清零、gripper 直通、
  跟踪中断阈值触发(3cm/20° 边界)。
- HybridPolicy 状态机单测:delegate→块循环→检查点→done/give_up;
  Astra 侧用可注入的假 LLM。
- 真机冒烟经 mission console(先纯 `umi-replay` 基线,再 `hybrid`)。

## 里程碑

1. `_client` + fake 服务协议测试(不碰硬件)
2. `_anchor` 单测
3. `VlaPolicy` 注册 + 纯 VLA 真机基线(与 neo 栈对照成功率)
4. `HybridPolicy` 状态机 + 假 LLM 单测
5. 真机 hybrid 首跑(放杯子任务)
6. 文档/CHANGELOG/控制台接入

## 明确不做

- 移植 200Hz 控制器、三相机对齐门、录制栈(消费端"重"的部分)
- VLA 后训练(LeRobot 导出数据已就绪,另行立项)
- FLARE 式旁路监督(范式 3,验证范式 1 后再议)
