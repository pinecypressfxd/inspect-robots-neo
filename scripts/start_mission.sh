#!/usr/bin/env bash
# 一键启动真机任务会话(2026-09-21):
#   1. VLA 推理服务 :10055(serve_rlt_inference,stage2_rlt_pi05_dualarm)
#   2. 日志视图     :8300(inspect-robots view logs --serve)
#   3. 任务台       :8400(mission console;金山云网关 mog-10-a,走 /v1/responses)
# 已在运行的服务自动跳过,重复执行安全。日志在 /tmp/*.log。
set -euo pipefail

REPO=/home/dell/fengxuedong/projects/opensource/inspect-robots
VLA_DIR=/home/dell/fengxuedong/projects/neo_manipulation/.worktree/RLtoken
VLA_CFG="$VLA_DIR/neo_manipulation/training/rl_token_training/configs/realworld/stage2_rlt_pi05_dualarm.json"
VLA_PY="$VLA_DIR/.venv/bin/python"

# --- 密钥:把 export JINSHANYUN_CHATGPT_KEY=<key> 写进 ~/.nerorc(chmod 600) ---
if [ -f "$HOME/.nerorc" ]; then . "$HOME/.nerorc"; fi
if [ -z "${JINSHANYUN_CHATGPT_KEY:-}" ]; then
  echo "!! JINSHANYUN_CHATGPT_KEY 未设置:agent/harness 运行会在起跑时失败" >&2
  echo "   编辑 ~/.nerorc 写入 export JINSHANYUN_CHATGPT_KEY=<你的key> 后重跑本脚本" >&2
fi

up() { curl -s -m 2 -o /dev/null --fail-with-body "$1" >/dev/null 2>&1 || return 1; }

# 0) 厂商 SDK 自愈:pyAgxArm 不在 PyPI、不能进 lockfile,任何 uv sync 都会把它
#    从 venv 剔除(症状:run 一启动就 EmbodimentFault: No module named 'pyAgxArm')。
#    这里检测不到就重装(uv pip 不裁剪环境)。
cd "$REPO"
if ! uv run --no-sync python -c "import pyAgxArm" >/dev/null 2>&1; then
  echo "[..] pyAgxArm 被 uv sync 清掉了,重装厂商 SDK"
  uv pip install -e "$VLA_DIR/third_party/pyAgxArm" -q
  uv run --no-sync python -c "import pyAgxArm" >/dev/null 2>&1 \
    && echo "[ok] pyAgxArm 已恢复" || echo "[!!] pyAgxArm 重装失败,手动跑: uv pip install -e $VLA_DIR/third_party/pyAgxArm" >&2
else
  echo "[ok] pyAgxArm 在位"
fi

# 1) VLA 推理服务(配置里的 stage1.config_path 与 .rlt_runtime 都是相对路径,必须 cd)
if up "http://127.0.0.1:10055/result/latest"; then
  echo "[ok] VLA 推理已在 :10055"
else
  echo "[..] 启动 VLA 推理 :10055(模型加载约 1-2 分钟,日志 /tmp/serve_rlt_inference.log)"
  (
    cd "$VLA_DIR"
    nohup "$VLA_PY" -m neo_manipulation.training.rl_token_training.app.serve_rlt_inference \
      --config "$VLA_CFG" --host 127.0.0.1 --port 10055 \
      >/tmp/serve_rlt_inference.log 2>&1 &
  )
  ready=0
  for _ in $(seq 1 90); do
    if up "http://127.0.0.1:10055/result/latest"; then ready=1; break; fi
    sleep 2
  done
  [ "$ready" = 1 ] && echo "[ok] VLA 推理就绪" || echo "[!!] VLA 推理 3 分钟未就绪,查 /tmp/serve_rlt_inference.log" >&2
fi

# 2) 日志视图 :8300
if up "http://127.0.0.1:8300/"; then
  echo "[ok] 日志视图已在 :8300"
else
  echo "[..] 启动日志视图 :8300(日志 /tmp/ir-view-8300.log)"
  (
    cd "$REPO"
    nohup uv run --no-sync inspect-robots view logs --serve >/tmp/ir-view-8300.log 2>&1 &
  )
  sleep 3
  up "http://127.0.0.1:8300/" && echo "[ok] http://127.0.0.1:8300/" || echo "[!!] 视图未就绪,查 /tmp/ir-view-8300.log" >&2
fi

# 3) 任务台 :8400(页面上选 agent / harness / vla)
if up "http://127.0.0.1:8400/"; then
  echo "[ok] 任务台已在 :8400"
else
  echo "[..] 启动任务台 :8400(日志 /tmp/ir-console-8400.log)"
  (
    cd "$REPO"
    nohup uv run --no-sync python plugins/inspect-robots-nero/scripts/mission_console.py \
      --model mog-10-a \
      --base-url https://kspmas.ksyun.com/v1 \
      --api-key-env JINSHANYUN_CHATGPT_KEY \
      >/tmp/ir-console-8400.log 2>&1 &
  )
  sleep 3
  up "http://127.0.0.1:8400/" && echo "[ok] http://127.0.0.1:8400/" || echo "[!!] 任务台未就绪,查 /tmp/ir-console-8400.log" >&2
fi

echo
echo "全部就绪:打开 http://127.0.0.1:8400/ 开始。机械臂上电后先点 Home(两夹爪会全开)。"
