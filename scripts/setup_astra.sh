#!/usr/bin/env bash
# 把 GPT-6-Astra（金山云网关）接到 inspect-robots 的最快路径。
#
#   用法：bash setup_astra.sh <你的API_KEY>
#
# 做三件事：
#   1. 写入密钥到 ~/.nerorc（chmod 600，不进 git）
#   2. 校验仓库里的网关参数与新值一致
#   3. 起一次最小 agent 任务，验证网关连通（不动机械臂）
#
# 之后用 scripts/start_mission.sh 起完整任务会话。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NERORC="${HOME}/.nerorc"
KEY_ENV=JINSHANYUN_CHATGPT_KEY
MODEL=mog-10-a
BASE_URL=https://kspmas.ksyun.com/v1

if [ $# -lt 1 ] || [ -z "${1:-}" ]; then
  cat >&2 <<USAGE
用法: bash $(basename "$0") <API_KEY>

API_KEY 从金山云 kspmas 网关后台获取（同 GPT-6-Astra 用的那个账号）。
不要把它贴进聊天、邮件或提交到 git。
USAGE
  exit 2
fi
KEY="$1"

# --- 1. 密钥落盘 ---
touch "$NERORC"
chmod 600 "$NERORC"
if grep -q "^export ${KEY_ENV}=" "$NERORC"; then
  # 覆盖旧值（就地替换，不追加第二行）
  tmp="$(mktemp)"
  sed "s|^export ${KEY_ENV}=.*|export ${KEY_ENV}=${KEY}|" "$NERORC" >"$tmp"
  mv "$tmp" "$NERORC"
  chmod 600 "$NERORC"
  echo "[ok] 已更新 ${NERORC} 里的 ${KEY_ENV}"
else
  printf 'export %s=%s\n' "$KEY_ENV" "$KEY" >>"$NERORC"
  echo "[ok] 已写入 ${KEY_ENV} 到 ${NERORC}"
fi
# shellcheck disable=SC1090
. "$NERORC"

# --- 2. 校验仓库参数 ---
cd "$REPO"
miss=0
check() { # name expected file
  if ! grep -q "$2" "$3"; then
    echo "[!!] $3 里没有 $1=$2，交接配置可能已漂移" >&2
    miss=1
  fi
}
check "model"     "$MODEL"    scripts/start_mission.sh
check "base-url"  "$BASE_URL" scripts/start_mission.sh
check "api-key"   "$KEY_ENV"  scripts/start_mission.sh
if [ "$miss" = 0 ]; then
  echo "[ok] start_mission.sh 的网关参数与本脚本一致"
fi

# --- 3. 最小连通性验证：不启动双臂，只跑一次 LLM 往返 ---
echo "[..] 验证网关连通（模型 ${MODEL} @ ${BASE_URL}）"
uv run --no-sync python - "$MODEL" "$BASE_URL" "$KEY_ENV" <<'PY'
import os, sys
from inspect_robots_agent._llm import AssistantMessage, ChatClient, resolve_provider
from inspect_robots_agent._responses import ResponsesClient

model, base_url, key_env = sys.argv[1:4]
if not os.environ.get(key_env):
    sys.exit(f"环境变量 {key_env} 为空")
provider = resolve_provider(
    model=model, base_url=base_url, api_key_env=key_env, env=dict(os.environ)
)
tools = [
    {
        "type": "function",
        "function": {
            "name": "noop",
            "description": "no-op",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]
messages = [{"role": "user", "content": "reply with the single word: ok"}]
# The kspmas gateway rejects tools + reasoning_effort on /chat/completions,
# so /v1/responses is the channel that must work; chat is reported for info.
for label, client in (
    ("responses", ResponsesClient(provider)),
    ("chat", ChatClient(provider)),
):
    try:
        reply = client.complete(messages, tools)
        assert isinstance(reply, AssistantMessage)
        print(f"[ok] {label} 通道可用（工具调用 schema 被接受）")
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001 - try each channel and report
        print(f"[..] {label} 通道不可用: {type(exc).__name__}: {str(exc)[:200]}")
sys.exit("[!!] 两个通道都失败：确认 key 有效、账户有额度、能访问 kspmas.ksyun.com")
PY

echo
echo "完成。起完整任务会话：bash scripts/start_mission.sh"