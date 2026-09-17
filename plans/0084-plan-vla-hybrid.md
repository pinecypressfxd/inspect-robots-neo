# 0084 VLA Hybrid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Plugin `inspect-robots-vla`: a pure VLA policy (`umi-replay`) speaking the live :10055 wire protocol, and a Helix-style hybrid policy (`hybrid`) where Astra decomposes tasks and delegates skill segments to the VLA.

**Spec:** `plans/0084-vla-hybrid-policy.md` (protocol facts are BINDING; they were verified against the live service and `serve_rlt_inference` source).

**Global Constraints**

- New package `plugins/inspect-robots-vla/`, import `inspect_robots_vla`, conventions per `plugins/inspect-robots-nero/` (hatch pyproject, entry points `inspect_robots.policies`, ruff extend root, mypy strict + `[[tool.mypy.overrides]]` ignore for `httpx.*` if needed, plugin static version).
- Deps: `inspect-robots`, `numpy`, `scipy`, `httpx`. No torch, no vendor SDK. `uv lock` after pyproject changes.
- Gates per task: `uv run --no-sync ruff check plugins/inspect-robots-vla`, `ruff format --check`, `uv run --no-sync mypy --config-file plugins/inspect-robots-vla/pyproject.toml plugins/inspect-robots-vla/src/inspect_robots_vla plugins/inspect-robots-vla/tests`, `uv run --no-sync python -m pytest plugins/inspect-robots-vla/tests -q`.
- All tunables in `src/inspect_robots_vla/_config.py`, every policy kwarg overridable via `-P k=v`.
- Prose: no em dashes. Commit per task on the feature branch.

---

### Task 1: Package skeleton + `_client.py` protocol client

**Files:** `pyproject.toml` (+`uv.lock`), `__init__.py` (registry factory), `_config.py`, `_client.py`, tests `test_client.py` (+ tiny in-test fake server via `http.server` on an ephemeral port).

**Interfaces:**
- `_config.py`: `VLA_BASE_URL = "http://127.0.0.1:10055"`, `VLA_SUBMIT_TIMEOUT_S = 10.0`, `VLA_POLL_INTERVAL_S = 0.05`, `VLA_POLL_TIMEOUT_S = 30.0`, `TRACKING_ABORT_POS_M = 0.03`, `TRACKING_ABORT_ROT_DEG = 20.0`, `CHECKPOINT_INTERVAL_S = 5.0`, `MAX_SKILL_SECONDS = 60.0`, `CHUNK_STEPS = 20`, `ACTION_DIM_VLA = 14`(per-arm xyz3+rpy3+gripper1 ×2).
- `_client.VlaClient(base_url, *, timeout_s, poll_interval_s, poll_timeout_s, transport=None)` (httpx injectable transport for fakes):
  - `submit(images: Mapping[str, np.ndarray] | None, state: np.ndarray, task: str, *, request_id: int) -> None` — POST `/submit`, body NPZ bytes (numpy `savez` to a buffer): fields `image{N}` (HWC uint8 RGB → **CHW uint8**, in sorted key order, 2..3 entries allowed), `state` float32, `task` str, `request_id`; Content-Type per spec. Server closes on garbage → any `httpx.TransportError`/bad status raises `VlaServiceError` (new, subclass `RuntimeError`) with the base_url and remedy text.
  - `poll(after_request_id: int) -> VlaChunk | None` — GET `/result/latest?after_request_id=N`; 204/empty → None; NPZ `{request_id, actions float32 (m,14), action_format, status}`; `status != "ok"` → `VlaServiceError`; wrong `action_format` → `VlaServiceError`; returns frozen `VlaChunk(request_id: int, deltas: np.ndarray (m,14))`.
  - `infer(images, state, task, *, request_id) -> VlaChunk` — submit then poll-loop until the response `request_id >= request_id` or timeout (`VlaServiceError` on timeout).
- Tests: fake server returning a canned (20,14) NPZ; happy path; 204-then-result; timeout; error status; wrong format; garbage-submit disconnect simulated as transport error; CHW conversion assert; state/task passthrough assert (fake server echoes received NPZ fields).

### Task 2: `_anchor.py` chunk re-anchoring

**Files:** `_anchor.py`, tests `test_anchor.py`.

**Interfaces:**
- `rpy_to_rot6d(rpy) / rot6d_to_rpy(rot6d)` — scipy `Rotation`; round-trip tests.
- `anchor_chunk(eef_state: np.ndarray (20,), chunk: VlaChunk) -> np.ndarray (m, 20)` — per arm: xyz deltas cumsum onto anchor xyz; rpy deltas cumsum in rpy space then converted per-step to rot6d; gripper passthrough (absolute). Output layout = embodiment 20-dim order. Non-finite anywhere → `VlaServiceError`.
- `tracking_error(target20: np.ndarray, observed20: np.ndarray) -> tuple[float, float]` — (pos_m, rot_deg) using rot6d→relative rotation angle. Boundary tests at 3 cm / 20°.
- Tests: pure deltas zero → identity targets; cumsum correctness; gripper passthrough; re-anchor resets drift (feed two chunks with the same anchor → no accumulation); non-finite rejection.

### Task 3: `VlaPolicy` (registry `umi-replay`)

**Files:** `policy.py` (VlaPolicy part), `__init__.py` entry point, tests `test_vla_policy.py` with fake client injected.

**Interfaces:**
- `vla_policy(base_url=VLA_BASE_URL, *, prompt, submit_images=("left_rgbd","right_rgbd","chest_rgbd"), state_key="eef_state", **tunables) -> VlaPolicy`.
- `VlaPolicy(PolicyBase)`: `info` (chunked, control_hz=30); `bind(embodiment_info)` records state field + labels; `next_action(observation)` — build state from `observation.state[state_key]`, images per submit_images, `task=prompt`(+scene instruction if policy receives it via bound task envelope or fallback prompt kwarg), request_id monotonically increasing, `anchor_chunk` the result, return `ActionChunk`. Transient `VlaServiceError` → retry once then raise `PolicyError` (core taxonomy). Track per-step `last commanded` internally to build next `state` fallback when observation lacks the key.
- Tests: fake client scripted chunks; assertion of submitted task/state/images; ActionChunk shape/length; error→retry→PolicyError path.

### Task 4: `HybridPolicy` (registry `hybrid`)

**Files:** `policy.py` (HybridPolicy), tests `test_hybrid_policy.py` (fake LLM + fake VLA client).

**Interfaces:**
- `hybrid_policy(model, base_url, api_key_env, *, vla_base_url, prompt, llm=None, vla=None, **tunables)`; both backends injectable.
- Reuses the agent plugin's chat client via a thin seam: constructor takes `llm` (an object with `.complete(messages, tools)`; default builds one from inspect_robots_agent._llm with the -P args).
- State machine: `PLANNING` (LLM call with tools `delegate_skill(subgoal, max_seconds)`, `done(summary, hindsight)`, `give_up(reason, hindsight)`; system prompt = nero docs + hybrid role text) → `EXECUTING` (chunk loop exactly like VlaPolicy, but the task string sent to the VLA is the **subgoal**; every CHECKPOINT_INTERVAL_S or on tracking-abort or chunk end, back to `DECIDING`: one LLM call with the latest observation + a compact skill progress note: steps executed, last tracking error, chunks used) → repeat until `done`/`give_up`/budget.
- Tracking abort: after each executed chunk step compare commanded target vs observed eef_state (`tracking_error`); > thresholds → end skill segment, report to LLM as `skill_interrupted: pos_err, rot_err`.
- `MAX_SKILL_SECONDS` cap per delegation; global LLM-call budget (reuse agent plugin's constant).
- Tests: scripted fake LLM returning delegate→delegate→done; tracking-abort triggers DECIDING with interrupt note; give_up path; budget exhaustion forces give_up; subgoal string passed as VLA task; done chunk has `request_stop` meta (mirror agent `_stop`).

### Task 5: README + CHANGELOG + doctor conformance + mission-console note

- Plugin README (safety: VLA chunks pass the same four clamp layers; usage `--policy umi-replay -P prompt=...` baseline and `--policy hybrid -P model=gpt-6-astra -P base_url=... -P api_key_env=EXPLABS_API_KEY -P prompt=<task>`); `inspect-robots doctor --policy umi-replay` clean (no runtime deps beyond httpx); CHANGELOG `**Plugins:**`; mission console README note (hybrid flags passthrough); notes dir update (06-vla混合策略.md with the live protocol table).
- Manual hardware checklist (do NOT execute): baseline `umi-replay` run on the cup task; then `hybrid` run.

---

## Verification

- All plugin gates green each task; `uv run --no-sync inspect-robots list policies` shows `umi-replay` and `hybrid` after Task 3/4.
- Protocol conformance proven against the in-test fake server byte-for-byte (NPZ field names, CHW, endpoint paths) — the fake mirrors the live capture from `plans/0084-vla-hybrid-policy.md`.
- Real-10055 smoke (operator-run): `curl`-level submit with a captured state npy is the acceptance probe before any arm motion.
