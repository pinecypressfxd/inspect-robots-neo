"""Tunable defaults for the :10055 VLA wire client and the hybrid policy.

Every value here is a constructor default, overridable per run with ``-P k=v``
through the policy factories (plan 0084). Wire facts (field names, CHW layout,
``xyz_rpy`` delta actions) are verified against the live service and the
``serve_rlt_inference`` source; see ``plans/0084-vla-hybrid-policy.md``.
"""

from __future__ import annotations

#: The VLA checkpoint's finetuning prompt, verbatim from the training data's
#: task field (meta/tasks.jsonl of VR-2026-0831-v0 and VR-2026-0901-v0, both
#: carry this one string; single task_index 0). The skill policy generalizes
#: only near this exact phrasing: near-variants (extra commas, "then") are
#: distribution shifts that stall the rollout after the trained half.
TRAINED_SKILL_PROMPT = "put the cup on the pad with left arm and take away the cup with right arm"

VLA_BASE_URL = "http://127.0.0.1:10055"
#: ``ros2_nero_inference.yaml`` ``umi_replay_http.server_url``. The replay
#: service speaks the same ``xyz_rpy`` wire and also requires
#: ``replay_proprioception_json`` on ``/submit``.
VLA_REPLAY_URL = "http://127.0.0.1:10054"
VLA_SUBMIT_TIMEOUT_S = 10.0
VLA_POLL_INTERVAL_S = 0.05
VLA_POLL_TIMEOUT_S = 30.0

# In-chunk tracking-abort thresholds (2026-09-17 design conversation).
TRACKING_ABORT_POS_M = 0.03
TRACKING_ABORT_ROT_DEG = 20.0
# Hybrid policy cadence: how often the planner sees execution progress.
CHECKPOINT_INTERVAL_S = 5.0
# Cap on one delegate_skill segment before the planner regains control.
MAX_SKILL_SECONDS = 60.0

CHUNK_STEPS = 20
# Per arm xyz (3) + rpy (3) + gripper (1), two arms.
ACTION_DIM_VLA = 14
#: The checkpoint (PaliGemma-224) expects square 224x224 images; the
#: client resizes each camera frame before packing (verified live 2026-09-18:
#: 224 answers 200, native 480x640 fails inside the model). The resize matches
#: the ROS inference client: aspect-preserving bilinear, leftover rows padded
#: on top (640x480 -> 224x168 content, 56 black rows above).
VLA_IMAGE_SIZE = 224
#: The service's slot semantics when the checkpoint config declares no
# image_features (verified in serve_rlt_inference._image_keys_from_request):
# image0=left, image1=right, image2=chest. NOT alphabetical.
VLA_CAMERA_SLOTS: tuple[str, ...] = ("left_rgbd", "right_rgbd", "chest_rgbd")
# The only action_format this client can decode; the live service emits it.
ACTION_FORMAT = "xyz_rpy"

# VlaPolicy defaults (plan 0084): the nero camera triple, the 20-dim EE state
# key those cameras' checkpoint was trained against, and its control rate.
#: Open-loop chunk playback rate. The RLT stage2 config's action_freq is
#: 10 Hz; the base VLA checkpoints and ``ros2_nero_inference.yaml`` are 30 Hz.
#: ``schedule=tracked`` follows the yaml, so those runs pass ``control_hz=30``.
VLA_CONTROL_HZ = 10.0
#: ``ros2_nero_inference.yaml``: model steps at 30 Hz, a new chunk every 5 Hz,
#: held for 300 ms (preview) then blended for 300 ms (transition). The
#: controller looks ``lookahead_ms`` ahead; that field is 150.
VLA_ACTION_HZ = 30.0
#: ``ros2_nero_inference.yaml`` ``state_history_stride``. At ``VLA_ACTION_HZ``
#: the older of the two last-action frames is this many action steps back
#: (3 / 30 Hz = 100 ms). The model server treats the pair as already sampled.
VLA_STATE_HISTORY_STRIDE = 3
VLA_INFERENCE_HZ = 5.0
VLA_PREVIEW_S = 0.3
VLA_TRANSITION_S = 0.3
VLA_LOOKAHEAD_S = 0.15
#: The hybrid's declared rate: its analytic move_to segments are tuned at the
#: embodiment's default pace, and its delegated VLA chunks inherit it.
CONTROL_HZ = 30.0
SUBMIT_IMAGES = ("left_rgbd", "right_rgbd", "chest_rgbd")
STATE_KEY = "eef_state"
