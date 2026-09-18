"""Tunable defaults for the :10055 VLA wire client and the hybrid policy.

Every value here is a constructor default, overridable per run with ``-P k=v``
through the policy factories (plan 0084). Wire facts (field names, CHW layout,
``xyz_rpy`` delta actions) are verified against the live service and the
``serve_rlt_inference`` source; see ``plans/0084-vla-hybrid-policy.md``.
"""

from __future__ import annotations

VLA_BASE_URL = "http://127.0.0.1:10055"
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
#: 224 answers 200, native 480x640 fails inside the model).
VLA_IMAGE_SIZE = 224
#: The service's slot semantics when the checkpoint config declares no
# image_features (verified in serve_rlt_inference._image_keys_from_request):
# image0=left, image1=right, image2=chest. NOT alphabetical.
VLA_CAMERA_SLOTS: tuple[str, ...] = ("left_rgbd", "right_rgbd", "chest_rgbd")
# The only action_format this client can decode; the live service emits it.
ACTION_FORMAT = "xyz_rpy"

# VlaPolicy defaults (plan 0084): the nero camera triple, the 20-dim EE state
# key those cameras' checkpoint was trained against, and its control rate.
CONTROL_HZ = 30.0
SUBMIT_IMAGES = ("left_rgbd", "right_rgbd", "chest_rgbd")
STATE_KEY = "eef_state"
