"""Hardware constants for the nero embodiment, carried from the working bring-up.

Values mirror ``neo_manipulation/evaluation/robot_itl_evaluation/config/
ros2_nero_inference.yaml`` (RLtoken branch); every value is a constructor
default, overridable per run with ``-E`` arguments.
"""

from __future__ import annotations

CAN_CHANNELS: dict[str, str] = {"left": "can_left", "right": "can_right"}
FIRMWARE_VERSION = "v112"
SPEED_PERCENT = 20
GRIPPER_FORCE = 0.3
GRIPPER_WIDTH_MIN_M = 0.0
GRIPPER_WIDTH_MAX_M = 0.09
GRIPPER_INIT_WIDTH_M = 0.09

HOME_LEFT: tuple[float, ...] = (0.56, 0.92, -1.38, 1.90, -0.46, 0.04, 0.20)
HOME_RIGHT: tuple[float, ...] = (-0.56, 0.92, 1.38, 1.90, 0.46, -0.04, 0.20)

CONTROL_HZ = 30.0
RESET_SETTLE_TOL_RAD = 0.05
RESET_SETTLE_TIMEOUT_S = 10.0
CAMERA_MAX_AGE_S = 0.5

# Firmware per-joint limits (rad), verbatim from pyAgxArm's
# ROBOT_JOINT_LIMIT_PRESET_RAD["nero"]. The URDF mirrors these EXACTLY, so IK
# can converge with commands sitting on the firmware fault edge; commanding
# past it makes the firmware drop the arm (disable = gravity fall). Every
# commanded joint value is clamped into these limits shrunk by the margin.
FIRMWARE_JOINT_LIMITS: tuple[tuple[float, float], ...] = (
    (-2.705261, 2.705261),
    (-1.745330, 1.745330),
    (-2.757621, 2.757621),
    (-1.012291, 2.146755),
    (-2.757621, 2.757621),
    (-0.733039, 0.959932),
    (-1.570797, 1.570797),
)
JOINT_LIMIT_SAFETY_MARGIN_RAD = 0.05
#: This rig has no brakes: disable() de-energizes the joints and the arms
#: FALL under gravity. Automatic close/abort paths therefore leave the arms
#: ENABLED and holding by default; disabling is the operator's explicit call
#: (mission console E-STOP or arm_tools.py disable --yes).
DISABLE_ON_CLOSE = False

ACTION_DIM = 20
ROT6D_BOUNDS = (-1.0, 1.0)
# Per-control-tick safety rate limits in native units: 1 cm position, rot6d
# component, and 1 cm gripper width. The agent policy scales these down
# further with max_speed_frac.
DEFAULT_MAX_STEP: tuple[float | None, ...] = (
    (0.01,) * 3 + (0.05,) * 6 + (0.01,) + (0.01,) * 3 + (0.05,) * 6 + (0.01,)
)

DIM_LABELS: tuple[str, ...] = (
    "left_x",
    "left_y",
    "left_z",
    "left_r1",
    "left_r2",
    "left_r3",
    "left_r4",
    "left_r5",
    "left_r6",
    "left_gripper",
    "right_x",
    "right_y",
    "right_z",
    "right_r1",
    "right_r2",
    "right_r3",
    "right_r4",
    "right_r5",
    "right_r6",
    "right_gripper",
)

# Provenance: ros2_nero_inference.yaml "camera.streams" (live wiring). Each
# entry is the V4L2 by-path color node of a RealSense D405; the neo stack
# reads these as V4L2 mmap devices (YUYV), not via pyrealsense2.
CAMERA_DEFAULTS: dict[str, dict[str, object]] = {
    "left_rgbd": {
        "device": "/dev/v4l/by-path/pci-0000:80:14.0-usb-0:11.2:1.0-video-index4",
        "width": 640,
        "height": 480,
        "fps": 30,
        "pixel_format": "YUYV",
    },
    "right_rgbd": {
        "device": "/dev/v4l/by-path/pci-0000:80:14.0-usb-0:2.2:1.0-video-index4",
        "width": 640,
        "height": 480,
        "fps": 30,
        "pixel_format": "YUYV",
    },
    "chest_rgbd": {
        "device": "/dev/v4l/by-path/pci-0000:00:0d.0-usb-0:2.2:1.0-video-index4",
        "width": 640,
        "height": 480,
        "fps": 30,
        "pixel_format": "YUYV",
    },
}

DUAL_NERO_DOCS = """Dual Nero arms bolted to a shared shelf; actions address
both arms in one 20-dim vector: [left xyz (m, base frame), left rot6d,
left gripper width (m), right xyz, right rot6d, right gripper width].
rot6d is the first two columns of the target rotation matrix (6 numbers, each in [-1, 1]).
Gripper width 0 means closed, 0.09 means open.
The state field eef_state mirrors the action layout from current forward kinematics.
The state field joint_pos is [left j1..j7, right j1..j7, left width, right width] (16 dims).
Base frame: shelf base at the left arm's root; +x forward, +z up.
Collision geometry: the gripper assembly extends 0.03 m BELOW the TCP that
the xyz targets describe; the hand's lowest point is 3 cm under the commanded
pose, and the rear of the assembly protrudes as well. A horizontal move that
passes this lowest point through an obstacle (basket or box rims, table edge)
snags the arm. Motion pattern near any container: move in axis-separated
segments only: raise z FIRST, then translate in xy at constant z, then lower
z LAST once centered over the opening. Never combine forward xy motion with
descending z, and never translate horizontally at a height where the point
0.03 m below the TCP would cross a rim; before any xy translation near a
container, target z at least 0.05 m above the highest rim you can see.
Calibrated working band (operator-measured): the standard start height
already sits about 0.05 m above the box rim, and the 3 cm protrusion clears
there -- do NOT lift far above the box before reaching over it, and descend
at most 0.05 m below that start height (the lowest safe TCP is at rim
height); work inside this band instead of climbing high above it.
Carrying: after closing on an object, rise only about 0.05 m above the
highest obstacle on the way to the target, then translate directly (a
diagonal path is fine) -- do not climb to a high transport altitude first.
Reachability feedback: a straight-down gripper orientation is reachable only
over part of the workspace. When a motion does not change the state at all,
the commanded pose was unreachable and the arms held; reposition closer or
change the approach orientation instead of repeating the same target."""
