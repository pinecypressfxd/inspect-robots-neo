"""Constructor and spaces behavior for the nero embodiment."""

from __future__ import annotations

import math

import pytest

from inspect_robots.conformance import assert_embodiment_conformant
from inspect_robots_nero import nero_embodiment
from inspect_robots_nero._config import ACTION_DIM, DIM_LABELS


def test_action_space_shape_and_semantics() -> None:
    embodiment = nero_embodiment()
    box = embodiment.info.action_space
    assert box.shape == (ACTION_DIM,)
    semantics = box.semantics
    assert semantics is not None
    assert semantics.control_mode == "eef_abs_pose"
    assert semantics.rotation_repr == "rot6d"
    assert semantics.gripper == "continuous"
    assert semantics.frame == "base"
    assert semantics.dim_labels == DIM_LABELS
    assert semantics.max_step is not None and len(semantics.max_step) == ACTION_DIM


def test_action_bounds_are_finite_and_ordered() -> None:
    box = nero_embodiment().info.action_space
    assert box.low is not None and box.high is not None
    assert all(math.isfinite(v) for v in box.low.ravel().tolist())
    assert all(math.isfinite(v) for v in box.high.ravel().tolist())
    assert bool((box.low <= box.high).all())


def test_observation_space_declares_agent_required_fields() -> None:
    space = nero_embodiment().info.observation_space
    by_key = {field.key: field for field in space.state.fields}
    assert by_key["eef_state"].shape == (ACTION_DIM,)
    assert by_key["joint_pos"].shape == (16,)
    cameras = {camera.name: camera for camera in space.cameras}
    assert set(cameras) == {"left_rgbd", "right_rgbd", "chest_rgbd"}
    for camera in cameras.values():
        assert (camera.height, camera.width) == (480, 640)


def test_conformance_passes_declaratively() -> None:
    assert_embodiment_conformant(nero_embodiment().info)


def test_capabilities_declare_self_paced_and_resettable() -> None:
    capabilities = nero_embodiment().info.capabilities
    assert "self_paced" in capabilities and "resettable" in capabilities


def test_docs_are_offered() -> None:
    docs = nero_embodiment().info.docs
    assert docs is not None and "rot6d" in docs


def test_invalid_control_hz_lists_fix_hint() -> None:
    with pytest.raises(ValueError, match="-E control_hz"):
        nero_embodiment(control_hz=0)


def test_invalid_camera_name_lists_valid_names() -> None:
    with pytest.raises(ValueError, match="left_rgbd"):
        nero_embodiment(cameras={"wrist": "/dev/video0"})


def test_invalid_camera_override_device() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        nero_embodiment(cameras={"left_rgbd": ""})


def test_invalid_workspace_bounds() -> None:
    with pytest.raises(ValueError, match="workspace"):
        nero_embodiment(workspace_high=(1.0, 1.0))  # wrong length
