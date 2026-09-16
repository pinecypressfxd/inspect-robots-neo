"""Portable pieces of the native V4L2 mmap capture."""

from __future__ import annotations

import pytest

from inspect_robots_nero._v4l2 import (
    V4L2_PIX_FMT_YUYV,
    VIDIOC_DQBUF,
    VIDIOC_QUERYCAP,
    VIDIOC_REQBUFS,
    VIDIOC_S_FMT,
    V4L2MmapCapture,
    fourcc_to_str,
)


def test_fourcc_round_trip() -> None:
    assert fourcc_to_str(V4L2_PIX_FMT_YUYV) == "YUYV"


def test_ioctl_request_numbers_match_the_linux_abi() -> None:
    # Stable kernel ABI values; a struct layout mistake shifts these.
    assert VIDIOC_QUERYCAP == 0x80685600
    assert VIDIOC_S_FMT == 0xC0D05605
    assert VIDIOC_REQBUFS == 0xC0145608
    assert VIDIOC_DQBUF == 0xC0585611


def test_unsupported_pixel_format_is_rejected() -> None:
    with pytest.raises(ValueError, match="pixel format"):
        V4L2MmapCapture("/dev/null", width=640, height=480, fps=30, pixel_format="MJPG")


def test_read_before_start_raises() -> None:
    capture = V4L2MmapCapture("/dev/null", width=640, height=480, fps=30)
    with pytest.raises(RuntimeError, match="not been started"):
        capture.read()


def test_release_is_idempotent_before_start() -> None:
    capture = V4L2MmapCapture("/dev/null", width=640, height=480, fps=30)
    capture.release()
    capture.release()
