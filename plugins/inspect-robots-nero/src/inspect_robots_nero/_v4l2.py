"""Native V4L2 mmap capture for the D405 color nodes.

Ported from the bring-up checkout's ``v4l2_mmap_capture.py``: OpenCV's V4L2
backend (5.0.0) freezes all but the first camera when several devices stream
concurrently on this rig and can segfault on reopen, while the hand-rolled
ioctl/mmap path runs six cameras at once. Only the streaming core is kept:
capability check, strict YUYV format negotiation, mmap buffers, select-based
reads, and YUYV-to-BGR conversion (cv2 is used for the conversion only).
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import fcntl
import mmap
import os
import select
from dataclasses import dataclass
from typing import Any

import numpy as np

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1
V4L2_FIELD_NONE = 1

V4L2_CAP_VIDEO_CAPTURE = 0x00000001
V4L2_CAP_STREAMING = 0x04000000
V4L2_CAP_DEVICE_CAPS = 0x80000000

V4L2_BUF_FLAG_ERROR = 0x00000040

V4L2_PIX_FMT_YUYV = ord("Y") | (ord("U") << 8) | (ord("Y") << 16) | (ord("V") << 24)

_PIXEL_FORMATS = {"YUYV": V4L2_PIX_FMT_YUYV}


def fourcc_to_str(value: int) -> str:
    """Render a V4L2 fourcc integer as its four characters."""
    return "".join(chr((int(value) >> shift) & 0xFF) for shift in (0, 8, 16, 24))


class TimeVal(ctypes.Structure):
    """C ``struct timeval`` as the V4L2 ABI defines it."""

    _fields_ = [
        ("tv_sec", ctypes.c_long),
        ("tv_usec", ctypes.c_long),
    ]


class V4L2Capability(ctypes.Structure):
    """C ``struct v4l2_capability``."""

    _fields_ = [
        ("driver", ctypes.c_uint8 * 16),
        ("card", ctypes.c_uint8 * 32),
        ("bus_info", ctypes.c_uint8 * 32),
        ("version", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("device_caps", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]


class V4L2PixFormat(ctypes.Structure):
    """C ``struct v4l2_pix_format``."""

    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("pixelformat", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("bytesperline", ctypes.c_uint32),
        ("sizeimage", ctypes.c_uint32),
        ("colorspace", ctypes.c_uint32),
        ("priv", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("ycbcr_enc", ctypes.c_uint32),
        ("quantization", ctypes.c_uint32),
        ("xfer_func", ctypes.c_uint32),
    ]


class _V4L2FormatUnion(ctypes.Union):
    _fields_ = [  # noqa: RUF012 - the ctypes idiom for union layouts
        ("pix", V4L2PixFormat),
        ("raw_data", ctypes.c_uint8 * 200),
    ]


class V4L2Format(ctypes.Structure):
    """C ``struct v4l2_format``."""

    _fields_ = [
        ("type", ctypes.c_uint32),
        ("_padding", ctypes.c_uint32),
        ("fmt", _V4L2FormatUnion),
    ]


class V4L2RequestBuffers(ctypes.Structure):
    """C ``struct v4l2_requestbuffers``."""

    _fields_ = [
        ("count", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class V4L2TimeCode(ctypes.Structure):
    """C ``struct v4l2_timecode``."""

    _fields_ = [
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("frames", ctypes.c_uint8),
        ("seconds", ctypes.c_uint8),
        ("minutes", ctypes.c_uint8),
        ("hours", ctypes.c_uint8),
        ("userbits", ctypes.c_uint8 * 4),
    ]


class _V4L2BufferUnion(ctypes.Union):
    _fields_ = [  # noqa: RUF012 - the ctypes idiom for union layouts
        ("offset", ctypes.c_uint32),
        ("userptr", ctypes.c_ulong),
        ("planes", ctypes.c_void_p),
        ("fd", ctypes.c_int32),
    ]


class _V4L2BufferRequestUnion(ctypes.Union):
    _fields_ = [  # noqa: RUF012 - the ctypes idiom for union layouts
        ("request_fd", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


class V4L2Buffer(ctypes.Structure):
    """C ``struct v4l2_buffer``."""

    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("bytesused", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("timestamp", TimeVal),
        ("timecode", V4L2TimeCode),
        ("sequence", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("m", _V4L2BufferUnion),
        ("length", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
        ("request", _V4L2BufferRequestUnion),
    ]


class V4L2Fract(ctypes.Structure):
    """C ``struct v4l2_fract``."""

    _fields_ = [
        ("numerator", ctypes.c_uint32),
        ("denominator", ctypes.c_uint32),
    ]


class V4L2CaptureParm(ctypes.Structure):
    """C ``struct v4l2_captureparm``."""

    _fields_ = [
        ("capability", ctypes.c_uint32),
        ("capturemode", ctypes.c_uint32),
        ("timeperframe", V4L2Fract),
        ("extendedmode", ctypes.c_uint32),
        ("readbuffers", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
    ]


class _V4L2StreamParmUnion(ctypes.Union):
    _fields_ = [  # noqa: RUF012 - the ctypes idiom for union layouts
        ("capture", V4L2CaptureParm),
        ("raw_data", ctypes.c_uint8 * 200),
    ]


class V4L2StreamParm(ctypes.Structure):
    """C ``struct v4l2_streamparm``."""

    _fields_ = [
        ("type", ctypes.c_uint32),
        ("parm", _V4L2StreamParmUnion),
    ]


def _ioc(direction: int, type_: str, nr: int, size: int) -> int:
    return (direction << 30) | (ord(type_) << 8) | nr | (size << 16)


def _ior(type_: str, nr: int, data_type: type[ctypes.Structure]) -> int:
    return _ioc(2, type_, nr, ctypes.sizeof(data_type))


def _iow(
    type_: str,
    nr: int,
    data_type: type[ctypes.Structure] | type[ctypes._SimpleCData[Any]],
) -> int:
    return _ioc(1, type_, nr, ctypes.sizeof(data_type))


def _iowr(type_: str, nr: int, data_type: type[ctypes.Structure]) -> int:
    return _ioc(3, type_, nr, ctypes.sizeof(data_type))


VIDIOC_QUERYCAP = _ior("V", 0, V4L2Capability)
VIDIOC_S_FMT = _iowr("V", 5, V4L2Format)
VIDIOC_REQBUFS = _iowr("V", 8, V4L2RequestBuffers)
VIDIOC_QUERYBUF = _iowr("V", 9, V4L2Buffer)
VIDIOC_QBUF = _iowr("V", 15, V4L2Buffer)
VIDIOC_DQBUF = _iowr("V", 17, V4L2Buffer)
VIDIOC_STREAMON = _iow("V", 18, ctypes.c_int)
VIDIOC_STREAMOFF = _iow("V", 19, ctypes.c_int)
VIDIOC_S_PARM = _iowr("V", 22, V4L2StreamParm)


def _ioctl(fd: int, request: int, arg: ctypes.Structure | ctypes._SimpleCData[Any]) -> None:
    try:
        fcntl.ioctl(fd, request, arg, True)
    except OSError as exc:
        errno_value = exc.errno if exc.errno is not None else 0
        raise OSError(errno_value, os.strerror(errno_value)) from exc


@dataclass(slots=True)
class _MappedBuffer:
    memory: mmap.mmap
    length: int


class V4L2MmapCapture:
    """One V4L2 device: mmap streaming with ``read() -> (ok, bgr)`` semantics.

    ``read()`` blocks up to ``timeout_s`` and returns ``(False, empty)`` on
    timeout or spurious empty buffers, mirroring what the camera grab loop
    expects from a capture source. Format negotiation is strict: a device
    that cannot deliver the exact requested format fails at ``start()``.
    """

    def __init__(
        self,
        device: str,
        *,
        width: int,
        height: int,
        fps: int,
        pixel_format: str = "YUYV",
        buffer_count: int = 4,
    ) -> None:
        if pixel_format not in _PIXEL_FORMATS:
            known = ", ".join(sorted(_PIXEL_FORMATS))
            raise ValueError(f"unsupported pixel format {pixel_format!r}; known: {known}")
        self._device = device
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._pixelformat = _PIXEL_FORMATS[pixel_format]
        self._buffer_count = int(buffer_count)
        self._fd: int | None = None
        self._buffers: list[_MappedBuffer] = []
        self._streaming = False
        self._bytesperline = width * 2

    def start(self) -> None:
        """Open the device, negotiate the format, map and queue buffers, stream."""
        self._fd = os.open(self._device, os.O_RDWR | os.O_NONBLOCK)
        try:
            self._configure_device()
            self._map_buffers()
            self._queue_all_buffers()
            self._stream_on()
        except Exception:
            self.release()
            raise

    def release(self) -> None:
        """Stream off, unmap buffers, and close the device; safe to call twice."""
        if self._fd is not None and self._streaming:
            buf_type = ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
            with contextlib.suppress(OSError):
                _ioctl(self._fd, VIDIOC_STREAMOFF, buf_type)
            self._streaming = False
        for mapped in self._buffers:
            mapped.memory.close()
        self._buffers.clear()
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def read(self, *, timeout_s: float = 1.0) -> tuple[bool, np.ndarray]:
        """Grab one BGR frame; ``(False, empty)`` on timeout or empty buffer."""
        if self._fd is None:
            raise RuntimeError("V4L2 capture has not been started")
        readable, _, _ = select.select([self._fd], [], [], timeout_s)
        if not readable:
            return False, np.zeros((0,))
        buf = V4L2Buffer()
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = V4L2_MEMORY_MMAP
        try:
            _ioctl(self._fd, VIDIOC_DQBUF, buf)
        except OSError as exc:
            if exc.errno == errno.EAGAIN:
                return False, np.zeros((0,))
            raise
        mapped = self._buffers[buf.index]
        try:
            if buf.flags & V4L2_BUF_FLAG_ERROR:
                if buf.bytesused == 0 and buf.sequence == 0:
                    # Some UVC devices emit empty error sentinels at stream start.
                    return False, np.zeros((0,))
                raise RuntimeError(
                    f"V4L2 driver error buffer from {self._device}: "
                    f"sequence={int(buf.sequence)}, flags=0x{int(buf.flags):08x}"
                )
            if buf.bytesused <= 0:
                return False, np.zeros((0,))
            # mmap slicing copies the payload, so the driver can reuse this
            # buffer while conversion runs after the requeue.
            payload = mapped.memory[: buf.bytesused]
        finally:
            _ioctl(self._fd, VIDIOC_QBUF, buf)
        return True, _yuyv_to_bgr(payload, self._width, self._height, self._bytesperline)

    def _configure_device(self) -> None:
        assert self._fd is not None
        caps = V4L2Capability()
        _ioctl(self._fd, VIDIOC_QUERYCAP, caps)
        effective_caps = (
            caps.device_caps if caps.capabilities & V4L2_CAP_DEVICE_CAPS else caps.capabilities
        )
        if not effective_caps & V4L2_CAP_VIDEO_CAPTURE:
            raise RuntimeError(f"{self._device} is not a V4L2 capture device")
        if not effective_caps & V4L2_CAP_STREAMING:
            raise RuntimeError(f"{self._device} does not support V4L2 streaming I/O")

        fmt = V4L2Format()
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        fmt.fmt.pix.width = self._width
        fmt.fmt.pix.height = self._height
        fmt.fmt.pix.pixelformat = self._pixelformat
        fmt.fmt.pix.field = V4L2_FIELD_NONE
        _ioctl(self._fd, VIDIOC_S_FMT, fmt)
        width = int(fmt.fmt.pix.width)
        height = int(fmt.fmt.pix.height)
        pixelformat = int(fmt.fmt.pix.pixelformat)
        if width != self._width or height != self._height or pixelformat != self._pixelformat:
            raise RuntimeError(
                f"V4L2 camera format mismatch for {self._device}: requested "
                f"{fourcc_to_str(self._pixelformat)} {self._width}x{self._height}, "
                f"actual {fourcc_to_str(pixelformat)} {width}x{height}"
            )
        self._bytesperline = int(fmt.fmt.pix.bytesperline) or self._width * 2

        parm = V4L2StreamParm()
        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        parm.parm.capture.timeperframe.numerator = 1
        parm.parm.capture.timeperframe.denominator = max(self._fps, 1)
        _ioctl(self._fd, VIDIOC_S_PARM, parm)

    def _map_buffers(self) -> None:
        assert self._fd is not None
        req = V4L2RequestBuffers()
        req.count = self._buffer_count
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        req.memory = V4L2_MEMORY_MMAP
        _ioctl(self._fd, VIDIOC_REQBUFS, req)
        if req.count < 2:
            raise RuntimeError(f"{self._device} returned too few V4L2 mmap buffers")
        for index in range(req.count):
            buf = V4L2Buffer()
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            buf.memory = V4L2_MEMORY_MMAP
            buf.index = index
            _ioctl(self._fd, VIDIOC_QUERYBUF, buf)
            memory = mmap.mmap(
                self._fd,
                buf.length,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
                offset=buf.m.offset,
            )
            self._buffers.append(_MappedBuffer(memory=memory, length=buf.length))

    def _queue_all_buffers(self) -> None:
        assert self._fd is not None
        for index in range(len(self._buffers)):
            buf = V4L2Buffer()
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            buf.memory = V4L2_MEMORY_MMAP
            buf.index = index
            _ioctl(self._fd, VIDIOC_QBUF, buf)

    def _stream_on(self) -> None:
        assert self._fd is not None
        buf_type = ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
        _ioctl(self._fd, VIDIOC_STREAMON, buf_type)
        self._streaming = True


def _yuyv_to_bgr(payload: bytes, width: int, height: int, bytesperline: int) -> np.ndarray:
    import cv2

    stride = bytesperline or width * 2
    expected = stride * height
    if len(payload) < expected:
        raise RuntimeError(
            f"V4L2 YUYV payload is shorter than expected ({len(payload)} < {expected})"
        )
    raw = np.frombuffer(payload[:expected], dtype=np.uint8)
    rows = raw.reshape((height, stride))[:, : width * 2]
    yuyv = rows.reshape((height, width, 2))
    return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
