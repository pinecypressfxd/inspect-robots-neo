"""D405 color stream over OpenCV V4L2, one keep-latest daemon thread per camera.

The bring-up reads the D405 color stream as a plain V4L2 device (YUYV), so
this module needs no vendor SDK. ``read()`` hands out the newest frame and
rejects stale ones so the observation is never silently old.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from typing import Protocol, cast

import numpy as np


class _Capture(Protocol):
    """Frame source for the grab loop: ``read`` grabs one (ok, bgr) frame.

    Injected captures only ever need ``read``; ``release`` is called solely
    on captures this camera opened itself (see ``stop``).
    """

    def read(self) -> tuple[bool, np.ndarray]:
        """Grab one frame; ``(False, <empty>)`` when the device has none."""
        ...

    def release(self) -> None:
        """Free the device; only meaningful for camera-opened captures."""
        ...


class D405Camera:
    """One camera: background grab loop plus staleness-checked reads."""

    def __init__(
        self,
        name: str,
        device: str,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        pixel_format: str = "YUYV",
        capture: object | None = None,
        clock: Callable[[], float] = time.monotonic,
        poll_s: float = 0.005,
        first_frame_timeout_s: float = 3.0,
    ) -> None:
        if not math.isfinite(first_frame_timeout_s) or first_frame_timeout_s <= 0:
            raise ValueError(
                f"first_frame_timeout_s must be positive and finite, got {first_frame_timeout_s!r}"
            )
        self.name = name
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.pixel_format = pixel_format
        self._injected = capture
        self._capture: object | None = capture
        self._clock = clock
        self._poll_s = poll_s
        self._first_frame_timeout_s = float(first_frame_timeout_s)
        self._lock = threading.Lock()
        self._latest: tuple[np.ndarray, float] | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def _open(self) -> object:
        import cv2

        capture = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not capture.isOpened():
            raise RuntimeError(
                f"could not open camera {self.name!r} at {self.device}; check the by-path node "
                "and pass -E cameras=<name>=<device> to override it"
            )
        # VideoWriter_fourcc is runtime-present on cv2 4.x/5.x but absent from
        # the 5.0 stubs, hence the targeted ignore.
        capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*self.pixel_format),  # type: ignore[attr-defined]
        )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        return capture

    def start(self) -> None:
        """Open the device (unless injected) and start the grab thread."""
        if self._running:
            return
        import cv2  # noqa: F401 - warm the import so the first stamp precedes the caller's read()

        if self._capture is None:
            self._capture = self._open()
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name=f"nero-camera-{self.name}", daemon=True
        )
        self._thread.start()

    def _loop(self) -> None:
        import cv2

        capture = self._capture
        assert capture is not None  # start() always assigns before launching the thread
        source = cast(_Capture, capture)
        while self._running:
            ok, frame = source.read()
            if not ok:
                time.sleep(self._poll_s)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._latest = (rgb, self._clock())

    def read(self, *, max_age_s: float) -> tuple[np.ndarray, float]:
        """Return (rgb frame, stamp); TimeoutError when absent or stale.

        The wait for the FIRST frame uses the separate, larger
        ``first_frame_timeout_s`` budget (USB warm-up on the D405 color node
        takes ~0.7 s, longer than a sane freshness budget); once frames have
        arrived, ``max_age_s`` bounds their staleness.
        """
        with self._lock:
            latest = self._latest
        if latest is None:
            deadline = self._clock() + self._first_frame_timeout_s
            while True:
                with self._lock:
                    latest = self._latest
                if latest is not None:
                    break
                if not self._running or self._clock() >= deadline:
                    raise TimeoutError(
                        f"camera {self.name!r} produced no frames within "
                        f"{self._first_frame_timeout_s:g}s; check the device node and cabling"
                    )
                time.sleep(self._poll_s)
        frame, stamp = latest
        age = self._clock() - stamp
        if age > max_age_s:
            raise TimeoutError(
                f"camera {self.name!r} frame is stale: {age:.2f}s old, "
                f"exceeding max_age_s={max_age_s:g}"
            )
        return frame, stamp

    def stop(self) -> None:
        """Stop the grab thread; release only captures this object opened itself."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._injected is None and self._capture is not None:
            cast(_Capture, self._capture).release()
        self._capture = None
        self._latest = None
