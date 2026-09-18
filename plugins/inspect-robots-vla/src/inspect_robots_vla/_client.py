"""Blocking submit/poll client for the live :10055 VLA inference wire.

The peer is ``serve_rlt_inference`` (the user-modified RLT stage2 process);
the wire facts are verified against the live service and its source (plan
0084, ``plans/0084-vla-hybrid-policy.md``):

- ``POST /submit`` with an NPZ body: ``image{N}`` CHW uint8 RGB (sorted
  camera-key order, 2..3 frames), ``state`` float32, ``task`` str,
  ``request_id``. The service closes the connection on payloads it cannot
  decode, so any transport failure is a wire failure here.
- ``GET /result/latest?after_request_id=N`` answers 204 until a result newer
  than ``N`` exists, then an NPZ with ``request_id`` int64, ``actions``
  float32 (m, 14) delta EE pose (gripper absolute), ``action_format``, and
  ``status``.

Every failure mode (disconnect, bad HTTP status, non-ok status, format or
shape mismatch, undecodable body, poll timeout) funnels into
[VlaServiceError][inspect_robots_vla._client.VlaServiceError] carrying the
service base URL and a remedy, never a raw httpx/zipfile exception.
"""

from __future__ import annotations

import io
import time
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

from ._config import (
    ACTION_DIM_VLA,
    ACTION_FORMAT,
    VLA_POLL_INTERVAL_S,
    VLA_POLL_TIMEOUT_S,
    VLA_SUBMIT_TIMEOUT_S,
)

_NPZ_CONTENT_TYPE = "application/x-npz"
_MIN_IMAGES = 2
_MAX_IMAGES = 3
_REMEDY = (
    "remedy: check that serve_rlt_inference is listening on the expected port "
    "and that the submitted NPZ matches its checkpoint config (image count, "
    "state dim); the service drops the connection on payloads it cannot decode"
)


class VlaServiceError(RuntimeError):
    """A VLA service call failed: transport, HTTP status, or wire format."""


@dataclass(frozen=True)
class VlaChunk:
    """One decoded action chunk: per-step delta EE poses, gripper absolute.

    ``deltas`` is float32 ``(m, 14)``: per arm xyz (3) + rpy (3) + gripper (1),
    two arms, exactly as the service emitted it. Re-anchoring to absolute
    targets is the anchor module's job, not the client's.
    """

    request_id: int
    deltas: np.ndarray


def _encode_payload(
    images: Mapping[str, np.ndarray] | None,
    state: np.ndarray,
    task: str,
    request_id: int,
) -> bytes:
    """Build the /submit NPZ body, converting HWC uint8 frames to CHW.

    Image fields are written ``image0..imageN`` in sorted camera-key order;
    a caller-side encoding mistake raises before any bytes hit the wire.
    """
    fields: dict[str, Any] = {
        "state": np.asarray(state, dtype=np.float32),
        "task": np.array(task),
        "request_id": np.int64(request_id),
    }
    if images is not None:
        if not _MIN_IMAGES <= len(images) <= _MAX_IMAGES:
            raise VlaServiceError(
                f"invalid /submit payload: expected 2..3 images, got {len(images)} "
                f"({sorted(images) if images else 'none'})"
            )
        for index, key in enumerate(sorted(images)):
            frame = np.asarray(images[key])
            if frame.ndim != 3 or frame.dtype != np.uint8:
                raise VlaServiceError(
                    f"invalid /submit payload: image {key!r} must be HWC uint8 RGB, "
                    f"got shape {frame.shape} dtype {frame.dtype}"
                )
            fields[f"image{index}"] = frame.transpose(2, 0, 1)
    buffer = io.BytesIO()
    np.savez(buffer, **fields)
    return buffer.getvalue()


def _decode_chunk(base_url: str, content: bytes) -> VlaChunk:
    """Parse one /result/latest NPZ body into a VlaChunk, or explain why not."""
    try:
        with np.load(io.BytesIO(content)) as npz:
            request_id = int(npz["request_id"].item())
            actions = np.asarray(npz["actions"], dtype=np.float32)
            action_format = str(npz["action_format"].item())
            status = str(npz["status"].item())
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        raise VlaServiceError(
            f"VLA service at {base_url} returned an undecodable /result/latest "
            f"body ({exc!r}); {_REMEDY}"
        ) from exc
    if status != "ok":
        raise VlaServiceError(
            f"VLA service at {base_url} reported status {status!r} instead of "
            f"'ok' for request {request_id}"
        )
    if action_format != ACTION_FORMAT:
        raise VlaServiceError(
            f"VLA service at {base_url} returned action_format {action_format!r}; "
            f"this client only decodes {ACTION_FORMAT!r}"
        )
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM_VLA:
        raise VlaServiceError(
            f"VLA service at {base_url} returned actions shaped {actions.shape}; "
            f"expected (m, {ACTION_DIM_VLA})"
        )
    return VlaChunk(request_id=request_id, deltas=actions)


class VlaClient:
    """Blocking submit/poll client for one VLA inference service.

    One in-flight request at a time (the rollout loop is synchronous); pass
    ``transport`` to inject an httpx transport for fakes.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = VLA_SUBMIT_TIMEOUT_S,
        poll_interval_s: float = VLA_POLL_INTERVAL_S,
        poll_timeout_s: float = VLA_POLL_TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._poll_interval_s = poll_interval_s
        self._poll_timeout_s = poll_timeout_s
        self._http = httpx.Client(base_url=self._base_url, timeout=timeout_s, transport=transport)

    def submit(
        self,
        images: Mapping[str, np.ndarray] | None,
        state: np.ndarray,
        task: str,
        *,
        request_id: int,
    ) -> None:
        """POST one observation to /submit; the result arrives via poll()."""
        body = _encode_payload(images, state, task, request_id)
        try:
            response = self._http.post(
                "/submit", content=body, headers={"Content-Type": _NPZ_CONTENT_TYPE}
            )
        except httpx.TransportError as exc:
            raise VlaServiceError(
                f"VLA service at {self._base_url} dropped the /submit connection "
                f"({exc!r}); {_REMEDY}"
            ) from exc
        if not 200 <= response.status_code < 300:
            raise VlaServiceError(
                f"VLA service at {self._base_url} answered /submit with HTTP "
                f"{response.status_code}; {_REMEDY}"
            )

    def poll(self, after_request_id: int) -> VlaChunk | None:
        """GET /result/latest once; None while no newer result exists."""
        try:
            response = self._http.get(
                "/result/latest", params={"after_request_id": after_request_id}
            )
        except httpx.TransportError as exc:
            raise VlaServiceError(
                f"VLA service at {self._base_url} dropped the /result/latest "
                f"connection ({exc!r}); {_REMEDY}"
            ) from exc
        if response.status_code == 204 or not response.content:
            return None
        if not 200 <= response.status_code < 300:
            raise VlaServiceError(
                f"VLA service at {self._base_url} answered /result/latest with "
                f"HTTP {response.status_code}; {_REMEDY}"
            )
        return _decode_chunk(self._base_url, response.content)

    def infer(
        self,
        images: Mapping[str, np.ndarray] | None,
        state: np.ndarray,
        task: str,
        *,
        request_id: int,
    ) -> VlaChunk:
        """Submit, then poll until this request's result (or the deadline).

        Results older than ``request_id`` (a straggler from a previous
        request) are skipped; exceeding ``poll_timeout_s`` raises
        VlaServiceError.
        """
        self.submit(images, state, task, request_id=request_id)
        deadline = time.monotonic() + self._poll_timeout_s
        while True:
            chunk = self.poll(request_id)
            if chunk is not None and chunk.request_id >= request_id:
                return chunk
            if time.monotonic() >= deadline:
                raise VlaServiceError(
                    f"VLA service at {self._base_url} produced no result for "
                    f"request {request_id} within {self._poll_timeout_s}s; remedy: "
                    "check the service is not stuck on an earlier inference and "
                    "raise VLA_POLL_TIMEOUT_S if chunks are legitimately slower"
                )
            time.sleep(self._poll_interval_s)

    def close(self) -> None:
        """Release the underlying HTTP connections."""
        self._http.close()
