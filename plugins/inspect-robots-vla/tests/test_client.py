"""Protocol tests for the :10055 VLA wire client against an in-test fake service.

The fake mirrors the live capture in ``plans/0084-vla-hybrid-policy.md``:
POST /submit with an NPZ body (image{N} CHW uint8, state float32, task,
request_id), GET /result/latest?after_request_id=N answering 204 until an NPZ
result {request_id, actions (20,14), action_format, status} is ready.
"""

from __future__ import annotations

import dataclasses
import io
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast

import httpx
import numpy as np
import pytest

from inspect_robots_vla._client import VlaClient, VlaServiceError


def _npz_bytes(**fields: Any) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **fields)
    return buffer.getvalue()


def _result_bytes(
    request_id: int = 7,
    actions: np.ndarray | None = None,
    *,
    action_format: str = "xyz_rpy",
    status: str = "ok",
) -> bytes:
    chunk = np.zeros((20, 14), dtype=np.float32) if actions is None else actions
    return _npz_bytes(
        request_id=np.int64(request_id),
        actions=chunk,
        action_format=np.array(action_format),
        status=np.array(status),
    )


def _hwc(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(4, 8, 3), dtype=np.uint8)


class _FakeVlaServer(ThreadingHTTPServer):
    """Scriptable stand-in for serve_rlt_inference on the :10055 wire."""

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _FakeVlaHandler)
        self.base_url = f"http://127.0.0.1:{self.server_address[1]}"
        self.received: list[dict[str, np.ndarray]] = []
        self.content_types: list[str] = []
        self.get_paths: list[str] = []
        # Each GET pops the front: None -> 204, bytes -> 200 with that body.
        # An exhausted script keeps answering 204 (nothing new yet).
        self.results: list[bytes | None] = []
        self.disconnect_on_submit = False
        self.submit_status = 200

    def make_client(self, **kwargs: Any) -> VlaClient:
        options: dict[str, Any] = {"poll_interval_s": 0.001, "poll_timeout_s": 0.2}
        options.update(kwargs)
        return VlaClient(self.base_url, **options)


class _FakeVlaHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        server = cast(_FakeVlaServer, self.server)
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if server.disconnect_on_submit:
            # The live service closes the socket on payloads it cannot decode.
            self.close_connection = True
            return
        server.content_types.append(self.headers.get("Content-Type", ""))
        with np.load(io.BytesIO(body)) as npz:
            server.received.append({name: np.asarray(npz[name]) for name in npz.files})
        self.send_response(server.submit_status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        server = cast(_FakeVlaServer, self.server)
        server.get_paths.append(self.path)
        body = server.results.pop(0) if server.results else None
        if body is None:
            self.send_response(204)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-npz")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


@pytest.fixture()
def fake_vla() -> Iterator[_FakeVlaServer]:
    server = _FakeVlaServer()
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)


def test_submit_converts_hwc_to_chw_and_passes_fields(fake_vla: _FakeVlaServer) -> None:
    images = {"left_rgbd": _hwc(1), "right_rgbd": _hwc(2), "chest_rgbd": _hwc(3)}
    state = np.linspace(-1.0, 1.0, 20, dtype=np.float32)
    client = fake_vla.make_client()
    client.submit(images, state, "pick up the cup", request_id=5)
    client.close()

    (payload,) = fake_vla.received
    assert set(payload) == {"image0", "image1", "image2", "state", "task", "request_id"}
    # Sorted key order: chest_rgbd < left_rgbd < right_rgbd.
    for index, _key in enumerate(("chest_rgbd", "left_rgbd", "right_rgbd")):
        # Nearest-resized to the checkpoint's 224 square, then CHW.
        assert payload[f"image{index}"].shape == (3, 224, 224)
        assert payload[f"image{index}"].dtype == np.uint8
        assert payload[f"image{index}"].dtype == np.uint8
    np.testing.assert_array_equal(payload["state"], state)
    assert payload["state"].dtype == np.float32
    assert str(payload["task"].item()) == "pick up the cup"
    assert int(payload["request_id"].item()) == 5
    assert fake_vla.content_types == ["application/x-npz"]


def test_submit_without_images_omits_image_fields(fake_vla: _FakeVlaServer) -> None:
    client = fake_vla.make_client()
    client.submit(None, np.zeros(20, dtype=np.float32), "task", request_id=1)
    client.close()

    (payload,) = fake_vla.received
    assert set(payload) == {"state", "task", "request_id"}


@pytest.mark.parametrize("images", [{}, {"left": _hwc(1)}])
def test_submit_rejects_out_of_range_image_counts(
    fake_vla: _FakeVlaServer, images: dict[str, np.ndarray]
) -> None:
    client = fake_vla.make_client()
    with pytest.raises(VlaServiceError, match=r"expected 2\.\.3 images"):
        client.submit(images, np.zeros(20, dtype=np.float32), "task", request_id=1)
    client.close()
    assert fake_vla.received == []


def test_submit_rejects_non_hwc_uint8_frames(fake_vla: _FakeVlaServer) -> None:
    images = {"left": _hwc(1), "right": np.zeros((4, 8, 3), dtype=np.float32)}
    client = fake_vla.make_client()
    with pytest.raises(VlaServiceError, match="HWC uint8"):
        client.submit(images, np.zeros(20, dtype=np.float32), "task", request_id=1)
    client.close()
    assert fake_vla.received == []


def test_submit_disconnect_raises_with_base_url_and_remedy(
    fake_vla: _FakeVlaServer,
) -> None:
    fake_vla.disconnect_on_submit = True
    client = fake_vla.make_client()
    with pytest.raises(VlaServiceError, match=fake_vla.base_url) as info:
        client.submit(None, np.zeros(20, dtype=np.float32), "task", request_id=1)
    client.close()
    assert "remedy" in str(info.value)


def test_poll_decodes_latest_result(fake_vla: _FakeVlaServer) -> None:
    actions = np.arange(20 * 14, dtype=np.float32).reshape(20, 14) / 7.0
    fake_vla.results = [_result_bytes(request_id=7, actions=actions)]
    client = fake_vla.make_client()
    chunk = client.poll(0)
    client.close()

    assert fake_vla.get_paths == ["/result/latest?after_request_id=0"]
    assert chunk is not None
    assert chunk.request_id == 7
    np.testing.assert_array_equal(chunk.deltas, actions)
    assert chunk.deltas.shape == (20, 14)
    assert chunk.deltas.dtype == np.float32
    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.request_id = 8  # type: ignore[misc]


@pytest.mark.parametrize("body", [None, b""])
def test_poll_returns_none_until_a_result_exists(
    fake_vla: _FakeVlaServer, body: bytes | None
) -> None:
    fake_vla.results = [body, _result_bytes(request_id=3)]
    client = fake_vla.make_client()
    assert client.poll(2) is None
    assert client.poll(2) is not None
    client.close()


def test_infer_submits_then_blocks_until_this_request(fake_vla: _FakeVlaServer) -> None:
    images = {"chest_rgbd": _hwc(3), "left_rgbd": _hwc(1), "right_rgbd": _hwc(2)}
    fake_vla.results = [
        None,  # nothing yet
        _result_bytes(request_id=4),  # stale result for a previous request
        _result_bytes(request_id=5),
    ]
    client = fake_vla.make_client()
    chunk = client.infer(images, np.zeros(20, dtype=np.float32), "pour", request_id=5)
    client.close()

    assert chunk.request_id == 5
    assert chunk.deltas.shape == (20, 14)
    (payload,) = fake_vla.received
    assert int(payload["request_id"].item()) == 5
    assert str(payload["task"].item()) == "pour"


def test_infer_times_out_when_no_result_arrives(fake_vla: _FakeVlaServer) -> None:
    client = fake_vla.make_client(poll_timeout_s=0.1)
    with pytest.raises(VlaServiceError, match="produced no result for request 9"):
        client.infer(None, np.zeros(20, dtype=np.float32), "task", request_id=9)
    client.close()


def test_poll_error_status_raises(fake_vla: _FakeVlaServer) -> None:
    fake_vla.results = [_result_bytes(request_id=2, status="inference_failed")]
    client = fake_vla.make_client()
    with pytest.raises(VlaServiceError, match="inference_failed"):
        client.poll(1)
    client.close()


def test_poll_wrong_action_format_raises(fake_vla: _FakeVlaServer) -> None:
    fake_vla.results = [_result_bytes(request_id=2, action_format="joint_position")]
    client = fake_vla.make_client()
    with pytest.raises(VlaServiceError, match="action_format"):
        client.poll(1)
    client.close()


def test_poll_wrong_action_width_raises(fake_vla: _FakeVlaServer) -> None:
    fake_vla.results = [_result_bytes(request_id=2, actions=np.zeros((20, 7), np.float32))]
    client = fake_vla.make_client()
    with pytest.raises(VlaServiceError, match="expected \\(m, 14\\)"):
        client.poll(1)
    client.close()


def test_poll_garbage_body_raises(fake_vla: _FakeVlaServer) -> None:
    fake_vla.results = [b"\x00garbage-not-an-npz"]
    client = fake_vla.make_client()
    with pytest.raises(VlaServiceError, match="undecodable"):
        client.poll(1)
    client.close()


def test_poll_bare_npy_body_raises_vla_service_error() -> None:
    # A 200 body holding a bare .npy (not NPZ) makes np.load return an ndarray,
    # whose context-manager use raises TypeError; that must surface as a
    # VlaServiceError, not leak the raw TypeError.
    buffer = io.BytesIO()
    np.save(buffer, np.zeros((20, 14), dtype=np.float32))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=buffer.getvalue())

    client = VlaClient(
        "http://127.0.0.1:10055",
        transport=httpx.MockTransport(cast(Callable[[httpx.Request], httpx.Response], handler)),
    )
    with pytest.raises(VlaServiceError, match="undecodable"):
        client.poll(0)
    client.close()


def test_transport_is_injectable_and_bad_statuses_raise() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(503, text="overloaded")

    client = VlaClient(
        "http://127.0.0.1:10055",
        transport=httpx.MockTransport(cast(Callable[[httpx.Request], httpx.Response], handler)),
    )
    with pytest.raises(VlaServiceError, match="HTTP 503"):
        client.submit(None, np.zeros(20, dtype=np.float32), "task", request_id=1)
    with pytest.raises(VlaServiceError, match="HTTP 503"):
        client.poll(0)
    client.close()
    assert calls == ["/submit", "/result/latest"]
