import json
import urllib.request

from inspect_robots_pai.primitives.clients import sam


def test_unreachable_service_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SAM3_SERVICE_URL", "http://127.0.0.1:9")
    assert sam(b"img", "cup", tmp_path) == []
    assert (tmp_path / "client_errors.jsonl").is_file()


def test_non_list_json_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SAM3_SERVICE_URL", "http://example.test/sam")

    class _Resp:
        def read(self) -> bytes:
            return json.dumps({"masks": []}).encode()

        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=2: _Resp())
    assert sam(b"img", "cup", tmp_path) == []
    assert (tmp_path / "client_errors.jsonl").is_file()
