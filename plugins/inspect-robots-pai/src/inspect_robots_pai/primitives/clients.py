"""HTTP clients for perception and action services; failures return empty lists."""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path
from typing import Any


def _append_error(trial_dir: Path, name: str, error: str) -> None:
    path = Path(trial_dir) / "client_errors.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"name": name, "error": error}) + "\n")


def _post(env_name: str, payload: dict[str, Any], trial_dir: Path) -> list[Any]:
    url = os.environ.get(env_name)
    if not url:
        _append_error(trial_dir, env_name, f"missing environment variable {env_name}")
        return []
    try:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            raw = resp.read()
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            _append_error(trial_dir, env_name, "response is not a JSON list")
            return []
        return parsed
    except Exception as exc:
        _append_error(trial_dir, env_name, str(exc))
        return []


def sam(image: bytes, prompt: str, trial_dir: Path) -> list[Any]:
    payload = {
        "image": base64.b64encode(image).decode("ascii"),
        "prompt": prompt,
    }
    return _post("SAM3_SERVICE_URL", payload, trial_dir)


def molmo(image: bytes, prompt: str, trial_dir: Path) -> list[Any]:
    payload = {
        "image": base64.b64encode(image).decode("ascii"),
        "prompt": prompt,
    }
    return _post("MOLMO_SERVICE_URL", payload, trial_dir)


def grasp(image: bytes, trial_dir: Path) -> list[Any]:
    payload = {"image": base64.b64encode(image).decode("ascii")}
    return _post("GRASPNET_SERVICE_URL", payload, trial_dir)


def vla(observation: dict[str, Any], trial_dir: Path) -> list[Any]:
    return _post("VLA_SERVICE_URL", observation, trial_dir)
