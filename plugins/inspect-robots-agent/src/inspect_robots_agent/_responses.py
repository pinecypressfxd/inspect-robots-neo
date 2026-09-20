"""Stateless OpenAI Responses client with chat-history translation."""

from __future__ import annotations

import time
from typing import Any, cast

import httpx

from inspect_robots_agent._llm import AssistantMessage, Provider, ToolCall

from ._capture import WireCapture


class ResponsesClient:
    """Blocking Responses client with raw reasoning-item replay and bounded retries.

    Chat-format messages remain the policy transcript. This boundary translates
    them to Responses input items and preserves raw output items required to
    replay reasoning-backed function calls.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        backoff_s: float = 1.0,
        transport: httpx.BaseTransport | None = None,
        capture: WireCapture | None = None,
    ):
        self._provider = provider
        self._max_retries = max_retries
        self._backoff_s = backoff_s
        self._capture = capture
        self._raw_items_by_call_id: dict[str, list[dict[str, Any]]] = {}
        headers = {}
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"
        self._http = httpx.Client(
            base_url=provider.base_url,
            headers=headers,
            timeout=timeout_s,
            transport=transport,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float | None = None,
        reasoning_effort: str | float | None = None,
    ) -> AssistantMessage:
        """Return one assistant turn for the translated chat-format history."""
        history_call_ids = _history_call_ids(messages)
        self._raw_items_by_call_id = {
            call_id: items
            for call_id, items in self._raw_items_by_call_id.items()
            if call_id in history_call_ids
        }
        body: dict[str, Any] = {
            "model": self._provider.model,
            "input": _translate_messages(messages, self._raw_items_by_call_id),
            "tools": _translate_tools(tools),
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if temperature is not None:
            body["temperature"] = temperature
        if reasoning_effort is not None:
            body["reasoning"] = {"effort": reasoning_effort}

        last_error = "unknown error"
        for attempt in range(self._max_retries):
            t_start = time.time() if self._capture is not None else 0.0
            try:
                response = self._http.post("/responses", json=body)
            except httpx.TransportError as exc:
                if self._capture is not None:
                    self._capture.record(
                        attempt=attempt,
                        endpoint="/responses",
                        request=body,
                        status=None,
                        response_text=None,
                        error=str(exc),
                        t_start=t_start,
                        duration_s=time.time() - t_start,
                    )
                last_error = str(exc)
            else:
                if self._capture is not None:
                    self._capture.record(
                        attempt=attempt,
                        endpoint="/responses",
                        request=body,
                        status=response.status_code,
                        response_text=response.text,
                        error=None,
                        t_start=t_start,
                        duration_s=time.time() - t_start,
                    )
                if response.status_code == 200:
                    message, output = _parse_response(response.json())
                    for item in output:
                        if item.get("type") == "function_call":
                            self._raw_items_by_call_id[str(item["call_id"])] = output
                    return message
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                if response.status_code == 400 and "summary" in response.text:
                    # Proxied Responses endpoints flap on the function_call_output
                    # summary field: one request demands it ("Missing required
                    # parameter: 'input[N].summary'"), the next rejects it
                    # ("Unknown parameter: 'input[N].summary'"). Flip the field
                    # once and retry immediately instead of failing the trial.
                    if "Unknown parameter" in response.text and _has_summaries(body):
                        body = _strip_summaries(body)
                        continue
                    if "Missing required parameter" in response.text and not _has_summaries(body):
                        body = _add_summaries(body)
                        continue
                if response.status_code not in (429,) and response.status_code < 500:
                    raise RuntimeError(f"LLM request rejected — {last_error}")
            if attempt + 1 < self._max_retries:
                time.sleep(self._backoff_s * 2**attempt)
        raise RuntimeError(f"LLM request failed after {self._max_retries} attempts — {last_error}")

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._http.close()


def _has_summaries(body: dict[str, Any]) -> bool:
    """True when any input function_call_output item carries a summary."""
    return any(
        item.get("type") == "function_call_output" and "summary" in item
        for item in body.get("input", [])
        if isinstance(item, dict)
    )


def _strip_summaries(body: dict[str, Any]) -> dict[str, Any]:
    """Rebuild input with summary fields dropped from function_call_output items."""
    input_items = body.get("input", [])
    body = dict(body)
    body["input"] = [
        {key: value for key, value in item.items() if key != "summary"}
        if isinstance(item, dict) and item.get("type") == "function_call_output"
        else item
        for item in input_items
    ]
    return body


def _add_summaries(body: dict[str, Any]) -> dict[str, Any]:
    """Rebuild input adding summary=output where a function_call_output lacks one."""
    input_items = body.get("input", [])
    body = dict(body)
    rebuilt = []
    for item in input_items:
        if (
            isinstance(item, dict)
            and item.get("type") == "function_call_output"
            and "summary" not in item
        ):
            patched = dict(item)
            patched["summary"] = str(item.get("output", ""))
            rebuilt.append(patched)
        else:
            rebuilt.append(item)
    body["input"] = rebuilt
    return body


def _history_call_ids(messages: list[dict[str, Any]]) -> set[str]:
    """Collect every function call id still referenced by the submitted history."""
    call_ids: set[str] = set()
    for message in messages:
        for call in message.get("tool_calls") or []:
            call_ids.add(str(call["id"]))
        if message.get("role") == "tool" and message.get("tool_call_id") is not None:
            call_ids.add(str(message["tool_call_id"]))
    return call_ids


def _translate_messages(
    messages: list[dict[str, Any]],
    raw_items_by_call_id: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Translate canonical chat messages to stateless Responses input items."""
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "tool":
            output = message["content"]
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": output,
                    # Some proxied Responses endpoints validate a summary on
                    # function_call_output (400 "Missing required parameter:
                    # 'input[N].summary'"); official OpenAI tolerates the extra
                    # field, so it rides along unconditionally.
                    "summary": str(output),
                }
            )
            continue

        tool_calls = message.get("tool_calls") or []
        if role == "assistant" and tool_calls:
            cached = raw_items_by_call_id.get(str(tool_calls[0]["id"]))
            if cached is not None:
                items.extend(cached)
                continue

        content = message.get("content")
        if isinstance(content, list):
            items.append({"role": role, "content": _translate_content_parts(content)})
        elif role != "assistant" or content:
            items.append({"role": role, "content": content})

        if role == "assistant":
            for call in tool_calls:
                function = call["function"]
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": function["name"],
                        "arguments": function["arguments"],
                    }
                )
    return items


def _translate_content_parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate chat text and image parts to Responses input content parts."""
    translated: list[dict[str, Any]] = []
    for part in parts:
        if part["type"] == "text":
            translated.append({"type": "input_text", "text": part["text"]})
        elif part["type"] == "image_url":
            translated.append({"type": "input_image", "image_url": part["image_url"]["url"]})
    return translated


def _translate_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten chat function schemas and explicitly retain non-strict behavior."""
    translated: list[dict[str, Any]] = []
    for tool in tools:
        function = tool["function"]
        translated.append(
            {
                "type": "function",
                "name": function["name"],
                "description": function["description"],
                "parameters": function["parameters"],
                "strict": False,
            }
        )
    return translated


def _parse_response(payload: dict[str, Any]) -> tuple[AssistantMessage, list[dict[str, Any]]]:
    """Parse usable assistant output while retaining the exact raw item list."""
    if payload.get("status") == "failed":
        error = payload.get("error")
        message = error.get("message") if isinstance(error, dict) else None
        raise RuntimeError(f"LLM response failed — {message or 'unknown error'}")

    output = cast(list[dict[str, Any]], payload["output"])
    texts: list[str] = []
    calls: list[ToolCall] = []
    for item in output:
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    texts.append(str(part["text"]))
        elif item.get("type") == "function_call":
            calls.append(
                ToolCall(
                    id=str(item["call_id"]),
                    name=str(item["name"]),
                    arguments=str(item["arguments"]),
                )
            )
    return (
        AssistantMessage(content="".join(texts) if texts else None, tool_calls=tuple(calls)),
        output,
    )
