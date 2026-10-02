"""The LLM adapter: an OpenAI-compatible chat client behind the priority gate.

`POST {base_url}/chat/completions` with `stream: true` (server-sent events), optional `tools`;
it works unchanged against vLLM (the default from phase 2) and Ollama (now). Every request
first takes a slot from the `PriorityGate` (voice > proactive > background, reserved voice
slots). With `[llm] send_priority` the request carries vLLM's `priority` field (what the
OpenAI SDK sends as `extra_body`), from `[llm].priority` for its class.

Each request is recorded in `requests` (without the messages) for the admin endpoint, with its
time to first token. A server that cannot be reached or answers an error raises
`LlmUnavailable`.
"""

import json
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from assistant_brain.adapters.priority_gate import PriorityGate, RequestClass
from assistant_core.config import LlmConfig

REQUEST_LOG_SIZE = 500
CONNECT_TIMEOUT_S = 3.0
READ_TIMEOUT_S = 60.0
"""The longest silence between two streamed chunks before the server counts as gone."""


class LlmUnavailable(Exception):
    """The LLM server could not be reached, or it answered with an error."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str = ""
    """JSON text, as streamed by the server."""


@dataclass
class ChatResult:
    request_id: str
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    queued_ms: float = 0.0
    ttft_ms: float | None = None
    total_ms: float = 0.0


@dataclass
class TextDelta:
    text: str


class LlmClient:
    def __init__(self, config: LlmConfig, gate: PriorityGate) -> None:
        self.config = config
        self.gate = gate
        self.requests: deque[dict[str, Any]] = deque(maxlen=REQUEST_LOG_SIZE)
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(READ_TIMEOUT_S, connect=CONNECT_TIMEOUT_S)
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    def build_request(
        self,
        messages: list[dict[str, Any]],
        cls: RequestClass,
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"model": self.config.model, "messages": messages, "stream": True}
        if tools:
            body["tools"] = tools
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if self.config.reasoning_effort:
            body["reasoning_effort"] = self.config.reasoning_effort
        if self.config.send_priority:
            body["priority"] = getattr(self.config.priority, cls)
        return body

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        cls: RequestClass,
        *,
        result: ChatResult | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        label: str = "",
    ) -> AsyncIterator[TextDelta]:
        """Stream the reply's text; `result` (if given) is filled in as it goes.

        Raises `LlmUnavailable` if the server is down or answers an error.
        """
        result = result if result is not None else ChatResult(f"llm-{uuid.uuid4().hex[:10]}")
        body = self.build_request(messages, cls, tools=tools, max_tokens=max_tokens)
        entry: dict[str, Any] = {
            "id": result.request_id,
            "class": cls,
            "label": label,
            "model": body["model"],
            "fields": sorted(k for k in body if k != "messages"),
            "outcome": "queued",
        }
        if "priority" in body:
            entry["priority"] = body["priority"]
        self.requests.append(entry)
        queued_at = time.monotonic()
        async with self.gate.slot(result.request_id, cls):
            started = time.monotonic()
            result.queued_ms = (started - queued_at) * 1000
            entry["queued_ms"] = round(result.queued_ms, 1)
            entry["outcome"] = "running"
            try:
                async for delta in self._stream(body, result, started):
                    yield delta
            except LlmUnavailable as exc:
                entry["outcome"] = "error"
                entry["error"] = str(exc)
                raise
            except BaseException:
                entry["outcome"] = "cancelled"
                raise
            finally:
                result.total_ms = (time.monotonic() - started) * 1000
                entry["total_ms"] = round(result.total_ms, 1)
                if result.ttft_ms is not None:
                    entry["ttft_ms"] = round(result.ttft_ms, 1)
            entry["outcome"] = "ok"
            entry["finish_reason"] = result.finish_reason
            entry["chars"] = len(result.text)

    async def complete(
        self, messages: list[dict[str, Any]], cls: RequestClass, **kwargs: Any
    ) -> ChatResult:
        result = ChatResult(f"llm-{uuid.uuid4().hex[:10]}")
        async for _ in self.stream_chat(messages, cls, result=result, **kwargs):
            pass
        return result

    async def _stream(
        self, body: dict[str, Any], result: ChatResult, started: float
    ) -> AsyncIterator[TextDelta]:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        calls: dict[int, ToolCall] = {}
        try:
            async with self._http.stream("POST", url, json=body) as response:
                if response.status_code >= 400:
                    detail = (await response.aread()).decode(errors="replace")[:300]
                    raise LlmUnavailable(f"{url} answered {response.status_code}: {detail}")
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise LlmUnavailable(f"{url}: {chunk['error']}")
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if choice.get("finish_reason"):
                            result.finish_reason = choice["finish_reason"]
                        for call in delta.get("tool_calls") or []:
                            index = call.get("index", len(calls))
                            fn = call.get("function") or {}
                            known = calls.get(index)
                            if known is None:
                                known = calls[index] = ToolCall(call.get("id", ""), "")
                            known.name += fn.get("name") or ""
                            known.arguments += fn.get("arguments") or ""
                        text = delta.get("content")
                        if text:
                            if result.ttft_ms is None:
                                result.ttft_ms = (time.monotonic() - started) * 1000
                            result.text += text
                            yield TextDelta(text)
        except httpx.HTTPError as exc:
            raise LlmUnavailable(f"{url}: {type(exc).__name__}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LlmUnavailable(f"{url}: bad stream chunk: {exc}") from exc
        result.tool_calls = [calls[i] for i in sorted(calls)]
        if result.tool_calls and result.ttft_ms is None:
            result.ttft_ms = (time.monotonic() - started) * 1000
