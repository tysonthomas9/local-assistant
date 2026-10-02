"""The admin/debug endpoint: loopback-only HTTP with JSON bodies (`[net].admin_bind`, 8771).

    GET  /health            {"ok": true, "engine": ..., "sessions": n}
    GET  /sessions          the connected edges and their dialog state
    GET  /turns             the turn log (states with times, LLM queued/TTFT/total, outcome)
    GET  /llm/gate          the priority gate: in flight, waiting, its event log
    GET  /llm/requests      the LLM request log (fields sent, `priority`, timings, outcome)
    POST /llm/background    {"count": n, "prompt"?: str, "max_tokens"?: int}: start n real
                            background-class LLM requests (they queue behind voice turns)
    POST /say               {"device": id, "text": str} or {"device": id, "prompt": str}:
                            proactive speech (`speech.request`), queued while a turn runs

It binds only to a loopback address and answers only loopback peers. A small HTTP/1.1 server
on asyncio streams (one request per connection) keeps the brain free of a web framework.
"""

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable
from typing import Any

from assistant_brain.adapters.llm import LlmClient, LlmUnavailable
from assistant_brain.bus import EventBus
from assistant_brain.console import emit
from assistant_brain.sessions import SessionManager
from assistant_brain.turnlog import TurnLog
from assistant_contracts.events import SpeechRequest
from assistant_link.server import is_loopback

MAX_BODY = 64 * 1024
BACKGROUND_PROMPT = (
    "Write a detailed, multi-paragraph summary of the history of clocks and timekeeping."
)
BACKGROUND_MAX_TOKENS = 300


class HttpError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status


Route = Callable[[dict[str, Any]], Awaitable[Any]]

REASONS = {
    200: "OK",
    202: "Accepted",
    400: "Bad Request",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    413: "Payload Too Large",
    500: "Internal Server Error",
}


class AdminServer:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        engine_name: str,
        sessions: SessionManager,
        turns: TurnLog,
        bus: EventBus,
        llm: LlmClient | None,
    ) -> None:
        if not is_loopback(host):
            raise ValueError(f"the admin endpoint binds only to loopback, not {host!r}")
        self.host = host
        self.port = port
        self.engine_name = engine_name
        self.sessions = sessions
        self.turns = turns
        self.bus = bus
        self.llm = llm
        self.background: set[asyncio.Task[None]] = set()
        self._started = 0
        self._server: asyncio.Server | None = None
        self.routes: dict[tuple[str, str], Route] = {
            ("GET", "/health"): self.health,
            ("GET", "/sessions"): self.list_sessions,
            ("GET", "/turns"): self.list_turns,
            ("GET", "/llm/gate"): self.gate,
            ("GET", "/llm/requests"): self.llm_requests,
            ("POST", "/llm/background"): self.start_background,
            ("POST", "/say"): self.say,
        }

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._serve, self.host, self.port)
        if self.port == 0:
            self.port = self._server.sockets[0].getsockname()[1]

    async def close(self) -> None:
        for task in list(self.background):
            task.cancel()
        for task in list(self.background):
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    # ------------------------------------------------------------ HTTP

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        status, payload = 500, {"error": "internal"}
        try:
            peer = writer.get_extra_info("peername")
            if not peer or not is_loopback(str(peer[0])):
                raise HttpError(403, "loopback only")
            method, path, body = await self._read(reader)
            route = self.routes.get((method, path))
            if route is None:
                known = {p for _, p in self.routes}
                raise HttpError(405 if path in known else 404, f"{method} {path}")
            status, payload = 200, await route(body)
        except HttpError as exc:
            status, payload = exc.status, {"error": str(exc)}
        except (asyncio.IncompleteReadError, ConnectionError):
            writer.close()
            return
        except Exception as exc:
            status, payload = 500, {"error": f"{type(exc).__name__}: {exc}"}
        data = json.dumps(payload, default=str).encode()
        head = (
            f"HTTP/1.1 {status} {REASONS.get(status, 'Error')}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(data)}\r\n"
            "Connection: close\r\n\r\n"
        )
        with contextlib.suppress(ConnectionError):
            writer.write(head.encode() + data)
            await writer.drain()
        writer.close()

    async def _read(self, reader: asyncio.StreamReader) -> tuple[str, str, dict[str, Any]]:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 10)
        lines = head.decode("latin-1").split("\r\n")
        try:
            method, target, _ = lines[0].split(" ", 2)
        except ValueError as exc:
            raise HttpError(400, "bad request line") from exc
        headers = {
            k.strip().lower(): v.strip() for k, _, v in (line.partition(":") for line in lines[1:])
        }
        length = int(headers.get("content-length") or 0)
        if length > MAX_BODY:
            raise HttpError(413, "body too large")
        body: dict[str, Any] = {}
        if length:
            raw = await asyncio.wait_for(reader.readexactly(length), 10)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise HttpError(400, f"bad JSON: {exc}") from exc
            if not isinstance(parsed, dict):
                raise HttpError(400, "the body must be a JSON object")
            body = parsed
        return method.upper(), target.split("?", 1)[0], body

    # ------------------------------------------------------------ routes

    async def health(self, body: dict[str, Any]) -> Any:
        del body
        return {"ok": True, "engine": self.engine_name, "sessions": len(self.sessions.sessions)}

    async def list_sessions(self, body: dict[str, Any]) -> Any:
        del body
        return [s.as_dict() for s in self.sessions.sessions.values()]

    async def list_turns(self, body: dict[str, Any]) -> Any:
        del body
        return self.turns.as_list()

    def _llm(self) -> LlmClient:
        if self.llm is None:
            raise HttpError(404, f"the {self.engine_name} engine has no LLM")
        return self.llm

    async def gate(self, body: dict[str, Any]) -> Any:
        del body
        gate = self._llm().gate
        return {**gate.snapshot(), "log": [e.as_dict() for e in gate.log]}

    async def llm_requests(self, body: dict[str, Any]) -> Any:
        del body
        return list(self._llm().requests)

    async def start_background(self, body: dict[str, Any]) -> Any:
        llm = self._llm()
        count = body.get("count", 1)
        prompt = body.get("prompt", BACKGROUND_PROMPT)
        max_tokens = body.get("max_tokens", BACKGROUND_MAX_TOKENS)
        if not isinstance(count, int) or not 1 <= count <= 32:
            raise HttpError(400, "count must be an integer from 1 to 32")
        if not isinstance(prompt, str) or not isinstance(max_tokens, int):
            raise HttpError(400, "prompt must be text and max_tokens an integer")
        ids: list[str] = []
        for _ in range(count):
            self._started += 1
            label = f"background-{self._started}"
            task = asyncio.create_task(self._background(llm, prompt, max_tokens, label))
            self.background.add(task)
            task.add_done_callback(self.background.discard)
            ids.append(label)
            await asyncio.sleep(0)  # queue them in order
        emit("BACKGROUND", count=count)
        return {"started": count, "labels": ids}

    async def _background(self, llm: LlmClient, prompt: str, max_tokens: int, label: str) -> None:
        messages = [{"role": "user", "content": prompt}]
        try:
            result = await llm.complete(messages, "background", max_tokens=max_tokens, label=label)
        except LlmUnavailable as exc:
            emit("BACKGROUND", {"error": str(exc)}, label=label, outcome="error")
            return
        emit("BACKGROUND", label=label, outcome="ok", id=result.request_id, chars=len(result.text))

    async def say(self, body: dict[str, Any]) -> Any:
        device, text, prompt = body.get("device"), body.get("text"), body.get("prompt")
        if not isinstance(device, str) or not device:
            raise HttpError(400, "device is required")
        if not (isinstance(text, str) and text) and not (isinstance(prompt, str) and prompt):
            raise HttpError(400, "text or prompt is required")
        session = self.sessions.sessions.get(device)
        if session is None:
            raise HttpError(404, f"device {device!r} is not connected")
        request = SpeechRequest(
            assistant_id=session.assistant.id,
            text=text if isinstance(text, str) and text else None,
            prompt=prompt if isinstance(prompt, str) and prompt else None,
            mode="verbatim" if isinstance(text, str) and text else "llm",
            target=f"device:{device}",
        )
        await self.bus.publish(request)
        return {"ok": True, "state": session.dialog.state if session.dialog else None}
