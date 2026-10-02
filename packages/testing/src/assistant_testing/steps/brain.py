"""Brain steps: the real brain (`python -m assistant_brain`) with the real turn engine.

The brain is the EdgeLink server here: its process is named `server` (like the link server
console), so the link and edge steps work against it unchanged (`start_edge_agent`,
`start_link_client`, `client_receives`, `kill_process: {process: server}`,
`restart_process`, `client_reconnects_within`, ...). What the brain did is read from its event
lines (`assistant_brain.console`) and from its loopback admin endpoint: the turn log, the LLM
priority gate and the LLM request log.

LLM features use the real `reachy-gemma4` on the stack's own LLM server (`scripts/llm_server.sh`
on GPU0 only: vLLM by default, Ollama with `[llm] server = "ollama"`; the system Ollama service
is never used). `llm: stack` (the default) uses the one serving on 127.0.0.1:8773 (the gate
starts it there; it must be the configured kind) or else starts one of the scenario's own on a
free port; `llm: test` uses a separate one the scenario starts and may kill
(`start_llm_server`, either kind). Either way the model must sit entirely on GPU0
(`assistant_testing.llm_server`), else the step fails. A scenario's own server frees GPU0
first (an Ollama unloads the model, a vLLM sleeps); the stack's vLLM is woken again when a later
scenario needs it.
"""

import asyncio
import contextlib
import json
import math
import os
import signal
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Literal

from assistant_testing import edge_host, llm_server
from assistant_testing.features.context import ScenarioContext
from assistant_testing.features.registry import step
from assistant_testing.processes import ManagedProcess
from assistant_testing.steps import edge_host as edge_host_steps
from assistant_testing.steps.edge import (
    _SAMPLER,
    SAMPLE_PERIOD_S,
    _remember_start,
    _robot_state,
    _rotation,
    check_sampling,
)
from assistant_testing.steps.link import (
    DEV_TOKEN,
    SERVER,
    _client_name,
    _expect,
    _free_port,
    _get_lines,
    _Line,
    _Link,
)

STACK_LLM = llm_server.STACK_URL
"""The stack's LLM server (the gate starts it; `scripts/llm_server.sh`)."""
LLM_MODEL = llm_server.MODEL
LLM_PROCESS = "llm"
STACK_LLM_PROCESS = "llm-stack"


def _server(ctx: ScenarioContext, server: str | None) -> llm_server.ServerKind:
    """`server`, else the configured `[llm] server` (profile ci, ASSISTANT__LLM__SERVER)."""
    if server is None:
        return llm_server.configured_server(ctx.repo_root)
    if server not in ("vllm", "ollama"):
        raise AssertionError(f"unknown LLM server {server!r} (vllm or ollama)")
    return server


async def _start_llm(
    ctx: ScenarioContext, name: str, server: llm_server.ServerKind, within_s: float | None
) -> str:
    """A real LLM server of the scenario's own (`scripts/llm_server.sh --server <server>`) on
    a free port; up, with `reachy-gemma4` loaded entirely on GPU0."""
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    script = str(ctx.repo_root / llm_server.SCRIPT)
    await ctx.processes.start(name, [script, "--server", server, "--port", str(port)])
    await _wait_llm_up(url, within_s or llm_server.START_TIMEOUT_S[server])
    print(await asyncio.to_thread(llm_server.load_and_check, url, LLM_MODEL, server))
    return url


async def ensure_llm_server(ctx: ScenarioContext) -> str:
    """The LLM server of this scenario: the stack's on 127.0.0.1:8773 if it serves (it must be
    the configured kind; a sleeping vLLM is woken), else the scenario's own (stopped in
    teardown); `reachy-gemma4` checked to be entirely on GPU0."""
    llm = ctx.state.get("llm")
    if llm is not None:
        return llm["url"]
    server = _server(ctx, None)
    if await asyncio.to_thread(llm_server.up, STACK_LLM):
        url, owned = STACK_LLM, False
        print(await asyncio.to_thread(llm_server.load_and_check, url, LLM_MODEL, server))
    else:
        url, owned = await _start_llm(ctx, STACK_LLM_PROCESS, server, None), True
    ctx.state["llm"] = {"url": url, "owned": owned, "server": server}
    return url


def _brain(ctx: ScenarioContext) -> dict[str, Any]:
    brain = ctx.state.get("brain")
    if brain is None:
        raise AssertionError("no brain started; use start_brain first")
    return brain


def _http(method: str, url: str, body: Any = None, timeout_s: float = 10.0) -> Any:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise AssertionError(f"{method} {url}: {exc.code} {exc.read().decode()[:300]}") from exc


async def admin(ctx: ScenarioContext, path: str, body: Any = None) -> Any:
    """GET (or POST `body` to) the brain's admin endpoint."""
    url = f"http://127.0.0.1:{_brain(ctx)['admin_port']}{path}"
    return await asyncio.to_thread(_http, "GET" if body is None else "POST", url, body)


# ---------------------------------------------------------------- the brain


@step("start_brain")
async def start_brain(
    ctx: ScenarioContext,
    engine: Literal["echo", "basic"] = "echo",
    llm: Literal["stack", "test"] = "stack",
    follow_up_s: float | None = None,
    speech: bool = False,
    set: dict[str, Any] | None = None,
) -> None:
    """Start the real brain on free loopback ports (EdgeLink and admin), profile `ci`.

    `engine: echo` answers with the input (no models); `engine: basic` asks the real LLM:
    the stack's LLM server (`llm: stack`, see `ensure_llm_server`) or the scenario's own that it
    may kill (`llm: test`, `start_llm_server`
    first). `speech: true` (basic engine) transcribes voice turns and speaks the replies with
    the scenario's speech server (`speech_server_running` or `start_speech_server` first);
    without it replies are speak text. `set` overrides config keys
    (`{"llm.max_concurrency": 3}`).
    """
    port, admin_port = _free_port(), _free_port()
    ctx.state["link"] = _Link(port=port, token=DEV_TOKEN)
    ctx.state["brain"] = {"admin_port": admin_port, "engine": engine}
    overrides = dict(set or {})
    if follow_up_s is not None:
        overrides["brain.follow_up_s"] = follow_up_s
    if engine == "basic":
        if llm == "stack":
            base = await ensure_llm_server(ctx)
            kind = ctx.state["llm"]["server"]
        else:
            base, kind = _test_llm(ctx)["url"], _test_llm(ctx)["server"]
        overrides.setdefault("llm.base_url", f"{base}/v1")
        overrides.setdefault("llm.model", LLM_MODEL)
        overrides.setdefault("llm.server", kind)
        ctx.state["brain"]["llm"] = {"url": base, "server": kind}
        overrides.setdefault("engine.speech", speech)
        if speech:
            from assistant_testing.steps.speech import speech_url

            url = speech_url(ctx)
            overrides.setdefault("stt.base_url", url)
            overrides.setdefault("tts.base_url", url)
    elif speech:
        raise AssertionError("speech needs the basic engine")
    argv = [sys.executable, "-m", "assistant_brain", "--config-dir", "config", "--profile", "ci"]
    argv += ["--engine", engine, "--host", "127.0.0.1", "--port", str(port)]
    argv += ["--admin-port", str(admin_port), "--token", DEV_TOKEN]
    for key, value in overrides.items():
        argv += ["--set", f"{key}={json.dumps(value)}"]
    await ctx.processes.start(
        SERVER, argv, env={"PYTHONUNBUFFERED": "1"}, ready_line=r"^LISTENING ", ready_timeout=30
    )


@step("brain_output_clean")
async def brain_output_clean(ctx: ScenarioContext) -> None:
    """The brain printed no traceback, handler failure or BRAIN-ERROR so far."""
    bad = [
        line
        for line in ctx.processes.get(SERVER).lines
        if "Traceback" in line
        or "handler failed" in line
        or line.startswith("BRAIN-ERROR")
        or '"level": "error"' in line
    ]
    assert not bad, "brain output has errors:\n" + "\n".join(bad)


# ---------------------------------------------------------------- what the edge saw


@step("edge_shows_reply")
async def edge_shows_reply(
    ctx: ScenarioContext,
    client: str,
    text: str | None = None,
    containing: str | None = None,
    within_s: float = 60.0,
) -> None:
    """The edge shows the next reply text (`SAY`, from `speak.begin.text`): exactly `text`,
    or containing `containing` (case-insensitive); any non-empty reply if neither is given."""
    line = await _expect(ctx, _client_name(client), {"SAY"}, within_s, what="a reply (SAY)")
    reply = str((line.payload or {}).get("text") or "")
    print(f"edge {client} shows reply: {reply!r}")
    assert reply.strip(), "the reply is empty"
    if text is not None:
        assert reply == text, f"reply {reply!r}, want {text!r}"
    if containing is not None:
        assert containing.lower() in reply.lower(), f"reply {reply!r} lacks {containing!r}"


def _attention_lines(proc: ManagedProcess) -> list[_Line]:
    return [line for line in _get_lines(proc, "RECV") if line.fields.get("type") == "attention"]


@step("attention_sequence_is")
async def attention_sequence_is(
    ctx: ScenarioContext, client: str, states: list[str], within_s: float = 30.0
) -> None:
    """The edge receives exactly these `attention` states, in this order, next (each is
    consumed; another state in between fails)."""
    name = _client_name(client)
    proc = ctx.processes.get(name)
    first = last = -1
    for state in states:
        line = await _expect(
            ctx,
            name,
            {"RECV"},
            within_s,
            fields={"type": "attention"},
            payload={"state": state},
            what=f"attention {state}",
        )
        first = line.index if first < 0 else first
        last = line.index
    got = [
        (line.payload or {}).get("state")
        for line in _attention_lines(proc)
        if first <= line.index <= last
    ]
    print(f"edge {client} attention: {' -> '.join(map(str, got))}")
    assert got == states, f"attention states {got}, want exactly {states}"


@step("flood_finished")
async def flood_finished(ctx: ScenarioContext, client: str, within_s: float = 10.0) -> None:
    """The client's frame flood was sent to its end (FLOOD-DONE)."""
    await _expect(ctx, _client_name(client), {"FLOOD-DONE"}, within_s, what="FLOOD-DONE")


# ---------------------------------------------------------------- the turn log


@step("turn_logged")
async def turn_logged(
    ctx: ScenarioContext,
    kind: Literal["text", "voice", "proactive"] | None = None,
    outcome: Literal["finished", "interrupted", "error", "abandoned"] = "finished",
    reply: str | None = None,
    states: list[str] | None = None,
    ttft: bool = False,
    max_ttft_ms: float | None = None,
    error_contains: str | None = None,
    within_s: float = 60.0,
) -> None:
    """The brain's turn log (admin `/turns`) has the latest turn ended with `outcome`, of
    `kind`, with this `reply` and these turn `states` in order. `ttft`: the LLM's time to
    first token, queue time and total time are logged (TTFT > 0, at most `max_ttft_ms`)."""
    deadline = time.monotonic() + within_s
    turns: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        turns = await admin(ctx, "/turns")
        if turns and turns[-1].get("outcome") is not None:
            break
        await asyncio.sleep(0.2)
    assert turns, "the turn log is empty"
    turn = turns[-1]
    print(f"turn: {json.dumps(turn)}")
    assert turn["outcome"] == outcome, f"outcome {turn['outcome']!r}, want {outcome!r}"
    if kind is not None:
        assert turn["kind"] == kind, f"kind {turn['kind']!r}, want {kind!r}"
    if reply is not None:
        assert turn["reply_text"] == reply, f"reply {turn['reply_text']!r}, want {reply!r}"
    if states is not None:
        got = [s["state"] for s in turn["states"]]
        assert got == states, f"turn states {got}, want {states}"
    if error_contains is not None:
        assert error_contains in (turn.get("error") or ""), f"error {turn.get('error')!r}"
    if ttft:
        llm = turn.get("llm") or {}
        print(
            f"LLM: queued {llm.get('queued_ms')} ms, TTFT {llm.get('ttft_ms')} ms, "
            f"total {llm.get('total_ms')} ms"
        )
        assert (llm.get("ttft_ms") or 0) > 0, f"no time to first token logged: {llm}"
        assert llm.get("queued_ms") is not None, f"no queue time logged: {llm}"
        assert llm.get("total_ms") is not None, f"no total time logged: {llm}"
        assert llm["total_ms"] >= llm["ttft_ms"], llm
        if max_ttft_ms is not None:
            assert llm["ttft_ms"] <= max_ttft_ms, f"TTFT {llm['ttft_ms']} ms > {max_ttft_ms}"


@step("brain_says")
async def brain_says(
    ctx: ScenarioContext, client: str, text: str | None = None, prompt: str | None = None
) -> None:
    """Proactive speech through the admin endpoint (`POST /say`): spoken at once when the
    device is idle, queued while a turn runs."""
    body: dict[str, Any] = {"device": client}
    if text is not None:
        body["text"] = text
    if prompt is not None:
        body["prompt"] = prompt
    answer = await admin(ctx, "/say", body)
    print(f"say: {json.dumps(answer)}")


@step("speech_queued")
async def speech_queued(ctx: ScenarioContext, client: str, within_s: float = 5.0) -> None:
    """The brain queued proactive speech for `client` behind a running turn (SPEECH-QUEUED)."""
    line = await _expect(
        ctx, SERVER, {"SPEECH-QUEUED"}, within_s, fields={"device": client}, what="SPEECH-QUEUED"
    )
    print(line.text)
    assert line.fields.get("state") != "idle", f"queued while idle: {line.text}"


@step("brain_state_is")
async def brain_state_is(
    ctx: ScenarioContext, client: str, state: str, within_s: float = 30.0
) -> None:
    """The client's session (admin `/sessions`) is in turn `state` within `within_s`."""
    deadline = time.monotonic() + within_s
    got = None
    while time.monotonic() < deadline:
        sessions = await admin(ctx, "/sessions")
        got = next((s["state"] for s in sessions if s["device_id"] == client), None)
        if got == state:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"session of {client} is {got!r}, want {state!r}")


# ---------------------------------------------------------------- the LLM priority gate


@step("start_background_llm_requests")
async def start_background_llm_requests(
    ctx: ScenarioContext, count: int, max_tokens: int = 300
) -> None:
    """The brain starts `count` real background-class LLM requests (admin `/llm/background`)."""
    answer = await admin(ctx, "/llm/background", {"count": count, "max_tokens": max_tokens})
    assert answer.get("started") == count, answer


@step("llm_gate_state")
async def llm_gate_state(
    ctx: ScenarioContext, in_flight: int, waiting: int, within_s: float = 30.0
) -> None:
    """The priority gate (admin `/llm/gate`) has `in_flight` requests running and `waiting`
    queued, within `within_s`."""
    deadline = time.monotonic() + within_s
    gate: dict[str, Any] = {}
    while time.monotonic() < deadline:
        gate = await admin(ctx, "/llm/gate")
        queued = sum(len(ids) for ids in gate["waiting"].values())
        if gate["in_flight"] == in_flight and queued == waiting:
            print(f"gate: {json.dumps({k: v for k, v in gate.items() if k != 'log'})}")
            return
        await asyncio.sleep(0.1)
    snapshot = {k: v for k, v in gate.items() if k != "log"}
    raise AssertionError(f"gate {snapshot}, want in_flight {in_flight}, waiting {waiting}")


@step("llm_requests_done")
async def llm_requests_done(ctx: ScenarioContext, count: int, within_s: float = 300.0) -> None:
    """All `count` LLM requests in the request log finished ok (none still queued/running)."""
    deadline = time.monotonic() + within_s
    requests: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        requests = await admin(ctx, "/llm/requests")
        if len(requests) >= count and all(
            r["outcome"] not in ("queued", "running") for r in requests
        ):
            break
        await asyncio.sleep(0.5)
    outcomes = [r["outcome"] for r in requests]
    print(f"{len(requests)} LLM requests: {outcomes}")
    assert len(requests) == count, f"{len(requests)} requests, want {count}"
    assert all(o == "ok" for o in outcomes), f"not all requests ended ok: {requests}"


def _gate_log(gate: dict[str, Any]) -> list[dict[str, Any]]:
    return list(gate["log"])


@step("llm_gate_never_exceeded")
async def llm_gate_never_exceeded(ctx: ScenarioContext, max_in_flight: int) -> None:
    """From the gate's event log: never more than `max_in_flight` requests ran at once."""
    gate = await admin(ctx, "/llm/gate")
    peak = max((e["in_flight"] for e in _gate_log(gate)), default=0)
    print(f"gate peak in flight: {peak} (gate's own count {gate['max_in_flight_seen']})")
    assert peak <= max_in_flight, f"{peak} requests ran at once, more than {max_in_flight}"
    assert gate["max_in_flight_seen"] <= max_in_flight, gate["max_in_flight_seen"]


@step("voice_request_admitted_first")
async def voice_request_admitted_first(ctx: ScenarioContext) -> None:
    """From the gate's event log: the voice request was admitted before every background
    request that was already waiting when it arrived (and some were waiting)."""
    log = _gate_log(await admin(ctx, "/llm/gate"))
    for event in log:
        print(f"  gate {event['kind']:9} {event['cls']:10} {event['request_id']} "
              f"in_flight={event['in_flight']} waiting={event['waiting']}")  # fmt: skip
    voice = [e for e in log if e["cls"] == "voice"]
    assert voice, "no voice request went through the gate"
    arrived = voice[0]["t"]
    voice_id = voice[0]["request_id"]
    admitted_voice = next(
        e["t"] for e in log if e["request_id"] == voice_id and e["kind"] == "admitted"
    )
    admitted = {e["request_id"]: e["t"] for e in log if e["kind"] == "admitted"}
    waiting = [
        e["request_id"]
        for e in log
        if e["kind"] == "queued"
        and e["cls"] == "background"
        and e["t"] <= arrived
        and admitted.get(e["request_id"], math.inf) > arrived
    ]
    print(f"background requests waiting when the voice request arrived: {waiting}")
    assert waiting, "no background request was waiting: the scenario did not saturate the gate"
    early = [r for r in waiting if admitted.get(r, math.inf) < admitted_voice]
    assert not early, f"background requests admitted before the voice request: {early}"


@step("llm_request_log")
async def llm_request_log(
    ctx: ScenarioContext,
    priority: bool,
    classes: dict[str, int] | None = None,
) -> None:
    """Every request in the LLM request log carries vLLM's `priority` field if `priority`,
    and none does otherwise; `classes` gives the value each class must carry."""
    requests = await admin(ctx, "/llm/requests")
    assert requests, "the LLM request log is empty"
    for request in requests:
        sent = "priority" in request["fields"]
        print(f"  {request['id']} {request['class']}: fields {request['fields']}")
        assert sent is priority, f"{request['id']} priority sent={sent}, want {priority}"
        if priority and classes is not None and request["class"] in classes:
            want = classes[request["class"]]
            assert request.get("priority") == want, f"{request} priority, want {want}"


# ---------------------------------------------------------------- the LLM server itself


def _brain_llm(ctx: ScenarioContext) -> dict[str, Any]:
    llm = _brain(ctx).get("llm")
    if llm is None:
        raise AssertionError("the brain has no LLM server (start_brain with engine: basic)")
    return llm


def _metrics(url: str) -> dict[str, float]:
    """vLLM's Prometheus gauges of interest, summed over their labels."""
    with urllib.request.urlopen(f"{url}/metrics", timeout=5) as response:
        text = response.read().decode()
    found: dict[str, float] = {}
    for line in text.splitlines():
        for name in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
            if line.startswith(name + "{") or line.startswith(name + " "):
                found[name] = found.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
    return found


@step("llm_server_queue")
async def llm_server_queue(
    ctx: ScenarioContext, running: int, min_waiting: int, within_s: float = 60.0
) -> None:
    """The brain's LLM server (vLLM, its `/metrics`) runs exactly `running` requests and has at
    least `min_waiting` more waiting in its own queue."""
    llm = _brain_llm(ctx)
    assert llm["server"] == "vllm", f"the brain's LLM server is {llm['server']}, not vllm"
    deadline = time.monotonic() + within_s
    gauges: dict[str, float] = {}
    while time.monotonic() < deadline:
        gauges = await asyncio.to_thread(_metrics, llm["url"])
        now_running = gauges.get("vllm:num_requests_running", 0)
        now_waiting = gauges.get("vllm:num_requests_waiting", 0)
        if now_running == running and now_waiting >= min_waiting:
            print(f"vLLM at {llm['url']}: {now_running:.0f} running, {now_waiting:.0f} waiting")
            return
        await asyncio.sleep(0.1)
    raise AssertionError(
        f"vLLM gauges {gauges}, want {running} running and at least {min_waiting} waiting"
    )


@step("voice_overtook_background")
async def voice_overtook_background(
    ctx: ScenarioContext,
    at_least: int | None = None,
    at_most: int | None = None,
    first: bool = False,
    margin_s: float = 0.25,
) -> None:
    """From the LLM request log (the brain's clock): the background requests that were sent to
    the server BEFORE the voice request but got their first token more than `margin_s` AFTER
    it, i.e. the ones the voice request overtook inside the server. There are at least
    `at_least` / at most `at_most` of them. `first`: also none of the background requests
    still waiting in the server when the voice request was sent (no first token yet) got its
    first token more than `margin_s` before the voice request's: the voice turn took the first
    free slot."""
    requests = await admin(ctx, "/llm/requests")
    voice = [r for r in requests if r["class"] == "voice"]
    assert len(voice) == 1, f"want exactly one voice request, got {voice}"
    v = voice[0]
    assert v.get("first_token_s") is not None, f"the voice request has no first token: {v}"
    earlier = [r for r in requests if r["class"] == "background" and r["sent_s"] < v["sent_s"]]
    overtaken = [
        r for r in earlier if (r.get("first_token_s") or math.inf) > v["first_token_s"] + margin_s
    ]
    for r in sorted(requests, key=lambda r: r.get("first_token_s") or math.inf):
        token = r.get("first_token_s")
        print(f"  {r['label'] or r['class']:14} priority {r.get('priority', '-'):>2}  sent "
              f"{r['sent_s'] - v['sent_s']:+7.2f} s  first token "
              f"{'-' if token is None else f'{token - v["sent_s"]:+7.2f} s'}")  # fmt: skip
    print(
        f"the voice request (TTFT {v.get('ttft_ms')} ms) was sent after {len(earlier)} "
        f"background requests and overtook {len(overtaken)} of them"
    )
    waiting = [r for r in earlier if (r.get("first_token_s") or math.inf) > v["sent_s"]]
    ahead = [r["label"] for r in waiting if r["first_token_s"] < v["first_token_s"] - margin_s]
    print(f"{len(waiting)} were still waiting in the server; {len(ahead)} of them went first")
    if first:
        assert waiting, "no background request was waiting in the server: nothing to overtake"
        assert not ahead, f"background requests waiting in the server went first: {ahead}"
    if at_least is not None:
        assert len(overtaken) >= at_least, f"overtook {len(overtaken)}, want >= {at_least}"
    if at_most is not None:
        assert len(overtaken) <= at_most, f"overtook {len(overtaken)}, want <= {at_most}"


THROUGHPUT_PROMPT = (
    "[{nonce}] Write a long, detailed story about robot number {i} exploring a city at night. "
    "Keep going for many paragraphs."
)


def _stream_tokens(url: str, model: str, prompt: str, max_tokens: int) -> dict[str, float]:
    """One streamed chat request: time to first token, completion tokens, seconds."""
    body = {"model": model, "stream": True, "stream_options": {"include_usage": True},
            "max_tokens": max_tokens, "reasoning_effort": "none",
            "messages": [{"role": "user", "content": prompt}]}  # fmt: skip
    request = urllib.request.Request(
        f"{url}/v1/chat/completions", data=json.dumps(body).encode(), method="POST"
    )
    request.add_header("Content-Type", "application/json")
    started = time.monotonic()
    first: float | None = None
    chunks = 0
    tokens = 0
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            chunk = json.loads(line[5:])
            usage = chunk.get("usage")
            if usage:
                tokens = int(usage.get("completion_tokens") or 0)
            for choice in chunk.get("choices") or []:
                if (choice.get("delta") or {}).get("content"):
                    chunks += 1
                    if first is None:
                        first = time.monotonic()
    ended = time.monotonic()
    return {"ttft_ms": round(((first or ended) - started) * 1000, 1),
            "tokens": tokens or chunks, "s": ended - started, "started": started,
            "ended": ended}  # fmt: skip


@step("llm_parallel_throughput")
async def llm_parallel_throughput(
    ctx: ScenarioContext, count: int = 4, max_tokens: int = 300, min_tokens: int | None = None
) -> None:
    """Straight to the brain's LLM server: one real streamed request alone, then `count` at
    once (each prompt unique, so nothing is reused from the prefix cache; reasoning off). Each
    must produce at least `min_tokens` (default 80% of `max_tokens`) and the aggregate rate
    with `count` in parallel must beat the single stream. Recorded under `llm_throughput` in
    the scenario's timings (`timings_recorded`)."""
    llm = _brain_llm(ctx)
    floor = min_tokens if min_tokens is not None else int(max_tokens * 0.8)
    nonce = f"{time.time_ns():x}"

    def prompt(i: int) -> str:
        return THROUGHPUT_PROMPT.format(nonce=nonce, i=i)

    single = await asyncio.to_thread(_stream_tokens, llm["url"], LLM_MODEL, prompt(0), max_tokens)
    runs = await asyncio.gather(*(
        asyncio.to_thread(_stream_tokens, llm["url"], LLM_MODEL, prompt(i), max_tokens)
        for i in range(1, count + 1)
    ))  # fmt: skip
    wall = max(r["ended"] for r in runs) - min(r["started"] for r in runs)
    total = sum(int(r["tokens"]) for r in runs)
    single_rate = single["tokens"] / single["s"]
    aggregate = total / wall
    result = {
        "server": llm["server"], "max_tokens": max_tokens,
        "single": {"ttft_ms": single["ttft_ms"], "tokens": single["tokens"],
                   "tokens_per_s": round(single_rate, 1)},
        "parallel": count, "parallel_tokens": total, "parallel_wall_s": round(wall, 2),
        "aggregate_tokens_per_s": round(aggregate, 1),
        "per_stream_tokens_per_s": [round(r["tokens"] / r["s"], 1) for r in runs],
        "parallel_ttft_ms": [r["ttft_ms"] for r in runs],
    }  # fmt: skip
    ctx.state.setdefault("timings", {})["llm_throughput"] = result
    print(f"LLM throughput ({llm['server']} at {llm['url']}): {json.dumps(result)}")
    short = [r["tokens"] for r in [single, *runs] if r["tokens"] < floor]
    assert not short, f"requests ended early ({short} tokens, want >= {floor})"
    assert aggregate > single_rate, (
        f"{count} in parallel give {aggregate:.1f} tokens/s, no more than one alone "
        f"({single_rate:.1f})"
    )


# ---------------------------------------------------------------- a test LLM server


def _test_llm(ctx: ScenarioContext) -> dict[str, Any]:
    llm = ctx.state.get("test_llm")
    if llm is None:
        raise AssertionError("no test LLM server; use start_llm_server first")
    return llm


async def _wait_llm_up(url: str, within_s: float) -> None:
    deadline = time.monotonic() + within_s
    while time.monotonic() < deadline:
        if await asyncio.to_thread(llm_server.up, url):
            return
        await asyncio.sleep(0.5)
    raise AssertionError(f"the LLM server at {url} did not come up in {within_s} s")


@step("start_llm_server")
async def start_llm_server(
    ctx: ScenarioContext, server: str | None = None, within_s: float | None = None
) -> None:
    """Start a real LLM server of the scenario's own (`scripts/llm_server.sh`: `server` vllm or
    ollama, default the configured `[llm] server`) on GPU0 on a free loopback port, wait until
    it answers (default 600 s for vLLM, 30 s for Ollama) and check `reachy-gemma4` is loaded
    entirely on GPU0. It frees GPU0 first (the stack's Ollama unloads the model, the stack's
    vLLM sleeps) and can be killed for real (`kill_llm_server`) without touching the stack's
    server."""
    kind = _server(ctx, server)
    url = await _start_llm(ctx, LLM_PROCESS, kind, within_s)
    ctx.state["test_llm"] = {"url": url, "server": kind}


@step("kill_llm_server")
async def kill_llm_server(ctx: ScenarioContext) -> None:
    """Kill the scenario's LLM server (SIGKILL) and its children (Ollama's model runner,
    vLLM's engine core); its port is closed."""
    proc = ctx.processes.get(LLM_PROCESS)
    pgid = proc.pid
    proc.send_signal(signal.SIGKILL)
    await proc.wait(10)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)  # the children it spawned (same process group)
    url = _test_llm(ctx)["url"]
    if await asyncio.to_thread(llm_server.up, url):
        raise AssertionError(f"the LLM server at {url} still answers after the kill")
    print(f"the {_test_llm(ctx)['server']} at {url} was killed (SIGKILL)")


@step("restart_llm_server")
async def restart_llm_server(ctx: ScenarioContext, within_s: float | None = None) -> None:
    """Start the killed LLM server again on the same port; the model is loaded entirely on
    GPU0 again."""
    llm = _test_llm(ctx)
    await ctx.processes.restart(LLM_PROCESS)
    await _wait_llm_up(llm["url"], within_s or llm_server.START_TIMEOUT_S[llm["server"]])
    print(await asyncio.to_thread(llm_server.load_and_check, llm["url"], LLM_MODEL, llm["server"]))


# ---------------------------------------------------------------- the robot


ATTENTION_POSES = {"listening": (0.0, -5.0), "thinking": (7.0, -3.0), "speaking": (0.0, 0.0)}
SETTLE_WINDOW_S = 0.5
"""How long after a move's end the head may still be reaching its pose."""
"""What the reachy body's MotionArbiter.attend does per state: (roll, pitch) from neutral."""


def _matrix(pose: dict[str, float]) -> list[list[float]]:
    return _rotation(pose)


def _degrees(pose: dict[str, float]) -> str:
    return " ".join(f"{k} {math.degrees(pose[k]):.1f}" for k in ("roll", "pitch", "yaw"))


def _mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _angle(a: list[list[float]], b: list[list[float]]) -> float:
    trace = sum(a[k][i] * b[k][i] for i in range(3) for k in range(3))
    return math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1) / 2))))


async def _start_sampler(ctx: ScenarioContext) -> ManagedProcess:
    host = edge_host_steps.host_of(ctx)
    count = ctx.state["samplers"] = ctx.state.get("samplers", 0) + 1
    name = f"state-sampler-{count}"
    url = f"http://127.0.0.1:{edge_host_steps.DAEMON_PORT}"
    args = ["-c", _SAMPLER, url, str(SAMPLE_PERIOD_S)]
    if host.ssh is not None:
        python = f"{edge_host.REMOTE_VENV}/bin/python"
        sampler = await ctx.processes.start(
            name, [python, *args], ssh=host.ssh, remote_cwd=f"{edge_host.EDGE_DIR}/src"
        )
    else:
        sampler = await ctx.processes.start(name, [sys.executable, *args])
    await _expect(ctx, name, {"STATE"}, 15, what="the robot-side state sampler")
    return sampler


@step("robot_pose_follows")
async def robot_pose_follows(
    ctx: ScenarioContext,
    client: str,
    text: str,
    states: list[str],
    tolerance_deg: float = 6.0,
    max_head_deg: float = 10.0,
    within_s: float = 120.0,
) -> None:
    """Type `text` to the agent; its head follows the turn's attention `states` (each a MOTION
    line with the time its pose was reached). A sampler next to the daemon reads the robot's
    state every 50 ms meanwhile: at each pose the head is within `tolerance_deg` of the
    target (neutral turned by the state's roll/pitch) and closer to it than neutral is, and it
    never turns more than `max_head_deg` from neutral. The Lite holds a small pose with a
    steady error of up to about 5 degrees measured from the woken neutral (servo backlash plus
    the neutral reading's own bias; 1.5-4.8 seen), hence the 6 degree default.
    Remembers the pose before (for `robot_back_at_rest`)."""
    _remember_start(ctx, await _robot_state(ctx))
    name = _client_name(client)
    agent = ctx.processes.get(name)
    consumed = ctx.state["link"].consumed.setdefault(name, set())
    consumed.update(line.index for line in _get_lines(agent, "MOTION"))  # before this turn
    sampler = await _start_sampler(ctx)
    motions: list[_Line] = []
    try:
        await agent.write_line(text)
        for state in states:
            line = await _expect(
                ctx, name, {"MOTION", "MOTION-ERROR"}, within_s, fields={"attention": state},
                what=f"MOTION attention={state}",
            )  # fmt: skip
            assert line.tag == "MOTION", f"the attention move failed: {line.text}"
            assert line.fields.get("moved") == "true", f"the head did not move: {line.text}"
            motions.append(line)
        await asyncio.sleep(SETTLE_WINDOW_S + 3 * SAMPLE_PERIOD_S)
    finally:
        await sampler.stop()
    samples = [p for line in _get_lines(sampler, "STATE") if (p := line.payload) is not None]
    span = samples[-1]["t"] - samples[0]["t"] if samples else 0.0
    print(f"sampler: {len(samples)} samples over {span:.1f} s")
    check_sampling(samples, span)
    moves = [line.payload or {} for line in motions]
    for line in motions:
        print(f"  {line.text}")
    posed = [m for m in moves if m.get("state") in ATTENTION_POSES]
    assert posed, "no attention pose was moved to"
    # The neutral pose: read by the arbiter at `t_neutral` (after wake_up() for a robot at
    # rest), before the first pose; the first sample from then on is the reference.
    start = posed[0]["t_neutral"]
    neutral_samples = [s for s in samples if s["t"] >= start]
    assert neutral_samples, "no sample at the neutral pose"
    neutral = _matrix(neutral_samples[0]["head_pose"])
    print(f"  neutral head pose: {_degrees(neutral_samples[0]['head_pose'])}")
    for move in posed:
        roll, pitch = ATTENTION_POSES[move["state"]]
        target = _mul(neutral, _matrix({"roll": math.radians(roll), "pitch": math.radians(pitch),
                                         "yaw": 0.0}))  # fmt: skip
        # The servos trail the commanded move a little and the next move may start at once:
        # the pose counts as reached if the head comes within tolerance during the move or
        # just after it (`SETTLE_WINDOW_S`).
        window = [s for s in samples
                  if move["t_start"] <= s["t"] <= move["t_reached"] + SETTLE_WINDOW_S]  # fmt: skip
        assert window, f"no sample while {move['state']} was moved to"
        reached = min(window, key=lambda s: _angle(target, _matrix(s["head_pose"])))
        error = _angle(target, _matrix(reached["head_pose"]))
        turned = _angle(neutral, _matrix(reached["head_pose"]))
        print(
            f"  {move['state']} closest at +{reached['t'] - move['t_reached']:.2f} s: "
            f"{_degrees(reached['head_pose'])}"
        )
        print(
            f"  {move['state']}: target roll {roll} pitch {pitch} deg; measured "
            f"{turned:.1f} deg from neutral, {error:.1f} deg off the target"
        )
        assert error <= tolerance_deg, f"{move['state']}: {error:.1f} deg off its target pose"
        commanded = _angle(neutral, target)
        if commanded > 0.5:  # a head that stayed put must not pass: it turned toward the pose
            assert error < commanded, (
                f"{move['state']}: the head is no closer to its pose ({error:.1f} deg off) than "
                f"neutral is ({commanded:.1f} deg)"
            )
    end = moves[-1].get("t_reached", samples[-1]["t"])
    during = [s for s in samples if start <= s["t"] <= end]
    peak = max(_angle(neutral, _matrix(s["head_pose"])) for s in during)
    print(f"largest head turn from neutral while attending: {peak:.1f} deg")
    assert peak <= max_head_deg + 0.5, f"the head turned {peak:.1f} deg (limit {max_head_deg})"
