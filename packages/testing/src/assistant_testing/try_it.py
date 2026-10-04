"""Try the assistant on the robot, hands-free: `scripts/try_it.sh [--listen MODE]`.

Starts everything the conversation needs, the same way the robot's e2e features do:

- the LLM server (vLLM with reachy-gemma4 on GPU0) and the speech server (Parakeet and
  Qwen3-TTS on GPU1), reusing the ones on 127.0.0.1:8773 / :8772 if they already serve;
- the commit checked out here (HEAD) synced to the robot's machine;
- the reachy-mini daemon, the brain and the SSH tunnels;
- the edge agent with the reachy body (inside Reachy Edge.app on a Mac).

`--listen` picks the listening mode: `wake_word` (the default: say "hey jarvis"), `open_mic`
(just talk) or `push_to_talk` (debugging: an empty Enter starts listening, the next one stops).
A typed line is sent as a typed question in every mode. It shows what was heard and the
replies. Ctrl-C stops everything it started (the robot goes to rest and its motors are turned
off first), sweeps the robot's machine and this PC for leftovers and releases the robot. A
legacy assistant still holding the robot is stopped first (SIGINT, so it shuts down cleanly).
One user of the robot at a time: the hw-run lock of the e2e features is held while it runs.
The e2e test aids (the energy trigger, armed only on fed audio) are never used here.
"""

import argparse
import asyncio
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from assistant_core.config import load_config
from assistant_edge.agent import LISTEN_MODES, ListenMode
from assistant_testing import edge_host
from assistant_testing.features.context import ScenarioContext
from assistant_testing.processes import ManagedProcess
from assistant_testing.steps import brain as brain_steps
from assistant_testing.steps import edge as edge_steps
from assistant_testing.steps import edge_host as edge_host_steps
from assistant_testing.steps import speech as speech_steps
from assistant_testing.steps.link import SERVER, _client_name

DEVICE = "reachy"
LEGACY = (
    # The legacy conversation app first (it shuts down on SIGINT), then its daemon.
    r"^[^ ]*[Pp]ython[0-9.]* [^ ]*local_backend/run_app\.py",
    r"^[^ ]*[Pp]ython[0-9.]* [^ ]*reachy_mini_conversation_app",
    r"^[^ ]*[Pp]ython[0-9.]* [^ ]*local_backend/run_daemon\.py",
)
SHOWN = {
    "WAKE", "MIC-OPEN", "MIC-CLOSE", "FOLLOW-UP", "SAY", "CONSOLE-ERROR",
    "TRACKING-MODE", "TRACKING", "VOICE-TURN",
}  # fmt: skip


def _repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
    )
    return Path(out.stdout.strip())


def _stop_legacy(host: edge_host.EdgeHost) -> list[str]:
    """SIGINT the legacy assistant on this PC and on the robot's machine; wait for it."""
    stopped: list[str] = []
    for pattern in LEGACY:
        script = (
            f'pids=$(pgrep -f {json.dumps(pattern)}); [ -z "$pids" ] && exit 0; '
            'ps -o pid=,command= -p $(echo $pids | tr " " ,); kill -INT $pids; '
            "for i in $(seq 1 30); do kill -0 $pids 2>/dev/null || exit 0; sleep 1; done; "
            "echo STILL-RUNNING"
        )
        runs = [subprocess.run(["sh", "-c", script], capture_output=True, text=True, timeout=60)]
        if host.ssh is not None:
            done = host.run(script, timeout_s=60)
            runs.append(subprocess.CompletedProcess([], done.returncode, done.stdout, ""))
        for run in runs:
            lines = [line.strip() for line in run.stdout.splitlines() if line.strip()]
            if "STILL-RUNNING" in lines:
                raise SystemExit(f"the legacy assistant did not stop on SIGINT: {lines}")
            stopped += [host.scrub(line) for line in lines]
    return stopped


def _fields(text: str) -> tuple[str, dict[str, str], dict[str, object]]:
    head, _, rest = text.partition(" {")
    tag, *words = head.split()
    fields = dict(word.split("=", 1) for word in words if "=" in word)
    payload: dict[str, object] = {}
    if rest:
        with contextlib.suppress(ValueError):
            payload = json.loads("{" + rest)
    return tag, fields, payload


def _show(source: str, text: str) -> str | None:
    """The line to show for an output line of the brain or the edge (None: not shown)."""
    if "Traceback" in text or text.startswith("BRAIN-ERROR"):
        return f"  [{source} error] {text}"
    tag, fields, payload = _fields(text)
    if source == "brain":
        return f"you:   {payload.get('text') or '(nothing heard)'}" if tag == "TRANSCRIPT" else None
    if tag not in SHOWN:
        return None
    if tag == "SAY":
        return f"robot: {payload.get('text', '')}"
    if tag == "WAKE":
        word = re.search(r" word=(.*?) model=", text)  # the word may have spaces
        return f"  [wake word: {word[1] if word else '?'} ({fields.get('score', '?')})]"
    if tag == "MIC-OPEN":
        cut = ", interrupting the robot" if fields.get("barge_in") == "true" else ""
        return f"  [listening ({fields.get('reason', '?')}{cut})]"
    if tag == "MIC-CLOSE":
        return f"  [stopped listening ({fields.get('reason', '?')})]"
    if tag == "FOLLOW-UP":
        return "  [follow-up: just talk, no wake word needed]"
    if tag == "TRACKING-MODE":
        return f"  [person tracking: {fields.get('mode', '?')}]"
    if tag == "TRACKING":
        camera = " (no camera)" if fields.get("camera") == "false" else ""
        return f"  [tracking {fields.get('state', '?')} ({fields.get('reason', '?')}){camera}]"
    if tag == "VOICE-TURN":
        if fields.get("turned") != "true":
            return f"  [no turn toward the voice ({fields.get('why', '?')})]"
        return f"  [turned {payload.get('yaw_deg', '?')} deg toward the voice]"
    return f"  [edge] {text}"


async def _typed(edge: ManagedProcess, listen: ListenMode) -> None:
    """Typed lines go to the edge: text is a typed question; in push_to_talk an empty line
    starts listening and the next one stops (`/ptt down`, `/ptt up`)."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    talking = False
    while raw := await reader.readline():
        text = raw.decode(errors="replace").strip()
        if text:
            await edge.write_line(text)
        elif listen == "push_to_talk":
            talking = not talking
            await edge.write_line("/ptt down" if talking else "/ptt up")


async def _converse(ctx: ScenarioContext, stop: asyncio.Event, listen: ListenMode) -> str:
    """Show the conversation until `stop` is set or the brain or the edge exits."""
    edge = ctx.processes.get(_client_name(DEVICE))
    sources: list[tuple[str, ManagedProcess]] = [
        ("brain", ctx.processes.get(SERVER)),
        ("edge", edge),
    ]
    typing = asyncio.create_task(_typed(edge, listen)) if sys.stdin.isatty() else None
    try:
        return await _show_until(sources, stop)
    finally:
        if typing is not None:
            typing.cancel()


async def _show_until(sources: list[tuple[str, ManagedProcess]], stop: asyncio.Event) -> str:
    seen = {name: len(proc.lines) for name, proc in sources}
    while not stop.is_set():
        for name, proc in sources:
            for text in proc.lines[seen[name] :]:
                if (shown := _show(name, text)) is not None:
                    print(shown, flush=True)
            seen[name] = len(proc.lines)
            if not proc.running:
                return f"the {name} exited ({proc.proc.returncode})"
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), 0.1)
    return "Ctrl-C"


async def _start(ctx: ScenarioContext, listen: ListenMode) -> None:
    await edge_host_steps.robot_host_found(ctx)
    await edge_host_steps.code_synced_to_edge_host(ctx)
    await speech_steps.llm_serves(ctx)
    await speech_steps.speech_server_running(ctx)
    await edge_host_steps.start_reachy_daemon(ctx)
    await brain_steps.start_brain(ctx, engine="basic", speech=True)
    await edge_steps.start_edge_agent(
        ctx, id=DEVICE, body="reachy", where="edge_host", listen=listen, within_s=180
    )


def _report_motors(ctx: ScenarioContext, host: edge_host.EdgeHost) -> None:
    """Just before the daemon stops (after its rest hook): print the motors' state."""
    rest = ctx.processes.before_stop.get(edge_host_steps.DAEMON)
    if rest is None:
        return
    status = f"curl -s -m 3 {edge_host.DAEMON_API}/motors/status"

    async def rest_and_report() -> None:
        await rest()
        done = await asyncio.to_thread(host.run, status, 30)
        print(f"robot at rest; motors: {' '.join(done.stdout.split()) or 'no answer'}")

    ctx.processes.before_stop[edge_host_steps.DAEMON] = rest_and_report


async def _main(listen: ListenMode) -> int:
    repo = _repo_root()
    os.environ.setdefault("ASSISTANT_TEST_RUN", f"try-it-{int(time.time())}-{os.getpid()}")
    host = await asyncio.to_thread(edge_host.resolve, repo)
    for line in await asyncio.to_thread(_stop_legacy, host):
        print(f"stopped the legacy assistant (SIGINT): {line}")
    locked = await asyncio.to_thread(edge_host.acquire_lock, host, os.getpid())
    print(locked)
    # Run from an hw feature, the gate's run already holds the lock: it is the gate's to release.
    owned = locked.startswith("hw-run lock taken")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    first: list[float] = []

    def on_signal() -> None:
        # One Ctrl-C reaches both uv and this process (uv forwards it): only a later one counts.
        if first and time.monotonic() - first[0] > 2:
            print("\nstill stopping, please wait (the robot is being put to rest)", flush=True)
        first[:] = first or [time.monotonic()]
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        loop.add_signal_handler(sig, on_signal)
    ctx = ScenarioContext(repo, Path(__file__))
    reason, rc = "Ctrl-C", 0
    try:
        for line in await asyncio.to_thread(edge_host.sweep, host):
            print(f"stopped leftover of an earlier run: {line}")
        starting = asyncio.create_task(_start(ctx, listen))
        waiting = asyncio.create_task(stop.wait())
        await asyncio.wait({starting, waiting}, return_when=asyncio.FIRST_COMPLETED)
        waiting.cancel()
        if not starting.done():
            starting.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await starting
        else:
            starting.result()
            hint = {
                "wake_word": 'say "hey jarvis", then your question',
                "open_mic": "just talk; talk over the robot to interrupt it",
                "push_to_talk": "press Enter to start listening, Enter again to stop",
            }[listen]
            print(f"\nready, listening: {listen}: {hint}. Ctrl-C stops everything.\n", flush=True)
            print("(a typed line is sent as a typed question)\n", flush=True)
            reason = await _converse(ctx, stop, listen)
            rc = 0 if reason == "Ctrl-C" else 1
    except Exception as error:  # reported; teardown below still runs
        reason, rc = f"failed: {error}", 1
    finally:
        print(f"\nstopping ({reason}): the robot goes to rest, motors off", flush=True)
        _report_motors(ctx, host)
        await ctx.aclose()
        left = await asyncio.to_thread(edge_host.sweep, host)
        for line in left:
            print(f"stopped leftover: {line}")
        if owned:
            print(await asyncio.to_thread(edge_host.release_lock, host))
        print(f"stopped; {len(left)} leftover process(es) on this PC and {host.label}")
    return rc


def main(argv: list[str] | None = None) -> int:
    config = load_config(Path("config"))
    parser = argparse.ArgumentParser(
        prog="scripts/try_it.sh", description="Try the assistant on the robot, hands-free."
    )
    parser.add_argument(
        "--listen",
        choices=LISTEN_MODES,
        default=config.edge.listen.mode,
        help=f"listening mode (default {config.edge.listen.mode}, from config [edge.listen])",
    )
    args = parser.parse_args(argv)
    listen: ListenMode = args.listen
    return asyncio.run(_main(listen))


if __name__ == "__main__":
    sys.exit(main())
