"""Steps for the try-it launcher (`scripts/try_it.sh`, `assistant_testing.try_it`).

The launcher runs as a whole, the way a user runs it: a child process of the scenario whose
output (the `you:` / `robot:` lines and the `[...]` notes) the steps read. Its client name
for `speaker_plays` is `try_it`. Stopping it is a Ctrl-C (SIGINT to its process group, as a
terminal sends it), also in teardown, so the robot is put to rest whatever the outcome.
"""

import re
import signal

from assistant_edge.agent import ListenMode
from assistant_testing.features.context import ScenarioContext
from assistant_testing.features.registry import step
from assistant_testing.processes import ManagedProcess
from assistant_testing.steps.link import _client_name
from assistant_testing.steps.listen import _mark
from assistant_testing.steps.speech import _golden, word_error_rate

CLIENT = "try_it"
STOP_S = 180.0
"""How long the launcher gets to stop everything after Ctrl-C."""


def _launcher(ctx: ScenarioContext) -> ManagedProcess:
    return ctx.processes.get(_client_name(CLIENT))


async def _ctrl_c(proc: ManagedProcess, within_s: float) -> int:
    proc.send_signal(signal.SIGINT)
    return await proc.wait(within_s)


@step("try_it_started")
async def try_it_started(
    ctx: ScenarioContext,
    listen: ListenMode | None = None,
    mode: ListenMode = "wake_word",
    within_s: float = 900.0,
) -> None:
    """Run `scripts/try_it.sh` (with `--listen listen` if given) until it is ready (it may
    start the LLM and speech servers itself: `within_s`); it listens in `mode`."""
    argv = [str(ctx.repo_root / "scripts" / "try_it.sh")]
    if listen is not None:
        argv += ["--listen", listen]
    name = _client_name(CLIENT)
    ctx.processes.before_stop[name] = lambda: _stop_in_teardown(ctx)
    await ctx.processes.start(name, argv, ready_line=r"^ready, listening: ", ready_timeout=within_s)
    ready = next(line for line in _launcher(ctx).lines if line.startswith("ready, listening: "))
    print(ready)
    assert ready.startswith(f"ready, listening: {mode}:"), f"want mode {mode}: {ready}"


async def _stop_in_teardown(ctx: ScenarioContext) -> None:
    proc = _launcher(ctx)
    if proc.running:
        await _ctrl_c(proc, STOP_S)


@step("try_it_shows")
async def try_it_shows(
    ctx: ScenarioContext,
    wake: bool = False,
    max_wer: float = 0.35,
    within_s: float = 60.0,
) -> None:
    """Since the mark (`speaker_plays`): the launcher showed the wake word (`wake`), what was
    heard (`you:`, the played golden WAV's text within `max_wer`) and a reply (`robot:`)."""
    proc, mark = _launcher(ctx), _mark(ctx, CLIENT)

    async def shown(pattern: str, what: str) -> str:
        regex = re.compile(pattern)
        _, text = await proc.wait_for_output(
            lambda line: regex.search(line) is not None,
            within_s,
            skip=lambda index: index < mark,
            what=what,
        )
        print(text)
        return text

    if wake:
        await shown(r"^  \[wake word: hey jarvis ", "the wake word")
    heard = (await shown(r"^you: ", "what was heard (you:)"))[len("you:") :].strip()
    fed = ctx.state.get("fed")
    assert fed is not None, "nothing played; use speaker_plays first"
    reference = _golden(ctx, fed["name"])[1]
    wer = word_error_rate(reference, heard)
    assert wer <= max_wer, f"heard {heard!r}: WER {wer:.3f} > {max_wer} against {reference!r}"
    reply = (await shown(r"^robot: ", "a reply (robot:)"))[len("robot:") :].strip()
    assert reply, "the reply is empty"


@step("try_it_stopped")
async def try_it_stopped(ctx: ScenarioContext, within_s: float = STOP_S) -> None:
    """Ctrl-C: the launcher exits 0 within `within_s`, having put the robot to rest with its
    motors off and left no process behind on this PC or the robot's machine."""
    proc = _launcher(ctx)
    code = await _ctrl_c(proc, within_s)
    tail = "\n".join(proc.lines[-12:])
    print(tail)
    assert code == 0, f"the launcher exited {code}:\n{tail}"
    motors = [line for line in proc.lines if line.startswith("robot at rest; motors: ")]
    assert motors, f"no motor state reported:\n{tail}"
    assert "disabled" in motors[-1], f"motors not off: {motors[-1]}"
    assert any(re.match(r"^stopped; 0 leftover process", line) for line in proc.lines), tail
