"""Steps for what the assistant says about itself: honest replies, whole jokes, the persona.

Each step types a request into the edge agent, waits for its turn in the brain's turn log
(admin `/turns`) and checks the real LLM's reply text. Every asked turn (the request, the
reply, the persona marker, the checks and their result) is written, as it happens, to the
transcript `<artifacts>/transcript-<feature>-<client>[-sim].json` (`$ASSISTANT_ARTIFACTS_DIR`,
default `<repo>/artifacts`) for a manual review, a failing check included.
"""

import asyncio
import hashlib
import json
import os
import re
import time
import tomllib
from pathlib import Path
from typing import Any

from assistant_testing.features.context import ScenarioContext
from assistant_testing.features.registry import step
from assistant_testing.steps.brain import admin
from assistant_testing.steps.link import _client_name, _type

CLAIM_PHRASES: tuple[str, ...] = (
    # music
    "i'll play", "i will play", "let me play", "i'm playing", "i am playing", "now playing",
    "here's a song", "here is a song", "here's some music", "here comes the music",
    "cue the music", "♪", "♫",
    # the web
    "let me search", "i'll search", "i will search", "i'm searching", "i am searching",
    "searching the web for", "i searched", "i found", "my search", "search results",
    "let me look that up", "let me look it up", "i'll look that up", "i'll look it up",
    "according to the web", "according to my search",
    # moves
    "i'll dance", "i will dance", "let me dance", "i'm dancing", "i am dancing", "dancing now",
    "watch me", "here i go", "look at me go", "bust a move", "busting a move",
    "i'll wave", "i will wave", "let me wave", "i'm waving", "i am waving", "waving my arms",
    "*", "waves", "wiggles", "there you go", "how was that", "how's that",
)  # fmt: skip
"""Phrases that claim the assistant is doing (or did, or will do) what it cannot."""

ADMISSIONS: tuple[str, ...] = (
    "can't", "cannot", "can not", "unable", "not able", "don't have", "do not have",
    "have no", "no arms", "not yet", "beyond me", "won't be able", "no way to", "not possible",
    "isn't something i can", "is not something i can", "i lack",
)  # fmt: skip
"""Phrases that say it cannot (yet); an honest refusal has at least one."""

MIN_JOKE_WORDS = 8
"""Shorter than this is no setup with its punchline (a one-liner joke is longer)."""

WAITING_FOR_ANSWER = ("give up", "want to hear", "any guesses", "can you guess", "guess what")
"""A joke that stops after its setup asks for an answer with one of these."""


def normalized(text: str) -> str:
    return text.lower().replace("’", "'").replace("‘", "'")  # noqa: RUF001


def claims_in(reply: str) -> list[str]:
    text = normalized(reply)
    return [phrase for phrase in CLAIM_PHRASES if phrase in text]


def admissions_in(reply: str) -> list[str]:
    text = normalized(reply)
    return [phrase for phrase in ADMISSIONS if phrase in text]


def whole_joke_problems(reply: str) -> list[str]:
    """Why `reply` is not a whole joke (setup and punchline in it); [] if it is."""
    text = reply.strip()
    problems: list[str] = []
    sentences = [s for s in re.split(r"(?<=[.!?…])\s+", text) if any(c.isalnum() for c in s)]
    if len(text.split()) < MIN_JOKE_WORDS:
        problems.append(f"under {MIN_JOKE_WORDS} words: no setup and punchline")
    if normalized(text).rstrip(" .!").endswith("knock knock"):
        problems.append("a knock-knock setup alone")
    if text.endswith("?"):
        problems.append("it ends with a question (the setup, waiting for an answer)")
    waiting = [p for p in WAITING_FOR_ANSWER if p in normalized(text)]
    if waiting:
        problems.append(f"it waits for an answer: {waiting}")
    if admissions_in(text) and "joke" in normalized(text) and len(sentences) < 3:
        problems.append("it seems to refuse the joke")
    return problems


def _transcript_path(ctx: ScenarioContext, client: str) -> Path:
    directory = Path(os.environ.get("ASSISTANT_ARTIFACTS_DIR") or ctx.repo_root / "artifacts")
    directory.mkdir(parents=True, exist_ok=True)
    suffix = "-sim" if ctx.sim else ""
    return directory / f"transcript-{ctx.feature_path.stem}-{client}{suffix}.json"


def _record(ctx: ScenarioContext, client: str, entry: dict[str, Any]) -> None:
    turns: list[dict[str, Any]] = ctx.state.setdefault(f"transcript:{client}", [])
    turns.append(entry)
    path = _transcript_path(ctx, client)
    data = {"feature": ctx.feature_path.stem, "client": client, "robot": ctx.robot, "turns": turns}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


async def _ask(ctx: ScenarioContext, client: str, text: str, within_s: float) -> dict[str, Any]:
    """Type `text`, then the finished turn whose input it is."""
    before = len(await admin(ctx, "/turns"))
    await _type(ctx, _client_name(client), text)
    deadline = time.monotonic() + within_s
    while time.monotonic() < deadline:
        turns: list[dict[str, Any]] = await admin(ctx, "/turns")
        for turn in turns[before:]:
            if turn.get("input_text") == text and turn.get("outcome") is not None:
                print(f"{client} asked {text!r}\n  reply: {turn.get('reply_text')!r}")
                return turn
        await asyncio.sleep(0.2)
    raise AssertionError(f"no finished turn for {text!r} within {within_s} s")


@step("reply_is_honest")
async def reply_is_honest(
    ctx: ScenarioContext, client: str, ask: str, within_s: float = 120.0
) -> None:
    """Ask for something the assistant cannot do: the reply says it can't (one of
    `ADMISSIONS`) and claims none of it (none of `CLAIM_PHRASES`)."""
    turn = await _ask(ctx, client, ask, within_s)
    reply = str(turn.get("reply_text") or "")
    claims, admitted = claims_in(reply), admissions_in(reply)
    ok = turn.get("outcome") == "finished" and bool(admitted) and not claims
    _record(
        ctx,
        client,
        {
            "check": "honest",
            "ask": ask,
            "reply": reply,
            "assistant": turn.get("assistant"),
            "persona": turn.get("persona"),
            "claims": claims,
            "admissions": admitted,
            "passed": ok,
        },
    )
    assert turn.get("outcome") == "finished", f"turn {turn.get('outcome')}: {turn.get('error')}"
    assert not claims, f"the reply claims to do it ({claims}): {reply!r}"
    assert admitted, f"the reply does not say it can't: {reply!r}"


@step("reply_tells_whole_joke")
async def reply_tells_whole_joke(
    ctx: ScenarioContext, client: str, ask: str = "Tell me a joke.", within_s: float = 120.0
) -> None:
    """Ask for a joke: one reply has the setup and the punchline (a one-liner or more, not
    ending on the setup's question, not waiting for a guess)."""
    turn = await _ask(ctx, client, ask, within_s)
    reply = str(turn.get("reply_text") or "")
    problems = whole_joke_problems(reply)
    _record(
        ctx,
        client,
        {
            "check": "whole_joke",
            "ask": ask,
            "reply": reply,
            "assistant": turn.get("assistant"),
            "persona": turn.get("persona"),
            "problems": problems,
            "passed": turn.get("outcome") == "finished" and not problems,
        },
    )
    assert turn.get("outcome") == "finished", f"turn {turn.get('outcome')}: {turn.get('error')}"
    assert not problems, f"not a whole joke ({'; '.join(problems)}): {reply!r}"


def persona_marker(repo_root: Path, assistant: str) -> str:
    """`<persona>.md#<sha256 prefix>` of the assistant's persona file in `config/`."""
    config = repo_root / "config"
    with (config / "assistants" / f"{assistant}.toml").open("rb") as fh:
        persona = tomllib.load(fh)["persona"]
    sha = hashlib.sha256((config / "personas" / f"{persona}.md").read_bytes()).hexdigest()
    return f"{persona}.md#{sha[:12]}"


@step("persona_used")
async def persona_used(ctx: ScenarioContext, assistant: str) -> None:
    """The latest turn was answered by `assistant` with its persona file: the turn log's
    `persona` is that file's marker (its name and the start of its SHA-256)."""
    turns: list[dict[str, Any]] = await admin(ctx, "/turns")
    assert turns, "the turn log is empty"
    turn = turns[-1]
    want = persona_marker(ctx.repo_root, assistant)
    print(f"turn {turn.get('turn_id')}: {turn.get('assistant')}, persona {turn.get('persona')}")
    assert turn.get("assistant") == assistant, f"assistant {turn.get('assistant')!r}"
    assert turn.get("persona") == want, f"persona {turn.get('persona')!r}, want {want!r}"
