"""Several assistants on one robot, picked by wake word ("Hey Jarvis" -> Jarvis, "Hey Marvin" -> Marvin).

Enabled by start_conversation.sh --assistants (REACHY_ASSISTANTS=1); run_app.py calls install() before the
app starts. Plan and design notes: MULTI_ASSISTANT_PLAN.md.

Registry: local_backend/assistants.json (name -> wake word, threshold, base profile, base voice; default).
Each assistant has its own folder, local_backend/state/assistants/<name>/:

  history.json     its conversation (last HISTORY_TURNS turns), replayed into every new session
  memory.v1.json   its long-term memory (remember / forget; the app's memory module is redirected here)
  lists.json       its lists (reachy_lists)
  books.json       its bookmarks (reachy_reader)
  style.json       its current style: a persona profile chosen with switch_persona / the web UI, and a voice
Reminders stay in one file with an "owner" field (reachy_scheduler); each assistant only sees its own.
The active assistant is saved in state/assistants/active.json, so the robot boots as whoever was active last.

Switching (the other assistant's wake word, detected in reachy_wake): the gate holds the mic audio (pre-roll
included), the current speech is flushed, and the session is restarted with the other assistant's profile
and voice (LocalStream.apply_personality, as the persona switcher does). A new session starts empty (the
speech server can't delete items), so in place of the startup greeting the assistant's saved turns are
replayed as conversation items; then the held audio is released and the server hears the request as usual.
Measured: a restart takes ~0.3 s.

Restyling within an assistant (switch_persona, the web UI's persona picker, the UI's voice picker) is saved
in that assistant's style.json; its name, wake word and data don't change.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import reachy_bridge

logger = logging.getLogger("reachy_assistants")

HERE = Path(__file__).resolve().parent
REGISTRY = Path(os.environ.get("REACHY_ASSISTANTS_FILE", HERE / "assistants.json"))
STATE = Path(os.environ.get("REACHY_ASSISTANTS_STATE", HERE / "state" / "assistants"))
HISTORY_TURNS = 20          # user turns kept per assistant (the server compacts beyond 30)
MAX_TEXT = 1200             # characters kept per message
HOLD_TIMEOUT_S = 10.0       # release the held mic audio even if a switch never completes

_lock = threading.RLock()
_registry: dict[str, Any] = {}
_active = ""
_web = False
_installed = False
_switching: str | None = None      # target of a router switch in progress
_skip_greeting_reply = False       # don't store the model's greeting as a conversation turn


# -- registry and folders ------------------------------------------------------------------------
def load_registry(path: Path = REGISTRY) -> dict[str, Any]:
    data = json.loads(path.read_text())
    assistants = data.get("assistants") or {}
    if not assistants:
        raise ValueError(f"No assistants in {path}")
    if data.get("default") not in assistants:
        data["default"] = next(iter(assistants))
    return data


def enabled() -> bool:
    return _installed


def names() -> list[str]:
    return list(_registry.get("assistants", {}))


def active() -> str:
    return _active


def spec(name: str | None = None) -> dict[str, Any]:
    return _registry["assistants"][name or _active]


def folder(name: str | None = None) -> Path:
    d = STATE / (name or _active)
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_file(filename: str, default: Path) -> Path:
    """Where a per-assistant file lives: the active assistant's folder, or `default` without the router."""
    return folder() / filename if _installed and _active else default


def _read_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return fallback


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    tmp.replace(path)


def style(name: str | None = None) -> dict[str, Any]:
    return _read_json(folder(name) / "style.json", {})


def set_style(name: str | None = None, **changes: Any) -> None:
    with _lock:
        s = style(name)
        s.update(changes)
        _write_json(folder(name) / "style.json", {k: v for k, v in s.items() if v})


def base_profile(name: str | None = None) -> str:
    return spec(name)["profile"] + ("_web" if _web else "")


def profile_for(name: str | None = None) -> str:
    p = style(name).get("profile") or spec(name)["profile"]
    return p + ("_web" if _web else "")


def voice_for(name: str | None = None) -> str | None:
    """A voice picked in the UI for this assistant, else None (the profile's own voice)."""
    return style(name).get("voice") or None


def wake_words() -> dict[str, float]:
    """openWakeWord model name -> threshold, for reachy_wake."""
    return {a["wake_word"]: float(a.get("threshold", 0.5)) for a in _registry["assistants"].values()}


def assistant_for_word(word: str) -> str | None:
    for name, a in _registry.get("assistants", {}).items():
        if a["wake_word"] == word:
            return name
    return None


def _set_active(name: str) -> None:
    global _active
    with _lock:
        _active = name
        _write_json(STATE / "active.json", {"active": name, "since": time.strftime("%Y-%m-%dT%H:%M:%S")})


# -- history -------------------------------------------------------------------------------------
def history(name: str | None = None) -> list[dict[str, str]]:
    h = _read_json(folder(name) / "history.json", [])
    return [m for m in h if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("text")]


def _trim(h: list[dict[str, str]]) -> list[dict[str, str]]:
    users = [i for i, m in enumerate(h) if m["role"] == "user"]
    if len(users) > HISTORY_TURNS:
        h = h[users[-HISTORY_TURNS]:]
    return h


def record(role: str, text: str, name: str | None = None) -> None:
    text = " ".join((text or "").split())[:MAX_TEXT]
    if not text or role not in ("user", "assistant"):
        return
    with _lock:
        h = history(name)
        if h and h[-1]["role"] == role == "assistant":
            h[-1]["text"] = (h[-1]["text"] + " " + text)[:MAX_TEXT]   # one reply spoken in several parts
        else:
            h.append({"role": role, "text": text, "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        _write_json(folder(name) / "history.json", _trim(h))


def clear_history(name: str | None = None) -> int:
    with _lock:
        n = len(history(name))
        _write_json(folder(name) / "history.json", [])
    return n


def replay_items(name: str | None = None) -> list[dict[str, Any]]:
    """Conversation items that rebuild this assistant's context in a fresh session."""
    items = []
    for m in history(name):
        kind = "input_text" if m["role"] == "user" else "output_text"
        items.append({"type": "message", "role": m["role"], "content": [{"type": kind, "text": m["text"]}]})
    return items


def identity_note(name: str | None = None) -> str:
    n = name or _active
    others = [o.title() for o in names() if o != n]
    word = spec(n)["wake_word"].replace("_", " ").title()
    note = (f"## WHO YOU ARE\nYour name is {n.title()}; the user talks to you by saying \"{word}\". "
            f"You live in a small Reachy Mini robot body")
    if others:
        note += (f", which you share with {', '.join(others)}: another assistant with its own conversations, "
                 f"memory and lists, which you can't see (and it can't see yours). If the user wants {others[0]}, "
                 f"tell them to say \"Hey {others[0]}\"")
    return note + ". If asked your name, you are " + n.title() + ", even while playing a character."


# -- the switch ----------------------------------------------------------------------------------
def on_activity(reason: str) -> None:
    """Bridge subscriber: route a wake word to its assistant."""
    if reason != "wake_word":
        return
    import reachy_wake
    gate = reachy_wake.GATE
    target = assistant_for_word(getattr(gate, "last_word", "") or "")
    if target is None or target == _active or target == _switching:
        return
    stream = reachy_bridge.stream()
    loop = getattr(stream, "_asyncio_loop", None)
    if stream is None or loop is None:
        return
    gate.hold()                                        # called on the app loop, inside Gate.process
    loop.create_task(switch_to(target))


async def switch_to(name: str) -> None:
    """Restart the session as `name`; the greeting hook replays its history and releases the held audio."""
    global _switching
    import reachy_wake
    _switching = name
    started = time.monotonic()
    try:
        try:
            import reachy_reader
            reachy_reader.READER.stop()
        except Exception:
            logger.debug("reader stop before switch failed", exc_info=True)
        stream = reachy_bridge.stream()
        stream.clear_audio_queue()                     # cut off the current assistant mid-sentence
        _set_active(name)
        stream._voice_override = voice_for(name)
        logger.info("Switching to %s (profile %s)", name, profile_for(name))
        await _orig_apply_personality(stream, profile_for(name))
        deadline = started + HOLD_TIMEOUT_S
        while reachy_wake.GATE is not None and reachy_wake.GATE.holding and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        if reachy_wake.GATE is not None and reachy_wake.GATE.holding:
            logger.warning("Switch to %s didn't finish in %.0f s; releasing the held audio", name, HOLD_TIMEOUT_S)
            reachy_wake.GATE.release()
        else:
            logger.info("Now talking to %s (switch took %.2f s)", name, time.monotonic() - started)
    except Exception:
        logger.exception("Switch to %s failed", name)
        if reachy_wake.GATE is not None:
            reachy_wake.GATE.release()
    finally:
        _switching = None


# -- patches -------------------------------------------------------------------------------------
_orig_apply_personality: Any = None


def install(registry_path: Path = REGISTRY) -> None:
    """Load the registry, pick the boot assistant and patch the app (call before the app starts)."""
    global _registry, _web, _installed, _orig_apply_personality
    if _installed:
        return
    from reachy_mini_conversation_app import config, console, memory, huggingface_realtime as hr

    _registry = load_registry(registry_path)
    _web = str(config.config.REACHY_MINI_CUSTOM_PROFILE or os.environ.get("REACHY_MINI_CUSTOM_PROFILE", "")).endswith("_web")
    last = _read_json(STATE / "active.json", {}).get("active")
    _set_active(last if last in _registry["assistants"] else _registry["default"])
    config.set_custom_profile(profile_for())
    _installed = True

    # memory: per assistant
    memory.memory_path_for_instance = lambda instance_path=None: folder() / memory.MEMORY_FILENAME

    # instructions: say who the assistant is (also while restyled as a persona)
    orig_instructions = hr.get_session_instructions

    def get_session_instructions(*args: Any, **kwargs: Any) -> str:
        return identity_note() + "\n\n" + orig_instructions(*args, **kwargs)
    hr.get_session_instructions = get_session_instructions

    # new session: replay the assistant's history; greet only on boot / restyle, not on a wake-word switch
    orig_greeting = hr.HuggingFaceRealtimeHandler._send_startup_greeting_prompt

    async def _send_startup_greeting_prompt(self: Any) -> None:
        global _skip_greeting_reply
        if self._startup_greeting_sent or not self.connection:
            return await orig_greeting(self)
        items = replay_items()
        for item in items:
            try:
                await self.connection.conversation.item.create(item=item)
            except Exception as e:
                logger.warning("History replay stopped: %r", e)
                break
        voice = voice_for()
        if voice and self.get_current_voice() != voice:
            try:
                await self.change_voice(voice)
            except Exception as e:
                logger.debug("voice %s not applied: %r", voice, e)
        logger.info("Session for %s: replayed %d items", _active, len(items))
        if _switching:
            self._startup_greeting_sent = True
            import reachy_wake
            if reachy_wake.GATE is not None:
                reachy_wake.GATE.release()
            return None
        _skip_greeting_reply = True
        return await orig_greeting(self)
    hr.HuggingFaceRealtimeHandler._send_startup_greeting_prompt = _send_startup_greeting_prompt

    # restyle (switch_persona, the UI's persona picker): saved for the active assistant
    _orig_apply_personality = console.LocalStream.apply_personality

    async def apply_personality(self: Any, profile: str | None) -> str:
        base = spec()["profile"]
        chosen = (profile or "").removesuffix("_web")
        set_style(profile="" if chosen in ("", base) else chosen, voice="")
        self._voice_override = None                    # a persona brings its own voice
        return await _orig_apply_personality(self, profile_for())
    console.LocalStream.apply_personality = apply_personality

    orig_change_voice = console.LocalStream.change_voice

    async def change_voice(self: Any, voice: str) -> str:
        result = await orig_change_voice(self, voice)
        if not str(result).lower().startswith("failed"):
            set_style(voice=voice)
        return result
    console.LocalStream.change_voice = change_voice

    # history capture
    orig_dispatch = console.LocalStream._dispatch_transcript

    def _dispatch_transcript(self: Any, role: str, text: str, final: bool) -> None:
        global _skip_greeting_reply
        orig_dispatch(self, role, text, final)
        if not final:
            return
        try:
            if role == "assistant":
                if _skip_greeting_reply:
                    _skip_greeting_reply = False
                    return
                import reachy_reader
                if reachy_reader.READER.reading:
                    return                              # book passages aren't conversation
            elif role == "user":
                _skip_greeting_reply = False
            record(role, text)
        except Exception:
            logger.debug("history capture failed", exc_info=True)
    console.LocalStream._dispatch_transcript = _dispatch_transcript

    reachy_bridge.subscribe(on_activity)
    logger.info("Assistants: %s; active %s (%s)", ", ".join(names()), _active, profile_for())
