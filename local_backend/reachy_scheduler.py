"""Reminder scheduler shared by the set_reminder / list_reminders / cancel_reminder tools.

Why not just let a tool sleep until the reminder is due? The conversation app only sends a tool's
result to the model when the tool finishes, and the speech server refuses new responses while a
tool result is pending, so a sleeping tool would freeze the conversation. Instead the tools return
immediately and this module's background thread fires each reminder by calling the app's own
JSON-RPC method `conversation.say` on ws://127.0.0.1:7860/rpc (served when the app runs with --ui).
The model then speaks the reminder in its own voice.

Reminders are kept in local_backend/state/reminders.json, so they survive an app restart. With several
assistants (--assistants), each reminder has an "owner"; an assistant only lists and cancels its own, and a
reminder that comes due while another assistant is active is announced as its owner's. One that
came due while the app was down is still announced if it is less than an hour late; older ones are
dropped (and logged).

Each reminder can ring a sound first (reachy_sounds: "chime" by default, "timer", "alarm", any
sound name, or "none"). An alarm repeats until stop_sound is called or ALARM_SECONDS pass; then
the robot speaks. If sound playback isn't available (no GStreamer bindings), it just speaks.

Settings: REACHY_APP_RPC_URL (default ws://127.0.0.1:7860/rpc), REACHY_REMINDERS_FILE,
REACHY_ALARM_SECONDS (default 30).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import websockets

logger = logging.getLogger("reachy_scheduler")

STATE_FILE = Path(os.environ.get("REACHY_REMINDERS_FILE", Path(__file__).parent / "state" / "reminders.json"))
RPC_URL = os.environ.get("REACHY_APP_RPC_URL", "ws://127.0.0.1:7860/rpc")
LATE_GRACE = timedelta(hours=1)
FIRE_PREFIX = "(Reminder due now"
ALARM_SECONDS = float(os.environ.get("REACHY_ALARM_SECONDS", 30))


def now() -> datetime:
    return datetime.now().astimezone()


def spoken_time(t: datetime) -> str:
    return t.strftime("%I:%M %p").lstrip("0")


def spoken_delta(d: timedelta) -> str:
    s = max(0, int(round(d.total_seconds())))
    if s < 90:
        return f"{s} seconds"
    m = round(s / 60)
    if m < 90:
        return f"{m} minutes"
    h, m = divmod(m, 60)
    return f"{h} hours" + (f" {m} minutes" if m else "")


_AT = re.compile(r"^\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?\s*$", re.I)


def parse_due(in_minutes: Any = None, at: Any = None, *, ref: datetime | None = None) -> datetime:
    """Due time from "in N minutes" or a clock time ("17:30", "5:30 pm", "5pm", "noon").

    A clock time that has already passed today means tomorrow.
    """
    ref = ref or now()
    if in_minutes not in (None, "") and at not in (None, ""):
        raise ValueError("Give either in_minutes or at, not both.")
    if in_minutes not in (None, ""):
        minutes = float(in_minutes)
        if not 0 < minutes <= 7 * 24 * 60:
            raise ValueError("in_minutes must be between 0 and one week.")
        return ref + timedelta(minutes=minutes)
    if at in (None, ""):
        raise ValueError("Say when: in_minutes or at.")
    text = str(at).strip().lower()
    if text in ("noon", "midday"):
        hour, minute = 12, 0
    elif text == "midnight":
        hour, minute = 0, 0
    else:
        m = _AT.match(text)
        if not m:
            raise ValueError(f"Couldn't understand the time {at!r}; use e.g. '17:30' or '5:30 pm'.")
        hour, minute, ampm = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").replace(".", "")
        if ampm:
            if not 1 <= hour <= 12:
                raise ValueError(f"Invalid time {at!r}.")
            hour = hour % 12 + (12 if ampm == "pm" else 0)
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError(f"Invalid time {at!r}.")
    due = ref.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return due if due > ref else due + timedelta(days=1)


async def rpc_say(text: str, url: str = RPC_URL, timeout: float = 10) -> None:
    """Call the app's `conversation.say` JSON-RPC method and wait for its reply."""
    req_id = uuid.uuid4().hex
    async with websockets.connect(url, open_timeout=5) as ws:
        await ws.send(json.dumps({"jsonrpc": "2.0", "id": req_id, "method": "conversation.say", "params": {"text": text}}))
        async with asyncio.timeout(timeout):
            while True:  # the server also pushes notifications to every client; wait for our reply
                msg = json.loads(await ws.recv())
                if msg.get("id") == req_id:
                    if "error" in msg:
                        raise RuntimeError(msg["error"].get("message", str(msg["error"])))
                    return


async def rpc_reachable(url: str = RPC_URL) -> bool:
    try:
        async with websockets.connect(url, open_timeout=2):
            return True
    except (OSError, websockets.exceptions.WebSocketException, asyncio.TimeoutError):
        return False


def _owner() -> str:
    try:
        import reachy_assistants
        return reachy_assistants.active() if reachy_assistants.enabled() else ""
    except ImportError:
        return ""


class Scheduler:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._items: list[dict[str, Any]] = self._load()

    # -- persistence ------------------------------------------------------------------------
    def _load(self) -> list[dict[str, Any]]:
        try:
            return json.loads(STATE_FILE.read_text())
        except FileNotFoundError:
            return []
        except (OSError, ValueError) as e:
            logger.warning("Couldn't read %s (%r); starting with no reminders", STATE_FILE, e)
            return []

    def _save(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._items, indent=1))
        tmp.replace(STATE_FILE)

    # -- public API (thread-safe) -------------------------------------------------------------
    def ensure_started(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="reachy-scheduler", daemon=True)
                self._thread.start()

    def add(self, message: str, due: datetime, sound: str = "chime") -> dict[str, Any]:
        item = {"id": uuid.uuid4().hex[:6], "message": message.strip(), "due": due.isoformat(),
                "created": now().isoformat(), "sound": (sound or "none").strip().lower()}
        if _owner():
            item["owner"] = _owner()
        with self._lock:
            self._items.append(item)
            self._items.sort(key=lambda x: x["due"])
            self._save()
        self._wake.set()
        self.ensure_started()
        return item

    def _mine(self, item: dict[str, Any]) -> bool:
        return item.get("owner", "") in ("", _owner())

    def pending(self) -> list[dict[str, Any]]:
        with self._lock:
            return [x for x in self._items if self._mine(x)]

    def cancel(self, which: str = "", all_: bool = False) -> list[dict[str, Any]]:
        which = which.strip().lower()
        with self._lock:
            if all_:
                removed = [x for x in self._items if self._mine(x)]
            else:
                removed = [x for x in self._items if self._mine(x) and which
                           and (x["id"] == which or which in x["message"].lower())]
            self._items = [x for x in self._items if x not in removed]
            if removed:
                self._save()
        self._wake.set()
        return removed

    # -- background thread --------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            with self._lock:
                nxt = self._items[0] if self._items else None
            if nxt is None:
                self._wake.wait(timeout=60)
                self._wake.clear()
                continue
            due = datetime.fromisoformat(nxt["due"])
            wait = (due - now()).total_seconds()
            if wait > 0:
                self._wake.wait(timeout=min(wait, 30))  # re-check at least every 30 s (clock changes, edits)
                self._wake.clear()
                continue
            with self._lock:
                if nxt not in self._items:
                    continue  # cancelled meanwhile
                self._items.remove(nxt)
                self._save()
            late = now() - due
            if late > LATE_GRACE:
                logger.warning("Dropping reminder %s (%r): %s late", nxt["id"], nxt["message"], spoken_delta(late))
                continue
            self._fire(nxt, late)

    def _ring(self, sound: str) -> None:
        """Play the reminder's sound and wait for it (an alarm rings until stopped or ALARM_SECONDS)."""
        if sound in ("", "none"):
            return
        try:
            import reachy_sounds  # needs GStreamer bindings; imported lazily so the scheduler works without them
        except Exception as e:
            logger.warning("No sound for reminders (%r); speaking only", e)
            return
        alarm = sound == "alarm"
        result = reachy_sounds.PLAYER.play(sound, repeat=0 if alarm else 1,
                                           max_seconds=ALARM_SECONDS if alarm else 60, wait=True)
        if "error" in result:
            logger.warning("Reminder sound %r: %s", sound, result["error"])

    def _fire(self, item: dict[str, Any], late: timedelta) -> None:
        self._ring(item.get("sound", "chime"))
        due = datetime.fromisoformat(item["due"])
        when = f", it was due at {spoken_time(due)}" if late > timedelta(minutes=2) else ""
        owner = item.get("owner", "")
        if owner and owner != _owner():
            when += f"; {owner.title()} set it, so say it's {owner.title()}'s reminder"
        text = f"{FIRE_PREFIX}{when}) {item['message']}"
        for attempt in range(1, 4):
            try:
                asyncio.run(rpc_say(text))
                logger.info("Reminder %s fired: %r", item["id"], item["message"])
                return
            except Exception as e:  # app busy, restarting, or started without --ui
                logger.warning("Reminder %s: conversation.say failed (attempt %d/3): %r", item["id"], attempt, e)
                threading.Event().wait(5)
        logger.error("Reminder %s (%r) could not be delivered", item["id"], item["message"])


SCHEDULER = Scheduler()
