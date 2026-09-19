"""Named lists (shopping, to-do, notes, ...) for the `lists` tool, stored in local_backend/state/lists.json.

Normally imported (not re-executed on profile reloads), thread-safe, atomic writes. List names are
normalised ("Shopping List" -> "shopping", "to do" / "todos" -> "todo").
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

STATE_FILE = Path(os.environ.get("REACHY_LISTS_FILE", Path(__file__).parent / "state" / "lists.json"))
_lock = threading.Lock()
_ALIASES = {"to do": "todo", "to-do": "todo", "todos": "todo", "to dos": "todo", "tasks": "todo", "task": "todo",
            "groceries": "shopping", "grocery": "shopping", "grocery list": "shopping", "shopping list": "shopping",
            "note": "notes"}


def list_key(name: str) -> str:
    n = " ".join((name or "").strip().lower().split())
    n = re.sub(r"^(my|the|our)\s+", "", n)
    n = _ALIASES.get(n, n)
    n = re.sub(r"(^|\s+)list$", "", n)             # "my list" -> "" -> the default list
    return _ALIASES.get(n, n) or "notes"


def _load() -> dict[str, list[str]]:
    try:
        data = json.loads(STATE_FILE.read_text())
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
        return data
    except FileNotFoundError:
        return {}
    except ValueError:   # corrupt file: keep a copy, start empty rather than failing every action
        backup = STATE_FILE.with_suffix(f".corrupt-{int(time.time())}.json")
        STATE_FILE.replace(backup)
        return {}


def _save(data: dict[str, list[str]]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    tmp.replace(STATE_FILE)


def add(name: str, items: list[str]) -> tuple[str, list[str], list[str]]:
    """Returns (list, added, already_there). Case-insensitive duplicates are skipped."""
    key = list_key(name)
    with _lock:
        data = _load()
        lst = data.setdefault(key, [])
        added, dup = [], []
        for it in (i.strip() for i in items if i and i.strip()):
            (dup if it.lower() in (x.lower() for x in lst) else added).append(it)
            if it not in dup:
                lst.append(it)
        _save(data)
    return key, added, dup


def remove(name: str, words: str) -> tuple[str, list[str]]:
    """Remove the item matching `words`: exact (or singular/plural) match first, then whole words, then substring."""
    key, w = list_key(name), (words or "").strip().lower()
    with _lock:
        data = _load()
        lst = data.get(key, [])
        forms = {w, w + "s", w + "es", w.rstrip("s")} | ({w[:-2]} if w.endswith("es") else set())
        exact = [x for x in lst if w and x.lower() in forms]
        whole = [x for x in lst if w and re.search(rf"\b{re.escape(w)}(?:e?s)?\b", x.lower())]
        gone = exact or whole or [x for x in lst if w and w in x.lower()]
        data[key] = [x for x in lst if x not in gone]
        if not data[key]:
            data.pop(key, None)
        _save(data)
    return key, gone


def read(name: str) -> tuple[str, list[str]]:
    key = list_key(name)
    with _lock:
        return key, list(_load().get(key, []))


def clear(name: str) -> tuple[str, int]:
    key = list_key(name)
    with _lock:
        data = _load()
        n = len(data.pop(key, []))
        _save(data)
    return key, n


def lists() -> dict[str, int]:
    with _lock:
        return {k: len(v) for k, v in _load().items()}
