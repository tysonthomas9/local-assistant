"""Book reader for the `read_book` tool: library, chapters, passages, bookmarks and paced reading.

Library: local_backend/books/*.txt (UTF-8). Project Gutenberg texts are cleaned (licence header and
footer removed), chapter headings are detected ("CHAPTER IV. ..." -> "Chapter 4. ..."; a table of
contents is skipped because its lines don't stand alone), and the text is split into ~110-word
passages on sentence boundaries. Bookmarks live in local_backend/state/books.json.

Reading: each passage is sent as an *out-of-band* response (`conversation: "none"`) through the
app's own response queue, with instructions to read it word for word. Measured on this stack: word
error rate 0.000 on a 148-word Alice passage, first audio 0.6 s, and the passage stays out of the
conversation history. Out-of-band responses also skip the speech server's "tool result pending"
check, so reading doesn't block the conversation. Pacing uses the bridge's playback clock. Any
barge-in ("interrupted" / user speech / an in-process "Hey Jarvis" detection) pauses the reader; the
bookmark then points at the passage that was playing, which is re-read on resume (or at the next one,
if the interruption came in the pause after a passage). A passage is only
sent when no in-band response is active or queued, so it can never be queued behind the user's turn.

Pauses between passages: measured live, the robot's audio board suppresses the microphone while the
robot speaks (mic level -39.9 dBFS during playback vs -35.1 in a quiet room: its echo cancellation
removes the robot's own voice, but also the user's). With back-to-back passages the user could never
be heard, so "Hey Jarvis, stop" failed. The reader therefore waits until a passage has finished
playing, then pauses GAP_S (plus ~0.6 s until the next passage's first audio) before continuing; the
user can speak in those pauses. Stopping via the tool cancels the passage on the speech server and
flushes local audio if a passage is still active there (otherwise the cancel would hit the model's
own reply).

Downloads (web profile): Gutenberg's own OPDS search feed + /ebooks/<id>.txt.utf-8.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import reachy_bridge

logger = logging.getLogger("reachy_reader")

HERE = Path(__file__).resolve().parent
BOOKS = Path(os.environ.get("REACHY_BOOKS_DIR", HERE / "books"))
BOOKMARKS = Path(os.environ.get("REACHY_BOOKMARKS_FILE", HERE / "state" / "books.json"))


def _bookmarks() -> Path:
    """This file, or the active assistant's own bookmarks with --assistants (reachy_assistants)."""
    try:
        import reachy_assistants
        return reachy_assistants.data_file("books.json", BOOKMARKS)
    except ImportError:
        return BOOKMARKS
PASSAGE_WORDS = 110   # ~40 s of speech, so there is a pause for the user at least that often
LEAD_S = 0.0          # send the next passage only when the current one has finished playing...
GAP_S = 1.5           # ...plus this pause, during which the microphone is not suppressed
GEN_TIMEOUT_S = 90
READ_INSTRUCTIONS = (
    "You are a narrator reading a book aloud. Read the text inside <passage> exactly as written, word for word, "
    "from the first word to the last. Do not add, skip, summarise or comment. Output only the passage text."
)
OPDS_SEARCH = "https://www.gutenberg.org/ebooks/search.opds/"
TEXT_URL = "https://www.gutenberg.org/ebooks/{id}.txt.utf-8"
ATOM = "{http://www.w3.org/2005/Atom}"

_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
_ROMAN_OK = re.compile(r"^m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$", re.I)
_NUMBER_WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
                 "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
# "CHAPTER IV.", "Chapter 12", "BOOK ONE", "Chapter the First", optionally followed by a title. The number must be a
# real number (valid roman numeral, digits or a number word), so "Part of the reason..." or "Chapter Mix" don't match,
# and a title after it starts with a capital (or punctuation), so "Book one was better than..." isn't a heading.
_HEADING = re.compile(r"^(chapter|book|part|stave|letter)\s+(?:the\s+)?([ivxlcdm]+|\d+|[a-z]+(?:-[a-z]+)?)\b\.?:?\s*"
                      r"(|(?-i:[^a-z\s]).{0,79})$", re.I)


def _roman(s: str) -> int | None:
    s = s.lower()
    if not s or not _ROMAN_OK.match(s):
        return None
    total = 0
    for a, b in zip(s, s[1:] + " "):
        v = _ROMAN[a]
        total += -v if b != " " and _ROMAN.get(b, 0) > v else v
    return total


def _heading_number(token: str) -> int | None:
    t = token.lower()
    if t.isdigit():
        return int(t)
    if t in _NUMBER_WORDS:
        return _NUMBER_WORDS.index(t) + 1
    tens, _, units = t.partition("-")               # "twenty", "Twenty-One"
    if tens in _TENS and (not units or units in _NUMBER_WORDS[:9]):
        return 20 + 10 * _TENS.index(tens) + (_NUMBER_WORDS.index(units) + 1 if units else 0)
    if t in _ORDINALS:
        return _ORDINALS.index(t) + 1
    return _roman(t) if token.isupper() else None   # headings write numerals in capitals ("Chapter IX"), not "Mix"


# -- parsing -------------------------------------------------------------------------------------
@dataclass
class Book:
    key: str                       # file stem
    title: str
    chunks: list[str] = field(default_factory=list)
    chunk_chapter: list[int] = field(default_factory=list)
    chapters: list[tuple[str, int]] = field(default_factory=list)   # (title, first chunk)


def _strip_gutenberg(text: str) -> tuple[str, str | None]:
    title = None
    m = re.search(r"^Title:\s*(.+)$", text, re.M)
    if m:
        title = m.group(1).strip()
    start = re.search(r"^\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", text, re.M | re.I)
    if start:
        text = text[start.end():]
    end = re.search(r"^\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", text, re.M | re.I)
    if end:
        text = text[:end.start()]
    return text, title


def _sentences(paragraph: str) -> list[str]:
    """Split after . ! ? (plus closing quotes/brackets) when the next word starts a sentence."""
    parts = re.split(r"([.!?][\"'”’)]*)\s+(?=[\"'“‘(]?[A-Z])", paragraph)
    out = ["".join(parts[i:i + 2]) for i in range(0, len(parts), 2)]
    return [x for x in out if x.strip()]


def parse(path: Path) -> Book:
    raw = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    body, title = _strip_gutenberg(raw)
    body = re.sub(r"\[Illustration[^\]]*\]", " ", body)   # Gutenberg picture markers
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", body) if p.strip()]
    book = Book(key=path.stem, title=title or path.stem.replace("_", " ").title())
    buf: list[str] = []
    words = 0
    chapter = -1

    def flush() -> None:
        nonlocal buf, words
        if buf:
            book.chunks.append(" ".join(buf))
            book.chunk_chapter.append(max(chapter, 0))
            buf, words = [], 0

    for p in paragraphs:
        m = _HEADING.match(p)
        num = _heading_number(m.group(2)) if m else None
        # A heading names one chapter; a paragraph naming several is a table of contents.
        if m and num is not None and len(p) < 100 and len(re.findall(
                r"\b(?:chapter|book|part|stave|letter)\s+(?:[IVXLCDM]+\b|\d+)", p, re.I)) <= 1:
            flush()
            label = f"{m.group(1).title()} {num}"
            heading = label + (f". {m.group(3).strip().rstrip('.')}" if m.group(3).strip() else "")
            chapter += 1
            book.chapters.append((heading, len(book.chunks)))
            buf, words = [heading + "."], len(heading.split())
            continue
        for sentence in _sentences(p) if len(p.split()) > PASSAGE_WORDS else [p]:
            n = len(sentence.split())
            if words and words + n > PASSAGE_WORDS:
                flush()
            buf.append(sentence)
            words += n
    flush()
    if not book.chapters and book.chunks:
        book.chapters.append(("Beginning", 0))
    return book


# -- library and bookmarks -------------------------------------------------------------------------
def library() -> list[Path]:
    BOOKS.mkdir(parents=True, exist_ok=True)
    return sorted(p for p in BOOKS.glob("*.txt"))


def _title_of(path: Path) -> str:
    head = path.read_text(encoding="utf-8", errors="replace")[:3000]
    m = re.search(r"^Title:\s*(.+)$", head, re.M)
    return m.group(1).strip() if m else path.stem.replace("_", " ").title()


_STOP = {"the", "a", "an", "of", "book", "novel", "story", "stories", "by", "please", "read", "me", "and", "in", "to"}


def _words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower().replace("’", "'").replace("'s", "")).split()


def find(query: str) -> Path | None:
    """Best local match for a title query: share of the query's key words found in the title, else fuzzy."""
    q = [w for w in _words(query or "") if w not in _STOP]
    best, score = None, 0.0
    for p in library():
        t = _words(_title_of(p))
        if not q:
            break
        overlap = sum(w in t for w in q) / len(q)
        s = max(overlap, SequenceMatcher(None, " ".join(q), " ".join(t)).ratio())
        if s > score:
            best, score = p, s
    return best if score >= 0.6 else None


def _load_marks() -> dict[str, Any]:
    try:
        return json.loads(_bookmarks().read_text())
    except (FileNotFoundError, ValueError):
        return {}


_marks_lock = threading.Lock()


def save_mark(key: str, chunk: int) -> None:
    with _marks_lock:
        _save_mark_locked(key, chunk)


def _save_mark_locked(key: str, chunk: int) -> None:
    marks = _load_marks()
    marks[key] = {"chunk": chunk, "updated": datetime.now().isoformat(timespec="seconds")}
    marks["_last"] = key
    _bookmarks().parent.mkdir(parents=True, exist_ok=True)
    tmp = _bookmarks().with_suffix(".tmp")
    tmp.write_text(json.dumps(marks, indent=1))
    tmp.replace(_bookmarks())


def bookmark(key: str) -> int:
    return int(_load_marks().get(key, {}).get("chunk", -1))


def last_book() -> str | None:
    return _load_marks().get("_last")


# -- Gutenberg ------------------------------------------------------------------------------------
def gutenberg_search(query: str, limit: int = 5) -> list[dict[str, str]]:
    import httpx
    r = httpx.get(OPDS_SEARCH, params={"query": query}, timeout=15, follow_redirects=True,
                  headers={"User-Agent": "reachy-mini-reader/1.0"})
    r.raise_for_status()
    out = []
    for e in ET.fromstring(r.text).findall(ATOM + "entry"):
        m = re.search(r"/ebooks/(\d+)\.opds$", e.findtext(ATOM + "id") or "")
        if m:
            out.append({"id": m.group(1), "title": " ".join((e.findtext(ATOM + "title") or "").split()),
                        "author": " ".join((e.findtext(ATOM + "content") or "").split())})
    return out[:limit]


def gutenberg_download(query: str) -> Path:
    """Fetch the best Gutenberg match for `query` as plain text into the library."""
    import httpx
    for hit in gutenberg_search(query):
        r = httpx.get(TEXT_URL.format(id=hit["id"]), timeout=30, follow_redirects=True,
                      headers={"User-Agent": "reachy-mini-reader/1.0"})
        if r.status_code != 200 or "START OF" not in r.text[:20000]:
            continue  # no plain-text edition (e.g. an illustration collection)
        name = re.sub(r"[^a-z0-9]+", "_", hit["title"].lower()).strip("_")[:60] or f"gutenberg_{hit['id']}"
        path = BOOKS / f"{name}.txt"
        BOOKS.mkdir(parents=True, exist_ok=True)
        path.write_text(r.text, encoding="utf-8")
        logger.info("Downloaded Gutenberg #%s %r to %s", hit["id"], hit["title"], path)
        return path
    raise LookupError(f"No plain-text book found on Project Gutenberg for {query!r}.")


# -- reading --------------------------------------------------------------------------------------
class Reader:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._gen = 0
        self.book: Book | None = None
        self.pos = 0
        self.reading = False
        self._inflight = False     # a passage response is active on the server (created, not yet done)

    def _send(self, text: str) -> None:
        s = reachy_bridge.stream()
        response = {"conversation": "none", "instructions": READ_INSTRUCTIONS, "output_modalities": ["audio"],
                    "input": [{"type": "message", "role": "user",
                               "content": [{"type": "input_text", "text": f"<passage>{text}</passage>"}]}]}
        reachy_bridge.run_in_app_loop(lambda: s.handler._safe_response_create(response=response))

    def start(self, book: Book, pos: int) -> None:
        with self._lock:
            self._gen += 1
            gen = self._gen
            self.book, self.pos, self.reading = book, pos, True
        save_mark(book.key, pos)
        threading.Thread(target=self._run, args=(gen,), daemon=True, name="reachy-reader").start()

    def stop(self) -> bool:
        with self._lock:
            was = self.reading
            self._gen += 1
            self.reading = False
            book, pos = self.book, self.pos
        if was and book is not None:
            save_mark(book.key, pos)
        return was

    async def stop_now(self) -> bool:
        """Stop from the app's event loop (the read_book tool): also cancel and flush a passage in progress.

        Only when a passage is really active on the server: the tool runs inside the model's own response, and
        the server cancels whichever response is active, so an unconditional cancel would cut off the model's
        "Okay" and drop its tool bookkeeping. (While a passage is active no in-band response can be.)"""
        inflight = self._inflight
        was = self.stop()
        if was and inflight:
            s = reachy_bridge.stream()
            try:
                await s.handler.connection.response.cancel()
            except Exception as e:  # nothing active, or connection gone
                logger.debug("response.cancel: %r", e)
            try:
                s.clear_audio_queue()
            except Exception as e:
                logger.debug("clear_audio_queue: %r", e)
        return was

    def _run(self, gen: int) -> None:
        interrupted = threading.Event()
        transcript_done = threading.Event()

        sent = threading.Event()

        def on_event(reason: str) -> None:
            # "wake_word" is detected in-process, before the speech server has even heard the user, so it
            # pauses the reader before the next passage can be queued behind the user's turn.
            if reason in ("interrupted", "user_speech_started", "wake_word"):
                self._inflight = False
                interrupted.set()
            elif reason == "response_created" and sent.is_set():
                self._inflight = True
            elif reason == "assistant_transcript_done":
                self._inflight = False
                sent.clear()
                transcript_done.set()

        def conversation_idle() -> bool:
            """No in-band response active or queued (the app sends queued responses one at a time)."""
            h = getattr(reachy_bridge.stream(), "handler", None)
            try:
                return h._response_done_event.is_set() and h._pending_responses.empty()
            except AttributeError:
                return True

        # Let the tool's spoken confirmation finish first.
        reachy_bridge.wait_for({"assistant_transcript_done"}, 8)
        time.sleep(0.3)
        unsubscribe = reachy_bridge.subscribe(on_event)
        try:
            while True:
                with self._lock:
                    if gen != self._gen or self.book is None:
                        return
                    book, pos = self.book, self.pos
                if pos >= len(book.chunks):
                    with self._lock:
                        if gen == self._gen:
                            save_mark(book.key, -1)   # finished: next time starts over
                    logger.info("Finished reading %s", book.title)
                    break
                t0 = time.monotonic()
                while not conversation_idle() and not interrupted.is_set() and gen == self._gen:
                    if time.monotonic() - t0 > GEN_TIMEOUT_S:
                        logger.warning("Conversation busy for %d s; pausing", GEN_TIMEOUT_S)
                        interrupted.set()
                    time.sleep(0.1)
                if interrupted.is_set() or gen != self._gen:
                    if gen == self._gen:
                        logger.info("Reading paused at passage %d of %s", pos, book.key)
                    break                     # the bookmark already points at this unsent passage
                transcript_done.clear()
                sent.set()
                self._send(book.chunks[pos])
                t0 = time.monotonic()
                while not transcript_done.is_set() and not interrupted.is_set() and gen == self._gen:
                    if time.monotonic() - t0 > GEN_TIMEOUT_S:
                        # e.g. the backend restarted (persona switch) or cancelled the passage: don't skip it
                        logger.warning("Passage %d: no transcript after %d s; pausing", pos, GEN_TIMEOUT_S)
                        interrupted.set()
                        break
                    time.sleep(0.1)
                while (not interrupted.is_set() and gen == self._gen
                       and reachy_bridge.audio_seconds_left() > LEAD_S):
                    time.sleep(0.2)
                played = not interrupted.is_set() and gen == self._gen
                if played:
                    interrupted.wait(GAP_S)   # a pause the user can speak into (see module docstring)
                if gen != self._gen:
                    break                     # stopped or restarted: stop()/start() own the bookmark
                if interrupted.is_set():
                    # cut off mid-passage: re-read it on resume; interrupted in the pause after it: go on from the next
                    resume_at = pos + 1 if played else pos
                    with self._lock:
                        if gen == self._gen:
                            self.pos = resume_at
                            save_mark(book.key, resume_at)
                    logger.info("Reading paused at passage %d of %s", resume_at, book.key)
                    break
                with self._lock:
                    if gen != self._gen:
                        break
                    self.pos = pos + 1
                    save_mark(book.key, pos + 1)
        finally:
            unsubscribe()
            with self._lock:
                if gen == self._gen:
                    self.reading = self._inflight = False

    def status(self) -> dict[str, Any]:
        with self._lock:
            b, pos, reading = self.book, self.pos, self.reading
        if b is None:
            return {"reading": False}
        ch = b.chunk_chapter[min(pos, len(b.chunks) - 1)] if b.chunks else 0
        return {"reading": reading, "book": b.title, "chapter": b.chapters[ch][0] if b.chapters else "",
                "progress_percent": round(100 * pos / max(1, len(b.chunks)))}


READER = Reader()
