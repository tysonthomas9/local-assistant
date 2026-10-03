"""Text-to-speech for MuseHandler: Qwen3-TTS (default), Kokoro-82M or macOS's built-in `say`.

All three turn a reply into 16-bit mono PCM at the robot speaker's rate, which the app plays as is.

Qwen3-TTS (`MUSE_TTS=qwen3`, run_app.py --tts qwen3): Qwen3-TTS 1.7B CustomVoice (8-bit, MLX),
run by qwen3_worker.py in the Kokoro venv (mlx-audio). The whole reply is rendered in one
streaming call, and each ~0.5 s chunk is resampled (with carried-over context, so there are no
clicks at chunk edges) and handed over as soon as it arrives: the robot starts talking after
about 0.3 s. MUSE_TTS_VOICE picks the speaker (default Aiden). The style instruction (how to say
it, never what to say) comes from MUSE_TTS_INSTRUCT, or the file MUSE_TTS_INSTRUCT_FILE names;
the default is QWEN3_INSTRUCT, and an empty value means none. If the worker can't start, the run
uses Kokoro (and `say` if Kokoro can't start either); if Qwen3 fails before a reply's first
audio, `say` speaks that reply.

Kokoro (`MUSE_TTS=kokoro`, run_app.py --tts kokoro): Kokoro-82M on MLX, in its own venv
(~/assistant-edge/muse-app/kokoro/.venv, see install.sh) as a long-lived worker process
(kokoro_worker.py) that loads the model once and warms it up. A reply is split into sentences and
each one is rendered, trimmed of Kokoro's ~0.3 s of leading silence, resampled from 24 kHz to the
speaker's rate and handed over as soon as it is ready, so playback starts after the first
sentence. If the worker can't start, the whole run uses `say`; if it fails mid-reply, `say`
speaks the rest of that reply. MUSE_TTS_VOICE picks the voice (default af_heart).

`say` (`MUSE_TTS=say`): `say -o reply.wav --file-format=WAVE --data-format=LEI16@<rate>` writes
the audio without playing it, so the app can push it to the robot's speaker (and wobble the head)
itself. MUSE_TTS_VOICE or MUSE_SAY_VOICE picks a voice (`say -v '?'` lists them); empty means the
system voice.

Reply text never appears on a command line or in a log line.
"""

from __future__ import annotations

import json
import logging
import os
import re
import select
import subprocess
import sys
import tempfile
import threading
import time
import wave
from math import gcd
from pathlib import Path
from typing import Iterator, Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)

SAY = "/usr/bin/say"
KOKORO_VOICE = "af_heart"  # Kokoro's best-rated English voice: warm, clear, natural pacing
KOKORO_VOICES = (  # the English voices in Kokoro-82M v1.0 (a = American, b = British; f/m)
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole", "af_nova",
    "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
    "am_michael", "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
)
QWEN3_SPEAKER = "Aiden"  # natural pace; Ryan comes out 1.5-2.5x slower for the same text
QWEN3_SPEAKERS = ("Aiden", "Ryan", "Eric", "Dylan", "Serena", "Vivian", "Uncle_Fu", "Ono_Anna", "Sohee")
QWEN3_INSTRUCT = "playful and cheeky, like a friendly cartoon robot"  # the user's pick (delivery only)
KOKORO_GAIN = 2.0  # Kokoro renders ~2.3x quieter (RMS) than `say`; this brings it close, peak ~0.8
MAX_SENTENCE_CHARS = 300  # longer sentences are cut at a comma/semicolon/space (Kokoro's limit is 510 phonemes)
_ABBREV = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|St|Prof|Sr|Jr|vs|etc|e\.g|i\.e|a\.m|p\.m|approx|No)\.$", re.IGNORECASE)


def say_command(text_file: str, out: str, sample_rate: int, voice: str | None) -> list[str]:
    cmd = [SAY, "-o", out, "--file-format=WAVE", f"--data-format=LEI16@{sample_rate}"]
    if voice:
        cmd += ["-v", voice]
    return cmd + ["-f", text_file]


def read_wav_int16(path: str | Path) -> tuple[int, np.ndarray]:
    """(sample rate, mono int16 samples) of a 16-bit PCM WAV."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"expected 16-bit PCM, got {8 * w.getsampwidth()}-bit")
        rate, channels = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels)[:, 0].copy()
    return rate, pcm


def synthesize(text: str, sample_rate: int = 16000, voice: str | None = None, timeout_s: float = 60) -> np.ndarray:
    """Render `text` with `say`; return mono int16 PCM at `sample_rate`."""
    voice = voice if voice is not None else (os.environ.get("MUSE_SAY_VOICE") or None)
    with tempfile.TemporaryDirectory(prefix="muse-say-") as tmp:
        text_file = os.path.join(tmp, "text.txt")
        out = os.path.join(tmp, "reply.wav")
        Path(text_file).write_text(text, encoding="utf-8")  # -f: no text on the command line
        subprocess.run(say_command(text_file, out, sample_rate, voice), check=True, timeout=timeout_s,
                       stdin=subprocess.DEVNULL, capture_output=True)
        rate, pcm = read_wav_int16(out)
    if rate != sample_rate:
        raise RuntimeError(f"say wrote {rate} Hz, asked for {sample_rate} Hz")
    return pcm


# ---------------------------------------------------------------------- text and audio helpers
def split_sentences(text: str, max_chars: int = MAX_SENTENCE_CHARS) -> list[str]:
    """Split a reply into sentences (at . ! ? and line breaks; not after Mr./Dr./e.g. ...)."""
    out: list[str] = []
    for line in text.splitlines():
        buf = ""
        for piece in re.split(r"(?<=[.!?])\s+", line.strip()):
            buf = f"{buf} {piece}" if buf else piece
            if not _ABBREV.search(buf):
                out.append(buf)
                buf = ""
        if buf:
            out.append(buf)
    sentences: list[str] = []
    for s in (s.strip() for s in out):
        while len(s) > max_chars:
            cut = max(s.rfind(", ", 0, max_chars), s.rfind("; ", 0, max_chars))
            if cut <= 0:
                cut = s.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars - 1
            sentences.append(s[: cut + 1].strip())
            s = s[cut + 1 :].strip()
        if s:
            sentences.append(s)
    return sentences


def trim_silence(audio: np.ndarray, rate: int, threshold: float = 0.01, lead_s: float = 0.03,
                 tail_s: float = 0.15) -> np.ndarray:
    """Cut silence before the first and after the last sample above `threshold` (keeping a margin)."""
    loud = np.flatnonzero(np.abs(audio) > threshold)
    if loud.size == 0:
        return audio[:0]
    start = max(0, int(loud[0]) - int(lead_s * rate))
    end = min(audio.size, int(loud[-1]) + 1 + int(tail_s * rate))
    return audio[start:end]


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Band-limited resampling (polyphase) of float mono audio."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if src_rate == dst_rate or audio.size == 0:
        return audio
    from scipy.signal import resample_poly

    g = gcd(src_rate, dst_rate)
    return resample_poly(audio, dst_rate // g, src_rate // g).astype(np.float32)


def to_int16(audio: np.ndarray, gain: float = 1.0) -> np.ndarray:
    return (np.clip(np.asarray(audio, dtype=np.float32) * gain, -1.0, 1.0) * 32767).astype(np.int16)


class StreamResampler:
    """resample() for audio that arrives in pieces: each piece is resampled with real samples on
    both sides (CONTEXT input samples kept from before, the last CONTEXT held back until more
    arrive), so the output matches resampling the whole signal at once, without edge clicks."""

    CONTEXT = 600  # input samples (25 ms at 24 kHz); the polyphase filter needs far fewer

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        g = gcd(src_rate, dst_rate)
        self.src_rate, self.dst_rate = src_rate, dst_rate
        self.up, self.down = dst_rate // g, src_rate // g
        self.ctx = self.CONTEXT // self.down * self.down  # whole filter phases
        self._done = np.zeros(0, np.float32)  # already emitted (the last self.ctx samples)
        self._pending = np.zeros(0, np.float32)

    def _run(self, n: int, pending: np.ndarray) -> np.ndarray:
        """Emit the output for the first n samples of `pending` (n a multiple of self.down)."""
        y = resample(np.concatenate([self._done, pending]), self.src_rate, self.dst_rate)
        start = self._done.size * self.up // self.down
        out = y[start : start + n * self.up // self.down]
        self._done = np.concatenate([self._done, pending[:n]])[-self.ctx :] if self.ctx else self._done
        return out

    def feed(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if self.src_rate == self.dst_rate:
            return audio
        self._pending = np.concatenate([self._pending, audio])
        n = (self._pending.size - self.ctx) // self.down * self.down
        if n <= 0:
            return np.zeros(0, np.float32)
        out = self._run(n, self._pending)
        self._pending = self._pending[n:]
        return out

    def flush(self) -> np.ndarray:
        if self.src_rate == self.dst_rate or self._pending.size == 0:
            return np.zeros(0, np.float32)
        n = -(-self._pending.size // self.down) * self.down
        tail = np.concatenate([self._pending, np.zeros(n - self._pending.size, np.float32)])
        out = self._run(n, tail)[: int(round(self._pending.size * self.up / self.down))]
        self._pending = np.zeros(0, np.float32)
        return out


# ---------------------------------------------------------------------- engines
class Tts(Protocol):
    name: str
    voice: Optional[str]

    def chunks(self, text: str, rate: int) -> Iterator[np.ndarray]:
        """int16 mono PCM at `rate`, one chunk per sentence (or one for the whole text)."""
        ...

    def voices(self) -> list[str]: ...

    def close(self) -> None: ...


class SayTts:
    """The whole reply in one `say` call."""

    name = "say"

    def __init__(self, voice: Optional[str] = None) -> None:
        self.voice = voice

    def chunks(self, text: str, rate: int) -> Iterator[np.ndarray]:
        yield synthesize(text, rate, voice=self.voice)

    def voices(self) -> list[str]:
        return []

    def close(self) -> None:
        pass


def child_env(prefix: str = sys.prefix) -> dict[str, str]:
    """os.environ without the parts that point into this venv (as exec_python.py does): the app
    venv's GStreamer bundle sets PYTHONPATH etc. to its own dirs, which the Kokoro venv must not see."""
    env = {}
    for key, value in os.environ.items():
        if prefix in value:
            value = os.pathsep.join(p for p in value.split(os.pathsep) if prefix not in p)
            if not value:
                continue
        env[key] = value
    return env


class KokoroEngine:
    """Client for kokoro_worker.py: start it, wait for `ready`, then one request at a time."""

    def __init__(self, cmd: list[str], start_timeout_s: float = 120.0, synth_timeout_s: float = 60.0) -> None:
        debug = os.environ.get("MUSE_TTS_DEBUG") == "1"
        self.synth_timeout_s = synth_timeout_s
        self._lock = threading.Lock()
        self._buf = b""
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=child_env(),
                                     stderr=None if debug else subprocess.DEVNULL)
        try:
            line = self._read_line(time.monotonic() + start_timeout_s)
            if not line.startswith("ready "):
                raise RuntimeError(f"kokoro worker: {line or 'exited'}")
            self.rate = int(line.split()[1])
        except BaseException:
            self.close()
            raise

    def _read(self, deadline: float) -> bytes:
        fd = self.proc.stdout.fileno()  # type: ignore[union-attr]
        left = deadline - time.monotonic()
        if left <= 0 or not select.select([fd], [], [], left)[0]:
            raise TimeoutError("kokoro worker timed out")
        data = os.read(fd, 1 << 16)
        if not data:
            raise EOFError("kokoro worker exited")
        return data

    def _read_line(self, deadline: float) -> str:
        while b"\n" not in self._buf:
            self._buf += self._read(deadline)
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode(errors="replace").strip()

    def _read_exact(self, n: int, deadline: float) -> bytes:
        while len(self._buf) < n:
            self._buf += self._read(deadline)
        data, self._buf = self._buf[:n], self._buf[n:]
        return data

    def synth(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        """float32 mono audio at self.rate."""
        with self._lock:
            if self.proc.poll() is not None:
                raise EOFError("kokoro worker exited")
            try:
                req = json.dumps({"text": text, "voice": voice, "speed": speed}) + "\n"
                self.proc.stdin.write(req.encode())  # type: ignore[union-attr]
                self.proc.stdin.flush()  # type: ignore[union-attr]
                deadline = time.monotonic() + self.synth_timeout_s
                header = self._read_line(deadline)
                if not header.startswith("ok "):
                    raise RuntimeError(f"kokoro worker: {header}")
                n = int(header.split()[1])
                return np.frombuffer(self._read_exact(4 * n, deadline), dtype="<f4").astype(np.float32)
            except (OSError, EOFError, TimeoutError, ValueError):
                self.close()  # out of step with the worker: don't reuse it
                raise

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()  # type: ignore[union-attr]
                self.proc.wait(3)
            except Exception:
                self.proc.kill()
                self.proc.wait()


class KokoroTts:
    """Kokoro sentence by sentence, with `say` as the fallback for the rest of a failed reply."""

    name = "kokoro"

    def __init__(self, engine: KokoroEngine, voice: str = KOKORO_VOICE, fallback: Optional[Tts] = None,
                 gain: float = KOKORO_GAIN) -> None:
        self.engine = engine
        self.voice = voice
        self.fallback = fallback or SayTts()
        self.gain = gain
        resample(np.zeros(240, np.float32), engine.rate, 16000)  # import scipy.signal now (~0.4 s), not on the first reply

    def chunks(self, text: str, rate: int) -> Iterator[np.ndarray]:
        sentences = split_sentences(text)
        for i, sentence in enumerate(sentences):
            try:
                audio = self.engine.synth(sentence, self.voice or KOKORO_VOICE)
            except Exception as e:
                logger.warning("Kokoro failed (%s); macOS say speaks the rest of this reply", type(e).__name__)
                yield from self.fallback.chunks(" ".join(sentences[i:]), rate)
                return
            audio = trim_silence(audio, self.engine.rate)
            if audio.size:
                yield to_int16(resample(audio, self.engine.rate, rate), self.gain)

    def voices(self) -> list[str]:
        return list(KOKORO_VOICES)

    def close(self) -> None:
        self.engine.close()


class Qwen3Engine(KokoroEngine):
    """Client for qwen3_worker.py (same start-up and framing as Kokoro's, but replies stream)."""

    def stream(self, text: str, speaker: str, instruct: Optional[str]) -> Iterator[np.ndarray]:
        """float32 mono chunks at self.rate, as the worker renders them."""
        with self._lock:
            if self.proc.poll() is not None:
                raise EOFError("qwen3 worker exited")
            finished = False
            try:
                req = json.dumps({"text": text, "speaker": speaker, "instruct": instruct or None}) + "\n"
                self.proc.stdin.write(req.encode())  # type: ignore[union-attr]
                self.proc.stdin.flush()  # type: ignore[union-attr]
                while True:
                    deadline = time.monotonic() + self.synth_timeout_s
                    header = self._read_line(deadline)
                    if header == "end":
                        finished = True
                        return
                    if header.startswith("err"):
                        finished = True  # the worker is ready for the next request
                        raise RuntimeError(f"qwen3 worker: {header}")
                    if not header.startswith("chunk "):
                        raise ValueError("qwen3 worker: bad header")
                    n = int(header.split()[1])
                    yield np.frombuffer(self._read_exact(4 * n, deadline), dtype="<f4").astype(np.float32)
            finally:
                if not finished:
                    self.close()  # failed or abandoned mid-reply: out of step with the worker


class Qwen3Tts:
    """Qwen3-TTS, the whole reply in one streaming call; `say` speaks it if Qwen3 fails before any audio."""

    name = "qwen3"

    def __init__(self, engine: Qwen3Engine, voice: str = QWEN3_SPEAKER, instruct: Optional[str] = QWEN3_INSTRUCT,
                 fallback: Optional[Tts] = None, gain: float = 1.0) -> None:
        self.engine = engine
        self.voice = voice
        self.instruct = instruct or None
        self.fallback = fallback or SayTts()
        self.gain = gain  # Qwen3 already renders at about Kokoro x2's level (peak ~0.8)
        resample(np.zeros(240, np.float32), engine.rate, 16000)  # import scipy.signal now, not on the first reply

    def chunks(self, text: str, rate: int) -> Iterator[np.ndarray]:
        text = " ".join(split_sentences(text))  # one line: mlx-audio would split at line breaks
        if not text:
            return
        rs = StreamResampler(self.engine.rate, rate)
        started = False
        try:
            for audio in self.engine.stream(text, self.voice or QWEN3_SPEAKER, self.instruct):
                if not started:
                    loud = np.flatnonzero(np.abs(audio) > 0.01)
                    if loud.size == 0:
                        continue  # leading silence
                    audio = audio[max(0, int(loud[0]) - int(0.03 * self.engine.rate)) :]
                    started = True
                out = rs.feed(audio)
                if out.size:
                    yield to_int16(out, self.gain)
        except Exception as e:
            if started:
                logger.warning("Qwen3 failed mid-reply (%s); the rest of this reply is dropped", type(e).__name__)
                return
            logger.warning("Qwen3 failed (%s); macOS say speaks this reply", type(e).__name__)
            yield from self.fallback.chunks(text, rate)
            return
        out = rs.flush()
        if out.size:
            yield to_int16(out, self.gain)

    def voices(self) -> list[str]:
        return list(QWEN3_SPEAKERS)

    def close(self) -> None:
        self.engine.close()


def qwen3_instruct() -> Optional[str]:
    """MUSE_TTS_INSTRUCT_FILE's contents, else MUSE_TTS_INSTRUCT, else QWEN3_INSTRUCT; empty means none."""
    path = os.environ.get("MUSE_TTS_INSTRUCT_FILE")
    if path:
        try:
            return " ".join(Path(path).read_text(encoding="utf-8").split()) or None
        except OSError as e:
            logger.warning("can't read MUSE_TTS_INSTRUCT_FILE (%s); using the default style", type(e).__name__)
    value = os.environ.get("MUSE_TTS_INSTRUCT")
    if value is not None:
        return " ".join(value.split()) or None
    return QWEN3_INSTRUCT


def worker_command(script: str, *args: str) -> list[str]:
    here = Path(__file__).resolve().parent
    python = os.environ.get("MUSE_KOKORO_PYTHON") or str(here.parent / "kokoro" / ".venv" / "bin" / "python")
    return [python, str(here / script), *args]


def kokoro_command(voice: str) -> list[str]:
    return worker_command("kokoro_worker.py", "--voice", voice)


def qwen3_command(speaker: str) -> list[str]:
    return worker_command("qwen3_worker.py", "--speaker", speaker)


def make_tts(backend: Optional[str] = None, voice: Optional[str] = None) -> Tts:
    """MUSE_TTS=qwen3 (default), kokoro or say; MUSE_TTS_VOICE picks the voice.
    Qwen3 falls back to Kokoro, Kokoro to say. Only one model is loaded at a time."""
    backend = (backend or os.environ.get("MUSE_TTS") or "qwen3").strip().lower()
    voice = voice or os.environ.get("MUSE_TTS_VOICE") or None
    if backend == "say":
        return SayTts(voice)
    if backend == "qwen3":
        speaker = {s.lower(): s for s in QWEN3_SPEAKERS}.get((voice or QWEN3_SPEAKER).lower())
        if speaker is None:
            logger.warning("unknown Qwen3 speaker %r; using %s", voice, QWEN3_SPEAKER)
            speaker = QWEN3_SPEAKER
        try:
            engine = Qwen3Engine(qwen3_command(speaker))
        except Exception as e:
            logger.warning("Qwen3-TTS unavailable (%s); using Kokoro", e)
            return make_tts("kokoro", None)
        return Qwen3Tts(engine, speaker, qwen3_instruct(), fallback=SayTts())
    if backend != "kokoro":
        raise ValueError(f"unknown TTS backend {backend!r} (qwen3, kokoro or say)")
    voice = voice or KOKORO_VOICE
    if voice not in KOKORO_VOICES:
        logger.warning("unknown Kokoro voice %r; using %s", voice, KOKORO_VOICE)
        voice = KOKORO_VOICE
    try:
        engine = KokoroEngine(kokoro_command(voice))
    except Exception as e:
        logger.warning("Kokoro unavailable (%s); using macOS say", e)
        return SayTts()
    return KokoroTts(engine, voice, fallback=SayTts())
