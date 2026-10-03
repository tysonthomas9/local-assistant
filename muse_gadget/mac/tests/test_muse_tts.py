"""Kokoro TTS path with fake engines: sentence splitting, resampling, fallback to `say`, streaming.

    cd muse_gadget/mac && <app venv>/bin/python -m pytest tests -q
"""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import muse_handler  # noqa: E402
import muse_tts  # noqa: E402
from muse_bridge import BridgeClient  # noqa: E402
from muse_tts import (  # noqa: E402
    KokoroEngine, KokoroTts, SayTts, child_env, make_tts, resample, split_sentences, trim_silence,
)
from muse_vad import EnergyVad, UtteranceSegmenter  # noqa: E402

KOKORO_RATE = 24000


def sine(seconds: float, rate: int = KOKORO_RATE, hz: float = 440.0, amp: float = 0.3) -> np.ndarray:
    return (amp * np.sin(2 * np.pi * hz * np.arange(int(seconds * rate)) / rate)).astype(np.float32)


def padded(audio: np.ndarray, lead_s: float = 0.3, tail_s: float = 0.45, rate: int = KOKORO_RATE) -> np.ndarray:
    """Like Kokoro's output: ~0.3 s of silence before the speech and ~0.45 s after."""
    return np.concatenate([np.zeros(int(lead_s * rate), np.float32), audio, np.zeros(int(tail_s * rate), np.float32)])


class FakeEngine:
    rate = KOKORO_RATE

    def __init__(self, fail_on: int | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.fail_on = fail_on

    def synth(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        self.calls.append((text, voice))
        if self.fail_on is not None and len(self.calls) > self.fail_on:
            raise EOFError("kokoro worker exited")
        return padded(sine(0.1 * len(text.split())))

    def close(self) -> None:
        pass


class FakeSay:
    name = "say"
    voice = None

    def __init__(self) -> None:
        self.texts: list[str] = []

    def chunks(self, text: str, rate: int):
        self.texts.append(text)
        yield np.full(rate // 10, 1000, np.int16)

    def voices(self) -> list[str]:
        return []

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------- sentence splitting
def test_split_sentences():
    assert split_sentences("The capital of France is Paris. It's on the Seine! Want more? Sure.") == [
        "The capital of France is Paris.", "It's on the Seine!", "Want more?", "Sure.",
    ]
    assert split_sentences("Dr. Smith met Mr. Jones, e.g. at 3.5 p.m. today.") == [
        "Dr. Smith met Mr. Jones, e.g. at 3.5 p.m. today.",
    ]
    assert split_sentences("First line\nSecond line.  \n\n  ") == ["First line", "Second line."]
    assert split_sentences("") == [] and split_sentences("  \n ") == []
    long = ", ".join(f"item number {i}" for i in range(40)) + "."
    parts = split_sentences(long, max_chars=100)
    assert len(parts) > 1 and all(len(p) <= 100 for p in parts)
    assert " ".join(parts) == long  # nothing lost, cut at commas
    assert all(len(p) <= 50 for p in split_sentences("x" * 120, max_chars=50))


# ---------------------------------------------------------------------- audio helpers
def test_resample_24k_to_16k_keeps_pitch_and_level():
    out = resample(sine(1.0), KOKORO_RATE, 16000)
    assert out.dtype == np.float32 and out.size == 16000
    spectrum = np.abs(np.fft.rfft(out))
    assert abs(np.argmax(spectrum) * 16000 / out.size - 440) < 2  # still 440 Hz (not 293 Hz)
    assert abs(np.abs(out[1000:-1000]).max() - 0.3) < 0.02
    assert resample(sine(0.1, 16000), 16000, 16000).size == 1600


def test_trim_silence_keeps_short_margins():
    trimmed = trim_silence(padded(sine(0.5)), KOKORO_RATE)
    assert abs(trimmed.size / KOKORO_RATE - (0.5 + 0.03 + 0.15)) < 0.01
    assert trim_silence(np.zeros(1000, np.float32), KOKORO_RATE).size == 0


# ---------------------------------------------------------------------- KokoroTts
def test_kokoro_renders_each_sentence_at_the_robot_rate():
    engine = FakeEngine()
    tts = KokoroTts(engine, "am_michael", fallback=FakeSay(), gain=2.0)
    chunks = list(tts.chunks("The capital of France is Paris. It sits on the Seine.", 16000))
    assert [t for t, _ in engine.calls] == ["The capital of France is Paris.", "It sits on the Seine."]
    assert {v for _, v in engine.calls} == {"am_michael"}
    assert len(chunks) == 2 and all(c.dtype == np.int16 for c in chunks)
    # 24 kHz -> 16 kHz, Kokoro's 0.3 s lead silence trimmed, gain 2 (0.3 -> 0.6 of full scale).
    expected_s = 0.1 * 6 + 0.03 + 0.15
    assert abs(chunks[0].size / 16000 - expected_s) < 0.01
    assert np.flatnonzero(np.abs(chunks[0]) > 300)[0] / 16000 < 0.05
    assert 0.55 < np.abs(chunks[0]).max() / 32767 < 0.65


def test_kokoro_failure_mid_reply_falls_back_to_say_for_the_rest():
    say = FakeSay()
    tts = KokoroTts(FakeEngine(fail_on=1), fallback=say)
    chunks = list(tts.chunks("One two. Three four. Five six.", 16000))
    assert len(chunks) == 2  # Kokoro's first sentence, then one `say` chunk
    assert say.texts == ["Three four. Five six."]


def test_make_tts_falls_back_to_say_when_kokoro_cannot_start(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("MUSE_KOKORO_PYTHON", str(tmp_path / "missing" / "python"))
    tts = make_tts("kokoro")
    assert isinstance(tts, SayTts) and "Kokoro unavailable" in caplog.text
    # A worker that starts but fails to load its model.
    bad = tmp_path / "bad.sh"
    bad.write_text("#!/bin/sh\necho 'err FileNotFoundError: no model'\n")
    bad.chmod(0o755)
    monkeypatch.setenv("MUSE_KOKORO_PYTHON", str(bad))
    assert isinstance(make_tts("kokoro"), SayTts)
    assert isinstance(make_tts("say", "Samantha"), SayTts) and make_tts("say", "Samantha").voice == "Samantha"
    with pytest.raises(ValueError):
        make_tts("espeak")


FAKE_WORKER = textwrap.dedent('''
    import json, os, sys
    import numpy as np
    out = sys.stdout.buffer
    out.write(b"ready 24000\\n"); out.flush()
    for line in sys.stdin:
        req = json.loads(line)
        if req["voice"] == "zz_bad":
            out.write(b"err LookupError\\n"); out.flush(); continue
        if req["text"] == "die":
            sys.exit(3)
        n = 240 * len(req["text"])
        out.write(b"ok %d\\n" % n + np.full(n, 0.25, "<f4").tobytes()); out.flush()
''')


def test_kokoro_engine_speaks_the_worker_protocol(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(FAKE_WORKER)
    engine = KokoroEngine([sys.executable, str(worker)])
    try:
        assert engine.rate == 24000
        audio = engine.synth("hello", "af_heart")
        assert audio.dtype == np.float32 and audio.size == 1200 and np.allclose(audio, 0.25)
        big = engine.synth("x" * 400, "af_heart")  # bigger than one pipe read
        assert big.size == 96000
        with pytest.raises(RuntimeError):
            engine.synth("hello", "zz_bad")
        assert engine.synth("ok", "af_heart").size == 480  # still in step after an error
        with pytest.raises(EOFError):
            engine.synth("die", "af_heart")
        with pytest.raises(EOFError):
            engine.synth("hello", "af_heart")
    finally:
        engine.close()
    assert engine.proc.poll() is not None


def test_child_env_drops_the_app_venvs_paths(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/app/venv/gst:/keep/me")
    monkeypatch.setenv("GST_PLUGIN_PATH", "/app/venv/only")
    env = child_env("/app/venv")
    assert env["PYTHONPATH"] == "/keep/me" and "GST_PLUGIN_PATH" not in env


# ---------------------------------------------------------------------- MuseHandler streaming
class FakeMedia:
    def get_output_audio_samplerate(self) -> int:
        return 16000


class FakeDeps:
    class reachy_mini:  # noqa: N801
        media = FakeMedia()


class GatedTts:
    """Two sentences; the second is only rendered once the test opens the gate."""

    name = "kokoro"

    def __init__(self) -> None:
        self.voice = "af_heart"
        self.gate = threading.Event()
        self.closed = False

    def chunks(self, text: str, rate: int):
        yield np.full(rate // 2, 100, np.int16)
        assert self.gate.wait(5)
        yield np.full(rate // 2, 200, np.int16)

    def voices(self) -> list[str]:
        return ["af_heart", "am_michael"]

    def close(self) -> None:
        self.closed = True


def test_playback_starts_after_the_first_sentence():
    async def run():
        tts = GatedTts()
        handler = muse_handler.MuseHandler(
            FakeDeps(), bridge=BridgeClient("http://127.0.0.1:9"), transcriber=lambda a: "",
            tts=tts, segmenter=UtteranceSegmenter(EnergyVad()), log_transcripts=False,
        )
        startup = asyncio.create_task(handler.start_up())
        while not handler._is_connected():
            await asyncio.sleep(0.01)
        speaking = asyncio.create_task(handler.say("One. Two."))
        for _ in range(200):
            if handler.output_queue.qsize():
                break
            await asyncio.sleep(0.01)
        first = [handler.output_queue.get_nowait() for _ in range(handler.output_queue.qsize())]
        assert not speaking.done()  # the second sentence is still being rendered
        tts.gate.set()
        await speaking
        second = [handler.output_queue.get_nowait() for _ in range(handler.output_queue.qsize())]
        busy_s = handler._busy_until - time.monotonic()
        voices = (await handler.get_available_voices(), await handler.change_voice("am_michael"),
                  await handler.change_voice("nope"), handler.get_current_voice())
        await handler.shutdown()
        await startup
        return first, second, busy_s, voices, tts

    first, second, busy_s, voices, tts = asyncio.run(run())
    assert sum(p.size for _, p in first) == 8000 and all(p.max() == 100 for _, p in first)
    assert sum(p.size for _, p in second) == 8000 and all(p.max() == 200 for _, p in second)
    assert all(rate == 16000 for rate, _ in first + second)
    assert busy_s > 0.9  # mic stays closed for both sentences (1 s) plus the tail
    assert voices[0] == ["af_heart", "am_michael"]
    assert voices[1] == "Voice set to am_michael." and voices[2].startswith("Unknown voice")
    assert voices[3] == "am_michael"
    assert not tts.closed  # injected: the caller owns it


def test_handler_loads_and_closes_its_own_tts(monkeypatch):
    tts = GatedTts()
    monkeypatch.setattr(muse_tts, "make_tts", lambda: tts)

    async def run():
        handler = muse_handler.MuseHandler(
            FakeDeps(), startup_voice="am_michael", bridge=BridgeClient("http://127.0.0.1:9"),
            transcriber=lambda a: "", segmenter=UtteranceSegmenter(EnergyVad()),
        )
        startup = asyncio.create_task(handler.start_up())
        while not handler._is_connected():
            await asyncio.sleep(0.01)
        voice = handler.get_current_voice()
        await handler.shutdown()
        await startup
        return voice, handler.tts

    voice, after = asyncio.run(run())
    assert voice == "am_michael" and tts.closed and after is None


def test_run_app_takes_the_tts_flags(monkeypatch):
    import run_app

    for key in ("MUSE_TTS", "MUSE_TTS_VOICE"):
        monkeypatch.delenv(key, raising=False)
    rest = run_app.take_own_flags(["run_app.py", "--tts", "say", "--no-camera", "--voice", "Samantha"])
    assert rest == ["run_app.py", "--no-camera"]
    assert os.environ["MUSE_TTS"] == "say" and os.environ["MUSE_TTS_VOICE"] == "Samantha"
    with pytest.raises(SystemExit):
        run_app.take_own_flags(["run_app.py", "--tts", "espeak"])
    with pytest.raises(SystemExit):
        run_app.take_own_flags(["run_app.py", "--voice"])


def test_stt_loads_and_runs_on_one_thread_while_tts_loads_in_parallel(monkeypatch):
    # MLX (parakeet) raises "There is no Stream(cpu, 1) in current thread" on any other thread.
    import muse_stt

    threads: dict[str, set[int]] = {"load": set(), "run": set()}

    def make_transcriber():
        threads["load"].add(threading.get_ident())

        def transcribe(audio):
            threads["run"].add(threading.get_ident())
            return ""

        return "fake-mlx", transcribe

    def make_tts():
        time.sleep(0.05)  # keep a second executor thread busy while STT loads
        return GatedTts()

    monkeypatch.setattr(muse_stt, "make_transcriber", make_transcriber)
    monkeypatch.setattr(muse_tts, "make_tts", make_tts)

    async def run():
        handler = muse_handler.MuseHandler(
            FakeDeps(), bridge=BridgeClient("http://127.0.0.1:9"), segmenter=UtteranceSegmenter(EnergyVad()),
        )
        startup = asyncio.create_task(handler.start_up())
        while not handler._is_connected():
            await asyncio.sleep(0.01)
        for _ in range(5):
            await asyncio.gather(*(asyncio.to_thread(time.sleep, 0.01) for _ in range(4)))  # churn the default pool
            await handler._run_turn(np.zeros(16000, np.float32))
        await handler.shutdown()
        await startup

    asyncio.run(run())
    assert len(threads["load"]) == 1 and threads["run"] == threads["load"]
