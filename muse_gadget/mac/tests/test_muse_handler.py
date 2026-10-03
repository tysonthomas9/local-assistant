"""MuseHandler unit tests: run on the PC with the pinned app's venv (no robot, no Mac, no MLX).

    cd muse_gadget/mac && <app venv>/bin/python -m pytest tests -q
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from reachy_mini_conversation_app.streaming import AdditionalOutputs  # noqa: E402

import fake_bridge  # noqa: E402
import muse_handler  # noqa: E402
from muse_bridge import BridgeClient, BridgeError, spoken_error  # noqa: E402
from muse_tts import read_wav_int16, say_command  # noqa: E402
from muse_vad import EnergyVad, SileroVad, UtteranceSegmenter, silero_model_path, to_mono_16k  # noqa: E402

WAV = HERE / "fixtures" / "utterance.wav"  # "Hello robot. What is the weather like today?" (macOS say, 16 kHz)


@pytest.fixture
def bridge_url():
    server = fake_bridge.serve(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def mic_frames(seconds_silence: float = 1.0, frame_s: float = 0.02):
    """The fixture as the robot mic delivers it: 16 kHz stereo float32 in short frames, padded with silence."""
    rate, pcm = read_wav_int16(WAV)
    assert rate == 16000
    speech = pcm.astype(np.float32) / 32768.0
    pad = np.zeros(int(seconds_silence * rate), dtype=np.float32)
    rng = np.random.default_rng(0)
    audio = np.concatenate([pad, speech, pad, pad]) + rng.normal(0, 0.001, len(pad) * 3 + len(speech)).astype(np.float32)
    stereo = np.stack([audio, audio], axis=1)
    step = int(frame_s * rate)
    return [(rate, stereo[i : i + step]) for i in range(0, len(stereo), step)]


class FakeMedia:
    def get_output_audio_samplerate(self) -> int:
        return 16000


class FakeDeps:
    class reachy_mini:  # noqa: N801
        media = FakeMedia()


def tone(text: str, rate: int) -> np.ndarray:
    n = int(rate * (0.5 + 0.02 * len(text)))
    return (np.sin(np.arange(n) * 2 * np.pi * 220 / rate) * 8000).astype(np.int16)


async def drive(handler: muse_handler.MuseHandler, frames) -> tuple[list, list]:
    transcripts: list[tuple[str, str, bool]] = []
    handler.set_transcript_observer(lambda role, text, final: transcripts.append((role, text, final)))
    startup = asyncio.create_task(handler.start_up())
    for _ in range(100):
        if handler._is_connected():
            break
        await asyncio.sleep(0.01)
    assert handler._is_connected()
    for frame in frames:
        await handler.receive(frame)
        await asyncio.sleep(0)
    outputs: list = []
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not any(r == "assistant" for r, _, _ in transcripts):
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.05)
    while not handler.output_queue.empty():
        outputs.append(handler.output_queue.get_nowait())
    await handler.shutdown()
    await asyncio.wait_for(startup, 5)
    return transcripts, outputs


def make_handler(bridge_url, transcribe, **kwargs):
    return muse_handler.MuseHandler(
        FakeDeps(), bridge=BridgeClient(bridge_url), transcriber=transcribe, synthesizer=tone,
        segmenter=UtteranceSegmenter(EnergyVad()), **kwargs,
    )


def test_recorded_utterance_becomes_a_spoken_reply(bridge_url, monkeypatch, caplog):
    monkeypatch.delenv("MUSE_LOG_TRANSCRIPTS", raising=False)
    caplog.set_level(logging.DEBUG)
    heard: list[np.ndarray] = []

    def transcribe(audio: np.ndarray) -> str:
        heard.append(audio)
        return "hello robot"

    handler = muse_handler.MuseHandler(
        FakeDeps(),
        bridge=BridgeClient(bridge_url),
        transcriber=transcribe,
        synthesizer=tone,
        segmenter=UtteranceSegmenter(EnergyVad()),
    )
    transcripts, outputs = asyncio.run(drive(handler, mic_frames()))

    # One utterance. The passes at pauses get the utterance so far; the check pass gets all of it.
    full = max(heard, key=len)
    assert all(np.array_equal(a, full[: len(a)]) for a in heard)
    speech_s = read_wav_int16(WAV)[1].size / 16000
    assert speech_s - 0.3 < full.size / 16000 < speech_s + 2.0
    assert ("user", "hello robot", True) in transcripts
    assert ("assistant", "You said: hello robot", True) in transcripts

    # No transcript text for the app to log, and none in any log record.
    assert not any(isinstance(o, AdditionalOutputs) for o in outputs)
    audio = [o for o in outputs if isinstance(o, tuple)]
    assert len(audio) == len(outputs)
    assert caplog.records
    assert not [r for r in caplog.records if "hello robot" in r.getMessage()]
    assert audio and all(rate == 16000 and pcm.dtype == np.int16 and pcm.ndim == 2 for rate, pcm in audio)
    assert sum(pcm.size for _, pcm in audio) == tone("You said: hello robot", 16000).size


def test_log_transcripts_opt_in_hands_the_text_to_the_app(bridge_url, monkeypatch):
    monkeypatch.setenv("MUSE_LOG_TRANSCRIPTS", "1")
    handler = make_handler(bridge_url, lambda a: "hello robot")
    assert handler.log_transcripts
    _, outputs = asyncio.run(drive(handler, mic_frames()))
    texts = [o.args[0] for o in outputs if isinstance(o, AdditionalOutputs)]
    assert {"role": "user", "content": "hello robot"} in texts
    assert {"role": "assistant", "content": "You said: hello robot"} in texts
    assert not make_handler(bridge_url, lambda a: "", log_transcripts=False).log_transcripts


def test_run_app_turns_the_flag_into_the_env(monkeypatch):
    import run_app

    monkeypatch.delenv("MUSE_LOG_TRANSCRIPTS", raising=False)
    monkeypatch.setattr(run_app, "install", lambda: (_ for _ in ()).throw(SystemExit(0)))
    monkeypatch.setattr(sys, "argv", ["run_app.py", "--no-camera", "--log-transcripts"])
    with pytest.raises(SystemExit):
        run_app.main()
    assert sys.argv == ["run_app.py", "--no-camera"]
    assert os.environ["MUSE_LOG_TRANSCRIPTS"] == "1"


def test_run_poc_redacts_transcripts_in_its_logs():
    script = (HERE.parent.parent / "run_poc.sh").read_text()
    redact = [line for line in script.splitlines() if line.startswith("redact()")]
    assert len(redact) == 1
    lines = (
        "2026-10-03 10:00:00 INFO console: role=user content=hello robot, my secret plan\n"
        "2026-10-03 10:00:02 INFO console: role=assistant content=You said: hello robot\n"
        "2026-10-03 10:00:03 INFO muse_handler: MuseHandler: heard 11 chars in 120 ms\n"
    )
    out = subprocess.run(["bash", "-c", redact[0] + "\nredact"], input=lines, capture_output=True,
                         text=True, check=True).stdout
    assert "hello robot" not in out and "secret" not in out
    assert "role=user content=<redacted>" in out and "role=assistant content=<redacted>" in out
    assert "heard 11 chars in 120 ms" in out
    # Every path to a log file on the PC goes through redact.
    tees = [line for line in script.splitlines() if "$LOGDIR/$RUN" in line and ("tee" in line or ">>" in line)]
    assert tees and all("redact" in line for line in tees)


def test_multi_line_transcripts_are_redacted_whole(bridge_url, monkeypatch):
    # console.py logs `role=%s content=%s`; a multi-line text must not leak lines past the first.
    monkeypatch.setenv("MUSE_LOG_TRANSCRIPTS", "1")
    handler = make_handler(bridge_url, lambda a: "hello robot\nmy secret plan\r\n  part two")
    _, outputs = asyncio.run(drive(handler, mic_frames()))
    texts = [o.args[0] for o in outputs if isinstance(o, AdditionalOutputs)]
    assert {"role": "user", "content": "hello robot my secret plan part two"} in texts
    assert texts and all("\n" not in t["content"] and "\r" not in t["content"] for t in texts)
    script = (HERE.parent.parent / "run_poc.sh").read_text()
    redact = next(line for line in script.splitlines() if line.startswith("redact()"))
    log = "".join("INFO console: role=%s content=%s\n" % (t["role"], t["content"]) for t in texts)
    out = subprocess.run(["bash", "-c", redact + "\nredact"], input=log, capture_output=True,
                         text=True, check=True).stdout
    assert "secret" not in out and "part two" not in out and "hello" not in out
    assert out.count("content=<redacted>") == len(texts)


def test_mic_is_ignored_while_the_robot_speaks(bridge_url):
    calls: list[int] = []

    def transcribe(audio: np.ndarray) -> str:
        calls.append(audio.size)
        return "hello robot"

    handler = muse_handler.MuseHandler(
        FakeDeps(), bridge=BridgeClient(bridge_url), transcriber=transcribe, synthesizer=tone,
        segmenter=UtteranceSegmenter(EnergyVad()), barge_in=False,
    )
    # The same utterance twice, back to back: the second arrives while the first reply plays.
    frames = mic_frames(seconds_silence=1.0)
    asyncio.run(drive(handler, frames + frames))
    assert len([c for c in calls if c == max(calls)]) == 1, "half-duplex: the second one isn't heard"


def test_say_speaks_verbatim_without_a_turn():
    async def run():
        handler = muse_handler.MuseHandler(
            FakeDeps(), bridge=BridgeClient("http://127.0.0.1:9"), transcriber=lambda a: "",
            synthesizer=tone, segmenter=UtteranceSegmenter(EnergyVad()), log_transcripts=False,
        )
        spoken: list = []
        handler.set_transcript_observer(lambda role, text, final: spoken.append((role, text)))
        with pytest.raises(RuntimeError):
            await handler.say("hi")
        startup = asyncio.create_task(handler.start_up())
        while not handler._is_connected():
            await asyncio.sleep(0.01)
        await handler.say("Hello there")
        items = [handler.output_queue.get_nowait() for _ in range(handler.output_queue.qsize())]
        await handler.shutdown()
        await startup
        return items, spoken

    items, spoken = asyncio.run(run())
    assert spoken == [("assistant", "Hello there")]
    assert all(isinstance(i, tuple) for i in items)
    assert sum(pcm.size for _, pcm in items) == tone("Hello there", 16000).size


def test_bridge_errors_become_spoken_messages(bridge_url):
    with pytest.raises(BridgeError) as err:
        BridgeClient("http://127.0.0.1:9", timeout_s=2).turn("hi")
    assert err.value.code == "unreachable"
    assert "isn't running" in spoken_error(err.value)
    assert BridgeClient(bridge_url).turn("hi there") == "You said: hi there"
    with pytest.raises(BridgeError) as err:
        BridgeClient(bridge_url).turn("   ")
    assert err.value.code == "bad_request"


def test_say_command_renders_pcm_at_the_robot_rate_without_text_on_the_command_line():
    cmd = say_command("/tmp/t.txt", "/tmp/o.wav", 16000, "Samantha")
    assert cmd[:5] == ["/usr/bin/say", "-o", "/tmp/o.wav", "--file-format=WAVE", "--data-format=LEI16@16000"]
    assert cmd[-2:] == ["-f", "/tmp/t.txt"] and "-v" in cmd


def test_mic_frames_are_downmixed_and_resampled():
    stereo = np.stack([np.full(480, 0.5, np.float32), np.zeros(480, np.float32)], axis=1)
    out = to_mono_16k(48000, stereo)
    assert out.dtype == np.float32 and out.size == 160 and np.allclose(out, 0.5)
    assert np.allclose(to_mono_16k(16000, np.full(10, 16384, np.int16)), 0.5)


@pytest.mark.skipif(silero_model_path() is None, reason="silero-vad wheel not installed")
def test_silero_finds_the_recorded_utterance():
    seg = UtteranceSegmenter(SileroVad(silero_model_path()))
    utterances = []
    for _, frame in mic_frames():
        utterances += seg.feed(to_mono_16k(16000, frame))
    assert len(utterances) == 1


def test_launcher_swaps_the_backend():
    import run_app
    from reachy_mini_conversation_app import huggingface_realtime

    original = huggingface_realtime.HuggingFaceRealtimeHandler
    try:
        run_app.install()
        assert huggingface_realtime.HuggingFaceRealtimeHandler is muse_handler.MuseHandler
        assert os.environ["HF_REALTIME_CONNECTION_MODE"] == "local"
    finally:
        huggingface_realtime.HuggingFaceRealtimeHandler = original


def test_exec_python_drops_the_trampoline_venvs_paths():

    env = dict(os.environ, PYTHONPATH=f"{sys.prefix}/gst:/keep/me", MUSE_PROBE=f"{sys.prefix}/only")
    out = subprocess.run(
        [sys.executable, str(HERE.parent / "exec_python.py"), "/bin/sh", "-c", 'echo "$PYTHONPATH|${MUSE_PROBE-unset}"'],
        env=env, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out == "/keep/me|unset"


class FakeWorker:
    """Stands in for a loaded STT/TTS worker: a real child process that close() stops."""

    name = "fake"
    voice = None

    def __init__(self) -> None:
        self.proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])

    def __call__(self, audio):
        return ""

    def voices(self):
        return []

    def close(self) -> None:
        self.proc.terminate()
        self.proc.wait(5)


@pytest.mark.parametrize("failing", ["tts", "stt"])
def test_start_up_closes_the_loaded_worker_when_the_other_fails(monkeypatch, failing):
    import muse_stt
    import muse_tts

    started: list[FakeWorker] = []

    def load_stt():
        if failing == "stt":
            raise RuntimeError("stt failed to load")
        w = FakeWorker()
        started.append(w)
        return "fake-stt", w

    def load_tts():
        if failing == "tts":
            raise RuntimeError("tts failed to load")
        w = FakeWorker()
        started.append(w)
        return w

    monkeypatch.setattr(muse_stt, "make_transcriber", load_stt)
    monkeypatch.setattr(muse_tts, "make_tts", load_tts)
    handler = muse_handler.MuseHandler(FakeDeps(), bridge=BridgeClient("http://127.0.0.1:9"),
                                       segmenter=UtteranceSegmenter(EnergyVad()))
    with pytest.raises(RuntimeError, match=f"{failing} failed"):
        asyncio.run(handler.start_up())
    assert len(started) == 1
    assert started[0].proc.poll() is not None, "the worker that did load was stopped"
    assert not handler._owns_stt and not handler._owns_tts


# ------------------------------------------------------------------ streamed replies
class SlowStream:
    """A bridge that streams two sentences with a gap, like Muse writing; records when each went out."""

    def __init__(self, gap_s=0.6, lines=None, think_s=0.0):
        import http.server
        import json as _json

        self.sent_at: list[float] = []
        outer = self
        self.lines = lines if lines is not None else [{"text": "First sentence."}, {"text": "Second one."}]

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                outer.paths.append(self.path)
                outer.texts.append(_json.loads(body or b"{}").get("text"))
                time.sleep(think_s)
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                try:
                    for i, line in enumerate(outer.lines):
                        if i:
                            time.sleep(gap_s)
                        self.wfile.write(_json.dumps(line).encode() + b"\n")
                        self.wfile.flush()
                        outer.sent_at.append(time.perf_counter())
                    self.wfile.write(b'{"done": true}\n')
                except OSError:
                    pass  # the client left (a barge-in or a continuation)
                self.close_connection = True

            def log_message(self, *args):
                pass

        self.paths: list[str] = []
        self.texts: list[str] = []
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def run_streamed_turn(bridge):
    spoken: list[tuple[str, float]] = []

    def synth(text, rate):
        spoken.append((text, time.perf_counter()))
        return tone(text, rate)

    handler = muse_handler.MuseHandler(FakeDeps(), bridge=BridgeClient(bridge.url), transcriber=lambda a: "hello",
                                       synthesizer=synth, segmenter=UtteranceSegmenter(EnergyVad()))

    async def main():
        await handler._run_turn(np.zeros(16000, np.float32), time.perf_counter() - 0.5)
        while not handler.output_queue.empty():
            handler.output_queue.get_nowait()
    asyncio.run(main())
    return spoken


def test_first_sentence_is_spoken_while_muse_is_still_writing(caplog):
    caplog.set_level(logging.INFO)
    bridge = SlowStream(gap_s=0.6)
    try:
        spoken = run_streamed_turn(bridge)
    finally:
        bridge.close()
    assert bridge.paths == ["/turn?stream=1"]
    assert [t for t, _ in spoken] == ["First sentence.", "Second one."]
    assert spoken[0][1] < bridge.sent_at[1], "the first sentence was rendered before the second arrived"
    timing = [r.getMessage() for r in caplog.records if "end of speech to first audio" in r.getMessage()]
    assert len(timing) == 1 and "hello" not in timing[0]


def test_stream_error_before_text_is_spoken_as_before():
    bridge = SlowStream(lines=[{"done": True, "error": "timeout"}])
    try:
        spoken = run_streamed_turn(bridge)
    finally:
        bridge.close()
    assert [t for t, _ in spoken] == ["Muse is taking too long to answer."]


def test_stream_client_reads_sentences_and_errors(bridge_url):
    got = []
    assert BridgeClient(bridge_url).turn_stream("hi there", got.append) == "You said: hi there"
    assert got == ["You said: hi there"]
    with pytest.raises(BridgeError) as err:
        BridgeClient("http://127.0.0.1:9", timeout_s=2).turn_stream("hi", got.append)
    assert err.value.code == "unreachable"


def test_end_silence_default_and_env(monkeypatch):
    import muse_vad

    monkeypatch.delenv(muse_vad.END_SILENCE_ENV, raising=False)
    assert muse_vad.end_silence_from_env() == 0.8 == muse_vad.END_SILENCE_S
    monkeypatch.setenv(muse_vad.END_SILENCE_ENV, "0.3")
    assert muse_vad.end_silence_from_env() == 0.3
    for bad in ("abc", "0", "9"):
        monkeypatch.setenv(muse_vad.END_SILENCE_ENV, bad)
        assert muse_vad.end_silence_from_env() == 0.8


# ------------------------------------------------------------------ barge-in and early speech-to-text
def burst(seconds: float, level: float = 0.1, seed: int = 1) -> np.ndarray:
    return (np.random.default_rng(seed).normal(0, level, int(seconds * 16000))).astype(np.float32)


def quiet(seconds: float, level: float = 0.003) -> np.ndarray:
    return burst(seconds, level, seed=2)


async def paced(handler, audio: np.ndarray, frame_s: float = 0.02) -> None:
    """Feed mono 16 kHz audio as the robot mic would, in real time."""
    step = int(frame_s * 16000)
    for i in range(0, len(audio), step):
        chunk = audio[i : i + step]
        await handler.receive((16000, np.stack([chunk, chunk], axis=1)))
        await asyncio.sleep(frame_s)


async def started(handler):
    task = asyncio.create_task(handler.start_up())
    for _ in range(100):
        if handler._is_connected():
            break
        await asyncio.sleep(0.01)
    return task


def barge_handler(url, spoken, heard=None, **kwargs):
    def synth(text, rate):
        spoken.append(text)
        return tone(text, rate)

    def transcribe(audio):
        if heard is not None:
            heard.append(audio.size)
        return "hello"

    return muse_handler.MuseHandler(FakeDeps(), bridge=BridgeClient(url), transcriber=transcribe, synthesizer=synth,
                                    segmenter=UtteranceSegmenter(EnergyVad()), **kwargs)


def test_barge_in_stops_the_reply_and_the_new_utterance_is_the_next_turn(caplog):
    caplog.set_level(logging.INFO)
    bridge = SlowStream(gap_s=1.5, lines=[{"text": "First sentence of a long reply."}, {"text": "Never heard."}])
    spoken: list[str] = []
    handler = barge_handler(bridge.url, spoken, barge_in=True, barge_in_stop_ms=300)
    cleared: list[float] = []

    def clear():
        cleared.append(time.monotonic())
        while not handler.output_queue.empty():
            handler.output_queue.get_nowait()

    handler._clear_queue = clear

    async def main():
        startup = await started(handler)
        await paced(handler, np.concatenate([burst(0.6), quiet(1.0)]))
        for _ in range(200):
            if handler._armed:
                break
            await asyncio.sleep(0.01)
        assert handler._armed, "the mic opens once the reply's first audio is queued"
        await paced(handler, quiet(0.3))
        assert not cleared, "quiet room: no barge-in"
        await paced(handler, burst(0.15, seed=3))
        assert not cleared, "it keeps speaking for the stop delay"
        await paced(handler, np.concatenate([burst(0.4, seed=4), quiet(1.0)]))
        assert cleared, "stopped"
        fade = handler.output_queue.get_nowait()
        for _ in range(500):
            if len(bridge.paths) == 2 and not handler._turn_active and handler._utterances.empty():
                break
            await asyncio.sleep(0.01)
        await handler.shutdown()
        await asyncio.wait_for(startup, 5)
        return fade

    try:
        rate, fade = asyncio.run(main())
    finally:
        bridge.close()
    assert rate == 16000 and fade.shape[1] == int(muse_handler.FADE_S * 16000)
    assert abs(int(fade[0, -1])) < 200, "faded to silence"
    assert bridge.paths == ["/turn?stream=1", "/turn?stream=1"], "the barge-in utterance was the next turn"
    assert spoken[0] == "First sentence of a long reply."
    assert spoken.count("Never heard.") == 1, "dropped from the first reply; spoken only in the second"
    logs = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("MuseHandler: barge-in after ") for m in logs)
    stop = [m for m in logs if "reply stopped" in m]
    assert len(stop) == 1 and "(faded)" in stop[0]
    assert not [m for m in logs if "hello" in m or "First sentence" in m]


def test_no_self_barge_in_when_only_the_robot_is_heard(caplog):
    """The mic hears only quiet (the echo the XVF3800 leaves): the whole reply plays, the margin is logged."""
    caplog.set_level(logging.INFO)
    bridge = SlowStream(gap_s=0.3, lines=[{"text": "One."}, {"text": "Two."}])
    spoken: list[str] = []
    handler = barge_handler(bridge.url, spoken, barge_in=True)

    async def main():
        startup = await started(handler)
        await paced(handler, np.concatenate([burst(0.6), quiet(1.0)]))
        for _ in range(200):
            if handler._armed:
                break
            await asyncio.sleep(0.01)
        await paced(handler, quiet(max(0.5, handler._play_end - time.monotonic() + 1.5)))
        await handler.shutdown()
        await asyncio.wait_for(startup, 5)

    try:
        asyncio.run(main())
    finally:
        bridge.close()
    assert spoken == ["One.", "Two."]
    logs = [r.getMessage() for r in caplog.records]
    assert not [m for m in logs if "barge-in after" in m]
    assert any(m.startswith("MuseHandler: reply played with the mic open, no barge-in; peak VAD 0.00") for m in logs)


def test_text_is_ready_at_end_of_speech_from_the_pause_pass(caplog):
    caplog.set_level(logging.INFO)
    bridge = SlowStream(gap_s=0.1, lines=[{"text": "Hi."}])
    spoken: list[str] = []
    heard: list[int] = []
    handler = barge_handler(bridge.url, spoken, heard, barge_in=False, stt_at_pauses=True)

    async def main():
        startup = await started(handler)
        # A short pause (not the end), more speech, then the end.
        await paced(handler, np.concatenate([burst(0.5), quiet(0.25), burst(0.5, seed=5), quiet(1.1)]))
        for _ in range(300):
            if spoken and any("matches the full pass" in r.getMessage() for r in caplog.records):
                break
            await asyncio.sleep(0.01)
        await handler.shutdown()
        await asyncio.wait_for(startup, 5)

    try:
        asyncio.run(main())
    finally:
        bridge.close()
    logs = [r.getMessage() for r in caplog.records]
    ready = [m for m in logs if m.startswith("MuseHandler: end of speech to text ready")]
    assert len(ready) == 1 and ready[0].endswith("(transcribed at the pause)")
    # The pass at the first pause was stale (more speech came); the turn used the pass at the last pause,
    # which covers both bursts.
    assert len(heard) == 3 and heard[0] < 16000 < heard[1] <= heard[2]
    assert any(m.startswith("MuseHandler: text from the pause matches the full pass: yes") for m in logs)


def test_barge_in_settings_from_env(monkeypatch):
    monkeypatch.delenv("MUSE_BARGE_IN", raising=False)
    monkeypatch.delenv("MUSE_BARGE_IN_STOP_MS", raising=False)
    assert muse_handler.barge_in_from_env() is True
    assert muse_handler.barge_in_stop_ms_from_env() == 350
    monkeypatch.setenv("MUSE_BARGE_IN", "0")
    assert muse_handler.barge_in_from_env() is False
    for bad in ("x", "-1", "2001"):
        monkeypatch.setenv("MUSE_BARGE_IN_STOP_MS", bad)
        assert muse_handler.barge_in_stop_ms_from_env() == 350
    monkeypatch.setenv("MUSE_BARGE_IN_STOP_MS", "250")
    assert muse_handler.barge_in_stop_ms_from_env() == 250


def test_drop_stream_ends_the_turn_with_what_was_said():
    bridge = SlowStream(gap_s=2.0)
    client = BridgeClient(bridge.url)
    got: list[str] = []

    def on_sentence(s):
        got.append(s)
        threading.Timer(0.05, client.drop_stream).start()

    try:
        t0 = time.perf_counter()
        reply = client.turn_stream("hi", on_sentence)
        took = time.perf_counter() - t0
    finally:
        bridge.close()
    assert reply == "First sentence." and got == ["First sentence."]
    assert took < 1.0, "didn't wait for the second sentence"


def test_a_pause_pass_is_dropped_when_its_utterance_is():
    """A pause pass must only ever be used by the utterance it came from."""
    handler = muse_handler.MuseHandler(FakeDeps(), bridge=BridgeClient("http://127.0.0.1:9"),
                                       transcriber=lambda a: "x", synthesizer=tone,
                                       segmenter=UtteranceSegmenter(EnergyVad()), barge_in=False)

    async def main():
        startup = await started(handler)
        stale = asyncio.get_running_loop().create_future()
        stale.set_result("old")
        handler._early = (stale, 12)
        await paced(handler, quiet(0.1))            # nothing in progress: dropped
        after_quiet = handler._early
        handler._early = (stale, 12)
        handler._busy = True                        # the mic closes (a turn starts): dropped
        await paced(handler, burst(0.1))
        after_closed = handler._early
        await handler.shutdown()
        await asyncio.wait_for(startup, 5)
        return after_quiet, after_closed

    assert asyncio.run(main()) == (None, None)


def speech_with_pauses() -> np.ndarray:
    """Talk, short pauses (under the end silence), a long pause (the end), then talk again."""
    return np.concatenate([quiet(0.3), burst(0.6), quiet(0.3), burst(0.4, seed=6), quiet(0.6), burst(0.5, seed=7),
                           quiet(1.2), burst(0.7, seed=8), quiet(1.2)])


def utterances_seen(stt_at_pauses: bool) -> tuple[list[np.ndarray], int]:
    """The utterances MuseHandler queued as turns, and how many STT calls it made."""
    calls: list[int] = []

    def slow_transcribe(audio):
        calls.append(audio.size)
        time.sleep(0.05)
        return "x"

    handler = muse_handler.MuseHandler(FakeDeps(), bridge=BridgeClient("http://127.0.0.1:9"),
                                       transcriber=slow_transcribe, synthesizer=tone,
                                       segmenter=UtteranceSegmenter(EnergyVad()), barge_in=False,
                                       stt_at_pauses=stt_at_pauses)
    turns: list[np.ndarray] = []

    async def record(turn):
        if turn.early is not None:
            await turn.early
        turns.append(turn.utterance)

    handler._run_turn = record

    async def main():
        startup = await started(handler)
        await paced(handler, speech_with_pauses())
        await asyncio.sleep(0.2)
        await handler.shutdown()
        await asyncio.wait_for(startup, 5)

    asyncio.run(main())
    return turns, len(calls)


def test_pause_passes_never_end_or_split_an_utterance():
    """The utterance boundaries are the VAD's alone, with or without speech-to-text at pauses."""
    vad_only = UtteranceSegmenter(EnergyVad()).feed(speech_with_pauses())
    with_pauses, pause_calls = utterances_seen(stt_at_pauses=True)
    full_pass, full_calls = utterances_seen(stt_at_pauses=False)
    assert len(vad_only) == 2
    for got in (with_pauses, full_pass):
        assert len(got) == len(vad_only)
        assert all(np.array_equal(a, b) for a, b in zip(got, vad_only))
    assert pause_calls > 2, "pause passes ran (at the short pauses and the ends)"
    assert full_calls == 0, "default: no pause passes (the turn itself does the one full pass)"


def test_one_full_pass_is_the_default(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    monkeypatch.delenv("MUSE_STT_AT_PAUSES", raising=False)
    bridge = SlowStream(gap_s=0.1, lines=[{"text": "Hi."}])
    spoken: list[str] = []
    heard: list[int] = []
    handler = barge_handler(bridge.url, spoken, heard, barge_in=False)
    assert not handler.stt_at_pauses
    monkeypatch.setenv("MUSE_STT_AT_PAUSES", "1")
    assert barge_handler(bridge.url, []).stt_at_pauses, "--stt-at-pauses turns them on"

    async def main():
        startup = await started(handler)
        await paced(handler, np.concatenate([burst(0.5), quiet(0.25), burst(0.5, seed=5), quiet(1.1)]))
        for _ in range(300):
            if spoken:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)
        await handler.shutdown()
        await asyncio.wait_for(startup, 5)

    try:
        asyncio.run(main())
    finally:
        bridge.close()
    logs = [r.getMessage() for r in caplog.records]
    assert len(heard) == 1 and heard[0] > 16000, "one pass, over both bursts"
    ready = [m for m in logs if m.startswith("MuseHandler: end of speech to text ready")]
    assert len(ready) == 1 and ready[0].endswith("(transcribed after the end silence)")
    assert not [m for m in logs if "matches the full pass" in m]


# ------------------------------------------------------------------ floor control: the user's voice always wins
def numbered(texts):
    """A transcriber that returns the given texts in order (one per pass), optionally slowly."""
    queue = list(texts)

    def transcribe(audio):
        text, delay = queue.pop(0)
        time.sleep(delay)
        return text
    return transcribe


def floor_handler(url, spoken, transcribe, deps=None, **kwargs):
    def synth(text, rate):
        spoken.append(text)
        return tone(text, rate)

    return muse_handler.MuseHandler(deps or FakeDeps(), bridge=BridgeClient(url), transcriber=transcribe,
                                    synthesizer=synth, segmenter=UtteranceSegmenter(EnergyVad()), barge_in=True,
                                    **kwargs)


async def settle(handler, until, timeout_s=8.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not until():
        await asyncio.sleep(0.02)


def floor_logs(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("MuseHandler")]


@pytest.mark.parametrize("slow_stt", [False, True], ids=["while-muse-thinks", "while-transcribing"])
def test_speech_before_the_reply_joins_the_two_utterances_into_one_turn(caplog, slow_stt):
    """Talk, pause past the end silence, talk again before the reply: one turn with both texts."""
    caplog.set_level(logging.INFO)
    bridge = SlowStream(lines=[{"text": "The answer."}], think_s=1.2)
    spoken: list[str] = []
    handler = floor_handler(bridge.url, spoken, numbered([("first part", 0.9 if slow_stt else 0.0),
                                                          ("second part", 0.0)]))

    async def main():
        startup = await started(handler)
        await paced(handler, np.concatenate([burst(0.5), quiet(1.0)]))   # utterance 1 ends, its turn starts
        await paced(handler, np.concatenate([burst(0.5, seed=5), quiet(1.0)]))   # more, before any reply
        await settle(handler, lambda: spoken and not handler._turn_active and not handler._pending)
        await handler.shutdown()
        await asyncio.wait_for(startup, 5)

    try:
        asyncio.run(main())
    finally:
        bridge.close()
    assert bridge.texts[-1] == "first part second part", "both utterances went to Muse as one turn"
    if slow_stt:
        assert bridge.texts == ["first part second part"], "the first was held back before it was sent"
    else:
        assert bridge.texts == ["first part", "first part second part"], "the first was sent, then dropped"
    assert spoken == ["The answer."], "only the joined turn's reply was spoken"
    logs = floor_logs(caplog)
    assert any(m.startswith("MuseHandler: continuation: speech ") for m in logs)
    assert any(m == "MuseHandler: continuation: 10 + 11 chars sent as one turn" for m in logs)
    assert not [m for m in logs if any(t in m for t in ("first part", "second part", "The answer."))], \
        "lengths and timings only"


def test_speech_while_muse_thinks_is_never_lost(caplog):
    """A follow-up too short to be a turn (a cough, "uh") still lets the held text go to Muse."""
    caplog.set_level(logging.INFO)
    bridge = SlowStream(lines=[{"text": "The answer."}], think_s=1.0)
    spoken: list[str] = []
    handler = floor_handler(bridge.url, spoken, numbered([("first part", 0.0)]))

    async def main():
        startup = await started(handler)
        await paced(handler, np.concatenate([burst(0.5), quiet(1.0)]))
        await paced(handler, np.concatenate([burst(0.15, seed=5), quiet(1.0)]))   # under the 0.3 s minimum
        await settle(handler, lambda: spoken and not handler._turn_active and not handler._pending)
        await handler.shutdown()
        await asyncio.wait_for(startup, 5)

    try:
        asyncio.run(main())
    finally:
        bridge.close()
    assert bridge.texts == ["first part", "first part"], "dropped at the follow-up, then sent again alone"
    assert spoken == ["The answer."]
    assert any("the follow-up wasn't a turn" in m for m in floor_logs(caplog))


class MovingManager:
    """Pollen's MovementManager as MuseHandler sees it: a running move, a queue, clear_move_queue()."""

    def __init__(self):
        self.state = type("State", (), {"current_move": None})()
        self.move_queue: list = []
        self.cleared: list[float] = []

    def clear_move_queue(self):
        self.cleared.append(time.monotonic())
        self.state.current_move = None
        self.move_queue.clear()


class DanceQueueMove:
    pass


class BreathingMove:
    pass


def moving_deps(mm):
    return type("Deps", (), {"reachy_mini": FakeDeps.reachy_mini, "movement_manager": mm})()


def start_move(handler, mm, move=None):
    mm.state.current_move = move or DanceQueueMove()
    handler._muse_moved_at = time.monotonic() - 1.0   # as _dispatch_robot_tool sets it, a second ago


def test_speech_stops_a_move_muse_started_and_refuses_its_next_ones(caplog):
    caplog.set_level(logging.INFO)
    mm = MovingManager()
    handler = floor_handler("http://127.0.0.1:9", [], numbered([("x", 0.0)]), deps=moving_deps(mm))

    async def main():
        startup = await started(handler)
        start_move(handler, mm)
        await paced(handler, quiet(0.5))
        assert not mm.cleared, "no speech: the move keeps going"
        await paced(handler, burst(0.2))
        assert len(mm.cleared) == 1, "speech onset: stopped at once"
        refused = await handler._dispatch_robot_tool("dance", {"move": "simple_nod"})
        await paced(handler, quiet(1.0))
        await handler.shutdown()
        await asyncio.wait_for(startup, 5)
        return refused

    refused = asyncio.run(main())
    assert "error" in refused, "a move Muse asks for while the user talks isn't started"
    logs = floor_logs(caplog)
    assert any(m.startswith("MuseHandler: move stopped by user speech, ") for m in logs)


def test_no_move_no_stop_and_breathing_is_not_a_move():
    mm = MovingManager()
    handler = floor_handler("http://127.0.0.1:9", [], numbered([("x", 0.0)]), deps=moving_deps(mm))

    async def main():
        startup = await started(handler)
        await paced(handler, burst(0.2))                 # nothing running: nothing to stop
        await paced(handler, quiet(1.0))
        start_move(handler, mm, BreathingMove())         # Pollen's idle breathing isn't Muse's move
        await paced(handler, burst(0.2, seed=5))
        await paced(handler, quiet(1.0))
        await handler.shutdown()
        await asyncio.wait_for(startup, 5)

    asyncio.run(main())
    assert mm.cleared == []


def test_the_robots_own_voice_never_stops_a_move(caplog):
    """A move runs while the reply plays and the mic hears only the echo: neither is stopped."""
    caplog.set_level(logging.INFO)
    mm = MovingManager()
    bridge = SlowStream(gap_s=0.3, lines=[{"text": "One."}, {"text": "Two."}])
    spoken: list[str] = []
    handler = floor_handler(bridge.url, spoken, numbered([("hello", 0.0)]), deps=moving_deps(mm))

    async def main():
        startup = await started(handler)
        await paced(handler, np.concatenate([burst(0.6), quiet(1.0)]))
        await settle(handler, lambda: handler._armed)
        start_move(handler, mm)
        await paced(handler, quiet(max(0.5, handler._play_end - time.monotonic() + 1.5)))
        await handler.shutdown()
        await asyncio.wait_for(startup, 5)

    try:
        asyncio.run(main())
    finally:
        bridge.close()
    assert spoken == ["One.", "Two."] and mm.cleared == []
    assert not [m for m in floor_logs(caplog) if "barge-in after" in m or "move stopped" in m]


def test_barge_in_also_stops_the_move_at_once():
    mm = MovingManager()
    bridge = SlowStream(gap_s=1.5, lines=[{"text": "First sentence of a long reply."}, {"text": "Later."}])
    spoken: list[str] = []
    handler = floor_handler(bridge.url, spoken, numbered([("hello", 0.0), ("stop", 0.0)]), deps=moving_deps(mm),
                            barge_in_stop_ms=300)
    handler._clear_queue = lambda: None

    async def main():
        startup = await started(handler)
        await paced(handler, np.concatenate([burst(0.6), quiet(1.0)]))
        await settle(handler, lambda: handler._armed)
        start_move(handler, mm)
        await paced(handler, burst(0.12, seed=3))
        moved_stop, barged = list(mm.cleared), handler._barge_at
        await paced(handler, np.concatenate([burst(0.4, seed=4), quiet(1.2)]))
        await settle(handler, lambda: len(bridge.paths) == 2 and not handler._turn_active)
        await handler.shutdown()
        await asyncio.wait_for(startup, 5)
        return moved_stop, barged

    try:
        moved_stop, barged = asyncio.run(main())
    finally:
        bridge.close()
    assert len(moved_stop) == 1 and barged is not None, "the move stops at once; the voice after the stop delay"
    assert bridge.texts == ["hello", "stop"], "a barge-in is still its own next turn (no joining)"
