"""MuseHandler unit tests: run on the PC with the pinned app's venv (no robot, no Mac, no MLX).

    cd muse_gadget/mac && <app venv>/bin/python -m pytest tests -q
"""

from __future__ import annotations

import asyncio
import os
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


def test_recorded_utterance_becomes_a_spoken_reply(bridge_url):
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

    assert len(heard) == 1, "one utterance, one STT call"
    speech_s = read_wav_int16(WAV)[1].size / 16000
    assert speech_s - 0.3 < heard[0].size / 16000 < speech_s + 2.0
    assert ("user", "hello robot", True) in transcripts
    assert ("assistant", "You said: hello robot", True) in transcripts

    audio = [o for o in outputs if isinstance(o, tuple)]
    texts = [o.args[0] for o in outputs if isinstance(o, AdditionalOutputs)]
    assert {"role": "user", "content": "hello robot"} in texts
    assert {"role": "assistant", "content": "You said: hello robot"} in texts
    assert audio and all(rate == 16000 and pcm.dtype == np.int16 and pcm.ndim == 2 for rate, pcm in audio)
    assert sum(pcm.size for _, pcm in audio) == tone("You said: hello robot", 16000).size


def test_mic_is_ignored_while_the_robot_speaks(bridge_url):
    calls: list[int] = []

    def transcribe(audio: np.ndarray) -> str:
        calls.append(audio.size)
        return "hello robot"

    handler = muse_handler.MuseHandler(
        FakeDeps(), bridge=BridgeClient(bridge_url), transcriber=transcribe, synthesizer=tone,
        segmenter=UtteranceSegmenter(EnergyVad()),
    )
    # The same utterance twice, back to back: the second arrives while the first reply plays.
    frames = mic_frames(seconds_silence=1.0)
    asyncio.run(drive(handler, frames + frames))
    assert len(calls) == 1


def test_say_speaks_verbatim_without_a_turn():
    async def run():
        handler = muse_handler.MuseHandler(
            FakeDeps(), bridge=BridgeClient("http://127.0.0.1:9"), transcriber=lambda a: "",
            synthesizer=tone, segmenter=UtteranceSegmenter(EnergyVad()),
        )
        with pytest.raises(RuntimeError):
            await handler.say("hi")
        startup = asyncio.create_task(handler.start_up())
        while not handler._is_connected():
            await asyncio.sleep(0.01)
        await handler.say("Hello there")
        items = [handler.output_queue.get_nowait() for _ in range(handler.output_queue.qsize())]
        await handler.shutdown()
        await startup
        return items

    items = asyncio.run(run())
    assert items[0].args[0] == {"role": "assistant", "content": "Hello there"}
    assert sum(pcm.size for _, pcm in items[1:]) == tone("Hello there", 16000).size


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
    import subprocess

    env = dict(os.environ, PYTHONPATH=f"{sys.prefix}/gst:/keep/me", MUSE_PROBE=f"{sys.prefix}/only")
    out = subprocess.run(
        [sys.executable, str(HERE.parent / "exec_python.py"), "/bin/sh", "-c", 'echo "$PYTHONPATH|${MUSE_PROBE-unset}"'],
        env=env, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out == "/keep/me|unset"
