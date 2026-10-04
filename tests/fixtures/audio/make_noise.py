"""Make the non-speech clips of the listening features: claps, music and room noise.

    uv run python tests/fixtures/audio/make_noise.py

`clap.wav`: five hand claps (bursts of band-limited noise, each a few sub-bursts with a fast
decay, as a real clap); `music.wav`: four seconds of instrumental music (chords with
harmonics, a bass line, a kick drum and a hi-hat, 120 bpm). Both 16 kHz mono s16le, -6 dBFS
peak, with digital silence before and after; synthesized from a fixed seed (deterministic),
no recording. They are played through a real speaker next to the robot: an open microphone
must not take either for speech.

`room_noise.wav`: six seconds of room noise (a fan's hum and its broadband rush) at
-28 dBFS RMS, the middle of a real room's -25 to -31 dBFS; `hey_jarvis_in_noise.wav` and
`what_time_is_it_in_noise.wav`: the golden utterances (made by assistant_speech.golden, run
that first) over the same noise, 1.5 s of it before and after: the wake word and the open mic
must work in it.

`hey_jarvis_then_what_time_is_it.wav`: the bare wake word, 3 s of quiet, then the question (a
wake followed by a pause); `tell_me_a_story_with_pause.wav`: "Please tell me a long story
about a" ... 1.3 s of quiet ... "little robot who learns to paint." (a pause mid-sentence,
which must not end the turn). Both joined from the golden utterances, no noise.
"""

import wave
from pathlib import Path

import numpy as np

RATE = 16000
OUT = Path(__file__).parent


def _write(name: str, audio: np.ndarray) -> None:
    audio = audio / (float(np.abs(audio).max()) or 1.0) * 0.5
    lead, tail = np.zeros(int(RATE * 0.3)), np.zeros(int(RATE * 1.0))
    pcm = (np.concatenate([lead, audio, tail]) * 32767).astype("<i2").tobytes()
    with wave.open(str(OUT / name), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(pcm)
    print(f"wrote {OUT / name}: {len(pcm) / 2 / RATE:.2f} s")


def _bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / RATE)
    spectrum[(freqs < lo) | (freqs > hi)] = 0
    return np.fft.irfft(spectrum, len(x))


def claps(rng: np.random.Generator) -> np.ndarray:
    out = np.zeros(int(RATE * 2.6))
    t = np.arange(int(RATE * 0.12)) / RATE
    for k in range(5):
        start = int(RATE * (0.05 + 0.5 * k))
        clap = np.zeros_like(t)
        for sub in range(4):  # a clap is a few reflections a millisecond or two apart
            delay = int(RATE * (0.0015 * sub + rng.uniform(0, 0.001)))
            burst = rng.standard_normal(len(t) - delay) * np.exp(-t[: len(t) - delay] / 0.012)
            clap[delay:] += burst * (0.8**sub)
        out[start : start + len(t)] += _bandpass(clap, 700, 4000)
    return out


def music(rng: np.random.Generator) -> np.ndarray:
    seconds, beat = 4.0, 0.5
    n = int(RATE * seconds)
    t = np.arange(n) / RATE
    out = np.zeros(n)
    chords = [
        (261.6, 329.6, 392.0),
        (220.0, 261.6, 329.6),
        (174.6, 220.0, 261.6),
        (196.0, 246.9, 293.7),
    ]
    for bar, chord in enumerate(chords):
        start, end = int(RATE * bar * 1.0), int(RATE * (bar + 1) * 1.0)
        seg = t[: end - start]
        env = np.exp(-seg / 0.6)
        for f in chord:  # a plucked chord: a few harmonics, decaying
            for h, amp in ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12)):
                out[start:end] += amp * env * np.sin(2 * np.pi * f * h * seg)
        bass = chord[0] / 2
        out[start:end] += 1.2 * np.exp(-seg / 0.3) * np.sin(2 * np.pi * bass * seg)
    kick_t = np.arange(int(RATE * 0.15)) / RATE
    kick = np.sin(2 * np.pi * (50 + 80 * np.exp(-kick_t / 0.03)) * kick_t) * np.exp(-kick_t / 0.05)
    hat_t = np.arange(int(RATE * 0.04)) / RATE
    for k in range(int(seconds / beat)):
        at = int(RATE * k * beat)
        out[at : at + len(kick)] += 2.0 * kick[: n - at]
        off = at + int(RATE * beat / 2)
        if off + len(hat_t) <= n:
            hat = _bandpass(rng.standard_normal(len(hat_t)), 5000, 7900) * np.exp(-hat_t / 0.01)
            out[off : off + len(hat_t)] += 0.6 * hat
    return out


NOISE_DBFS = -28.0


def room_noise(rng: np.random.Generator, seconds: float) -> np.ndarray:
    n = int(RATE * seconds)
    t = np.arange(n) / RATE
    rush = _bandpass(rng.standard_normal(n), 60, 6000)
    rumble = _bandpass(np.cumsum(rng.standard_normal(n)), 20, 400)
    rumble /= float(np.std(rumble)) or 1.0
    hum = sum(np.sin(2 * np.pi * f * t) / k for k, f in enumerate((120.0, 240.0, 360.0), 1))
    noise = rush + 0.8 * rumble + 0.3 * hum
    rms = float(np.sqrt(np.mean(noise**2)))
    return noise / rms * 10 ** (NOISE_DBFS / 20)


def _read(name: str) -> np.ndarray:
    with wave.open(str(OUT / name)) as wav:
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2") / 32768.0


def _write_raw(name: str, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    with wave.open(str(OUT / name), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(pcm)
    print(f"wrote {OUT / name}: {len(pcm) / 2 / RATE:.2f} s")


def in_noise(rng: np.random.Generator, speech: np.ndarray) -> np.ndarray:
    pad = int(RATE * 1.5)
    noise = room_noise(rng, (len(speech) + 2 * pad) / RATE)
    noise[pad : pad + len(speech)] += speech
    return noise


def joined(first: str, gap_s: float, second: str) -> np.ndarray:
    """`first` and `second` (golden WAVs, each with its own lead and tail of silence) with
    silence added between them, so the quiet from speech to speech is `gap_s` more."""
    return np.concatenate([_read(first), np.zeros(int(RATE * gap_s)), _read(second)])


def main() -> None:
    rng = np.random.default_rng(8)
    _write("clap.wav", claps(rng))
    _write("music.wav", music(rng))
    _write_raw("room_noise.wav", room_noise(rng, 6.0))
    for name in ("hey_jarvis_whats_your_name", "what_time_is_it"):
        stem = name.removesuffix("_whats_your_name")
        _write_raw(f"{stem}_in_noise.wav", in_noise(rng, _read(f"{name}.wav")))
    # hey_jarvis ends with 1.0 s of silence and what_time_is_it starts with 0.3 s: 3 s in all.
    _write_raw("hey_jarvis_then_what_time_is_it.wav",
               joined("hey_jarvis.wav", 1.7, "what_time_is_it.wav"))  # fmt: skip
    _write_raw("tell_me_a_story_with_pause.wav", joined("story_start.wav", 0.0, "story_end.wav"))


if __name__ == "__main__":
    main()
