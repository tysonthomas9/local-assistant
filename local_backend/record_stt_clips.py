"""Record speech-to-text test clips through the robot's microphone.

The robot speaks each prompt (Piper TTS), beeps, then records a 16 kHz clip with arecord
through the shared `reachymini_audio_src` device. Run under the `audio` group:

    sg audio -c "reachy_mini_conversation_app/.venv/bin/python local_backend/record_stt_clips.py"
"""

import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent / "tests" / "fixtures" / "stt"
VOICE = ROOT / "voices" / "en_US-lessac-medium.onnx"
PIPER = ROOT / ".venv" / "bin" / "python"
SINK, SRC = "plug:reachymini_audio_sink", "plug:reachymini_audio_src"
SECONDS = 6

CLIPS = [
    ("en_reachy_joke", "Please say: Hey Reachy, tell me a joke."),
    ("en_reachy_look_dance", "Please say: Reachy, look to your left and then dance."),
    ("en_weather_paris", "Please say: What's the weather like in Paris today?"),
    ("en_free", "Now say any sentence you like, in English."),
    ("kn_free_1", "Now say something in Kannada."),
    ("kn_free_2", "One more sentence in Kannada, please."),
]


def speak(text: str) -> None:
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        code = ("import wave,sys; from piper import PiperVoice; v=PiperVoice.load(sys.argv[1]);"
                "w=wave.open(sys.argv[2],'wb'); v.synthesize_wav(sys.argv[3], w); w.close()")
        subprocess.run([str(PIPER), "-c", code, str(VOICE), tmp.name, text], check=True, capture_output=True)
        subprocess.run(["aplay", "-q", "-D", SINK, tmp.name], check=True)


def beep() -> None:
    t = np.arange(int(16000 * 0.25)) / 16000
    tone = (0.3 * 32767 * np.sin(2 * np.pi * 880 * t)).astype(np.int16)
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        with wave.open(tmp.name, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(tone.tobytes())
        subprocess.run(["aplay", "-q", "-D", SINK, tmp.name], check=True)


def main(names: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, prompt in CLIPS:
        if names and name not in names:
            continue
        speak(prompt)
        beep()
        path = OUT / f"{name}.wav"
        subprocess.run(["arecord", "-q", "-D", SRC, "-f", "S16_LE", "-r", "16000", "-c", "1",
                        "-d", str(SECONDS), str(path)], check=True)
        with wave.open(str(path)) as w:
            a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float)
        print(f"{name}: {len(a)/16000:.1f}s, rms {20*np.log10(max(np.sqrt((a**2).mean()),1)/32768):.1f} dBFS", flush=True)
    speak("Thank you, that's all.")


if __name__ == "__main__":
    main(sys.argv[1:])
