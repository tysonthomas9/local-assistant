"""Make Reachy Mini say a sentence through its own speaker.

Usage: python say.py "Hello, I am Reachy Mini!"
"""

import subprocess
import sys
import tempfile
import wave
from pathlib import Path

from piper import PiperVoice
from reachy_mini import ReachyMini

VOICE = Path(__file__).parent / "voices" / "en_US-lessac-medium.onnx"
SPEAKER = "plughw:CARD=Audio,DEV=0"  # the "Reachy Mini Audio" USB card

text = " ".join(sys.argv[1:]) or "Hello! I am Reachy Mini."

voice = PiperVoice.load(VOICE)
with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
    with wave.open(tmp.name, "wb") as wav:
        voice.synthesize_wav(text, wav)

    with ReachyMini(media_backend="no_media") as mini:
        player = subprocess.Popen(["aplay", "-q", "-D", SPEAKER, tmp.name])
        # Wiggle the antennas while it talks.
        while player.poll() is None:
            mini.goto_target(antennas=[0.3, -0.3], duration=0.3)
            mini.goto_target(antennas=[-0.3, 0.3], duration=0.3)
        mini.goto_target(antennas=[0, 0], duration=0.3)
