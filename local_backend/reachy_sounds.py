"""Sound effects for the robot: alarm, timer, chime, bell, ... played on the robot's speaker.

Used by the play_sound / stop_sound tools and by reachy_scheduler (a reminder or timer rings
before the robot speaks). Everything is local:

  - generated sounds (alarm, timer, chime, bell, beep, success, error): synthesised with numpy on
    first use into local_backend/sounds/generated/ (git-ignored, deterministic, license-free);
  - the reachy_mini SDK's bundled sounds (wake_up, go_sleep, dance, confused, impatient, count);
  - your own files: any .wav/.mp3/.ogg/.flac in local_backend/sounds/custom/, by file name.

Playback is a GStreamer playbin into the shared `reachymini_audio_sink` dmix device (see
~/.asoundrc), so it mixes with the robot's voice and the radio. One sound plays at a time; a new
one replaces the old. Settings: REACHY_SOUND_VOLUME (0-100, default 70), REACHY_SOUND_SINK
(gst-launch sink description, e.g. "fakesink" for tests).
"""

from __future__ import annotations

import logging
import os
import threading
import time
import wave
from pathlib import Path

import numpy as np

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

logger = logging.getLogger("reachy_sounds")

HERE = Path(__file__).resolve().parent
GENERATED = HERE / "sounds" / "generated"
CUSTOM = HERE / "sounds" / "custom"
RATE = 16000  # the robot's audio board runs at 16 kHz
DEFAULT_SINK = "audioconvert ! audioresample ! alsasink device=reachymini_audio_sink"
AUDIO_EXT = (".wav", ".mp3", ".ogg", ".flac")


# -- synthesis -----------------------------------------------------------------------------------
def _t(seconds: float) -> np.ndarray:
    return np.arange(int(RATE * seconds)) / RATE


def _tone(freq: float, seconds: float, shape: str = "sine") -> np.ndarray:
    t = _t(seconds)
    x = np.sin(2 * np.pi * freq * t)
    if shape == "square":  # softened square: odd harmonics 1, 3, 5
        x = x + np.sin(2 * np.pi * 3 * freq * t) / 3 + np.sin(2 * np.pi * 5 * freq * t) / 5
    fade = min(len(t) // 10, int(RATE * 0.01))  # 10 ms fades, no clicks
    if fade:
        ramp = np.linspace(0, 1, fade)
        x[:fade] *= ramp
        x[-fade:] *= ramp[::-1]
    return x


def _bell(freq: float, seconds: float) -> np.ndarray:
    """Struck bell: inharmonic partials with exponential decay."""
    t = _t(seconds)
    partials = [(1.0, 1.0, 2.5), (2.76, 0.5, 4.0), (5.40, 0.25, 6.0), (0.5, 0.3, 1.5)]
    x = sum(a * np.exp(-d * t) * np.sin(2 * np.pi * freq * r * t) for r, a, d in partials)
    x[: int(RATE * 0.002)] *= np.linspace(0, 1, int(RATE * 0.002))
    return x


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(RATE * seconds))


def _cat(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts)


def _synthesise() -> dict[str, np.ndarray]:
    beep_pair = _cat(_tone(988, 0.12, "square"), _silence(0.06), _tone(988, 0.12, "square"), _silence(0.06))
    return {
        # Classic digital alarm: 4 fast double-beeps then a pause (one ~2 s cycle; repeated while ringing).
        "alarm": _cat(*[beep_pair] * 4, _silence(0.5)),
        # Kitchen timer: three bright dings.
        "timer": _cat(*[_cat(_bell(1568, 0.45), _silence(0.1)) for _ in range(3)], _bell(1568, 1.2)),
        # Gentle reminder chime: rising C6-E6-G6.
        "chime": _cat(_bell(1047, 0.35), _bell(1319, 0.35), _bell(1568, 1.4)),
        "bell": _bell(880, 2.0),
        "beep": _tone(880, 0.3),
        "success": _cat(_tone(784, 0.12), _tone(1175, 0.25)),
        "error": _cat(_tone(392, 0.18, "square"), _silence(0.04), _tone(294, 0.35, "square")),
    }


def ensure_generated() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    CUSTOM.mkdir(parents=True, exist_ok=True)
    for name, x in _synthesise().items():
        path = GENERATED / f"{name}.wav"
        if path.exists():
            continue
        pcm = (x / np.max(np.abs(x)) * 0.7 * 32767).astype(np.int16)  # peak -3 dBFS
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(pcm.tobytes())


def _sdk_assets() -> dict[str, Path]:
    try:
        import reachy_mini
    except ImportError:
        return {}
    assets = Path(reachy_mini.__file__).parent / "assets"
    names = {"wake_up": "wake_up", "go_sleep": "go_sleep", "dance1": "dance", "confused1": "confused",
             "impatient1": "impatient", "count": "count"}
    return {nice: assets / f"{stem}.wav" for stem, nice in names.items() if (assets / f"{stem}.wav").exists()}


def catalog() -> dict[str, Path]:
    """name -> file. Custom files override generated/SDK sounds with the same name."""
    ensure_generated()
    out = {p.stem: p for p in sorted(GENERATED.glob("*.wav"))}
    out.update(_sdk_assets())
    out.update({p.stem.lower(): p for p in sorted(CUSTOM.iterdir()) if p.suffix.lower() in AUDIO_EXT})
    return out


def resolve(name: str) -> tuple[str, Path] | None:
    """Exact name, else a close match ('alarm clock' -> alarm, 'kitchen timer' -> timer)."""
    cat = catalog()
    key = (name or "").strip().lower().replace(" ", "_")
    if key in cat:
        return key, cat[key]
    for k in cat:
        if k in key or key in k:
            return k, cat[k]
    return None


# -- playback ------------------------------------------------------------------------------------
class Player:
    def __init__(self) -> None:
        Gst.init(None)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.current: str | None = None

    def _play_once(self, path: Path, volume: float) -> bool:
        """Play one pass; False if stopped early."""
        player = Gst.ElementFactory.make("playbin", None)
        player.set_property("uri", path.as_uri())
        player.set_property("volume", volume)
        player.set_property("audio-sink", Gst.parse_bin_from_description(os.environ.get("REACHY_SOUND_SINK", DEFAULT_SINK), True))
        player.set_property("video-sink", Gst.ElementFactory.make("fakesink", None))
        player.set_state(Gst.State.PLAYING)
        bus = player.get_bus()
        try:
            while not self._stop.is_set():
                msg = bus.timed_pop_filtered(100 * Gst.MSECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
                if msg is None:
                    continue
                if msg.type == Gst.MessageType.ERROR:
                    logger.warning("Sound %s failed: %s", path.name, msg.parse_error()[0].message)
                return True
            return False
        finally:
            player.set_state(Gst.State.NULL)

    def _loop(self, name: str, path: Path, volume: float, repeat: int, max_seconds: float) -> None:
        start = time.monotonic()
        try:
            for i in range(repeat):
                if not self._play_once(path, volume):
                    break
                if time.monotonic() - start >= max_seconds:
                    break
        finally:
            with self._lock:
                if self.current == name:
                    self.current = None

    def play(self, name: str, *, volume_pct: float | None = None, repeat: int = 1,
             max_seconds: float = 60, wait: bool = False) -> dict:
        """Start a sound (replacing any playing one). repeat=0 means 'until stopped or max_seconds'."""
        found = resolve(name)
        if found is None:
            return {"error": f"No sound called {name!r}.", "available": sorted(catalog())}
        key, path = found
        volume = max(0.0, min(1.0, (volume_pct if volume_pct is not None
                                    else float(os.environ.get("REACHY_SOUND_VOLUME", 70))) / 100))
        self.stop()
        self._stop = threading.Event()
        repeat = repeat if repeat > 0 else 10_000
        with self._lock:
            self.current = key
            self._thread = threading.Thread(target=self._loop, args=(key, path, volume, repeat, max_seconds),
                                            daemon=True, name=f"reachy-sound-{key}")
            self._thread.start()
            thread = self._thread
        if wait:
            thread.join()
        return {"playing": key, "file": path.name, "volume": round(volume * 100)}

    def stop(self) -> str | None:
        with self._lock:
            current, thread = self.current, self._thread
        self._stop.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
        return current


PLAYER = Player()
