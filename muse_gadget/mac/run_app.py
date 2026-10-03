"""Run Pollen's conversation app (pinned, unmodified) with MuseHandler as its backend.

    python run_app.py [--log-transcripts] [--tts qwen3|kokoro|say] [--voice NAME] [--instruct TEXT]
                      [app args, e.g. --no-camera --debug]

--tts picks the reply voice engine (MUSE_TTS; default qwen3, which falls back to Kokoro and then
macOS `say` if it can't load), --voice its voice (MUSE_TTS_VOICE; Qwen3 default Aiden, Kokoro
default af_heart) and --instruct Qwen3's style instruction (MUSE_TTS_INSTRUCT; how to say it, not
what to say; "" for none). See muse_tts.py.

--log-transcripts (debugging only) lets the app log each turn's text (`role=... content=...`);
by default MuseHandler doesn't hand the text to the app's logger at all.

The app builds its backend in `main.run()` -> `build_handler()`, which imports
`HuggingFaceRealtimeHandler` from `reachy_mini_conversation_app.huggingface_realtime` each time
it is called. This launcher replaces that module attribute with `MuseHandler` before the app
starts, the same way local_backend/run_app.py wraps the app: no upstream file is edited.

The app only starts its audio loops once a Hugging Face realtime target is configured, so a
placeholder local target is set; nothing ever connects to it (MuseHandler talks to the Muse
bridge on MUSE_BRIDGE_URL, default http://127.0.0.1:48080). The app's --ui server would bind
0.0.0.0, so it is bound to 127.0.0.1 here as in local_backend/run_app.py.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Read by the app's config module at import time: set before any app import.
os.environ["HF_REALTIME_CONNECTION_MODE"] = "local"
os.environ["HF_REALTIME_WS_URL"] = "ws://127.0.0.1:9/muse-backend-unused"


def install() -> None:
    """Make the app build MuseHandler wherever it would build HuggingFaceRealtimeHandler."""
    import uvicorn
    from reachy_mini_conversation_app import huggingface_realtime

    from muse_handler import MuseHandler

    huggingface_realtime.HuggingFaceRealtimeHandler = MuseHandler  # type: ignore[misc]

    config_init = uvicorn.Config.__init__

    def config_init_on_loopback(self, app, *args, **kwargs):  # type: ignore[no-untyped-def]
        if kwargs.get("host") == "0.0.0.0":
            kwargs["host"] = "127.0.0.1"
        config_init(self, app, *args, **kwargs)

    uvicorn.Config.__init__ = config_init_on_loopback  # type: ignore[method-assign]


def take_own_flags(argv: list[str]) -> list[str]:
    """Strip our flags from argv (the app would reject them) and set their env vars."""
    rest: list[str] = []
    it = iter(argv)
    for a in it:
        if a == "--log-transcripts":
            os.environ["MUSE_LOG_TRANSCRIPTS"] = "1"
        elif a == "--instruct":
            value = next(it, None)
            if value is None:
                raise SystemExit("run_app: --instruct needs a value (\"\" for none)")
            os.environ["MUSE_TTS_INSTRUCT"] = value
        elif a in ("--tts", "--voice"):
            value = next(it, "")
            if not value:
                raise SystemExit(f"run_app: {a} needs a value")
            if a == "--tts" and value not in ("qwen3", "kokoro", "say"):
                raise SystemExit("run_app: --tts must be qwen3, kokoro or say")
            os.environ["MUSE_TTS" if a == "--tts" else "MUSE_TTS_VOICE"] = value
        else:
            rest.append(a)
    return rest


def main() -> int:
    sys.argv = take_own_flags(sys.argv)
    install()
    from reachy_mini_conversation_app.main import main as app_main

    sys.argv[0] = "reachy-mini-conversation-app"
    app_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
