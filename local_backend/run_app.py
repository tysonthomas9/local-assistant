"""Run reachy-mini-conversation-app with its web UI bound to 127.0.0.1.

The app hard-codes `uvicorn.Config(..., host="0.0.0.0", port=7860)` for `--ui`
(reachy_mini_conversation_app/main.py), so anyone on the LAN could open the settings page and
switch the robot back to the hosted backend or to a profile with internet tools. Set
REACHY_UI_HOST=0.0.0.0 to restore LAN access.

It also installs reachy_bridge (see that module), which lets our tools reach the live
conversation stream.

Upstream code is unchanged: this rewrites only a 0.0.0.0 host passed to uvicorn.Config and wraps
three LocalStream methods, then calls the normal app entry point with the same arguments.
"""

import os
import sys

import uvicorn

HOST = os.environ.get("REACHY_UI_HOST", "127.0.0.1")
_config_init = uvicorn.Config.__init__


def _config_init_on_host(self, app, *args, **kwargs):
    if kwargs.get("host") == "0.0.0.0":
        kwargs["host"] = HOST
    _config_init(self, app, *args, **kwargs)


uvicorn.Config.__init__ = _config_init_on_host

# Give our tools access to the live conversation stream (mute, wake word, reader, personas).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reachy_bridge  # noqa: E402

reachy_bridge.install()

# Optional: several assistants picked by wake word (start_conversation.sh --assistants). Before the wake gate,
# which takes its words from the assistant registry.
if os.environ.get("REACHY_ASSISTANTS"):
    import reachy_assistants  # noqa: E402

    reachy_assistants.install()

# Optional wake-word gate (start_conversation.sh --wake / REACHY_WAKE_WORD).
if os.environ.get("REACHY_WAKE_WORD"):
    import reachy_wake  # noqa: E402

    _install = "uv pip install --python reachy_mini_conversation_app/.venv/bin/python "
    if reachy_wake.uses_openwakeword(os.environ["REACHY_WAKE_WORD"]):
        try:
            import openwakeword  # noqa: F401
        except ImportError:
            sys.exit("This wake word needs openWakeWord in the app venv: " + _install + "openwakeword")
    else:
        try:
            import sentencepiece  # noqa: F401
            import sherpa_onnx  # noqa: F401
        except ImportError:
            sys.exit("--wake needs sherpa-onnx in the app venv: " + _install + "sherpa-onnx sentencepiece")
        if not (reachy_wake.KWS_MODEL / "tokens.txt").exists():
            sys.exit(f"--wake needs the keyword-spotting model in {reachy_wake.KWS_MODEL}: run local_backend/cache_models.sh")

    reachy_wake.install()

from reachy_mini_conversation_app.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
