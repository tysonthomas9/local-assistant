"""Run reachy-mini-conversation-app with its web UI bound to 127.0.0.1.

The app hard-codes `uvicorn.Config(..., host="0.0.0.0", port=7860)` for `--ui`
(reachy_mini_conversation_app/main.py), so anyone on the LAN could open the settings page and
switch the robot back to the hosted backend or to a profile with internet tools. Set
REACHY_UI_HOST=0.0.0.0 to restore LAN access.

Upstream code is unchanged: this rewrites only a 0.0.0.0 host passed to uvicorn.Config, then
calls the normal app entry point with the same arguments.
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

from reachy_mini_conversation_app.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
