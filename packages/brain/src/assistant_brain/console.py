"""The brain's event lines on stdout: `TAG key=value ... [json]`.

The same line format as the link and edge consoles (`assistant_link.console`), so operators
and the e2e features read one format everywhere. Tags: LISTENING, CONNECTED, DISCONNECTED,
REFUSED, WAKE, ATTENTION, TURN-START, TURN-STATE, TURN-END, SPEAK, SPEECH-QUEUED,
SPEECH-DROPPED, LLM-GATE, BACKGROUND, BRAIN-ERROR, STOPPED.
"""

import json
import sys
from typing import Any


def emit(tag: str, payload: Any = None, **fields: object) -> None:
    parts = [tag, *(f"{k}={v}" for k, v in fields.items())]
    if payload is not None:
        parts.append(payload if isinstance(payload, str) else json.dumps(payload, default=str))
    sys.stdout.write(" ".join(parts) + "\n")
    sys.stdout.flush()
