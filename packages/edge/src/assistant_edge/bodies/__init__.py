"""Body drivers, found through the `assistant.bodies` entry point group.

The edge package registers `console`, `null` and `sounddevice`; `assistant-robot-reachy`
registers `reachy`. A body's module is imported only when that body is chosen.
"""

from importlib.metadata import entry_points
from typing import Any

from assistant_contracts.body import BODY_ENTRY_POINT_GROUP


def available() -> list[str]:
    return sorted(ep.name for ep in entry_points(group=BODY_ENTRY_POINT_GROUP))


def load_body(kind: str, **options: Any) -> Any:
    """Construct the body registered as `kind`."""
    for ep in entry_points(group=BODY_ENTRY_POINT_GROUP):
        if ep.name == kind:
            return ep.load()(**options)
    raise ValueError(f"no body {kind!r}; installed: {', '.join(available()) or 'none'}")
