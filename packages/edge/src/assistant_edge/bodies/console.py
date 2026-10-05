"""ConsoleBody: a text console as a body (a real body type, not a test double).

What the user would hear is printed: the reply text of each speech stream (`SAY`), and
attention and expressions as text (`SHOW`). Input is typed on the agent's stdin: plain text
becomes `text.input`, `/wake <word> <score>` is a wake event, `/ptt down` and `/ptt up` are
push-to-talk (see `assistant_edge.agent`). There is no microphone and no camera, so a
`snapshot` is answered `result{ok: false}`. Speech audio runs on the real playback clock into
no device, so the brain sees the same `playback` events as from a speaker.
"""

import json
import sys

from assistant_contracts.capabilities import BodyCapabilities, Capabilities, MotionCaps
from assistant_contracts.common import AttentionState, LookTarget
from assistant_contracts.expressions import ExpressName
from assistant_edge.bodies.null import NullAudio, NullBody


def show(tag: str, **fields: object) -> None:
    sys.stdout.write(f"{tag} {json.dumps(fields)}\n")
    sys.stdout.flush()


class ConsoleMotion:
    """Shows attention and expressions as text lines."""

    async def attention(self, state: AttentionState, assistant: str | None) -> None:
        show("SHOW", attention=state, assistant=assistant)

    async def express(self, name: str, intensity: float = 1.0) -> bool:
        show("SHOW", express=name, intensity=intensity)
        return True

    async def look_at(self, target: LookTarget) -> bool:
        del target
        return False


class ConsoleBody(NullBody):
    kind = "console"

    def __init__(self) -> None:
        super().__init__()
        self.audio = NullAudio()
        self.motion = ConsoleMotion()

    async def start(self) -> BodyCapabilities:
        expressions = [name.value for name in ExpressName]
        return Capabilities(
            motion=MotionCaps(expressions=expressions, attention=True),
            speak_text=True,
        )

    def say(self, stream_id: int, text: str) -> None:
        """Print the reply text of a speech stream (what a speaker would say)."""
        show("SAY", stream=stream_id, text=text)
