"""BodyController: body intents on the bus -> `attention` / `express` / `look_at` to the edge.

The DialogManager publishes `body.intent` (kind attention) on every turn-state change; skills
(S5) publish express and look_at. The controller sends each to its device, drops what the
device's body cannot do (its `hello` capabilities), and sends an attention state only when it
changes (the same state twice in a row is one message).
"""

from collections.abc import Awaitable, Callable

from assistant_brain.bus import EventBus
from assistant_brain.console import emit
from assistant_contracts.capabilities import Capabilities
from assistant_contracts.common import AttentionState
from assistant_contracts.events import BodyIntent
from assistant_contracts.messages import Attention, Envelope, Express, LookAt

Sender = Callable[[str, Envelope], Awaitable[bool]]
"""Sends a message to a device; False if it is not connected."""


class BodyController:
    def __init__(self, bus: EventBus, send: Sender) -> None:
        self.send = send
        self.capabilities: dict[str, Capabilities] = {}
        self.attention: dict[str, AttentionState] = {}
        bus.subscribe(BodyIntent, self.on_intent)

    def device_connected(self, device_id: str, capabilities: Capabilities) -> None:
        """A (re)connected edge: its capabilities, and no attention state sent yet."""
        self.capabilities[device_id] = capabilities
        self.attention.pop(device_id, None)

    def device_gone(self, device_id: str) -> None:
        self.capabilities.pop(device_id, None)
        self.attention.pop(device_id, None)

    async def on_intent(self, intent: BodyIntent) -> None:
        caps = self.capabilities.get(intent.device_id)
        if caps is None:
            return
        motion = caps.motion
        message: Envelope
        if intent.kind == "attention":
            assert intent.state is not None
            if self.attention.get(intent.device_id) == intent.state:
                return
            self.attention[intent.device_id] = intent.state
            emit(
                "ATTENTION",
                device=intent.device_id,
                state=intent.state,
                sent=str(bool(motion and motion.attention)).lower(),
            )
            if motion is None or not motion.attention:
                return
            message = Attention(state=intent.state, assistant=intent.assistant)
        elif intent.kind == "express":
            if motion is None or intent.name is None or intent.name not in motion.expressions:
                return
            message = Express(name=intent.name, intensity=intent.intensity)
        else:
            if motion is None or intent.target is None or not motion.look_at:
                return
            message = LookAt(target=intent.target)
        await self.send(intent.device_id, message)
