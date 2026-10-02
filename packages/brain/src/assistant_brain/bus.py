"""The in-process, typed event bus (proposal section 5.4).

    bus = EventBus()
    bus.subscribe(TurnStarted, on_turn_started)       # by payload model
    bus.subscribe("turn.*", on_any_turn_event)        # or by topic / topic prefix
    await bus.publish(TurnStarted(session_id=..., turn_id=...))

Payloads are the Pydantic models of `assistant_contracts.events`. Handlers run in subscription
order, one after another, in the publisher's task; a handler that raises is logged and does not
stop the others (or the publisher). NATS can replace this later if skills move off the box.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar, overload

from assistant_contracts.events import BusEvent
from assistant_core.log import get_logger

E = TypeVar("E", bound=BusEvent)
Handler = Callable[[Any], Awaitable[None]]

log = get_logger("assistant_brain.bus")


@dataclass(frozen=True)
class _Subscription:
    pattern: str
    handler: Handler

    def matches(self, topic: str) -> bool:
        if self.pattern.endswith(".*"):
            return topic.startswith(self.pattern[:-1])
        return topic == self.pattern


class EventBus:
    def __init__(self) -> None:
        self._subs: list[_Subscription] = []

    @overload
    def subscribe(
        self, topic: type[E], handler: Callable[[E], Awaitable[None]]
    ) -> Callable[[], None]: ...

    @overload
    def subscribe(
        self, topic: str, handler: Callable[[BusEvent], Awaitable[None]]
    ) -> Callable[[], None]: ...

    def subscribe(self, topic: Any, handler: Any) -> Callable[[], None]:
        """Subscribe to a payload model's topic, an exact topic or a prefix (`turn.*`).

        Returns a function that removes the subscription.
        """
        pattern = topic if isinstance(topic, str) else topic.topic
        sub = _Subscription(pattern, handler)
        self._subs.append(sub)

        def unsubscribe() -> None:
            if sub in self._subs:
                self._subs.remove(sub)

        return unsubscribe

    async def publish(self, event: BusEvent, topic: str | None = None) -> None:
        """Deliver `event` on its model's topic (or `topic`, e.g. `scheduler.due.<skill>`)."""
        name = topic or event.topic
        for sub in list(self._subs):
            if not sub.matches(name):
                continue
            try:
                await sub.handler(event)
            except Exception:
                log.exception("bus handler failed", topic=name, handler=repr(sub.handler))
