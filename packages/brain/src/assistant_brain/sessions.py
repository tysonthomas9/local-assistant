"""SessionManager: the brain's EdgeLink handler, one EdgeSession per connected edge.

A device that reconnects gets a new session (a new `session_id` and an idle DialogManager);
the old one is closed when its connection goes away. After `welcome` the session tells the
edge its attention state (idle), so a restarted brain or a reconnected edge starts in a known
state. Messages and frames are routed to the session's DialogManager; a wake goes through the
Router first (the session's assistant follows the wake word).
"""

from dataclasses import dataclass, field

from assistant_brain.body_controller import BodyController
from assistant_brain.bus import EventBus
from assistant_brain.console import emit
from assistant_brain.dialog import DialogManager
from assistant_brain.engine import EngineSession, SessionInfo, TurnEngine
from assistant_brain.router import Router
from assistant_brain.turnlog import TurnLog
from assistant_contracts.capabilities import Capabilities
from assistant_contracts.events import (
    EdgeConnected,
    EdgeDisconnected,
    PrivacyChanged,
    SpeechRequest,
    WakeDetected,
    WakeRouted,
)
from assistant_contracts.frames import Frame
from assistant_contracts.messages import (
    EdgeEvent,
    Envelope,
    Error,
    Hello,
    Playback,
    Privacy,
    Result,
    TextInput,
    Vad,
    Wake,
    Welcome,
    WelcomeAudio,
)
from assistant_core.config import AssistantDef
from assistant_link.connection import Connection, LinkClosed


@dataclass
class EdgeSession:
    session_id: str
    device_id: str
    conn: Connection
    capabilities: Capabilities
    body_kind: str
    assistant: AssistantDef
    engine: EngineSession | None = None
    dialog: DialogManager | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, object]:
        dialog = self.dialog
        return {
            "session_id": self.session_id,
            "device_id": self.device_id,
            "body": self.body_kind,
            "assistant": self.assistant.id,
            "state": dialog.state if dialog else None,
            "muted": dialog.muted if dialog else False,
            "queued_speech": len(dialog.queue) if dialog else 0,
            "capabilities": self.capabilities.model_dump(mode="json"),
        }


class SessionManager:
    def __init__(
        self,
        *,
        bus: EventBus,
        router: Router,
        engine: TurnEngine,
        turns: TurnLog,
        follow_up_s: float,
    ) -> None:
        self.bus = bus
        self.router = router
        self.engine = engine
        self.turns = turns
        self.follow_up_s = follow_up_s
        self.sessions: dict[str, EdgeSession] = {}
        """device_id -> its current session."""
        self.body = BodyController(bus, self.send)
        bus.subscribe(SpeechRequest, self.on_speech_request)

    async def send(self, device_id: str, message: Envelope) -> bool:
        session = self.sessions.get(device_id)
        if session is None:
            return False
        try:
            await session.conn.send(message)
        except LinkClosed:
            return False
        return True

    def _current(self, conn: Connection) -> EdgeSession | None:
        session = self.sessions.get(conn.device_id)
        return session if session is not None and session.conn is conn else None

    # ------------------------------------------------------------ LinkHandler

    async def on_connect(self, conn: Connection, hello: Hello) -> Welcome:
        caps = hello.body.capabilities
        old = self.sessions.pop(hello.device_id, None)
        if old is not None:
            await self._close(old)
        session = EdgeSession(
            session_id=conn.session_id,
            device_id=hello.device_id,
            conn=conn,
            capabilities=caps,
            body_kind=hello.body.kind,
            assistant=self.router.default,
        )
        session.engine = await self.engine.open_session(
            SessionInfo(conn.session_id, hello.device_id, session.assistant)
        )

        async def send(message: Envelope, device: str = hello.device_id) -> bool:
            return await self.send(device, message)

        session.dialog = DialogManager(
            session, session.engine, self.engine.name, self.bus, self.turns, self.follow_up_s, send
        )
        self.sessions[hello.device_id] = session
        self.body.device_connected(hello.device_id, caps)
        emit(
            "CONNECTED",
            caps.model_dump(mode="json"),
            device=hello.device_id,
            session=conn.session_id,
            body=hello.body.kind,
        )
        await self.bus.publish(EdgeConnected(device_id=hello.device_id, capabilities=caps))
        return Welcome(
            session_id=conn.session_id,
            wake_words=self.router.wake_words(),
            follow_up_max_s=self.follow_up_s,
            audio=WelcomeAudio(speak_text=caps.speak_text),
        )

    async def on_welcomed(self, conn: Connection) -> None:
        session = self._current(conn)
        if session is not None and session.dialog is not None:
            await session.dialog.start()

    async def on_message(self, conn: Connection, message: Envelope) -> None:
        session = self._current(conn)
        if session is None or session.dialog is None:
            return
        dialog = session.dialog
        match message:
            case Wake():
                assistant = self.router.route(message.word)
                emit(
                    "WAKE",
                    device=session.device_id,
                    word=message.word,
                    score=message.score,
                    assistant=assistant.id,
                )
                await self.bus.publish(
                    WakeDetected(
                        device_id=session.device_id, word=message.word, score=message.score
                    )
                )
                session.assistant = assistant
                await self.bus.publish(
                    WakeRouted(
                        device_id=session.device_id,
                        assistant_id=assistant.id,
                        session_id=session.session_id,
                    )
                )
                await dialog.on_wake()
            case Vad():
                await dialog.on_vad(message)
            case TextInput():
                await dialog.on_text(message)
            case Playback():
                await dialog.on_playback(message)
            case Privacy():
                await dialog.on_privacy(message.muted)
                await self.bus.publish(
                    PrivacyChanged(
                        device_id=session.device_id,
                        muted=message.muted,
                        hard=message.hard,
                        until=message.until,
                    )
                )
            case Error():
                emit(
                    "EDGE-ERROR",
                    {"code": message.code, "message": message.message},
                    device=session.device_id,
                )
            case Result() | EdgeEvent():
                pass  # S5: skills await results and receive edge events
            case _:
                pass

    async def on_frame(self, conn: Connection, frame: Frame) -> None:
        session = self._current(conn)
        if session is not None and session.dialog is not None:
            await session.dialog.on_mic_frame(frame)

    async def on_disconnect(self, conn: Connection, code: int | None, reason: str) -> None:
        emit(
            "DISCONNECTED",
            {"reason": reason},
            device=conn.device_id,
            session=conn.session_id,
            code=code,
        )
        session = self._current(conn)
        if session is None:
            return  # replaced by a newer connection: already closed
        del self.sessions[conn.device_id]
        await self._close(session)
        self.body.device_gone(conn.device_id)
        await self.bus.publish(
            EdgeDisconnected(device_id=conn.device_id, capabilities=session.capabilities)
        )

    async def on_refused(self, device_id: str | None, code: str, detail: str) -> None:
        emit("REFUSED", {"detail": detail}, device=device_id, code=code)

    async def _close(self, session: EdgeSession) -> None:
        if session.dialog is not None:
            await session.dialog.close()

    # ------------------------------------------------------------ proactive speech

    async def on_speech_request(self, request: SpeechRequest) -> None:
        """`device:<id>` speaks on that device; `origin` and `room:` on every connected one
        (S7 resolves rooms and the origin device)."""
        if request.target.startswith("device:"):
            targets = [request.target.removeprefix("device:")]
        else:
            targets = list(self.sessions)
        for device_id in targets:
            session = self.sessions.get(device_id)
            if session is None or session.dialog is None:
                emit("SPEECH-DROPPED", device=device_id, reason="not_connected")
                continue
            await session.dialog.request_speech(request)

    async def close(self) -> None:
        for session in list(self.sessions.values()):
            await self._close(session)
        self.sessions.clear()
