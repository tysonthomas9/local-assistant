"""RobotLinkSession against a fake Muse VM over the real Noise transport."""

import asyncio
import json

from musegadget.link_client import MessageDecoder
from musegadget.noise import (
    ApplicationResponse, BodyChunk, NoiseFrameDecoder, NoiseXXResponder, ServiceFrame, encode_noise_frames,
)
from musegadget.noise.transport import decode_request_envelope, encode_response_envelope

from gadget import chat, restrict, robot
from gadget.link import RobotLinkSession
from musegadget.link_client import DeviceDescription


class Pipe:
    def __init__(self, inbox, outbox):
        self._inbox, self._outbox = inbox, outbox

    async def send(self, data):
        await self._outbox.put(data)

    async def recv(self):
        data = await self._inbox.get()
        if data is None:
            raise ConnectionError("closed")
        return data

    async def close(self):
        await self._outbox.put(None)


class FakeVm:
    def __init__(self, ws):
        self.ws = ws
        self.decoder = NoiseFrameDecoder()
        self.bodies: dict[int, bytearray] = {}
        self.requests: dict[int, object] = {}
        self.control = None
        self.register = None
        self.sub = None
        self.sub_headers = {}
        self.resets = []
        self.chat_bodies = []

    async def handshake(self):
        responder = NoiseXXResponder()
        responder.initialize()
        await self.ws.send(responder.read_message1_and_write_message2(await self.ws.recv()))
        responder.read_message3(await self.ws.recv())
        self.send_cipher, self.recv_cipher = responder.split()

    async def respond(self, stream_id, status, body: bytes, end=True):
        frame = ServiceFrame.response(stream_id, ApplicationResponse(status=status, body=body, end_body=end))
        for chunk in encode_noise_frames(encode_response_envelope(frame)):
            await self.ws.send(self.send_cipher.encrypt_with_ad(b"", chunk))

    async def chunk(self, stream_id, data: bytes, end=False):
        frame = ServiceFrame.body_chunk(stream_id, BodyChunk(data=data, end_body=end))
        for piece in encode_noise_frames(encode_response_envelope(frame)):
            await self.ws.send(self.send_cipher.encrypt_with_ad(b"", piece))

    async def events(self, *events):
        data = b"".join(json.dumps(e).encode() + b"\n" for e in events)
        await self.chunk(self.sub, data[:7])   # a line split across chunks
        await self.chunk(self.sub, data[7:])

    async def serve(self):
        messages = MessageDecoder()
        while True:
            try:
                raw = await self.ws.recv()
            except ConnectionError:
                return
            assembled = self.decoder.decode(self.recv_cipher.decrypt_with_ad(b"", raw))
            if assembled is None:
                continue
            frame = decode_request_envelope(assembled)
            sid = frame.stream_id
            if frame.kind == "reset":
                self.resets.append(sid)
                continue
            if frame.kind == "request":
                self.requests[sid] = frame.value
                self.bodies[sid] = bytearray(frame.value.body)
                if frame.value.path == "/link-control":
                    self.control = sid
                    await self.respond(sid, 200, b"", end=False)
                    continue
                done = frame.value.end_body
            else:
                if sid == self.control:
                    for m in messages.feed(frame.value.data):
                        if m.get("method") == "link.register":
                            self.register = m
                    continue
                self.bodies[sid] += frame.value.data
                done = frame.value.end_body
            if done:
                await self.answer(sid)

    async def answer(self, sid):
        req = self.requests[sid]
        if req.path == "/chat/subscribe":
            self.sub = sid
            self.sub_headers = {h.key.lower(): h.value for h in req.headers}
            self.sub_body = json.loads(bytes(self.bodies[sid]))
            await self.respond(sid, 200, b'{"type":"ack","ok":true}\n', end=False)
        elif req.path == "/chat/stream":
            body = json.loads(bytes(self.bodies[sid]))
            self.chat_bodies.append(body)
            await self.respond(sid, 200, json.dumps({"ok": True, "result": {"message_id": "u1"}}).encode())
            ev = {"type": "event", "seq": 10, "event": "delta.message_start",
                  "payload": {"message_id": "a1", "reply_to_message_id": "u1"}}
            await self.events(
                ev,
                {"type": "event", "seq": 11, "event": "delta.text_append",
                 "payload": {"message_id": "a1", "text": "Hello from "}},
                {"type": "event", "seq": 12, "event": "delta.text_append",
                 "payload": {"message_id": "a1", "text": "Muse."}},
                {"type": "event", "seq": 13, "event": "delta.message_done", "payload": {"message_id": "a1"}},
            )
        else:
            await self.respond(sid, 404, b"")


def test_turn_over_real_link_session():
    async def scenario():
        to_device, to_vm = asyncio.Queue(), asyncio.Queue()
        device_ws, vm_ws = Pipe(to_device, to_vm), Pipe(to_vm, to_device)

        async def connect(url, headers):
            return device_ws

        device = DeviceDescription(node_id="homelink-abcdef", display_name=restrict.display_name(),
                                   version="0.1.0", commands=restrict.command_specs())
        session = RobotLinkSession(noise_host="gw.example", vm_id="vm", vm_auth_token="tok",
                                   device=device, run_command=lambda *a: {"ok": True}, connect=connect)
        vm = FakeVm(vm_ws)
        stop = asyncio.Event()
        task = asyncio.ensure_future(session.run(stop))
        await vm.handshake()
        server = asyncio.ensure_future(vm.serve())
        for _ in range(100):
            if vm.register:
                break
            await asyncio.sleep(0.01)
        params = vm.register["params"]
        assert params["display_name"] == "Reachy Mini"
        assert set(params["commands_v2"]) == {"device.health", *robot.COMMANDS}

        reply = await chat.turn(session, "hi robot", chat.TurnOptions(settle_s=0.05))
        assert reply == "Hello from Muse."
        assert vm.sub_headers["accept"] == "application/x-ndjson"
        assert vm.sub_body == {"session_id": chat.DEFAULT_SESSION_ID}
        for _ in range(100):
            if vm.resets:
                break
            await asyncio.sleep(0.01)
        assert vm.resets == [vm.sub]   # the subscription is closed after the turn
        assert vm.chat_bodies[0]["device_id"] == "homelink-abcdef"
        assert vm.chat_bodies[0]["session_id"] == chat.DEFAULT_SESSION_ID
        assert vm.chat_bodies[0]["message"] == "hi robot"   # no note by default
        stop.set()
        await asyncio.wait_for(task, 2)
        server.cancel()

    asyncio.run(scenario())
