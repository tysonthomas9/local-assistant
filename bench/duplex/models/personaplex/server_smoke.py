"""Check that PersonaPlex's own server (moshi.server) runs locally: start it on loopback,
open /api/chat over WebSocket, wait for the handshake byte (sent once the voice and text
prompts are processed), stream 3 s of silence as Opus, count the frames that come back,
then shut down.

moshi.server ignores --host and binds every interface, and fetches config.json from the Hub
on start; both are patched here (loopback only, offline).
"""

import asyncio
import os
import sys
import threading
import time
from pathlib import Path

import aiohttp
import numpy as np
import sphn
from aiohttp import web

CKPT = Path(os.environ.get("DX_MODELS", os.path.expanduser("~/assistant-dxpoc/models"))) / "personaplex-7b-v1"
PORT = 8998


def serve() -> None:
    # moshi.server runs main() at import time, so patch first, then import.
    import huggingface_hub

    orig = web.run_app
    web.run_app = lambda app, **kw: orig(app, host="127.0.0.1", **{k: v for k, v in kw.items() if k != "host"})
    huggingface_hub.hf_hub_download = lambda repo, name, **kw: str(CKPT / name)
    sys.argv = ["moshi.server", "--port", str(PORT), "--static", "none",
                "--moshi-weight", str(CKPT / "model.safetensors"),
                "--mimi-weight", str(CKPT / "tokenizer-e351c8d8-checkpoint125.safetensors"),
                "--tokenizer", str(CKPT / "tokenizer_spm_32k_3.model"),
                "--voice-prompt-dir", str(CKPT / "voices")]
    import moshi.server  # noqa: F401


async def client() -> None:
    url = (f"http://127.0.0.1:{PORT}/api/chat?voice_prompt=NATF2.pt"
           "&text_prompt=You%20enjoy%20having%20a%20good%20conversation.")
    for _ in range(120):
        try:
            async with aiohttp.ClientSession() as s:
                async with s.ws_connect(url) as ws:
                    t0 = time.time()
                    msg = await ws.receive(timeout=120)
                    print(f"handshake: {msg.data!r} after {time.time() - t0:.1f}s", flush=True)
                    enc = sphn.OpusStreamWriter(24000)
                    got = 0
                    for _ in range(int(3 / 0.08)):
                        enc.append_pcm(np.zeros(1920, np.float32))
                        b = enc.read_bytes()
                        if b:
                            await ws.send_bytes(b"\x01" + b)
                        try:
                            m = await ws.receive(timeout=0.08)
                            if m.type == aiohttp.WSMsgType.BINARY and m.data[:1] == b"\x01":
                                got += 1
                        except asyncio.TimeoutError:
                            pass
                    print(f"audio messages received in 3 s: {got}", flush=True)
                    return
        except aiohttp.ClientConnectorError:
            await asyncio.sleep(2)
    print("server never came up", flush=True)


def client_then_exit() -> None:
    asyncio.run(client())
    os._exit(0)


if __name__ == "__main__":
    # aiohttp's run_app needs the main thread; the client runs beside it and exits the process.
    threading.Thread(target=client_then_exit, daemon=True).start()
    serve()
