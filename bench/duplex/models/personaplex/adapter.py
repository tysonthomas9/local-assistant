"""PersonaPlex-7B adapter: the moshi server's per-frame loop, run in-process.

Every 80 ms frame (1920 samples at 24 kHz) of user audio is Mimi-encoded, one LMGen step
samples the agent's text + audio tokens, and Mimi decodes one 80 ms frame of agent audio.
This is exactly what ``moshi.server`` does per WebSocket frame, minus Opus.
"""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

import numpy as np
import sentencepiece
import torch
from moshi.models import LMGen, loaders

CKPT = Path(os.environ.get("DX_MODELS", os.path.expanduser("~/assistant-dxpoc/models"))) / "personaplex-7b-v1"
# PersonaPlex's documented assistant-role prompt (its QA/interruption evaluation prompt).
TEACHER = "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."


class PersonaPlex:
    name = "personaplex-7b-v1"
    serving = ("true streaming full duplex: moshi.server (WebSocket, Opus frames, custom protocol); "
               "not OpenAI-Realtime; needs a native streaming adapter")
    in_sr = out_sr = 24000
    chunk_s = 0.08

    def __init__(self, voice: str = "NATF2.pt", prompt: str = TEACHER, seed: int = 42424242):
        self.voice, self.prompt, self.seed = voice, prompt, seed
        self.out: list = []

    def load(self) -> None:
        torch.manual_seed(self.seed)
        dev = "cuda"
        self.mimi = loaders.get_mimi(str(CKPT / loaders.MIMI_NAME), dev)
        self.other_mimi = loaders.get_mimi(str(CKPT / loaders.MIMI_NAME), dev)
        self.tok = sentencepiece.SentencePieceProcessor(str(CKPT / loaders.TEXT_TOKENIZER_NAME))
        lm = loaders.get_moshi_lm(str(CKPT / loaders.MOSHI_NAME), device=dev)
        lm.eval()
        self.lm_gen = LMGen(lm, audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
                            sample_rate=self.mimi.sample_rate, device=dev, frame_rate=self.mimi.frame_rate,
                            save_voice_prompt_embeddings=False, use_sampling=True,
                            temp=0.8, temp_text=0.7, top_k=250, top_k_text=25)
        self.frame = int(self.mimi.sample_rate / self.mimi.frame_rate)
        assert self.frame == int(self.chunk_s * self.in_sr)
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        voices = CKPT / "voices"
        if not voices.exists():
            with tarfile.open(CKPT / "voices.tgz") as t:
                t.extractall(CKPT)
        self.voice_path = str(voices / self.voice)
        with torch.no_grad():
            for _ in range(4):
                z = torch.zeros(1, 1, self.frame, device=dev)
                codes = self.mimi.encode(z)
                self.other_mimi.encode(z)
                for c in range(codes.shape[-1]):
                    t = self.lm_gen.step(codes[:, :, c:c + 1])
                    if t is not None:
                        self.mimi.decode(t[:, 1:9])
                        self.other_mimi.decode(t[:, 1:9])
        torch.cuda.synchronize()

    @torch.no_grad()
    def start(self, sc) -> None:
        torch.manual_seed(self.seed)
        self.lm_gen.load_voice_prompt_embeddings(self.voice_path)
        self.lm_gen.text_prompt_tokens = self.tok.encode(f"<system> {self.prompt} <system>")
        self.mimi.reset_streaming()
        self.other_mimi.reset_streaming()
        self.lm_gen.reset_streaming()
        self.lm_gen.step_system_prompts(self.mimi)
        self.mimi.reset_streaming()
        self.out = []

    @torch.no_grad()
    def feed(self, pcm: np.ndarray) -> None:
        x = torch.from_numpy(np.ascontiguousarray(pcm, dtype=np.float32)).to("cuda")[None, None]
        codes = self.mimi.encode(x)
        self.other_mimi.encode(x)
        for c in range(codes.shape[-1]):
            t = self.lm_gen.step(codes[:, :, c:c + 1])
            if t is None:
                continue
            y = self.mimi.decode(t[:, 1:9])
            self.other_mimi.decode(t[:, 1:9])
            self.out.append(("audio", y[0, 0].float().cpu().numpy()))
            tt = int(t[0, 0, 0].item())
            if tt not in (0, 3):
                self.out.append(("text", self.tok.id_to_piece(tt).replace("▁", " ")))

    def poll(self) -> list:
        o, self.out = self.out, []
        return o

    def end(self) -> None:
        pass

    def mem_stats(self) -> dict:
        return {"max_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                "max_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20)}


ADAPTER = PersonaPlex
