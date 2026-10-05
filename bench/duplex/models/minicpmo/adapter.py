"""MiniCPM-o 4.5 (AWQ int4) adapter: the model's own full-duplex mode, in-process.

Every 1 s of user audio: ``streaming_prefill`` (feed the chunk) then ``streaming_generate``
(the model decides listen/speak for this unit and, if speaking, returns up to ~1 s of audio).
That is the loop the official duplex example and web demo run.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

CKPT = Path(os.environ.get("DX_MODELS", os.path.expanduser("~/assistant-dxpoc/models"))) / "MiniCPM-o-4_5-awq"
# The duplex system prompt from the model card's duplex example.
DUPLEX_PROMPT = "Streaming Omni Conversation."
REF_WAV = CKPT / "assets" / "system_ref_audio.wav"  # the model's stock assistant voice


class MiniCPMo:
    name = "minicpm-o-4_5-awq"
    serving = ("full duplex in 1 s units: transformers remote code (as_duplex: streaming_prefill / "
               "streaming_generate) behind OpenBMB's WebRTC demo or llama.cpp-omni; no OpenAI-Realtime server")
    in_sr = 16000
    out_sr = 24000
    chunk_s = 1.0

    def __init__(self, dtype: str = "float16", seed: int = 1234, t2w_fp16: bool = False, n_timesteps: int = 10,
                 attn: str = "sdpa"):
        self.dtype = getattr(torch, dtype)
        self.seed = seed
        self.attn = attn
        self.t2w = {"enable_float16": t2w_fp16, "n_timesteps": n_timesteps}
        self.out: list = []
        self.costs: list = []

    def load(self) -> None:
        import librosa

        torch.manual_seed(self.seed)
        model = AutoModel.from_pretrained(
            str(CKPT), trust_remote_code=True, attn_implementation=self.attn, torch_dtype=self.dtype,
            init_vision=False, init_audio=True, init_tts=True,
        )
        model.eval().cuda()
        self.m = model.as_duplex(**self.t2w)
        self.ref, _ = librosa.load(str(REF_WAV), sr=16000, mono=True)
        # warm-up: one short session of silence
        self.start(None)
        for _ in range(3):
            self.feed(np.zeros(16000, np.float32))
        self.out = []

    @torch.no_grad()
    def start(self, sc) -> None:
        torch.manual_seed(self.seed)
        self.m.prepare(prefix_system_prompt=DUPLEX_PROMPT, ref_audio=self.ref, prompt_wav_path=str(REF_WAV))
        self.out = []

    @torch.no_grad()
    def feed(self, pcm: np.ndarray) -> None:
        self.m.streaming_prefill(audio_waveform=np.asarray(pcm, np.float32))
        r = self.m.streaming_generate(prompt_wav_path=str(REF_WAV), max_new_speak_tokens_per_chunk=20,
                                      decode_mode="sampling")
        if not r.get("is_listen"):
            self.costs.append({k: round(float(v), 3) for k, v in r.items() if k.startswith("cost_")})
        wav = r.get("audio_waveform")
        if r.get("is_listen") or wav is None or len(wav) == 0:
            self.out.append(("audio", np.zeros(int(self.chunk_s * self.out_sr), np.float32)))
        else:
            w = wav.detach().float().cpu().numpy() if torch.is_tensor(wav) else np.asarray(wav, np.float32)
            self.out.append(("audio", w.reshape(-1)))
        if r.get("text"):
            self.out.append(("text", r["text"]))

    def poll(self) -> list:
        o, self.out = self.out, []
        return o

    def end(self) -> None:
        if self.costs:
            import json
            print("minicpmo speak-chunk costs:", json.dumps(self.costs[:12]), flush=True)
        self.costs = []

    def mem_stats(self) -> dict:
        return {"max_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                "max_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20)}


ADAPTER = MiniCPMo
