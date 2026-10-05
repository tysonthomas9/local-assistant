"""Real-time scenario runner.

The runner owns the wall clock. Every ``adapter.chunk_s`` seconds it hands the adapter the
next slice of user audio (the scheduled TTS clips, or silence), then collects whatever the
adapter has produced. Model audio is placed on the output track at the moment it arrives
(or right after audio already queued, like a playback buffer), so latencies include the
model's own compute time. A ``flush`` event (barge-in) drops audio queued past "now".

Both tracks are recorded at 24 kHz: left = user (what the model heard), right = model.

Usage (from bench/duplex, inside a model's env):
    python -m dxbench.runner --adapter models/personaplex/adapter.py --clips CLIPS --out OUT
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .audio import REC_SR, frame_db, load_mono, resample, write_stereo
from .scenarios import SCENARIOS, Scenario, by_id

TALK_DB = -42.0  # model output above this (20 ms frames) counts as talking
HOP_S = 0.02
MAX_S = 240.0


class GpuMonitor(threading.Thread):
    """Samples total GPU memory used (nvidia-smi) and keeps the peak."""

    def __init__(self, gpu: str = "0", period: float = 0.25):
        super().__init__(daemon=True)
        self.gpu, self.period = gpu, period
        self.peak_mib = 0
        self.window_peak_mib = 0
        self._stop = threading.Event()

    def sample(self) -> int:
        out = subprocess.run(
            ["nvidia-smi", "-i", self.gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        try:
            return int(out.splitlines()[0])
        except (ValueError, IndexError):
            return 0

    def run(self) -> None:
        while not self._stop.is_set():
            v = self.sample()
            self.peak_mib = max(self.peak_mib, v)
            self.window_peak_mib = max(self.window_peak_mib, v)
            time.sleep(self.period)

    def reset_window(self) -> None:
        self.window_peak_mib = 0

    def stop(self) -> None:
        self._stop.set()


def load_adapter(path: str, kwargs: dict):
    spec = importlib.util.spec_from_file_location("dx_adapter", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dx_adapter"] = mod
    spec.loader.exec_module(mod)
    return mod.ADAPTER(**kwargs)


class Session:
    def __init__(self, adapter, sc: Scenario, clips: dict[str, np.ndarray]):
        self.a, self.sc, self.clips = adapter, sc, clips
        self.n = int(MAX_S * REC_SR)
        self.rec_in = np.zeros(self.n, np.float32)
        self.rec_out = np.zeros(self.n, np.float32)
        self.cursor = 0  # end of queued model audio (samples)
        self.talk = np.zeros(int(MAX_S / HOP_S) + 1, bool)  # per-frame talk flags of heard audio
        self.talk_upto = 0  # frames computed
        self.log: dict = {"scenario": sc.id, "clips": [], "text": [], "tools": [], "flushes": [],
                          "trigger_notes": []}
        self.lag: list[float] = []
        self.compute: list[float] = []

    # ---- model-talk tracking on the heard (<= now) part of the output track
    def _update_talk(self, now_s: float) -> None:
        upto = min(int(now_s / HOP_S), len(self.talk))
        if upto <= self.talk_upto:
            return
        hop = int(REC_SR * HOP_S)
        seg = self.rec_out[self.talk_upto * hop: upto * hop]
        self.talk[self.talk_upto:upto] = frame_db(seg, REC_SR, HOP_S)[: upto - self.talk_upto] > TALK_DB
        self.talk_upto = upto

    def _onset_after(self, t: float, min_frames: int = 10) -> float | None:
        """First time >= t where the model talks for min_frames consecutive frames (200 ms)."""
        i0 = int(t / HOP_S)
        run = 0
        for i in range(i0, self.talk_upto):
            run = run + 1 if self.talk[i] else 0
            if run >= min_frames:
                return (i - min_frames + 1) * HOP_S
        return None

    def _quiet_for(self, since: float, now: float, silence: float) -> bool:
        """Model spoke after `since` and has now been quiet (heard + queued) for `silence`."""
        q0 = int(now * REC_SR)
        if self.cursor > q0 and frame_db(self.rec_out[q0:self.cursor], REC_SR, HOP_S).max(initial=-120.0) > TALK_DB:
            return False  # speech still queued for playback
        i0 = int(since / HOP_S)
        spoke = np.nonzero(self.talk[i0:self.talk_upto])[0]
        if spoke.size == 0:
            return False
        last = (i0 + spoke[-1] + 1) * HOP_S
        return now - last >= silence

    def run(self) -> dict:
        a, sc = self.a, self.sc
        chunk = int(round(a.chunk_s * REC_SR))
        steps = list(sc.steps)
        si = 0
        cur: np.ndarray | None = None
        cur_pos = 0
        prev_start = prev_end = 0.0
        last_end: float | None = None
        ended_at = None
        a.start(sc)
        t0 = time.perf_counter()
        k = 0
        while True:
            p = k * chunk
            if p + chunk >= self.n:
                self.log["trigger_notes"].append("hit MAX_S")
                break
            tgt = p / REC_SR
            wait = t0 + tgt - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            now = time.perf_counter() - t0
            self.lag.append(max(0.0, now - tgt))
            self._update_talk(now)

            # --- schedule the next clip
            if cur is None and si < len(steps):
                st = steps[si]
                fire, note = False, None
                if st.trigger == "at":
                    fire = tgt >= st.t
                elif st.trigger == "gap":
                    fire = tgt >= prev_end + st.t
                elif st.trigger == "onset":
                    on = self._onset_after(prev_end)
                    if on is not None and tgt >= on + st.t:
                        fire, note = True, f"{st.clip}: model onset {on:.2f}s"
                    elif tgt >= prev_end + st.timeout:
                        fire, note = True, f"{st.clip}: TIMEOUT, model not talking"
                elif st.trigger == "done":
                    if self._quiet_for(prev_start, tgt, st.t):
                        fire = True
                    elif tgt >= prev_end + st.timeout:
                        fire, note = True, f"{st.clip}: TIMEOUT waiting for model to finish"
                if fire:
                    cur, cur_pos = self.clips[st.clip], 0
                    prev_start = tgt
                    self.log["clips"].append({"clip": st.clip, "role": st.role, "start": round(tgt, 3),
                                              "end": round(tgt + len(cur) / REC_SR, 3)})
                    if note:
                        self.log["trigger_notes"].append(note)
                    si += 1

            user = np.zeros(chunk, np.float32)
            if cur is not None:
                take = cur[cur_pos: cur_pos + chunk]
                user[: len(take)] = take
                cur_pos += chunk
                if cur_pos >= len(cur):
                    prev_end = prev_start + len(cur) / REC_SR
                    cur = None
                    if si >= len(steps):
                        last_end = prev_end
            self.rec_in[p: p + chunk] = user

            c0 = time.perf_counter()
            a.feed(resample(user, REC_SR, a.in_sr))
            evs = a.poll()
            self.compute.append(time.perf_counter() - c0)
            now = time.perf_counter() - t0
            for ev in evs:
                kind = ev[0]
                if kind == "audio":
                    x = resample(ev[1], a.out_sr, REC_SR)
                    s = max(int(now * REC_SR), self.cursor)
                    e = min(s + len(x), self.n)
                    self.rec_out[s:e] = x[: e - s]
                    self.cursor = e
                elif kind == "flush":
                    cut = int(now * REC_SR)
                    self.rec_out[cut:] = 0.0
                    self.cursor = cut
                    self.log["flushes"].append(round(now, 3))
                elif kind == "text":
                    self.log["text"].append([round(now, 3), ev[1]])
                elif kind == "tool":
                    self.log["tools"].append([round(now, 3), ev[1]])
            # --- end of run
            if last_end is not None and ended_at is None:
                if self._quiet_for(last_end, now, sc.tail) or now >= last_end + sc.tail_cap:
                    ended_at = now
            if ended_at is not None:
                break
            k += 1
        a.end()
        dur = max((k + 1) * chunk, self.cursor)
        lag = np.array(self.lag)
        comp = np.array(self.compute)
        self.log.update({
            "duration_s": round(dur / REC_SR, 2),
            "chunk_s": a.chunk_s,
            "feed_compute_ms": {"mean": round(1000 * comp.mean(), 1), "p95": round(1000 * np.percentile(comp, 95), 1),
                                "max": round(1000 * comp.max(), 1)},
            "behind_realtime_s": {"max": round(float(lag.max()), 3), "frac_late_100ms": round(float((lag > 0.1).mean()), 3)},
        })
        self.rec_in, self.rec_out = self.rec_in[:dur], self.rec_out[:dur]
        return self.log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--take", default="t1")
    ap.add_argument("--kw", default="{}", help="JSON kwargs for the adapter")
    args = ap.parse_args()

    mon = GpuMonitor(os.environ.get("DX_GPU", "0"))
    base = mon.sample()
    mon.start()
    t_load = time.perf_counter()
    adapter = load_adapter(args.adapter, json.loads(args.kw))
    adapter.load()
    load_s = time.perf_counter() - t_load
    after_load = mon.sample()

    clips = {p.stem: load_mono(str(p), REC_SR) for p in Path(args.clips).glob("*.wav")}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sel = SCENARIOS if args.scenarios == "all" else [by_id(s) for s in args.scenarios.split(",")]
    summary = {"adapter": adapter.name, "serving": getattr(adapter, "serving", ""),
               "baseline_mib": base, "after_load_mib": after_load, "load_s": round(load_s, 1),
               "scenarios": {}}
    for sc in sel:
        mon.reset_window()
        sess = Session(adapter, sc, clips)
        log = sess.run()
        log["peak_mib"] = mon.window_peak_mib
        if hasattr(adapter, "mem_stats"):
            log["torch_mem"] = adapter.mem_stats()
        d = out / sc.id
        d.mkdir(exist_ok=True)
        write_stereo(str(d / f"{args.take}.wav"), sess.rec_in, sess.rec_out)
        (d / f"{args.take}.json").write_text(json.dumps(log, indent=1))
        summary["scenarios"][sc.id] = {"peak_mib": log["peak_mib"], "duration_s": log["duration_s"],
                                       "behind": log["behind_realtime_s"], "notes": log["trigger_notes"]}
        print(json.dumps({sc.id: summary["scenarios"][sc.id]}), flush=True)
    mon.stop()
    summary["peak_mib"] = mon.peak_mib
    (out / f"summary_{args.take}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"peak_mib": mon.peak_mib, "after_load_mib": after_load, "baseline_mib": base}))


if __name__ == "__main__":
    main()
