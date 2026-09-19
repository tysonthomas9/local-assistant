"""Sample GPU memory and utilisation to a CSV, per GPU and per process.

Usage:
    python local_backend/gpu_monitor.py [--interval 2] [--out local_backend/logs/gpu.csv]   # record
    python local_backend/gpu_monitor.py --summary local_backend/logs/gpu.csv               # peaks
"""

import argparse
import csv
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DEFAULT_OUT = Path(__file__).parent / "logs" / "gpu.csv"
FIELDS = ["time", "kind", "gpu", "name", "mem_used_mib", "mem_total_mib", "util_pct"]


def query(args: list[str]) -> list[list[str]]:
    out = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def process_name(pid: str) -> str:
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return f"pid {pid}"
    for label, needles in (("speech-to-speech", ("speech-to-speech",)),
                           ("reachy-mini-daemon", ("reachy-mini-daemon", "run_daemon.py")),
                           ("reachy-mini-conversation-app", ("reachy-mini-conversation-app", "run_app.py")),
                           ("llama-server", ("llama-server",)), ("ollama", ("ollama",))):
        if any(n in cmd for n in needles):
            return f"{label} ({pid})"
    parts = cmd.split()
    return f"{parts[0].rsplit('/', 1)[-1]} ({pid})" if parts else f"pid {pid}"


def sample() -> list[dict]:
    now = datetime.now().isoformat(timespec="seconds")
    rows = []
    uuid_to_idx = {}
    for idx, uuid, used, total, util in query(["--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu"]):
        uuid_to_idx[uuid] = idx
        rows.append(dict(time=now, kind="gpu", gpu=idx, name="", mem_used_mib=used, mem_total_mib=total, util_pct=util))
    for uuid, pid, used in query(["--query-compute-apps=gpu_uuid,pid,used_memory"]):
        rows.append(dict(time=now, kind="proc", gpu=uuid_to_idx.get(uuid, "?"), name=process_name(pid),
                         mem_used_mib=used, mem_total_mib="", util_pct=""))
    return rows


def record(out: Path, interval: float) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        while True:
            try:
                w.writerows(sample())
                f.flush()
            except Exception as e:  # one bad sample (e.g. a process exiting mid-read) must not stop logging
                print(f"{datetime.now().isoformat(timespec='seconds')} sample failed: {e!r}", flush=True)
            time.sleep(interval)


def to_int(value: str) -> int | None:
    """nvidia-smi reports "[N/A]" (or blanks) for some fields on some GPUs/drivers."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def summary(path: Path) -> None:
    peak_gpu, total, util, peak_proc, seen = {}, {}, defaultdict(list), {}, set()
    with path.open() as f:
        for r in csv.DictReader(f):
            seen.add(r["time"])
            used = to_int(r["mem_used_mib"])
            if used is None:
                continue
            if r["kind"] == "gpu":
                peak_gpu[r["gpu"]] = max(peak_gpu.get(r["gpu"], 0), used)
                if (t := to_int(r["mem_total_mib"])) is not None:
                    total[r["gpu"]] = t
                if (u := to_int(r["util_pct"])) is not None:
                    util[r["gpu"]].append(u)
            else:
                key = (r["gpu"], r["name"])
                peak_proc[key] = max(peak_proc.get(key, 0), used)
    times = sorted(seen)
    print(f"{len(times)} samples, {times[0]} → {times[-1]}")
    for g in sorted(peak_gpu):
        u = util[g] or [0]
        t = total.get(g, 0)
        print(f"GPU {g}: peak {peak_gpu[g]/1024:.1f} / {t/1024:.1f} GiB "
              f"(headroom {(t-peak_gpu[g])/1024:.1f} GiB), util avg {sum(u)/len(u):.0f}% max {max(u)}%")
    for (g, name), m in sorted(peak_proc.items()):
        print(f"  GPU {g}  {name:45s} peak {m/1024:.1f} GiB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--summary", type=Path)
    a = p.parse_args()
    summary(a.summary) if a.summary else record(a.out, a.interval)
