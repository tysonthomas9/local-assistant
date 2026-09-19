"""Log every network connection made by the Reachy stack, to prove it stays on this machine.

Polls `ss` (TCP and UDP, including connection attempts) for the daemon, speech server and
conversation app, and writes one line per new connection to local_backend/logs/net_watch.log.
Any peer outside 127.0.0.0/8 or ::1 is marked EXTERNAL.

Processes started through `sg` (the daemon and the app, until you log in again after joining the
audio/video/dialout groups) run with a different group ID, and Linux then hides their sockets'
owning PID from other processes. Sockets owned by this user that `ss` can't attribute to any PID
are logged as "unattributed": they are *probably* the sg-launched daemon/app, but can also be any
other process of this user (Claude Code, a browser, curl) whose socket appeared between ss building
its PID table and reading the socket list, or an orphaned FIN-WAIT/LAST-ACK socket. Treat an
EXTERNAL "unattributed" line as a lead to check (`ss -tnpe`), not proof. Logging in again with the
groups (so nothing runs under sg) removes the ambiguity.

    python3 local_backend/net_watch.py [--interval 0.5]
"""

import argparse
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

LOG = Path(__file__).parent / "logs" / "net_watch.log"
PATTERNS = {
    "daemon": r"^[^ ]*python[0-9.]* [^ ]*(reachy-mini-daemon|local_backend/run_daemon\.py)",
    "speech": r"^[^ ]*python[0-9.]* [^ ]*speech-to-speech serve",
    "app": r"^[^ ]*python[0-9.]* [^ ]*(reachy-mini-conversation-app|local_backend/run_app\.py)",
}
UID = os.getuid()
LOCAL = re.compile(r"^(127\.|\[::1\]|\[::ffff:127\.|\*|0\.0\.0\.0|\[::\])")


def stack_pids() -> dict[str, str]:
    pids = {}
    for label, pattern in PATTERNS.items():
        for pid in subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split():
            pids[pid] = label
    return pids


def main(interval: float) -> None:
    LOG.parent.mkdir(exist_ok=True)
    seen = set()
    with LOG.open("a") as log:
        log.write(f"{datetime.now().isoformat(timespec='seconds')} watch started\n")
        log.flush()
        while True:
            pids = stack_pids()
            out = subprocess.run(["ss", "-tunaepH"], capture_output=True, text=True).stdout
            for line in out.splitlines():
                cols = line.split()
                if len(cols) < 7:
                    continue
                m = re.search(r"pid=(\d+)", line)
                if m and m.group(1) in pids:
                    owner = pids[m.group(1)]
                elif not m and f"uid:{UID} " in line:
                    owner = "unattributed"
                else:
                    continue
                proto, state, local, peer = cols[0], cols[1], cols[4], cols[5]
                if state == "LISTEN" or (proto == "udp" and state == "UNCONN" and peer.startswith(("0.0.0.0", "*", "[::]"))):
                    continue
                key = (owner, proto, local, peer)
                if key in seen:
                    continue
                seen.add(key)
                tag = "local" if LOCAL.match(peer) else "EXTERNAL"
                log.write(f"{datetime.now().isoformat(timespec='seconds')} {tag:8s} {owner:12s} {proto} {state:10s} {local} -> {peer}\n")
                log.flush()
            time.sleep(interval)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=0.5)
    main(p.parse_args().interval)
