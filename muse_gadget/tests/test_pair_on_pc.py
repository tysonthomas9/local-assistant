"""pair_on_pc.sh stops its pairing container when interrupted (fake docker and ssh; runs on the PC).

The script isn't in the container image, so this test skips there.
"""

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "pair_on_pc.sh"

pytestmark = pytest.mark.skipif(
    not SCRIPT.exists()
    or not Path("/run/dbus/system_bus_socket").is_socket()
    or not list(Path("/sys/class/bluetooth").glob("hci*")),
    reason="needs pair_on_pc.sh, a D-Bus system socket and a Bluetooth adapter (the script checks them)",
)

# `docker run` starts a detached "container" (like the real daemon does), so it
# outlives the client unless something runs `docker rm -f`.
FAKE_DOCKER = """#!/usr/bin/env bash
echo "$*" >> "$FAKE/docker.log"
case "$1" in
    build) exit 0 ;;
    run) setsid sleep 300 < /dev/null > /dev/null 2>&1 &
         echo $! > "$FAKE/container.pid"
         while kill -0 "$(cat "$FAKE/container.pid")" 2>/dev/null; do sleep 0.1; done
         exit 1 ;;
    rm) kill "$(cat "$FAKE/container.pid")" 2>/dev/null; exit 0 ;;
esac
exit 0
"""
FAKE_SSH = "#!/usr/bin/env bash\necho unpaired\n"


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.parametrize("sig,rc", [(signal.SIGTERM, 143), (signal.SIGINT, 130), (signal.SIGHUP, 129)])
def test_a_signal_stops_the_pairing_container(tmp_path, sig, rc):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("docker", FAKE_DOCKER), ("ssh", FAKE_SSH)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    runtime = tmp_path / "run"
    runtime.mkdir()
    token = tmp_path / "token"
    token.write_text("mgst_" + "a" * 42 + "A\n")
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", FAKE=str(tmp_path),
               XDG_RUNTIME_DIR=str(runtime))

    proc = subprocess.Popen(["bash", str(SCRIPT), "--token-file", str(token), "--timeout", "60"],
                            env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    pid_file = tmp_path / "container.pid"
    deadline = time.monotonic() + 10
    while not (pid_file.exists() and pid_file.read_text().strip()) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert pid_file.exists(), "the fake pairing container never started"
    container = int(pid_file.read_text())
    assert alive(container)
    time.sleep(0.2)

    try:
        proc.send_signal(sig)
        try:
            out, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            os.kill(container, signal.SIGKILL)  # else the fake client holds stdout open
            out, _ = proc.communicate()
        assert proc.returncode == rc, out
        log = (tmp_path / "docker.log").read_text()
        assert "--name muse-gadget-pair" in log
        assert "rm -f muse-gadget-pair" in log
        deadline = time.monotonic() + 5
        while alive(container) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not alive(container), "the pairing container kept running"
        assert not list(runtime.glob("muse-pair.*")), "the pairing state wasn't wiped"
    finally:
        if alive(container):
            os.kill(container, signal.SIGKILL)
