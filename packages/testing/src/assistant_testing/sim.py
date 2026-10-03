"""The simulated robot: Pollen's real reachy-mini daemon on a MuJoCo Reachy Mini, on this PC.

The e2e tier `sim` runs robot features against it, for fast iteration; every such feature
also runs on the physical robot (tier `hw`). Nothing about the stack under test changes: the
edge agent and its reachy body (the reachy-mini SDK) talk to a real daemon. What is simulated:

- the robot: `python -m assistant_robot_reachy.sim` = our daemon launcher (loopback, motor
  watchdog, rest on stop) with Pollen's `--sim --headless` MuJoCo backend, from the pinned env
  `.venv-sim` (the uv workspace plus the `sim` group: mujoco 3.3.0, reachy-mini 1.10.0);
- its sound card: PipeWire virtual devices named "Reachy Mini Audio (sim) ...", which the SDK
  finds by name exactly as it finds a Lite's USB card. The speaker is a virtual sink whose
  output is recorded to a WAV (`<artifacts>/sim/<name>-speaker.wav`); nothing reaches the
  PC's speakers. The microphone is a virtual source carrying a quiet pink-noise floor (the
  "room", about -65 dBFS), so it delivers a live, varying signal; voice comes from the golden
  WAVs fed at the agent's mic input (`/feed`). No acoustics: no echo, no wake word through
  the air, no direction of arrival;
- the robot's own machine: its processes get a HOME of their own (a temp dir), so this PC's
  ~/.asoundrc (written by Pollen's daemon for a real robot) cannot point the SDK at a USB
  card that is not there; the Hugging Face cache stays this user's (HF_HUB_OFFLINE=1).

Not simulated (hw-only): the camera (headless MuJoCo renders no frames), the XVF3800 board
and its echo canceller, real acoustics, the macOS edge host and Reachy Edge.app.

    python -m assistant_testing.sim prepare   # sync .venv-sim, check MuJoCo, PipeWire, datasets
    python -m assistant_testing.sim run       # the sim robot until Ctrl-C (sound card + daemon)
    python -m assistant_testing.sim sweep     # stop leftovers of a sim run; exit 1 if any

One sim at a time on this PC: it uses the daemon's port 8000 and fixed device names. `run`
and the tests start and stop only what they started and FAIL (never skip) if they cannot.
"""

import asyncio
import contextlib
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import wave
from array import array
from collections.abc import Awaitable, Callable
from math import log10, sqrt
from pathlib import Path

from assistant_testing.processes import ManagedProcess, ProcessGroup

SIM_VENV = ".venv-sim"
SIM_GROUP = "sim"
MUJOCO_VERSION = "3.3.0"
REACHY_MINI_VERSION = "1.10.0"
SIM_MODULE = "assistant_robot_reachy.sim"
DAEMON_READY = r"^SIM-READY at rest"
"""The sim daemon's ready line: its API serves and the simulated robot is at rest."""
DAEMON_PORT = 8000
DAEMON_URL = f"http://127.0.0.1:{DAEMON_PORT}"
EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"

CARD = "Reachy Mini Audio (sim)"
"""The sound card's name: the SDK picks the device whose name contains "Reachy Mini Audio"."""
SPEAKER = "reachy_sim_speaker"
SPEAKER_OUT = "reachy_sim_speaker_out"
ROOM = "reachy_sim_room"
MIC = "reachy_sim_mic"
NODE_PREFIX = "reachy_sim_"
ROOM_NOISE_VOLUME = 0.003
"""Pink noise into the microphone: about -55 dBFS peak, -65 dBFS mean."""
RATE = 16000
TOOLS = ("pw-loopback", "pw-record", "pw-cli", "gst-launch-1.0")

_PROCESS_PATTERN = (
    rf"^[^ ]*python[0-9.]* -m {re.escape(SIM_MODULE)}( |$)"
    rf"|^pw-loopback -n {NODE_PREFIX}"
    rf"|^pw-record --target {SPEAKER_OUT}"
    rf"|^gst-launch-1.0 .*device={ROOM}"
)
"""pgrep -f pattern (anchored, so no shell running it matches) for a sim run's processes."""


SIM_MARK = "ASSISTANT_SIM"
"""Set in the environment of the sim robot's processes, so the sweep finds a stray edge agent
(its command line is any edge agent's)."""

HOME_PREFIX = "assistant-sim-home."
"""The sim robot's temporary HOME, under the system temp dir."""


class SimUnavailable(AssertionError):
    """The sim cannot run here (missing env, tools, PipeWire or dataset): fail, never skip."""


def sim_python(repo_root: Path) -> Path:
    return repo_root / SIM_VENV / "bin" / "python"


def runtime_dir() -> str:
    """Where the user's PipeWire sockets are (a shell outside the desktop may lack it)."""
    return os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"


def hf_home() -> str:
    return os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")


def robot_env(home: Path) -> dict[str, str]:
    """The environment of the sim robot's processes (the daemon, the edge agent)."""
    return {
        "HOME": str(home),
        "HF_HOME": hf_home(),
        "HF_HUB_OFFLINE": "1",
        "XDG_RUNTIME_DIR": runtime_dir(),
        "PYTHONUNBUFFERED": "1",
        SIM_MARK: "1",
    }


def daemon_argv(repo_root: Path, args: list[str]) -> list[str]:
    return [str(sim_python(repo_root)), "-m", SIM_MODULE, *args]


def _tool_env() -> dict[str, str]:
    return {**os.environ, "XDG_RUNTIME_DIR": runtime_dir()}


def sim_nodes() -> list[str]:
    """PipeWire nodes of a sim sound card that exist now (any run's)."""
    done = subprocess.run(
        ["pw-cli", "ls", "Node"], capture_output=True, text=True, env=_tool_env(), check=False
    )
    found = re.findall(r'node\.name = "(' + NODE_PREFIX + r'[a-z_]+)"', done.stdout)
    return sorted(set(found))


def _daemon_answers() -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"{DAEMON_URL}/api/daemon/status", timeout=2):
            return True
    except OSError:
        return False


# ---------------------------------------------------------------- prepare


def prepare(repo_root: Path) -> str:
    """Sync the pinned sim env and check everything the sim needs; returns a facts line."""
    missing = [tool for tool in TOOLS if shutil.which(tool) is None]
    if missing:
        raise SimUnavailable(f"sim: missing tools {missing} (PipeWire and GStreamer tools)")
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env["UV_PROJECT_ENVIRONMENT"] = SIM_VENV
    sync = subprocess.run(
        ["uv", "sync", "-q", "--locked", "--no-default-groups", "--group", SIM_GROUP],
        cwd=repo_root, env=env, capture_output=True, text=True, check=False,
    )  # fmt: skip
    if sync.returncode != 0:
        raise SimUnavailable(f"sim: uv sync of {SIM_VENV} failed:\n{sync.stderr[-2000:]}")
    probe = subprocess.run(
        [str(sim_python(repo_root)), "-c",
         "import importlib.metadata as m, mujoco, assistant_robot_reachy.sim; "
         "print(m.version('mujoco'), m.version('reachy-mini'), "
         "assistant_robot_reachy.sim.__file__)"],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    words = probe.stdout.split()
    if probe.returncode != 0 or len(words) != 3:
        raise SimUnavailable(f"sim: {SIM_VENV} cannot import MuJoCo:\n{probe.stderr[-2000:]}")
    mujoco, reachy_mini, module = words
    if (mujoco, reachy_mini) != (MUJOCO_VERSION, REACHY_MINI_VERSION):
        raise SimUnavailable(
            f"sim: mujoco {mujoco}, reachy-mini {reachy_mini} in {SIM_VENV}; "
            f"want {MUJOCO_VERSION}, {REACHY_MINI_VERSION}"
        )
    if not Path(module).resolve().is_relative_to(repo_root.resolve()):
        raise SimUnavailable(f"sim: {SIM_MODULE} is not imported from this checkout: {module}")
    info = subprocess.run(
        ["pw-cli", "info", "0"], capture_output=True, text=True, env=_tool_env(), check=False
    )
    if info.returncode != 0:
        raise SimUnavailable(f"sim: PipeWire is not reachable: {info.stderr.strip()}")
    emotions = Path(hf_home()) / "hub" / ("datasets--" + EMOTIONS_DATASET.replace("/", "--"))
    moves = list(emotions.glob("snapshots/*/*.json"))
    if not moves:
        raise SimUnavailable(
            f"sim: Pollen's emotions dataset is not in the Hugging Face cache ({EMOTIONS_DATASET});"
            " cache it once: hf download --repo-type dataset " + EMOTIONS_DATASET
        )
    return (
        f"sim ready: mujoco={mujoco} reachy_mini={reachy_mini} env={SIM_VENV} "
        f"emotions={len(moves)} pipewire=ok"
    )


# ---------------------------------------------------------------- the sound card and the robot


def sound_card(wav: Path) -> list[tuple[str, list[str]]]:
    """(process name, argv) of the virtual sound card, in start order."""
    card = f'node.description="{CARD} %s"'
    return [
        ("sim:speaker", [
            "pw-loopback", "-n", SPEAKER, "-c", "2",
            "--capture-props", f"media.class=Audio/Sink node.name={SPEAKER} " + card % "speaker",
            "--playback-props", f"media.class=Audio/Source node.name={SPEAKER_OUT} "
            "node.description=\"reachy sim speaker output\"",
        ]),
        ("sim:mic", [
            "pw-loopback", "-n", MIC, "-c", "2",
            "--capture-props", f"media.class=Audio/Sink node.name={ROOM} "
            "node.description=\"reachy sim room\"",
            "--playback-props", f"media.class=Audio/Source node.name={MIC} " + card % "microphone",
        ]),
        ("sim:room", [
            "gst-launch-1.0", "-q", "audiotestsrc", "wave=pink-noise",
            f"volume={ROOM_NOISE_VOLUME}", "is-live=true", "!", "audioconvert", "!",
            "audioresample", "!", "pulsesink", f"device={ROOM}",
        ]),
        ("sim:recorder", [
            "pw-record", "--target", SPEAKER_OUT, "--rate", str(RATE), "--channels", "1",
            "--format", "s16", str(wav),
        ]),
    ]  # fmt: skip


def speaker_summary(wav: Path) -> str:
    """What the simulated speaker played: length, seconds with sound, peak level."""
    try:
        with wave.open(str(wav)) as w:
            pcm = array("h", w.readframes(w.getnframes()))
            rate = w.getframerate()
    except (OSError, EOFError, wave.Error) as exc:
        return f"no speaker recording ({type(exc).__name__}: {exc})"
    if len(pcm) == 0:
        return f"speaker recording {wav.name}: empty"
    frame = rate // 50
    loud = 32768 * 10 ** (-45 / 20)
    voiced = sum(
        sqrt(sum(x * x for x in pcm[i : i + frame]) / frame) > loud
        for i in range(0, len(pcm) - frame + 1, frame)
    )
    peak = 20 * log10(max(max(abs(x) for x in pcm) / 32768, 1e-6))
    return (
        f"speaker recording {wav.name}: {len(pcm) / rate:.1f} s, {voiced * 0.02:.1f} s with sound, "
        f"peak {peak:.1f} dBFS"
    )


class SimRobot:
    """One simulated robot: its home, its sound card and (once started) its daemon."""

    def __init__(self, repo_root: Path, processes: ProcessGroup, name: str) -> None:
        self.repo_root = repo_root
        self.processes = processes
        self.home = Path(tempfile.mkdtemp(prefix=HOME_PREFIX))
        artifacts = Path(os.environ.get("ASSISTANT_ARTIFACTS_DIR") or repo_root / "artifacts")
        self.wav = artifacts / "sim" / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', name)}-speaker.wav"
        self.agents: set[str] = set()
        """The processes that run "on the robot's machine" (the edge agents next to the daemon)."""

    @property
    def env(self) -> dict[str, str]:
        return robot_env(self.home)

    async def plug_in(self) -> None:
        """The sound card appears (fails if another sim's card is there)."""
        if found := await asyncio.to_thread(sim_nodes):
            raise SimUnavailable(
                f"another sim's sound card is present ({found}): one sim at a time"
            )
        self.wav.parent.mkdir(parents=True, exist_ok=True)
        self.wav.unlink(missing_ok=True)
        env = {"XDG_RUNTIME_DIR": runtime_dir()}
        for name, argv in sound_card(self.wav):
            await self.processes.start(name, argv, env=env)
            if name == "sim:mic":
                await self._nodes_present({SPEAKER, SPEAKER_OUT, ROOM, MIC})
        await asyncio.sleep(0.3)
        for name, _ in sound_card(self.wav):
            proc = self.processes.get(name)
            if not proc.running:
                raise SimUnavailable(f"{name} exited at once:\n{proc.output[-2000:]}")

    async def _nodes_present(self, want: set[str], within_s: float = 10.0) -> None:
        deadline = time.monotonic() + within_s
        while not want <= set(await asyncio.to_thread(sim_nodes)):
            if time.monotonic() > deadline:
                raise SimUnavailable(f"the sim sound card did not appear: {sorted(want)}")
            await asyncio.sleep(0.2)

    def unplug(self) -> str:
        """After the processes stopped: the recording's summary; the temp home is removed."""
        shutil.rmtree(self.home, ignore_errors=True)
        return speaker_summary(self.wav)


async def start_daemon(
    robot: SimRobot,
    args: list[str],
    *,
    name: str = "daemon",
    ready_timeout: float = 120.0,
    before_stop: Callable[[], Awaitable[None]] | None = None,
) -> ManagedProcess:
    """The sim daemon (Pollen's, MuJoCo, headless) with our launcher's options `args`."""
    if before_stop is not None:
        robot.processes.before_stop[name] = before_stop
    return await robot.processes.start(
        name,
        daemon_argv(robot.repo_root, args),
        env=robot.env,
        ready_line=DAEMON_READY,
        ready_timeout=ready_timeout,
    )


# ---------------------------------------------------------------- sweep and the CLI


def _marked_processes() -> list[str]:
    """This user's processes started with the sim mark in their environment."""
    found: list[str] = []
    mark = f"{SIM_MARK}=1".encode()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            if mark not in (proc / "environ").read_bytes().split(b"\0"):
                continue
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
        except OSError:
            continue
        found.append(f"{proc.name} {cmdline}")
    return found


def _leftover_processes() -> list[str]:
    done = subprocess.run(
        ["pgrep", "-af", _PROCESS_PATTERN], capture_output=True, text=True, check=False
    )
    found = [line for line in done.stdout.splitlines() if line.strip()]
    pids = {line.split()[0] for line in found}
    return found + [line for line in _marked_processes() if line.split()[0] not in pids]


def sweep() -> list[str]:
    """Stop every leftover of a sim run on this PC (the robot rested first); return them."""
    found = _leftover_processes()
    if found and _daemon_answers():
        from assistant_testing import edge_host

        if rested := edge_host.rest_robot(edge_host.EdgeHost(None)):
            found.append(f"sim robot put to rest: {rested}")
    for sig in (signal.SIGTERM, signal.SIGKILL):
        pids = [int(line.split()[0]) for line in _leftover_processes()]
        for pid in pids:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, sig)
        deadline = time.monotonic() + 10
        while _leftover_processes() and time.monotonic() < deadline:
            time.sleep(0.3)
    found += [f"sound card node: {node}" for node in sim_nodes()]
    for home in Path(tempfile.gettempdir()).glob(f"{HOME_PREFIX}*"):
        shutil.rmtree(home, ignore_errors=True)
        found.append(f"sim home: {home.name}")
    return found


DAEMON_ARGS = [
    "--no-wake-up-on-start", "--no-goto-sleep-on-stop", "--dataset-update-interval", "0",
    "--fastapi-host", "127.0.0.1", "--fastapi-port", str(DAEMON_PORT),
]  # fmt: skip


async def _run(repo_root: Path) -> int:
    print(prepare(repo_root), flush=True)
    if await asyncio.to_thread(_daemon_answers):
        print(f"sim: a daemon already answers at {DAEMON_URL}; stop it first", flush=True)
        return 1
    processes = ProcessGroup(cwd=repo_root)
    robot = SimRobot(repo_root, processes, "run")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        loop.add_signal_handler(sig, stop.set)
    from assistant_testing import edge_host

    async def rest() -> None:
        if rested := await asyncio.to_thread(edge_host.rest_robot, edge_host.EdgeHost(None)):
            print(f"sim robot put to rest: {rested}", flush=True)

    try:
        await robot.plug_in()
        await start_daemon(robot, DAEMON_ARGS, before_stop=rest)
        print(
            f"SIM READY api={DAEMON_URL} card={CARD!r} speaker->{robot.wav} "
            f"(run the edge agent with: HOME={robot.home} HF_HUB_OFFLINE=1); Ctrl-C stops it",
            flush=True,
        )
        daemon = processes.get("daemon")
        while not stop.is_set() and daemon.running:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), 1)
        if not daemon.running:
            print(f"sim: the daemon exited:\n{daemon.output[-3000:]}", flush=True)
            return 1
        return 0
    finally:
        await processes.stop_all()
        print(robot.unplug(), flush=True)
        print("sim stopped", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    repo_root = Path.cwd()
    try:
        if args[:1] == ["prepare"]:
            print(prepare(repo_root))
            return 0
        if args[:1] == ["run"]:
            return asyncio.run(_run(repo_root))
    except SimUnavailable as exc:
        print(str(exc))
        return 1
    if args[:1] == ["sweep"]:
        found = sweep()
        for line in found:
            print(f"stopped sim leftover: {line}")
        print(f"sim sweep: {len(found)} leftover(s)")
        return 1 if found else 0
    print("usage: python -m assistant_testing.sim prepare|run|sweep", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
