"""Run the reachy-mini daemon on a SIMULATED Reachy Mini (Pollen's MuJoCo backend), headless.

    python -m assistant_robot_reachy.sim [reachy-mini-daemon options...]

The same launcher as `assistant_robot_reachy.daemon` (loopback only, the motor watchdog, rest
on stop), with `--sim --headless` added: Pollen's own daemon drives a MuJoCo model of the
robot instead of the USB serial bus. It needs MuJoCo (the `sim` dependency group, synced into
`.venv-sim`, see `assistant_testing.sim`). Used by the e2e tier `sim`; never by a real robot.

One addition, upstream code unchanged (a wrap like the ones in `daemon`): Pollen's MuJoCo
backend has no motor modes (it always reports `enabled` and ignores a change), so the
simulated robot keeps the mode the daemon was asked for, starting `disabled` like a robot
at rest after `--no-wake-up-on-start`. The SDK's `wake_up()` / `goto_sleep()`, the REST
`/api/motors/set_mode/...` and the watchdog then see what they see on the real robot. MuJoCo's
servos have no torque switch: a simulated robot with its motors "off" still holds its pose.

MuJoCo starts the robot in Pollen's joint-space sleep pose, about 5 degrees of head pitch from
where `goto_sleep()` ends (a real robot at rest sits where `goto_sleep()` left it), and with no
head target: a `goto_sleep()` from there (close enough to skip its move) drifts the head to
the zero pose. So once the API serves, the simulated robot is moved to Pollen's sleep pose
(`/api/move/goto`, its `SLEEP_HEAD_POSE` and sleep antennas), put to rest the way the watchdog
does it (`goto_sleep`, then the motors off) and `SIM-READY` is printed: wait for that line,
not for uvicorn's.
"""

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from assistant_robot_reachy import daemon, watchdog

READY = "SIM-READY"

SIM_FLAGS = ("--sim", "--headless")


def _patch_motor_modes() -> None:
    from reachy_mini.daemon.backend.mujoco.backend import MujocoBackend
    from reachy_mini.io.protocol import MotorControlMode

    def get_motor_control_mode(self: Any) -> MotorControlMode:
        return getattr(self, "_sim_motor_mode", MotorControlMode.Disabled)

    def set_motor_control_mode(self: Any, mode: MotorControlMode) -> None:
        self._sim_motor_mode = MotorControlMode(mode)

    MujocoBackend.get_motor_control_mode = get_motor_control_mode
    MujocoBackend.set_motor_control_mode = set_motor_control_mode


def _goto_sleep_pose(api: watchdog.DaemonApi) -> None:
    from reachy_mini.daemon.backend.abstract import Backend

    move = {
        "head_pose": {"m": [float(v) for v in Backend.SLEEP_HEAD_POSE.flatten()]},
        "antennas": [float(v) for v in Backend.SLEEP_ANTENNAS_JOINT_POSITIONS],
        "duration": 1.0,
    }
    request = urllib.request.Request(
        f"{api.base}/api/move/goto",
        data=json.dumps(move).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5.0) as response:
        response.read()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        time.sleep(0.3)
        if api._call("/api/move/running") == []:
            return
    raise ValueError("the move to the sleep pose did not end within 10 s")


def _come_to_rest(base: str) -> None:
    api = watchdog.DaemonApi(base)
    deadline = time.monotonic() + 120
    while True:
        try:
            api.motors()
            break
        except (OSError, urllib.error.URLError):
            if time.monotonic() > deadline:
                print(f"{READY}-ERROR the daemon API never answered at {base}", flush=True)
                return
            time.sleep(0.2)
    try:
        _goto_sleep_pose(api)
        motors = api.rest()
    except (OSError, urllib.error.URLError, ValueError) as exc:
        print(f"{READY}-ERROR putting the simulated robot to rest: {exc}", flush=True)
        return
    print(f"{READY} at rest motors={motors} api={base}", flush=True)


def main() -> int:
    _patch_motor_modes()
    base = daemon._api_base(sys.argv[1:])
    threading.Thread(target=_come_to_rest, args=(base,), name="sim-rest", daemon=True).start()
    sys.argv[1:] = [*(flag for flag in SIM_FLAGS if flag not in sys.argv), *sys.argv[1:]]
    return daemon.main()


if __name__ == "__main__":
    raise SystemExit(main())
