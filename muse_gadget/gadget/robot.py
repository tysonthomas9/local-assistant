"""Robot commands Muse can call: ``reachy.*``, forwarded to Pollen's own app tools.

Each command maps to one tool of Pollen's conversation app (``play_emotion``,
``dance``, ``stop_dance``, ``move_head``, ``head_tracking``, ``robot_status``)
and is forwarded over loopback to the robot-tools endpoint that MuseHandler
serves inside the app while a run holds the robot (``mac/robot_tools.py``).
The gadget never moves the robot itself and never starts it: with no robot
session the endpoint isn't there, and the command answers "robot is asleep".

The endpoint wants the run's shared secret (``$MUSE_ROBOT_TOOLS_SECRET``, given
to the container at start) as ``Authorization: Bearer <secret>``.

The enums mirror Pollen's lists at the pinned app commit (f58523b):
``EMOTION_INTENTS`` in ``tools/play_emotion.py``, ``AVAILABLE_MOVES`` in
``reachy_mini_dances_library`` and ``MoveHead.DELTAS``. The endpoint checks
again, and Pollen's tools check a third time.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

log = logging.getLogger(__name__)

URL_ENV = "MUSE_ROBOT_TOOLS_URL"
SECRET_ENV = "MUSE_ROBOT_TOOLS_SECRET"
DEFAULT_URL = "http://host.containers.internal:48081"
TIMEOUT_S = 15.0
ASLEEP = "robot is asleep: no robot session is running, so it can't move right now"

EMOTIONS = (
    "random", "happy", "excited", "loving", "grateful", "success", "thinking", "attentive",
    "confused", "uncertain", "sad", "downcast", "lonely", "angry", "irritated", "displeased",
    "disgusted", "scared", "anxious", "surprised", "amazed", "calming", "relief", "impatient",
    "embarrassed", "bored", "tired", "sleepy", "yes", "yes_understanding", "no", "no_sad",
    "no_excited", "no_firm", "welcoming", "greeting", "goodbye", "go_away", "helpful", "dance",
    "electric", "dying",
)
DANCES = (
    "simple_nod", "head_tilt_roll", "side_to_side_sway", "dizzy_spin", "stumble_and_recover",
    "headbanger_combo", "interwoven_spirals", "sharp_side_tilt", "side_peekaboo", "yeah_nod",
    "uh_huh_tilt", "neck_recoil", "chin_lead", "groovy_sway_and_roll", "chicken_peck",
    "side_glance_flick", "polyrhythm_combo", "grid_snap", "pendulum_swing", "jackson_square",
)
DIRECTIONS = ("left", "right", "up", "down", "front")
# robot_status topics that stay on the robot: no wifi (IP address), account or app list.
STATUS_TOPICS = ("name", "software", "imu")

_TIMEOUT_MS = int(TIMEOUT_S * 1000) + 5000


def _param(type_: str, description: str, enum: tuple[str, ...] | None = None) -> dict:
    spec: dict = {"type": type_, "description": description}
    if enum:
        spec["enum"] = list(enum)
    return spec


# command -> (Pollen tool, spec)
COMMANDS: dict[str, tuple[str, dict]] = {
    "reachy.emotion": ("play_emotion", {
        "description": (
            "Make the robot body show an emotion with one of Pollen's recorded moves (a few "
            "seconds). Use it when a feeling fits your answer, or when asked to show one "
            "(e.g. 'show me you're happy')."
        ),
        "required": {"emotion": _param("string", "Emotion to show. One of: " + ", ".join(EMOTIONS) + ".", EMOTIONS)},
        "optional": {},
        "timeout_ms": _TIMEOUT_MS,
    }),
    "reachy.dance": ("dance", {
        "description": (
            "Make the robot dance one of Pollen's dance moves (head and antennas). Use it when "
            "asked to dance or to celebrate. Omit move for a random one."
        ),
        "required": {},
        "optional": {"move": _param("string", "Dance move. One of: " + ", ".join(DANCES) + ".", DANCES)},
        "timeout_ms": _TIMEOUT_MS,
    }),
    "reachy.stop_move": ("stop_dance", {
        "description": "Stop the robot's current dance or emotion and clear its queued moves. Use it when asked to stop moving.",
        "required": {},
        "optional": {},
        "timeout_ms": _TIMEOUT_MS,
    }),
    "reachy.look": ("move_head", {
        "description": "Turn the robot's head to look left, right, up, down, or back to the front. Use it when asked to look somewhere.",
        "required": {"direction": _param("string", "One of: left, right, up, down, front.", DIRECTIONS)},
        "optional": {},
        "timeout_ms": _TIMEOUT_MS,
    }),
    "reachy.head_tracking": ("head_tracking", {
        "description": (
            "Turn on or off the robot following the user's face with its head. Use it when asked "
            "to look at, follow or keep watching the user, or to stop. No image leaves the robot."
        ),
        "required": {"enabled": _param("boolean", "true to follow the user's face, false to stop.")},
        "optional": {},
        "timeout_ms": _TIMEOUT_MS,
    }),
    "reachy.status": ("robot_status", {
        "description": (
            "Read the robot's own status: 'name' (its name), 'software' (version, update "
            "available) or 'imu' (head tilt, moving or not, temperature). Also tells you if the "
            "robot is asleep."
        ),
        "required": {"topic": _param("string", "One of: name, software, imu.", STATUS_TOPICS)},
        "optional": {},
        "timeout_ms": _TIMEOUT_MS,
    }),
}


def command_specs() -> dict:
    return {name: spec for name, (_, spec) in COMMANDS.items()}


class BadParams(ValueError):
    pass


def tool_args(command: str, params: dict) -> tuple[str, dict]:
    """The Pollen tool and its arguments for one ``reachy.*`` command; BadParams if invalid."""
    if command not in COMMANDS:
        raise BadParams(f"unsupported command: {command}")
    if not isinstance(params, dict):
        raise BadParams("params must be an object")
    tool, spec = COMMANDS[command]
    allowed = set(spec["required"]) | set(spec["optional"])
    extra = sorted(set(params) - allowed)
    if extra:
        raise BadParams(f"unknown parameter: {', '.join(extra)}")
    args: dict = {}
    for name, pspec in {**spec["required"], **spec["optional"]}.items():
        if name not in params or params[name] is None:
            if name in spec["required"]:
                raise BadParams(f"missing parameter: {name}")
            continue
        value = params[name]
        if pspec["type"] == "boolean":
            if not isinstance(value, bool):
                raise BadParams(f"{name} must be true or false")
        elif not isinstance(value, str) or value not in pspec.get("enum", ()):
            raise BadParams(f"{name} must be one of: {', '.join(pspec['enum'])}")
        args[name] = value
    if tool == "stop_dance":
        args["dummy"] = True   # Pollen's schema requires it
    return tool, args


class RobotTools:
    """Forwards ``reachy.*`` commands to the app's robot-tools endpoint."""

    def __init__(self, url: str | None = None, secret: str | None = None, timeout_s: float = TIMEOUT_S) -> None:
        self.url = (url if url is not None else os.environ.get(URL_ENV) or DEFAULT_URL).rstrip("/")
        self.secret = secret if secret is not None else os.environ.get(SECRET_ENV, "")
        self.timeout_s = timeout_s

    def run(self, command: str, params: dict) -> dict:
        """Upstream-style result: ``{"ok": True, "payload": ...}`` or ``{"ok": False, "error": ...}``."""
        try:
            tool, args = tool_args(command, params)
        except BadParams as exc:
            result = {"ok": False, "error": str(exc)}
        else:
            result = self._forward(tool, args)
        log.info("robot command %s -> %s", command, "ok" if result["ok"] else "error: " + result["error"])
        return result

    def _forward(self, tool: str, args: dict) -> dict:
        if not self.secret:
            return {"ok": False, "error": ASLEEP}
        body = json.dumps({"tool": tool, "args": args}).encode()
        request = urllib.request.Request(
            self.url + "/tool", data=body, method="POST",
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.secret},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                answer = json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            if exc.code == 503:
                return {"ok": False, "error": ASLEEP}
            if exc.code == 401:
                return {"ok": False, "error": "the robot refused this gadget (wrong run secret)"}
            return {"ok": False, "error": f"robot tools answered HTTP {exc.code}"}
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            if isinstance(exc, TimeoutError) or isinstance(getattr(exc, "reason", None), TimeoutError):
                return {"ok": False, "error": "the robot didn't answer in time"}
            return {"ok": False, "error": ASLEEP}   # nothing listening: no run holds the robot
        except ValueError:
            return {"ok": False, "error": "robot tools sent a bad answer"}
        if not isinstance(answer, dict):
            return {"ok": False, "error": "robot tools sent a bad answer"}
        result = answer.get("result")
        if isinstance(result, dict) and result.get("error"):
            return {"ok": False, "error": str(result["error"])}
        if not answer.get("ok"):
            return {"ok": False, "error": str(answer.get("error") or "robot tool failed")}
        return {"ok": True, "payload": result if isinstance(result, dict) else {"result": result}}
