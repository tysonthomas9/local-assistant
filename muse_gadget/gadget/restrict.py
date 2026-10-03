"""What the robot gadget offers Muse, and how it names itself.

Muse may call only the commands in ``ALLOWED_COMMANDS``: ``device.health`` and
the robot's ``reachy.*`` commands (``gadget/robot.py``). The SDK's shell and
file commands (``system.run``, ``file.read``, ``file.write``) are removed from
the registration *and* refused by the executor, so a Muse that ignores the
registration still gets nothing.

The display name is a fixed, neutral name, never the machine's host name.
"""

from __future__ import annotations

import os

from musegadget import executor as upstream_executor

from gadget import robot

DEFAULT_DISPLAY_NAME = "Reachy Mini"
DISPLAY_NAME_ENV = "MUSE_DISPLAY_NAME"
ALLOWED_COMMANDS = frozenset({"device.health", *robot.COMMANDS})
BLOCKED_COMMANDS = frozenset({"system.run", "file.read", "file.write", "device.ota"})


def display_name() -> str:
    """The neutral name shown in the Muse app (``$MUSE_DISPLAY_NAME`` or "Reachy Mini")."""
    name = (os.environ.get(DISPLAY_NAME_ENV) or "").strip()
    return name[:64] if name else DEFAULT_DISPLAY_NAME


def command_specs() -> dict:
    """Upstream command specs, filtered to the allowed ones, plus the robot's commands."""
    specs = {
        name: spec
        for name, spec in upstream_executor.COMMAND_SPECS.items()
        if name in ALLOWED_COMMANDS and name not in BLOCKED_COMMANDS
    }
    specs.update(robot.command_specs())
    return specs


class RestrictedExecutor(upstream_executor.Executor):
    """Upstream executor that refuses everything but the allowed commands."""

    def __init__(self, account, robot_tools: robot.RobotTools | None = None) -> None:
        super().__init__(account)
        self.robot_tools = robot_tools or robot.RobotTools()

    def run(self, command: str, params: dict, timeout_ms: int | None = None) -> dict:
        if command not in ALLOWED_COMMANDS or command in BLOCKED_COMMANDS:
            return upstream_executor.error(f"unsupported command: {command}")
        if command == "device.health":
            return upstream_executor.ok(health())
        if command in robot.COMMANDS:
            return self.robot_tools.run(command, params)
        return super().run(command, params, timeout_ms)


def health() -> dict:
    """Upstream device health with the host name replaced by the display name."""
    data = upstream_executor.device_health()
    data["hostname"] = display_name()
    data.pop("model", None)
    return data
