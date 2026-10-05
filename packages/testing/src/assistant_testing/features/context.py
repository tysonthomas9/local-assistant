"""Per-scenario state handed to every step."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from assistant_testing.processes import ProcessGroup

Robot = Literal["sim", "hw"]
"""Which robot a robot scenario drives: the simulated one on this PC or the physical one."""


class ScenarioContext:
    """What steps share within one scenario. `aclose` stops every process that was started,
    then runs the `cleanups` (last added first)."""

    def __init__(self, repo_root: Path, feature_path: Path, robot: Robot | None = None) -> None:
        self.repo_root = repo_root
        self.feature_path = feature_path
        self.robot = robot
        self.processes = ProcessGroup(cwd=repo_root)
        self.state: dict[str, Any] = {}
        self.cleanups: list[Callable[[], None]] = []

    @property
    def sim(self) -> bool:
        """The scenario runs on the simulated robot (tier `sim`)."""
        return self.robot == "sim"

    async def aclose(self) -> None:
        try:
            await self.processes.stop_all()
        finally:
            while self.cleanups:
                self.cleanups.pop()()
