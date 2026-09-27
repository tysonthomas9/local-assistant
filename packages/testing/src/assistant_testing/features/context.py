"""Per-scenario state handed to every step."""

from pathlib import Path
from typing import Any

from assistant_testing.processes import ProcessGroup


class ScenarioContext:
    """What steps share within one scenario. `aclose` stops every process that was started."""

    def __init__(self, repo_root: Path, feature_path: Path) -> None:
        self.repo_root = repo_root
        self.feature_path = feature_path
        self.processes = ProcessGroup(cwd=repo_root)
        self.state: dict[str, Any] = {}

    async def aclose(self) -> None:
        await self.processes.stop_all()
