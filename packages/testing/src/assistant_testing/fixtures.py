"""pytest fixtures for real hardware, registered by the feature plugin.

`reachy_daemon` makes sure the real Reachy Mini daemon is running on the robot's machine
(this PC, or the edge host over SSH): it reuses a daemon that already answers, otherwise it
syncs the code under test there and starts one (with media, every socket on loopback). It
yields the daemon's base URL as reachable from here (through an `ssh -L` tunnel for an edge
host) and on teardown stops only what it started. When the robot or the daemon is unavailable
it FAILS the test, never skips it (tier `hw` means the robot is required) unless the gate runs
with GATE_NO_HW=1, which skips it.
"""

import asyncio
import os
from collections.abc import Iterator

import pytest

REACHY_DAEMON_URL = "http://127.0.0.1:8000"
"""The daemon's address on the robot's own machine."""


@pytest.fixture
def reachy_daemon(request: pytest.FixtureRequest) -> Iterator[str]:
    """The base URL of a running real Reachy Mini daemon."""
    if os.environ.get("GATE_NO_HW") == "1":
        pytest.skip("GATE_NO_HW=1: the hardware tier is off")
    from assistant_testing.features.context import ScenarioContext
    from assistant_testing.steps import edge_host as steps

    ctx = ScenarioContext(repo_root=request.config.rootpath, feature_path=request.path)

    async def start() -> str:
        await steps.robot_host_found(ctx)
        if steps.host_of(ctx).ssh is not None:
            await steps.code_synced_to_edge_host(ctx)
        return await steps.ensure_reachy_daemon(ctx)

    with asyncio.Runner() as runner:
        try:
            yield runner.run(start())
        finally:
            runner.run(ctx.aclose())
