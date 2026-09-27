"""pytest fixtures for real hardware, registered by the feature plugin.

`reachy_daemon` is a placeholder until S3 (robot-reachy) implements it. The contract it will
keep: make sure the real Reachy Mini daemon is running (start it if needed), yield its base URL,
and on teardown stop only a daemon it started itself. When the robot or the daemon is
unavailable it FAILS the test, never skips it (tier `hw` means the robot is required).
"""

import pytest

REACHY_DAEMON_URL = "http://127.0.0.1:8000"


@pytest.fixture
def reachy_daemon() -> str:
    """The base URL of a running real Reachy Mini daemon (implemented in S3)."""
    raise NotImplementedError(
        "reachy_daemon is a placeholder: task S3 (robot-reachy) implements it. It will start "
        "the real Reachy Mini daemon if needed, yield " + REACHY_DAEMON_URL + ", stop only a "
        "daemon it started, and fail (not skip) when the robot is missing."
    )
