"""Pure logic that the e2e features cannot pin down: backoff bounds and token parsing."""

import random

import pytest

from assistant_link.auth import DevTokenVerifier, bearer_token
from assistant_link.backoff import BACKOFF_MAX_S, BACKOFF_MIN_S, JITTER, backoff_delay

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("attempt", [1, 2, 3, 4, 5, 6, 10, 100, 10_000])
def test_backoff_stays_in_window(attempt: int) -> None:
    rng = random.Random(attempt)
    base = min(BACKOFF_MAX_S, BACKOFF_MIN_S * 2 ** min(attempt - 1, 32))
    low = max(BACKOFF_MIN_S, base * (1 - JITTER))
    high = min(BACKOFF_MAX_S, base * (1 + JITTER))
    for _ in range(200):
        assert low <= backoff_delay(attempt, rng) <= high


def test_backoff_grows_then_caps() -> None:
    class Middle(random.Random):
        def uniform(self, a: float, b: float) -> float:
            return (a + b) / 2

    delays = [backoff_delay(n, Middle()) for n in range(1, 8)]
    assert delays == [0.5, 1.0, 2.0, 4.0, 8.0, 10.0, 10.0]


def test_backoff_jitter_spreads_edges() -> None:
    rng = random.Random(7)
    assert len({round(backoff_delay(3, rng), 6) for _ in range(20)}) > 10


def test_backoff_rejects_attempt_zero() -> None:
    with pytest.raises(ValueError, match="attempt"):
        backoff_delay(0)


@pytest.mark.parametrize(
    ("header", "token"),
    [
        ("Bearer abc", "abc"),
        ("bearer  abc ", "abc"),
        ("Bearer", None),
        ("Bearer ", None),
        ("Basic abc", None),
        ("", None),
        (None, None),
    ],
)
def test_bearer_token(header: str | None, token: str | None) -> None:
    assert bearer_token(header) == token


async def test_dev_token_verifier() -> None:
    verifier = DevTokenVerifier("s3cret")
    assert await verifier.verify("s3cret", "desk")
    assert await verifier.verify("s3cret", None)
    with pytest.raises(ValueError, match="empty"):
        DevTokenVerifier("")
