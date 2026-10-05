"""Reconnect backoff: exponential from 0.5 s up to 10 s, with jitter (proposal section 5.1)."""

import random
from typing import Final

BACKOFF_MIN_S: Final = 0.5
BACKOFF_MAX_S: Final = 10.0
JITTER: Final = 0.2
"""Each delay is scaled by a random factor in [1 - JITTER, 1 + JITTER], then clamped."""


def backoff_delay(
    attempt: int,
    rng: random.Random | None = None,
    *,
    low: float = BACKOFF_MIN_S,
    high: float = BACKOFF_MAX_S,
) -> float:
    """Delay before reconnect attempt `attempt` (1 = the first retry after a drop).

    The base doubles per attempt (0.5, 1, 2, 4, 8, 10, 10, ...); jitter spreads edges that
    lost the brain at the same moment. The result is always within [low, high].
    """
    if attempt < 1:
        raise ValueError(f"attempt must be >= 1, got {attempt}")
    base = min(high, low * 2.0 ** min(attempt - 1, 32))
    factor = (rng or random).uniform(1.0 - JITTER, 1.0 + JITTER)
    return max(low, min(high, base * factor))
