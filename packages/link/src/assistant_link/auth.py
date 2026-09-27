"""Bearer-token checks for EdgeLink connections.

The server asks a `TokenVerifier` who a token belongs to. S2 ships `DevTokenVerifier` (one
shared development token); S7 swaps in the device registry (hashed, revocable per-device
tokens) behind the same Protocol.
"""

import hmac
from typing import Protocol


class TokenVerifier(Protocol):
    """Decides whether a bearer token may open an EdgeLink connection."""

    async def verify(self, token: str, device_id: str | None) -> bool:
        """True if `token` is valid (for `device_id`, when the verifier binds tokens to devices).

        `device_id` is None while the handshake has not sent `hello` yet.
        """
        ...


class DevTokenVerifier:
    """Accepts exactly one shared development token, for any device (loopback dev only)."""

    def __init__(self, token: str) -> None:
        if not token:
            raise ValueError("the dev token must not be empty")
        self._token = token.encode()

    async def verify(self, token: str, device_id: str | None) -> bool:
        del device_id
        return hmac.compare_digest(token.encode(), self._token)


def bearer_token(authorization: str | None) -> str | None:
    """The token of an `Authorization: Bearer <token>` header value, or None."""
    if not authorization:
        return None
    scheme, _, token = authorization.strip().partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        return None
    return token
