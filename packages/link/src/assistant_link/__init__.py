"""EdgeLink v1 transport: the brain-side WebSocket server and the edge-side client.

Import from the submodules: `assistant_link.server.LinkServer`, `assistant_link.client.LinkClient`,
`assistant_link.auth.TokenVerifier` (pluggable; S7 adds the device registry),
`assistant_link.connection.Connection` and `assistant_link.backoff.backoff_delay`. Both ends
accept an optional SSL context, and the client a certificate-pinning hook (used from S7).
The package imports nothing eagerly, so `python -m assistant_link.server` runs cleanly.
"""
