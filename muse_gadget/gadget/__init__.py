"""Reachy Mini wrapper around the pinned Muse Linux Device SDK.

Upstream files are never edited. This package restricts the gadget's commands,
hides the host name, and adds a loopback-only HTTP bridge (``/turn``) that the
robot's conversation app uses to send a text turn and get Muse's reply.
"""

__version__ = "0.1.0"
