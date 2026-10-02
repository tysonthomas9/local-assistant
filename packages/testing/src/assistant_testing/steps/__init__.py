"""Built-in feature steps. Importing this package registers them all."""

from assistant_testing.steps import brain, contracts, core, edge, edge_host, link, speech

__all__ = ["brain", "contracts", "core", "edge", "edge_host", "link", "speech"]
