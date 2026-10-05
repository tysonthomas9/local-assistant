"""E2E feature runner, real-process launchers and test helpers.

Never imported by production packages. There are deliberately no fakes here: feature files
drive real processes (brain, edge, daemon, model servers) and real codecs.
"""

from assistant_testing.processes import CompletedRun, ManagedProcess, ProcessGroup

__all__ = ["CompletedRun", "ManagedProcess", "ProcessGroup"]
