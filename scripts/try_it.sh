#!/usr/bin/env bash
# Try the assistant on the robot, hands-free. Run it from anywhere in the repo:
#
#   scripts/try_it.sh                      # wake word (the default): say "hey jarvis", then ask
#   scripts/try_it.sh --listen open_mic    # just talk; talk over the robot to interrupt it
#   scripts/try_it.sh --listen push_to_talk  # debugging: Enter starts / stops listening
#
# Starts (or reuses) the LLM server and the speech server, syncs this checkout's HEAD to the
# robot's machine (the [test.edge_host] ssh alias), starts the daemon, the brain, the SSH
# tunnels and the edge agent (inside Reachy Edge.app on a Mac) and shows what was heard and
# the replies. Ctrl-C stops everything it started: the robot goes to rest with its motors off
# and nothing is left running on this PC or the robot's machine. See
# packages/testing/src/assistant_testing/try_it.py.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
export UV_PROJECT_ENVIRONMENT=.venv-assistant
exec uv run --locked -q python -m assistant_testing.try_it "$@"
