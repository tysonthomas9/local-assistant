"""exec another Python inside Reachy Edge.app: `exec_python.py <python> [args...]`.

Reachy Edge.app's launcher always starts ~/assistant-edge/src/.venv-assistant/bin/python as its
child (scripts/edge_app_run.sh, branch redesign/architecture). The app must not be rebuilt, so
this script, run by that Python, replaces itself (same process, so the app stays responsible
for the mic) with the daemon's or the conversation app's own venv Python.

That venv's GStreamer bundle sets PYTHONPATH, GST_*, GI_* ... to its own directories when Python
starts; they are removed first, or the target Python would import that venv's packages.
"""

import os
import sys

if len(sys.argv) < 2:
    sys.exit("usage: exec_python.py <python> [args...]")
mine = sys.prefix
for key, value in list(os.environ.items()):
    if mine in value:
        kept = [part for part in value.split(os.pathsep) if mine not in part]
        if kept:
            os.environ[key] = os.pathsep.join(kept)
        else:
            del os.environ[key]
os.execv(sys.argv[1], sys.argv[1:])
