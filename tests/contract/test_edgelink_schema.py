"""JSON Schema snapshot of all 23 EdgeLink messages.

The snapshot for a protocol version is frozen: if any message schema changes, this test fails
until PROTOCOL_VERSION is bumped and a new snapshot is written with `just schema-snapshot`
(which never overwrites an existing version's file).
"""

import json
import os
from pathlib import Path

import pytest

from assistant_contracts.messages import MESSAGE_TYPES
from assistant_contracts.schema import edgelink_schemas, edgelink_schemas_json
from assistant_contracts.version import PROTOCOL_VERSION

pytestmark = pytest.mark.contract

SNAPSHOTS = Path(__file__).parent / "snapshots"
SNAPSHOT = SNAPSHOTS / f"edgelink-v{PROTOCOL_VERSION}.json"


def test_snapshot_covers_all_23_messages() -> None:
    assert len(edgelink_schemas()["messages"]) == len(MESSAGE_TYPES) == 23


def test_schema_matches_snapshot_for_this_protocol_version() -> None:
    current = edgelink_schemas_json()
    if not SNAPSHOT.exists():
        if os.environ.get("UPDATE_SCHEMA_SNAPSHOT") == "1":
            SNAPSHOT.write_text(current)
            return
        pytest.fail(
            f"no schema snapshot for protocol {PROTOCOL_VERSION}; run `just schema-snapshot`"
        )
    frozen = json.loads(SNAPSHOT.read_text())
    if frozen != json.loads(current):
        changed = sorted(
            name
            for name in set(frozen["messages"]) | set(edgelink_schemas()["messages"])
            if frozen["messages"].get(name) != edgelink_schemas()["messages"].get(name)
        )
        pytest.fail(
            f"EdgeLink message schemas changed ({', '.join(changed)}) but PROTOCOL_VERSION is "
            f"still {PROTOCOL_VERSION}. Bump PROTOCOL_VERSION in assistant_contracts.version "
            "and run `just schema-snapshot` to freeze the new version."
        )
