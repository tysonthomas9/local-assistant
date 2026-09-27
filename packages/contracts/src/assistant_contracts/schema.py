"""JSON Schema export of every EdgeLink message (used by the contract snapshot test).

python -m assistant_contracts.schema > snapshot.json
"""

import json
from typing import Any

from assistant_contracts.messages import MESSAGE_TYPES, message_type_name
from assistant_contracts.version import PROTOCOL_VERSION


def edgelink_schemas() -> dict[str, Any]:
    """`{"protocol_version": ..., "messages": {type: JSON Schema}}`, keys sorted."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "messages": {
            message_type_name(m): m.model_json_schema(mode="validation")
            for m in sorted(MESSAGE_TYPES, key=message_type_name)
        },
    }


def edgelink_schemas_json() -> str:
    return json.dumps(edgelink_schemas(), indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    print(edgelink_schemas_json(), end="")
