
"""Community summary helpers used by the graph build command."""

from typing import Any


def community_to_dict(community: Any) -> dict[str, Any]:
    created_at = getattr(community, "created_at", None)
    if hasattr(created_at, "isoformat"):
        created_at_value = created_at.isoformat()
    else:
        created_at_value = created_at

    return {
        "uuid": getattr(community, "uuid", None),
        "name": getattr(community, "name", None),
        "group_id": getattr(community, "group_id", None),
        "summary": getattr(community, "summary", None),
        "created_at": created_at_value,
    }
