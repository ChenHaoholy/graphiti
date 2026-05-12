import argparse
import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from graphiti_core.search.search_filters import ComparisonOperator, DateFilter, SearchFilters

try:
    from .graphiti_client import get_graphiti_client
    from .schema import EvidenceItem
except ImportError:
    from graphiti_client import get_graphiti_client
    from schema import EvidenceItem


DEFAULT_GROUP_ID = "us-iran-conflict"


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def to_utc_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def build_query(query: str | None, actor: str | None, event_type: str | None) -> str:
    parts = [part for part in [query, actor, event_type] if part]
    if parts:
        return " ".join(parts)
    return "Iran United States conflict"


def build_search_filter(start_time: str | None, end_time: str | None) -> SearchFilters | None:
    filters: list[DateFilter] = []
    start = parse_time(start_time)
    end = parse_time(end_time)

    if start is not None:
        filters.append(
            DateFilter(date=start, comparison_operator=ComparisonOperator.greater_than_equal)
        )
    if end is not None:
        filters.append(DateFilter(date=end, comparison_operator=ComparisonOperator.less_than_equal))

    if not filters:
        return None
    return SearchFilters(valid_at=[filters])


def parse_episode_content(content: str | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    if not content:
        return parsed

    for line in content.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            parsed[key.strip().lower()] = value.strip()

    return parsed


def parse_source_description(source_description: str | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    if not source_description:
        return parsed

    parts = [part.strip() for part in source_description.split(";") if part.strip()]
    if parts:
        parsed["source"] = parts[0]
    for part in parts[1:]:
        key, separator, value = part.partition("=")
        if separator:
            parsed[key.strip().lower()] = value.strip()
    return parsed


async def load_episode_map(graphiti, edges: list[Any]) -> dict[str, Any]:
    episode_uuids: list[str] = []
    for edge in edges:
        episodes = get_field(edge, "episodes", []) or []
        if isinstance(episodes, str):
            episodes = [episodes]
        episode_uuids.extend(str(uuid) for uuid in episodes if uuid)

    unique_uuids = list(dict.fromkeys(episode_uuids))
    if not unique_uuids:
        return {}

    try:
        episodes = await graphiti.nodes.episode.get_by_uuids(unique_uuids)
    except Exception:
        return {}

    return {episode.uuid: episode for episode in episodes}


def edge_time(edge: Any, episode_data: dict[str, str]) -> datetime | None:
    for value in [
        episode_data.get("event time (utc)"),
        get_field(edge, "reference_time"),
        get_field(edge, "valid_at"),
    ]:
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc)
        if isinstance(value, str):
            try:
                return parse_time(value)
            except ValueError:
                continue
    return None


def matches_filters(
    edge: Any,
    episode_data: dict[str, str],
    actor: str | None,
    event_type: str | None,
    start_time: str | None,
    end_time: str | None,
) -> bool:
    searchable_text = " ".join(
        [
            str(get_field(edge, "fact", "")),
            " ".join(f"{key}: {value}" for key, value in episode_data.items()),
        ]
    ).lower()

    if actor and actor.lower() not in searchable_text:
        return False
    if event_type and event_type.lower() not in searchable_text:
        return False

    current_time = edge_time(edge, episode_data)
    start = parse_time(start_time)
    end = parse_time(end_time)

    if start and current_time and current_time < start:
        return False
    if end and current_time and current_time > end:
        return False

    return True


def build_evidence_item(edge: Any, episode: Any, matched_query: str) -> EvidenceItem:
    fact = str(get_field(edge, "fact", "") or "")
    content = str(get_field(episode, "content", "") or "")
    source_description = str(get_field(episode, "source_description", "") or "")
    episode_data = parse_episode_content(content)
    source_data = parse_source_description(source_description)
    event_time = edge_time(edge, episode_data)

    source_name = episode_data.get("source") or source_data.get("source")
    source_url = episode_data.get("source url") or source_data.get("url")

    if content:
        evidence_content = f"{fact}\n\n{content}" if fact else content
    else:
        evidence_content = fact

    return EvidenceItem(
        content=evidence_content,
        source_name=source_name,
        source_url=source_url,
        event_time=to_utc_iso(event_time),
        matched_query=matched_query,
        relevance_note="Returned by Graphiti.search and linked to an ingested conflict episode.",
    )


async def search_evidence(
    query: str | None = None,
    actor: str | None = None,
    event_type: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    limit: int = 5,
    graphiti: Any | None = None,
) -> list[EvidenceItem]:
    matched_query = build_query(query, actor, event_type)
    search_filter = build_search_filter(start_time, end_time)
    num_results = max(limit * 3, limit, 10)
    owns_client = graphiti is None
    if graphiti is None:
        graphiti = get_graphiti_client()

    try:
        edges = await graphiti.search(
            query=matched_query,
            group_ids=[DEFAULT_GROUP_ID],
            num_results=num_results,
            search_filter=search_filter,
        )

        episode_map = await load_episode_map(graphiti, edges)
        evidence_items: list[EvidenceItem] = []

        for edge in edges:
            episodes = get_field(edge, "episodes", []) or []
            if isinstance(episodes, str):
                episodes = [episodes]
            episode = episode_map.get(episodes[0]) if episodes else None
            episode_data = parse_episode_content(get_field(episode, "content", None))

            if not matches_filters(edge, episode_data, actor, event_type, start_time, end_time):
                continue

            evidence_items.append(build_evidence_item(edge, episode, matched_query))
            if len(evidence_items) >= limit:
                break

        return evidence_items
    finally:
        if owns_client:
            await graphiti.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search Graphiti evidence for US-Iran conflict.")
    parser.add_argument("--query", help="Keyword query, for example: Iran sanctions")
    parser.add_argument("--actor", help="Simple actor filter, for example: IAEA")
    parser.add_argument("--event-type", help="Simple event type filter, for example: diplomacy")
    parser.add_argument("--start-time", help="UTC ISO start time, for example: 2025-04-15T00:00:00Z")
    parser.add_argument("--end-time", help="UTC ISO end time, for example: 2025-04-20T00:00:00Z")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of evidence items")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    items = await search_evidence(
        query=args.query,
        actor=args.actor,
        event_type=args.event_type,
        start_time=args.start_time,
        end_time=args.end_time,
        limit=args.limit,
    )
    print(json.dumps([item.model_dump() for item in items], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
