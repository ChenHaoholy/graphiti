"""Build or incrementally update the Graphiti graph from local events."""

import argparse
import asyncio
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from graphiti_core.nodes import EpisodeType
from pydantic import ValidationError

try:
    from .community_summary import community_to_dict
    from .constants import DEFAULT_GROUP_ID
    from .graphiti_client import get_graphiti_client
    from .schema import ConflictEvent
except ImportError:
    from community_summary import community_to_dict
    from constants import DEFAULT_GROUP_ID
    from graphiti_client import get_graphiti_client
    from schema import ConflictEvent


DEFAULT_EVENTS_PATH = "data/events.jsonl"
DEFAULT_STATE_PATH = "outputs/import_state.json"
DEFAULT_COMMUNITY_OUTPUT = "outputs/community_summaries.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incrementally build the Graphiti graph from data/events.jsonl."
    )
    parser.add_argument(
        "--events",
        default=DEFAULT_EVENTS_PATH,
        help=f"Event JSONL path. Default: {DEFAULT_EVENTS_PATH}",
    )
    parser.add_argument(
        "--state",
        default=DEFAULT_STATE_PATH,
        help=f"Persistent import state path. Default: {DEFAULT_STATE_PATH}",
    )
    parser.add_argument(
        "--no-communities",
        action="store_true",
        help="Skip community summary rebuild after import.",
    )
    parser.add_argument(
        "--community-output",
        default=DEFAULT_COMMUNITY_OUTPUT,
        help=f"Community summary output path. Default: {DEFAULT_COMMUNITY_OUTPUT}",
    )
    parser.add_argument(
        "--all-groups",
        action="store_true",
        help="Build community summaries for all groups instead of the project group.",
    )
    return parser.parse_args()


def iter_events(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if line:
                yield line_number, line


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "imported": {}}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"version": 1, "imported": {}}

    if not isinstance(data, dict):
        return {"version": 1, "imported": {}}
    data.setdefault("version", 1)
    data.setdefault("imported", {})
    return data


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def event_key(event: ConflictEvent) -> str:
    if event.source_url:
        return f"url:{event.source_url}"
    return f"id:{event.event_id}"


def stable_episode_uuid(event: ConflictEvent) -> str:
    seed = event.source_url or event.event_id
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def mark_imported(
    state: dict[str, Any],
    event: ConflictEvent,
    episode_uuid: str,
    status: str,
) -> None:
    state["imported"][event_key(event)] = {
        "event_id": event.event_id,
        "source_url": event.source_url,
        "episode_uuid": episode_uuid,
        "status": status,
        "updated_at": utc_now_iso(),
    }


async def episode_exists(graphiti, event: ConflictEvent, episode_uuid: str) -> bool:
    try:
        episodes = await graphiti.nodes.episode.get_by_uuids([episode_uuid])
        if episodes:
            return True
    except Exception:
        pass

    try:
        result = await graphiti.driver.execute_query(
            """
            MATCH (e:Episodic)
            WHERE e.source_description CONTAINS $source_url
               OR e.content CONTAINS $event_id
            RETURN e.uuid AS uuid
            LIMIT 1
            """,
            source_url=event.source_url,
            event_id=event.event_id,
        )
        return bool(result.records)
    except Exception:
        return False


async def add_event(graphiti, event: ConflictEvent, episode_uuid: str) -> None:
    await graphiti.add_episode(
        name=f"{event.event_type}: {event.event_id}",
        episode_body=event.to_episode_text(),
        source_description=(
            f"{event.source_name}; url={event.source_url}; "
            f"claim_type={event.claim_type}; confidence={event.confidence}"
        ),
        reference_time=event.reference_time(),
        source=EpisodeType.text,
        group_id=DEFAULT_GROUP_ID,
    )


async def write_community_summaries(
    graphiti,
    output_path: Path,
    all_groups: bool = False,
) -> tuple[int, int]:
    group_ids = None if all_groups else [DEFAULT_GROUP_ID]
    communities, community_edges = await graphiti.build_communities(group_ids=group_ids)
    payload = {
        "generated_at": utc_now_iso(),
        "group_id": None if all_groups else DEFAULT_GROUP_ID,
        "community_count": len(communities),
        "community_edge_count": len(community_edges),
        "communities": [community_to_dict(node) for node in communities],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(communities), len(community_edges)


async def build_graph(args: argparse.Namespace) -> dict[str, int]:
    events_path = Path(args.events)
    state_path = Path(args.state)
    community_output = Path(args.community_output)

    if not events_path.exists():
        raise FileNotFoundError(f"Event file not found: {events_path}")

    state = load_state(state_path)
    imported_state: dict[str, Any] = state["imported"]

    graphiti = get_graphiti_client()
    success_count = 0
    skipped_count = 0
    failed_count = 0
    parsed_count = 0

    try:
        print("Building indices and constraints...")
        await graphiti.build_indices_and_constraints()

        for line_number, raw_event in iter_events(events_path):
            try:
                event = ConflictEvent.model_validate_json(raw_event)
                parsed_count += 1
            except ValidationError as exc:
                failed_count += 1
                print(f"  [校验失败] 行 {line_number}: {exc}", file=sys.stderr)
                continue

            key = event_key(event)
            episode_uuid = stable_episode_uuid(event)
            if key in imported_state:
                skipped_count += 1
                continue

            if await episode_exists(graphiti, event, episode_uuid):
                skipped_count += 1
                mark_imported(state, event, episode_uuid, "found_existing")
                save_state(state_path, state)
                continue

            try:
                await add_event(graphiti, event, episode_uuid)
                success_count += 1
                mark_imported(state, event, episode_uuid, "imported")
                save_state(state_path, state)
            except Exception as exc:
                failed_count += 1
                print(f"  [导入失败] {event.event_id}: {exc}", file=sys.stderr)

        should_build_communities = (
            not args.no_communities and (success_count > 0 or not community_output.exists())
        )
        if should_build_communities:
            try:
                community_count, community_edge_count = await write_community_summaries(
                    graphiti,
                    community_output,
                    all_groups=args.all_groups,
                )
                print(f"  社区摘要：{community_count} 个社区 | {community_edge_count} 条关系", file=sys.stderr)
            except Exception as exc:
                print(f"  [警告] 社区摘要构建失败: {exc}", file=sys.stderr)
        elif args.no_communities:
            pass
        else:
            pass

    finally:
        await graphiti.close()

    return {
        "parsed": parsed_count,
        "success": success_count,
        "skipped": skipped_count,
        "failed": failed_count,
    }


async def main() -> None:
    args = parse_args()
    stats = await build_graph(args)
    print(f"[build] 解析 {stats['parsed']} | 导入 {stats['success']} | 跳过 {stats['skipped']} | 失败 {stats['failed']}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
