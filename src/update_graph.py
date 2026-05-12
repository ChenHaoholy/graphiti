"""
update_graph.py — 开发者入口：将新数据写入 Graphiti 图谱。

支持两种数据源：
  --source data/events.jsonl          本地 JSONL 文件
  --source gdelt --days 7 --limit 50  GDELT API 实时抓取

复用已有模块：fetch_gdelt、batch_ingest（仅读取逻辑）、graphiti_client。
写入完成后打印统计：成功数量、失败数量、跳过重复数量。
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── 本项目模块 ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError

try:
    from src.schema import ConflictEvent
    from src.graphiti_client import get_graphiti_client
except ImportError:
    from schema import ConflictEvent
    from graphiti_client import get_graphiti_client

# batch_ingest 中的写入函数（可被复用或参照）
try:
    from src.batch_ingest import ingest_event, iter_events
except ImportError:
    # 如果 batch_ingest 不存在，内联基本逻辑
    def iter_events(path: Path):
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    yield line_number, line

    async def ingest_event(graphiti, event: ConflictEvent) -> None:
        from graphiti_core.nodes import EpisodeType

        await graphiti.add_episode(
            name=f"{event.event_type}: {event.event_id}",
            episode_body=event.to_episode_text(),
            source_description=(
                f"{event.source_name}; url={event.source_url}; "
                f"claim_type={event.claim_type}; confidence={event.confidence}"
            ),
            reference_time=event.reference_time(),
            source=EpisodeType.text,
            # 移除 group_id，每个事件独立创建 episode
        )


# ── 并行导入配置 ──────────────────────────────────────────────
DEFAULT_CONCURRENCY = 5  # 并行数
REQUEST_DELAY = 0.5  # 请求间隔（秒）


# GDELT fetcher（按需导入，避免 gdelt 脚本缺失时报错）
def try_import_fetch_gdelt():
    try:
        from src.fetch_gdelt import run_fetch
        return run_fetch
    except ImportError:
        return None


# ── 去重记录（进程内全局，避免重复写入） ───────────────────────
_seen_ids: set[str] = set()


def is_duplicate(event: ConflictEvent) -> bool:
    """按 source_url 优先去重，其次按 event_id。"""
    if event.source_url:
        key = f"url:{event.source_url}"
        if key in _seen_ids:
            return True
        _seen_ids.add(key)
        return False
    if event.event_id:
        key = f"id:{event.event_id}"
        if key in _seen_ids:
            return True
        _seen_ids.add(key)
        return False
    return False


# ── 并行导入 worker ───────────────────────────────────────────
_semaphore: asyncio.Semaphore | None = None


async def import_single_event(
    graphiti,
    event: ConflictEvent,
    line_number: int,
) -> tuple[str, str | None, str | None]:
    """导入单个事件，返回 (event_id, status, error_msg)"""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(DEFAULT_CONCURRENCY)

    async with _semaphore:
        if is_duplicate(event):
            return (event.event_id or "", "skipped", "duplicate")

        try:
            await ingest_event(graphiti, event)
            return (event.event_id or "", "success", None)
        except Exception as exc:
            return (event.event_id or "", "failed", str(exc))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="[开发者] 将新事件数据写入 Graphiti 图谱。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 从本地 JSONL 导入（串行）
  python -m src.update_graph --source data/events.jsonl

  # 从本地 JSONL 导入（并行，默认5并发）
  python -m src.update_graph --source data/events.jsonl --parallel

  # 从 GDELT 实时抓取并导入
  python -m src.update_graph --source gdelt --days 7 --limit 50
        """,
    )
    parser.add_argument(
        "--source",
        choices=["jsonl", "gdelt"],
        default="jsonl",
        help="数据源类型：jsonl（本地文件）或 gdelt（GDELT API）",
    )
    parser.add_argument(
        "--path",
        default="data/events.jsonl",
        help="本地 JSONL 文件路径（仅 --source jsonl 时有效）",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="GDELT 回溯天数（仅 --source gdelt 时有效）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="GDELT 单次最大返回条数（仅 --source gdelt 时有效）",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="启用并行导入（默认5并发）",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"并行导入的并发数（默认 {DEFAULT_CONCURRENCY}）",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    global _semaphore, DEFAULT_CONCURRENCY
    if args.parallel:
        _semaphore = asyncio.Semaphore(args.concurrency)
        DEFAULT_CONCURRENCY = args.concurrency
        print(f"[Parallel mode] Concurrency: {args.concurrency}")

    graphiti = get_graphiti_client()

    try:
        print("Building indices and constraints...")
        await graphiti.build_indices_and_constraints()

        # ── 本地 JSONL ────────────────────────────────────────
        if args.source == "jsonl":
            source_path = Path(args.path)
            if not source_path.exists():
                print(f"错误：文件不存在 {source_path}", file=sys.stderr)
                sys.exit(1)

            print(f"Reading events from {source_path} ...")
            
            # 先解析所有事件
            events_to_import: list[tuple[int, ConflictEvent]] = []
            for line_number, raw_event in iter_events(source_path):
                try:
                    event = ConflictEvent.model_validate_json(raw_event)
                    events_to_import.append((line_number, event))
                except ValidationError as exc:
                    print(f"Line {line_number}: validation error — {exc}")

            total = len(events_to_import)
            print(f"Parsed {total} events, starting import...")

            if args.parallel:
                # 并行导入
                tasks = [
                    import_single_event(graphiti, event, line_num)
                    for line_num, event in events_to_import
                ]
                
                results = []
                for i, coro in enumerate(asyncio.as_completed(tasks)):
                    result = await coro
                    results.append(result)
                    done = i + 1
                    if result[1] == "success":
                        print(f"[{done}/{total}] imported {result[0]}")
                    elif result[1] == "skipped":
                        print(f"[{done}/{total}] skipped {result[0]}")
                    else:
                        print(f"[{done}/{total}] failed {result[0]}: {result[2]}")

                success_count = sum(1 for r in results if r[1] == "success")
                skip_count = sum(1 for r in results if r[1] == "skipped")
                failure_count = sum(1 for r in results if r[1] == "failed")
            else:
                # 串行导入（原有逻辑）
                success_count = 0
                failure_count = 0
                skip_count = 0
                for line_number, event in events_to_import:
                    if is_duplicate(event):
                        skip_count += 1
                        print(f"Line {line_number}: skipped duplicate {event.event_id or event.source_url}")
                        continue

                    try:
                        await ingest_event(graphiti, event)
                        success_count += 1
                        print(f"Line {line_number}: imported {event.event_id}")
                    except Exception as exc:
                        failure_count += 1
                        print(f"Line {line_number}: failed to ingest {event.event_id} — {exc}")

        # ── GDELT API ──────────────────────────────────────────
        elif args.source == "gdelt":
            success_count = 0
            failure_count = 0
            skip_count = 0
            run_fetch = try_import_fetch_gdelt()
            if run_fetch is None:
                print("错误：src/fetch_gdelt.py 不存在，无法使用 gdelt 数据源。", file=sys.stderr)
                sys.exit(1)

            gdelt_output = Path("data/gdelt_events_temp.jsonl")
            print(f"Fetching GDELT events (days={args.days}, limit={args.limit}) ...")
            try:
                run_fetch(days=args.days, limit=args.limit, output=str(gdelt_output))
            except Exception as exc:
                print(f"GDELT fetch failed: {exc}", file=sys.stderr)
                sys.exit(1)

            if not gdelt_output.exists():
                print("GDELT fetch returned no data.", file=sys.stderr)
                sys.exit(1)

            print(f"Importing GDELT events from {gdelt_output} ...")
            for line_number, raw_event in iter_events(gdelt_output):
                try:
                    event = ConflictEvent.model_validate_json(raw_event)
                except ValidationError as exc:
                    failure_count += 1
                    print(f"Line {line_number}: validation error — {exc}")
                    continue

                if is_duplicate(event):
                    skip_count += 1
                    print(f"Line {line_number}: skipped duplicate {event.event_id or event.source_url}")
                    continue

                try:
                    await ingest_event(graphiti, event)
                    success_count += 1
                    print(f"Line {line_number}: imported {event.event_id}")
                except Exception as exc:
                    failure_count += 1
                    print(f"Line {line_number}: failed to ingest {event.event_id} — {exc}")

            # 清理临时文件
            try:
                gdelt_output.unlink()
            except OSError:
                pass

    finally:
        await graphiti.close()

    print(f"\n===== Update Complete =====")
    print(f"  Success : {success_count}")
    print(f"  Skipped : {skip_count}")
    print(f"  Failed  : {failure_count}")


if __name__ == "__main__":
    asyncio.run(main())
