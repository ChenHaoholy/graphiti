"""
GDELT data adapter: fetch US-Iran conflict news via GDELT API and convert to ConflictEvent JSONL.

Uses GDELT 2.0 API (https://api.gdeltproject.org/api/v2/doc/doc).
No web scraping, no Neo4j access, no LLM API calls.
"""

import argparse
import json
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import requests

try:
    from .schema import ConflictEvent
except ImportError:
    from schema import ConflictEvent

DEFAULT_OUTPUT = "data/events.jsonl"
BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_MODE = "artlist"

QUERIES = [
    # 美伊核心冲突
    "Iran United States conflict military strike",
    "Iran sanctions oil economic",
    "Iran nuclear IAEA JCPOA deal",
    "Strait of Hormuz Iran military naval",
    "Iran missile drone attack",
    "Iran proxy militia Hezbollah Houthi Iraq Syria",
    # 中伊关系（补充中国视角）
    "China Iran strategic partnership trade oil",
    "China Iran diplomatic meeting Xi",
    "China Iran United Nations Security Council",
    "China Middle East diplomacy Iran",
    "China Iran military security cooperation",
    # 美中伊三角
    "China United States Iran tensions",
    "China response Iran attack US",
    "China Iran sanctions US pressure",
    # 中东局势外延（给推演提供上下文）
    "Israel Iran military strike",
    "Iran Israel Gaza war escalation",
    "Persian Gulf Iran military tension",
]

REQUEST_TIMEOUT = 30
MAX_RETRY = 3
BASE_RETRY_DELAY = 60  # 基础等待秒数（429错误时）


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_gdelt_datetime(raw: str | None) -> str | None:
    """Parse GDELT datetime to UTC ISO format."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except (ValueError, AttributeError):
        return None


def detect_event_type(title: str, text: str) -> str:
    """Simple rule-based event type detection."""
    content = (title + " " + text).lower()

    if re.search(r"sanction|treasury|economic", content):
        return "sanctions_action"
    if re.search(r"iaea|nuclear|uranium|enrichment|facility", content):
        return "nuclear_iaea"
    if re.search(r"attack|missile|drone|strike|military|bomb|assassination", content):
        return "military_strike"
    if re.search(r"talk|negotiation|diplomacy|mediation|security council", content):
        return "diplomatic_warning"
    if re.search(r"hormuz|shipping|oil tanker|persian gulf|maritime", content):
        return "maritime_incident"
    if re.search(r"proxy|militia|hezbollah|houthi| Kataib", content):
        return "proxy_attack_claim"
    return "general_tension"


def extract_actors(title: str, text: str) -> list[str]:
    """Enhanced actor extraction with multi-language support."""
    content = title + " " + text
    actors: list[str] = []

    # 多语言 actor 关键词 (英文 + 中文 + 阿拉伯文 + 俄文等)
    actor_keywords = [
        ("Iran", ["iran", "iranian", "伊朗", "إيران"]),
        ("United States", ["united states", "usa", "u.s.", "america", "american", "美国", "الولايات المتحدة", "США"]),
        ("Israel", ["israel", "israeli", "以色列", "إ以色列", "Израиль"]),
        ("China", ["china", "chinese", "中国", "北京", "الصين", "Китай"]),
        ("Russia", ["russia", "russian", "俄罗斯", "Россия"]),
        ("United Kingdom", ["britain", "british", "uk", "英国", "بريطانيا"]),
        ("UAE", ["uae", "emirates", "阿联酋", "迪拜", "الإمارات"]),
        ("Saudi Arabia", ["saudi", "沙特", "السعودية"]),
        ("Iraq", ["iraq", "iraqi", "伊拉克", "العراق"]),
        ("Syria", ["syria", "syrian", "叙利亚", "سوريا"]),
        ("Lebanon", ["lebanon", "lebanese", "黎巴嫩", "لبنان"]),
        ("Yemen", ["yemen", "yemeni", "也门", "اليمن"]),
        ("Hezbollah", ["hezbollah", "真主党", "حزب الله"]),
        ("Houthis", ["houthi", "houthies", "胡塞", "الحوثيين"]),
        ("IRGC", ["irgc", "revolutionary guard", "pasdaran", "伊斯兰革命卫队", "الحرس الثوري"]),
        ("Trump", ["trump", "特朗普", "川普", "ترامب"]),
        ("Putin", ["putin", "普京", "بوتين"]),
        ("Netanyahu", ["netanyahu", "内塔尼亚胡", "نتن ياهو"]),
        ("IAEA", ["iaea", "国际原子能机构"]),
        ("Qatar", ["qatar", "卡塔尔", "قطر"]),
        ("Pakistan", ["pakistan", "巴基斯坦", "باكستان"]),
    ]

    for actor_name, keywords in actor_keywords:
        for kw in keywords:
            if kw.lower() in content.lower():
                if actor_name not in actors:
                    actors.append(actor_name)
                break

    if not actors:
        actors.append("Unknown Actor")

    return actors


def detect_risk_tags(title: str, text: str) -> list[str]:
    """Detect risk tags based on content keywords."""
    content = (title + " " + text).lower()
    tags: list[str] = []

    tag_rules = {
        "military_escalation": r"attack|strike|missile|war|invasion|bombing",
        "nuclear_facility": r"nuclear|uranium|enrichment|facility|reactor",
        "retaliation_risk": r"retaliat|revenge|response|retaliation",
        "sanctions": r"sanction|treasury|economic|oil ban",
        "proxy_activity": r"proxy|militia|hezbollah|houthi|armed group",
        "maritime_security": r"hormuz|shipping|tanker|persian gulf|naval",
        "diplomatic_escalation": r"security council|diplomat|negotiat|talk",
        "civilian_risk": r"civilian|casualt|deaths|civilians killed",
        "regional_spillover": r"lebanon|syria|iraq|yemen|israel|gaza",
    }

    for tag, pattern in tag_rules.items():
        if re.search(pattern, content):
            tags.append(tag)

    return tags


def detect_confidence(title: str, text: str) -> str:
    """Estimate confidence based on source and content."""
    content = (title + " " + text).lower()

    official_keywords = r"government|official|minister|president|statement|announce"
    if re.search(official_keywords, content):
        return "high"

    witness_keywords = r"witness|resident|local|eyewitness|reported by"
    if re.search(witness_keywords, content):
        return "medium"

    return "low"


def detect_claim_type(title: str, text: str) -> str:
    """Detect claim type based on content."""
    content = (title + " " + text).lower()

    if re.search(r"official|government|strike back|declare|announce", content):
        return "official_statement"
    if re.search(r"claim|alleged|accuse|report|according to", content):
        return "media_report"
    if re.search(r"denied|refused|rejected|no comment", content):
        return "denial_response"
    return "news_report"


def parse_gdelt_article(raw_article: dict[str, Any], query: str) -> dict[str, Any] | None:
    """Convert a GDELT article to ConflictEvent dict."""
    try:
        title = raw_article.get("title", "") or ""
        url = raw_article.get("url", "") or ""
        raw_date = raw_article.get("seendate", "") or ""
        domain = raw_article.get("domain", "") or ""
        social_image = raw_article.get("socialimage", "") or ""

        if not title:
            return None

        article_date = parse_gdelt_datetime(raw_date)
        if not article_date:
            article_date = utc_now_iso()

        event_time = article_date
        published_at = article_date

        article_text = raw_article.get("text", "") or title

        actors = extract_actors(title, article_text)
        event_type = detect_event_type(title, article_text)
        risk_tags = detect_risk_tags(title, article_text)
        confidence = detect_confidence(title, article_text)
        claim_type = detect_claim_type(title, article_text)

        summary = article_text[:500].strip() if len(article_text) > 500 else article_text.strip()
        if len(article_text) > 500 and not summary.endswith("..."):
            summary += "..."

        event_id = f"GDELT-{uuid.uuid4().hex[:12].upper()}"

        locations: list[str] = []
        # 多语言 location 提取
        location_patterns = [
            ("Tehran, Iran", ["tehran", "德黑兰", "طهران"]),
            ("Natanz, Iran", ["natanz", "纳坦兹", "ناتانز"]),
            ("Strait of Hormuz", ["strait of hormuz", "霍尔木兹海峡", "مضيق هرمز"]),
            ("Persian Gulf", ["persian gulf", "波斯湾", "الخليج الفارسي", "الخلیج الفارسی"]),
            ("Baghdad, Iraq", ["baghdad", "巴格达", "بغداد"]),
            ("Damascus, Syria", ["damascus", "大马士革", "دمشق"]),
            ("Beirut, Lebanon", ["beirut", "贝鲁特", "بيروت"]),
            ("Sanaa, Yemen", ["sanaa", "萨那", "صنعاء"]),
            ("Red Sea", ["red sea", "红海", "البحر الأحمر"]),
            ("Washington, United States", ["washington", "华盛顿", "واشنطن"]),
            ("Geneva, Switzerland", ["geneva", "日内瓦", "جنيف"]),
            ("Vienna, Austria", ["vienna", "维也纳", "فيينا"]),
            ("Moscow, Russia", ["moscow", "莫斯科", "موسكو"]),
            ("Beijing, China", ["beijing", "北京", "بكين"]),
            ("Tel Aviv, Israel", ["tel aviv", "特拉维夫", "تل أبيب"]),
            ("Gaza Strip", ["gaza", "加沙", "غزة"]),
            ("Dubai, UAE", ["dubai", "迪拜", "دبي"]),
        ]
        for location, keywords in location_patterns:
            for kw in keywords:
                if kw.lower() in article_text.lower():
                    if location not in locations:
                        locations.append(location)
                    break

        targets: list[str] = []
        target_patterns = [
            ("Iranian nuclear program", ["iran", "nuclear", "uranium", "enrichment", "facility", "原子能", "核", "إيران", "نووي"]),
            ("Iranian Revolutionary Guard Corps", ["irgc", "revolutionary guard", "pasdaran", "伊斯兰革命卫队", "الحرس الثوري"]),
            ("US military personnel", ["u.s.", "usa", "american", "base", "forces", "troops", "美国", "美军", "القوات الأمريكية"]),
            ("US diplomatic facilities", ["embassy", "diplomat", "consulate", "大使馆", "外交", "السفارة"]),
            ("Oil infrastructure", ["oil", "refinery", "tanker", "油田", "炼油", "النفط"]),
            ("Israeli government", ["israeli", "government", "netanyahu", "以色列", "政府", "إسرائيلية"]),
            ("Suez Canal", ["suez", "苏伊士", "قناة السويس"]),
        ]
        for target, keywords in target_patterns:
            for kw in keywords:
                if kw.lower() in article_text.lower():
                    if target not in targets:
                        targets.append(target)
                    break

        return {
            "event_id": event_id,
            "event_type": event_type,
            "event_time": event_time,
            "published_at": published_at,
            "actors": actors,
            "targets": targets,
            "locations": locations,
            "summary": summary,
            "source_name": domain or "GDELT News",
            "source_url": url,
            "claim_type": claim_type,
            "confidence": confidence,
            "risk_tags": risk_tags,
        }
    except Exception:
        return None


def fetch_gdelt(query: str, limit: int = 50, days: int = 90, mode: str = DEFAULT_MODE) -> list[dict[str, Any]]:
    """Fetch articles from GDELT API for a given query."""
    params = {
        "query": query,
        "mode": mode,
        "maxrecords": limit,
        "format": "json",
        "sort": "DateDesc",
        "timespan": f"{days}d",
    }

    for attempt in range(MAX_RETRY):
        try:
            response = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)

            # 429 Too Many Requests - 等待后重试
            if response.status_code == 429:
                wait_time = BASE_RETRY_DELAY * (attempt + 1)
                print(f"  Rate limited (429), waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            data = response.json()

            articles = []
            if isinstance(data, dict):
                articles = data.get("articles", [])
            elif isinstance(data, list):
                articles = data

            return articles

        except requests.exceptions.Timeout:
            print(f"  Timeout on attempt {attempt + 1}/{MAX_RETRY}, retrying...")
            time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            print(f"  Request error: {e}")
            break
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  Parse error: {e}")
            break

    return []


def load_existing_event_ids(output_path: str) -> set[str]:
    """Load existing event IDs from output file for deduplication."""
    existing_ids: set[str] = set()
    existing_urls: set[str] = set()

    path = __import__("pathlib").Path(output_path)
    if not path.exists():
        return existing_ids

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if "event_id" in event:
                    existing_ids.add(event["event_id"])
                if "source_url" in event and event["source_url"]:
                    existing_urls.add(event["source_url"])
            except json.JSONDecodeError:
                continue

    return existing_ids | existing_urls


def deduplicate_events(
    events: list[dict[str, Any]],
    seen_urls: set[str],
) -> list[dict[str, Any]]:
    """Deduplicate events by source_url."""
    unique_events: list[dict[str, Any]] = []
    for event in events:
        url = event.get("source_url", "")
        if url and url in seen_urls:
            continue
        seen_urls.add(url)
        unique_events.append(event)
    return unique_events


def write_events_jsonl(events: list[dict[str, Any]], output_path: str) -> None:
    """Write events to JSONL file, one per line."""
    from pathlib import Path

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if path.exists() else "w"
    with open(path, mode, encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def run_fetch(*, days: int = 14, limit: int = 50, output: str = DEFAULT_OUTPUT) -> None:
    """Programmatic entry point for update_graph.py."""
    import time

    print(f"[GDELT] Fetching articles (days={days}, limit={limit}, output={output})...")
    seen_urls: set[str] = set()
    total_fetched = 0
    total_saved = 0

    for query in QUERIES:
        articles = fetch_gdelt(query, limit=limit, days=days, mode=DEFAULT_MODE)
        if not articles:
            print(f"[GDELT] No articles for '{query}'")
            time.sleep(15)  # GDELT免费API限制：请求间隔约15秒
            continue
        print(f"[GDELT] Fetched {len(articles)} articles for '{query}'")
        total_fetched += len(articles)

        events = [e for raw in articles if (e := parse_gdelt_article(raw, query))]
        events = deduplicate_events(events, seen_urls)
        if events:
            write_events_jsonl(events, output)
            total_saved += len(events)
        time.sleep(15)  # GDELT免费API限制：请求间隔约15秒

    print(f"[GDELT] Done. Fetched: {total_fetched}, Saved: {total_saved}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch US-Iran conflict news from GDELT and save as ConflictEvent JSONL."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="Time window in days (passed to GDELT timespan parameter)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum articles per query",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--mode",
        default=DEFAULT_MODE,
        choices=["artlist", "artimeline", "artransn", "timeline"],
        help="GDELT API output mode",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=QUERIES,
        help="Query strings (space-separated)",
    )
    args = parser.parse_args()

    print(f"Fetching GDELT articles for US-Iran conflict topics...")
    print(f"  Queries: {len(args.queries)}")
    print(f"  Days: {args.days}")
    print(f"  Limit per query: {args.limit}")
    print(f"  Mode: {args.mode}")
    print(f"  Output: {args.output}")

    seen_urls: set[str] = set()
    total_fetched = 0
    total_saved = 0

    for query in args.queries:
        print(f"\nQuerying: {query}")
        articles = fetch_gdelt(query, limit=args.limit, days=args.days, mode=args.mode)

        if not articles:
            print(f"  No articles returned for '{query}'")
            time.sleep(15)  # GDELT免费API限制
            continue

        print(f"  Fetched {len(articles)} articles")
        total_fetched += len(articles)

        events: list[dict[str, Any]] = []
        for raw in articles:
            event = parse_gdelt_article(raw, query)
            if event:
                events.append(event)

        events = deduplicate_events(events, seen_urls)
        print(f"  Converted {len(events)} events (after dedup)")

        if events:
            write_events_jsonl(events, args.output)
            total_saved += len(events)
            print(f"  Appended to {args.output}")

        time.sleep(15)  # GDELT免费API限制

    print(f"\n{'=' * 50}")
    print(f"Done. Fetched: {total_fetched}, Saved: {total_saved}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
