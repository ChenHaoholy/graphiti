"""
GDELT data adapter: fetch US-Iran conflict news via GDELT API and convert to ConflictEvent JSONL.

Uses GDELT 2.0 API (https://api.gdeltproject.org/api/v2/doc/doc).
Article body is fetched via trafilatura from the original URL.
No Neo4j access, no LLM API calls.
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
    import trafilatura
    _HAS_TRAFILATURA = True
except ImportError:
    _HAS_TRAFILATURA = False

BODY_FETCH_TIMEOUT = 15   # 每篇正文抓取超时秒数
BODY_FETCH_DELAY = 1.0    # 抓取正文的间隔（秒），避免对目标站限速

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
    # 中伊关系
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
BASE_RETRY_DELAY = 15


def fetch_article_body(url: str) -> str:
    """Fetch full article body from URL using trafilatura. Returns empty string on failure."""
    if not _HAS_TRAFILATURA or not url:
        return ""
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return ""
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        return (text or "").strip()
    except Exception:
        return ""


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


# 日期模式匹配优先级：从标题、正文、URL提取事件发生时间
# 注意：GDELT Doc API 只有 seendate（索引时间），没有原生的事件时间字段
_EVENT_DATE_PATTERNS = [
    # URL中的日期: /20260411.htm 或 20260411
    (re.compile(r'/(\d{8})\.htm', re.IGNORECASE), None),
    # ISO 完整格式: 2026-05-12T03:30:00Z
    (re.compile(r'\b(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\b'), None),
    # ISO 日期: 2026-05-12
    (re.compile(r'\b(\d{4}-\d{2}-\d{2})\b'), None),
    # "on May 12, 2026" / "May 12, 2026" / "base, January 15, 2026"
    (re.compile(r'(?<![/\d])(\w{3,9})\s+(\d{1,2}),?\s+(\d{4})\b', re.IGNORECASE), None),
    # "12 May 2026" / "5 May 2026" (day before month)
    (re.compile(r'(?<![/\d])(\d{1,2})\s+(\w{3,9})\s+(\d{4})\b', re.IGNORECASE), None),
    # 中文: 2026年5月12日 03:30 / 2026年5月12日
    (re.compile(r'(\d{4})年(\d{1,2})月(\d{1,2})日[日\s]*(\d{1,2}:\d{2})?'), None),
    # 中文无年份: 4月10日 / 4月10日电  ← 需结合 fallback 年份
    (re.compile(r'(\d{1,2})月(\d{1,2})日电?'), None),
]

_MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def _parse_8digit_date(text: str) -> datetime | None:
    """Parse YYYYMMDD from URL like /20260411.htm."""
    try:
        return datetime(int(text[:4]), int(text[4:6]), int(text[6:8]), 0, 0, tzinfo=timezone.utc)
    except ValueError:
        return None


def _try_parse_date(match: re.Match, fallback_year: int) -> datetime | None:
    """Convert a regex match to datetime. Returns None on failure."""
    text = match.group(0)

    # URL日期: /20260411.htm → 取匹配的8位数字
    if '.ht' in text.lower() and re.match(r'^/\d{8}\.htm', text, re.IGNORECASE):
        return _parse_8digit_date(match.group(1))

    # ISO 格式
    iso_match = re.search(r'\d{4}-\d{2}-\d{2}', text)
    if iso_match:
        iso_str = iso_match.group(0)
        if 'T' in text or 'Z' in text or '+' in text:
            try:
                dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
                return dt.astimezone(timezone.utc)
            except ValueError:
                pass
        try:
            return datetime.fromisoformat(iso_str).replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    groups = match.groups()
    if not groups:
        return None

    # 中文无年份: 4月10日 / 4月10日电
    if len(groups) == 2 and re.match(r'^\d{1,2}$', groups[0]) and re.match(r'^\d{1,2}$', groups[1]):
        try:
            month, day = int(groups[0]), int(groups[1])
            return datetime(fallback_year, month, day, 0, 0, tzinfo=timezone.utc)
        except ValueError:
            return None

    if len(groups) >= 3:
        # 中文完整: 2026年5月12日 [03:30]
        if "年" in text:
            try:
                year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                hour, minute = 0, 0
                if len(groups) >= 4 and groups[3]:
                    parts = groups[3].split(":")
                    hour = int(parts[0])
                    minute = int(parts[1]) if len(parts) > 1 else 0
                return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
            except ValueError:
                pass

        # 英文: Month Day, Year 或 Day Month Year
        month_str = groups[0] if not groups[0].isdigit() else groups[1]
        day_str = groups[1] if not groups[0].isdigit() else groups[0]
        year_str = groups[2]
        month_num = _MONTH_MAP.get(month_str.lower())
        if month_num is None:
            return None
        try:
            return datetime(int(year_str), month_num, int(day_str), 0, 0, tzinfo=timezone.utc)
        except ValueError:
            return None

    return None


def extract_event_time(title: str, body: str, fallback: str, *, url: str = "") -> str:
    """Extract event occurrence time from article title/body/URL.

    Falls back to `fallback` (published_at / seendate) if nothing found.
    Strategy: scan URL first (often has date), then title, then body.
    """
    # 从 fallback 提取年份作为无年份日期的默认值
    try:
        fb_dt = datetime.fromisoformat(fallback.replace("Z", "+00:00"))
        fallback_year = fb_dt.year
    except (ValueError, AttributeError):
        fallback_year = datetime.now(timezone.utc).year

    # URL → title → body 优先级递减
    search_text = f"{url}\n{title}\n{body[:500]}"
    now = datetime.now(timezone.utc)

    for pattern, _ in _EVENT_DATE_PATTERNS:
        for match in pattern.finditer(search_text):
            dt = _try_parse_date(match, fallback_year)
            if dt and 2000 <= dt.year <= now.year + 1:
                return dt.isoformat().replace("+00:00", "Z")

    return fallback


def detect_event_type(title: str, text: str) -> str:
    """Rule-based event type detection with expanded keyword coverage."""
    content = (title + " " + text).lower()

    if re.search(r"sanction|treasury|economic penalty|export control|embargo|freeze asset|financial restriction", content):
        return "sanctions_action"
    if re.search(r"iaea|nuclear|uranium|enrichment|centrifuge|natanz|fordow|reactor|nonproliferation|npt|atomic", content):
        return "nuclear_iaea"
    if re.search(r"attack|missile|drone|strike|military|bomb|assassination|airstrike|air strike|explosion|rocket|artillery|shoot down|intercept|raid|offensive|troops|soldier|combat|casualt", content):
        return "military_strike"
    if re.search(r"hormuz|shipping lane|oil tanker|persian gulf|maritime|vessel|strait|seize|blockade|naval|coast guard|port", content):
        return "maritime_incident"
    if re.search(r"proxy|militia|hezbollah|houthi|kataib|pmu|popular mobilization|islamic resistance|backed group|armed group", content):
        return "proxy_attack_claim"
    if re.search(r"talk|negotiation|diplomacy|mediation|security council|ceasefire|peace deal|envoy|diplomat|foreign minister|state department|warning|condemn|threat|demand|ultimatum|statement|sanction threat|response|retaliat", content):
        return "diplomatic_warning"
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


def parse_gdelt_article(raw_article: dict[str, Any], query: str, prefetched_body: str = "") -> dict[str, Any] | None:
    """Convert a GDELT article to ConflictEvent dict.
    
    prefetched_body: already-fetched full article text (pass empty string to skip body fetch).
    """
    try:
        title = raw_article.get("title", "") or ""
        url = raw_article.get("url", "") or ""
        raw_date = raw_article.get("seendate", "") or ""
        domain = raw_article.get("domain", "") or ""

        if not title:
            return None

        article_date = parse_gdelt_datetime(raw_date)
        if not article_date:
            article_date = utc_now_iso()

        # 正文优先级：传入正文 > GDELT text字段(通常为空) > fallback标题
        body_from_gdelt = raw_article.get("text", "") or ""
        article_text = prefetched_body or body_from_gdelt or title

        published_at = article_date                        # seendate = GDELT索引/收录时间 → 作为"发布时间"
        event_time = extract_event_time(title, article_text, article_date, url=url)  # 从URL/正文提取事件发生时间

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


def _normalize_gdelt_date(value: str) -> str:
    """Convert flexible date string to GDELT's YYYYMMDDHHMMSS format."""
    # Already 14-char GDELT format
    if re.match(r'^\d{14}$', value):
        return value
    # YYYYMMDD
    if re.match(r'^\d{8}$', value):
        return value + "000000"
    # ISO-like: try parsing
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.strftime("%Y%m%d%H%M%S")
    except ValueError:
        return value  # pass through, let GDELT reject it


def _fix_end_exclusive(end_date: str | None) -> str | None:
    """GDELT enddatetime is EXCLUSIVE: --end 20260410 means seendate < 2026-04-10 00:00:00 (excludes Apr 10 entirely).

    When user passes a bare YYYYMMDD end date, shift it forward by 1 day so the target
    date is included. Skip shifting if the input already has HHMMSS > 000000 (explicit
    end-of-day intent) or is already in GDELT 14-char format.
    """
    if not end_date:
        return None
    if re.match(r'^\d{14}$', end_date):
        return end_date
    m = re.match(r'^(\d{4})(\d{2})(\d{2})$', end_date)
    if not m:
        return end_date
    year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
    from calendar import monthrange
    _, last_day = monthrange(year, month)
    if day < last_day:
        day += 1
    else:
        month += 1
        day = 1
        if month > 12:
            month = 1
            year += 1
    return f"{year:04d}{month:02d}{day:02d}"


def fetch_gdelt(
    query: str,
    limit: int = 50,
    days: int = 90,
    mode: str = DEFAULT_MODE,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch articles from GDELT API for a given query.

    Use `start_date`/`end_date` for absolute range (YYYYMMDDHHMMSS or YYYYMMDD),
    otherwise fall back to relative `days` from now.
    """
    params: dict[str, Any] = {
        "query": query,
        "mode": mode,
        "maxrecords": limit,
        "format": "json",
        "sort": "DateDesc",
    }

    # 绝对时间段优先
    if start_date:
        params["startdatetime"] = _normalize_gdelt_date(start_date)
    # GDELT enddatetime is EXCLUSIVE: shift bare YYYYMMDD end date +1 day automatically
    if end_date:
        params["enddatetime"] = _normalize_gdelt_date(_fix_end_exclusive(end_date))
    if start_date or end_date:
        params.pop("timespan", None)
    else:
        params["timespan"] = f"{days}d"

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


def load_existing_urls(output_path: str) -> set[str]:
    """Load existing source URLs from output file for cross-run dedup."""
    urls: set[str] = set()
    path = __import__("pathlib").Path(output_path)
    if not path.exists():
        return urls
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                url = event.get("source_url", "")
                if url:
                    urls.add(url)
            except json.JSONDecodeError:
                continue
    return urls


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


def run_fetch(
    *,
    days: int = 14,
    limit: int = 50,
    output: str = DEFAULT_OUTPUT,
    fetch_body: bool = True,
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    """Programmatic entry point for update_graph.py.
    
    fetch_body=True: fetch full article body via trafilatura (slower but richer).
    fetch_body=False: use title only (fast, legacy behavior).
    """
    import time

    actual_end = _fix_end_exclusive(end_date) if end_date else None
    date_range = f"{start_date or '?'} ~ {actual_end or end_date or '?'}" if (start_date or end_date) else f"最近{days}天"
    print(f"[GDELT] Fetching articles (range={date_range}, limit={limit}, output={output}, fetch_body={fetch_body})...")
    if fetch_body and not _HAS_TRAFILATURA:
        print("[GDELT] WARNING: trafilatura not installed, falling back to title-only mode.")
    seen_urls: set[str] = load_existing_urls(output)   # 跨运行去重：已有URL跳过
    if seen_urls:
        print(f"[GDELT] Loaded {len(seen_urls)} existing URLs for dedup.")
    total_fetched = 0
    total_saved = 0

    for query in QUERIES:
        articles = fetch_gdelt(query, limit=limit, days=days, mode=DEFAULT_MODE,
                                start_date=start_date, end_date=end_date)
        if not articles:
            print(f"[GDELT] No articles for '{query}'")
            time.sleep(15)  # GDELT免费API限制：请求间隔约15秒
            continue
        print(f"[GDELT] Fetched {len(articles)} articles for '{query}'")
        total_fetched += len(articles)

        events = []
        for raw in articles:
            body = ""
            if fetch_body and _HAS_TRAFILATURA:
                url = raw.get("url", "")
                if url:
                    body = fetch_article_body(url)
                    if body:
                        print(f"  [body] {len(body)}chars from {raw.get('domain','?')}")
                    time.sleep(BODY_FETCH_DELAY)
            e = parse_gdelt_article(raw, query, prefetched_body=body)
            if e:
                events.append(e)

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
    parser.add_argument(
        "--no-fetch-body",
        action="store_true",
        default=False,
        help="Disable full-text fetching via trafilatura (title-only mode, faster)",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Start date (YYYYMMDD or YYYYMMDDHHMMSS or ISO). Overrides --days.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date (YYYYMMDD or YYYYMMDDHHMMSS or ISO). Overrides --days.",
    )
    args = parser.parse_args()

    fetch_body = not args.no_fetch_body

    print(f"Fetching GDELT articles for US-Iran conflict topics...")
    print(f"  Queries: {len(args.queries)}")
    if args.start or args.end:
        actual_end = _fix_end_exclusive(args.end) if args.end else None
        print(f"  Range: {args.start or '?'} ~ {actual_end or args.end or '?'} (absolute, end is inclusive)")
    else:
        print(f"  Days: {args.days} (relative from now)")
    print(f"  Limit per query: {args.limit}")
    print(f"  Output: {args.output}")
    print(f"  Fetch full body: {fetch_body} (trafilatura={'yes' if _HAS_TRAFILATURA else 'NOT INSTALLED'})")
    seen_urls: set[str] = load_existing_urls(args.output)   # 跨运行去重
    if seen_urls:
        print(f"[GDELT] Loaded {len(seen_urls)} existing URLs for dedup.")
    total_fetched = 0
    total_saved = 0

    for query in args.queries:
        print(f"\nQuerying: {query}")
        articles = fetch_gdelt(query, limit=args.limit, days=args.days, mode=args.mode,
                                start_date=args.start, end_date=args.end)

        if not articles:
            print(f"  No articles returned for '{query}'")
            time.sleep(15)
            continue

        print(f"  Fetched {len(articles)} articles")
        total_fetched += len(articles)

        events: list[dict[str, Any]] = []
        for raw in articles:
            body = ""
            if fetch_body and _HAS_TRAFILATURA:
                url = raw.get("url", "")
                if url:
                    body = fetch_article_body(url)
                    if body:
                        print(f"    [body] {len(body)}chars <- {raw.get('domain','?')}")
                    time.sleep(BODY_FETCH_DELAY)
            event = parse_gdelt_article(raw, query, prefetched_body=body)
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
